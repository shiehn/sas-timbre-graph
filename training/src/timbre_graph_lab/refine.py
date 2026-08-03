"""Closed-loop refinement: search against real renders instead of predicting.

Measured in C13b: the Jacobian gets a direction roughly right but cannot
predict what several interacting parameters do together, because Surge's
controls combine multiplicatively. Open-loop achieved a median 0.048
achieved-vs-requested cosine; seeding from the same solve and then refining
against actual renders reached 0.186 with 25 renders. A render is ~30 ms, so
searching is simply cheaper than being clever.

Structure:
  1. seed      - the damped least-squares solve from the measured Jacobian
  2. line search - scale that seed direction; fixes the magnitude error that
                   linear extrapolation always makes
  3. coordinate refine - adaptive Gaussian tweaks on live parameters, shrinking
                   the step when progress stalls

`measure` is injected — a closure over a Surge worker in production, a plain
function in tests — so the search logic is verifiable without audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from timbre_graph_lab.solver import MAX_PARAM_STEP, TRUST_RADIUS, solve_step, unit

# Achieved movement is rewarded along the requested axis and mildly penalised
# off it: a follower that lands somewhere else is not following the gesture,
# but demanding perfect purity would reject most of what a synth can do.
OFF_AXIS_PENALTY = 0.3
LINE_SCALES = (0.35, 0.7, 1.0, 1.5, 2.2, 3.0)

Measure = Callable[[np.ndarray], "np.ndarray | None"]


@dataclass
class RefineResult:
    dx: np.ndarray
    achieved: np.ndarray
    score: float
    cosine: float
    projection: float
    n_renders: int
    history: list[float] = field(default_factory=list)

    @property
    def moved(self) -> bool:
        """Whether any move beat standing still.

        Declining is a legitimate answer, not a failure: a kick asked to get
        "brighter" genuinely cannot, and the proposal's own policy is to fall
        back to no movement rather than invent one. Reporting must separate
        "declined" from "moved badly" or the two average together into
        nonsense.
        """
        return bool(np.any(np.abs(self.dx) > 1e-9))


def score_move(achieved: np.ndarray, axis: np.ndarray) -> tuple[float, float, float]:
    """(score, cosine, projection) of an achieved timbre delta against an axis."""
    if achieved is None:
        return -np.inf, 0.0, 0.0
    n = float(np.linalg.norm(achieved))
    if n < 1e-12:
        return 0.0, 0.0, 0.0
    proj = float(achieved @ axis)
    off = float(np.linalg.norm(achieved - proj * axis))
    return proj - OFF_AXIS_PENALTY * off, proj / n, proj


def _clamp(dx: np.ndarray, max_param_step: float, trust_radius: float) -> np.ndarray:
    dx = np.clip(dx, -max_param_step, max_param_step)
    n = float(np.linalg.norm(dx))
    return dx * (trust_radius / n) if n > trust_radius else dx


def refine(
    measure: Measure,
    J: np.ndarray,
    usable: np.ndarray,
    axis: np.ndarray,
    budget: int = 40,
    seed: int = 0,
    max_param_step: float = MAX_PARAM_STEP,
    trust_radius: float = TRUST_RADIUS,
) -> RefineResult:
    """Find the parameter move that best travels along `axis`, by measuring."""
    axis = unit(np.asarray(axis, dtype=np.float64))
    live = np.flatnonzero(np.asarray(usable, bool))
    n_params = J.shape[0]
    rng = np.random.default_rng(seed)

    best_dx = np.zeros(n_params)
    best_achieved = np.zeros(J.shape[1])
    best_score, best_cos, best_proj = 0.0, 0.0, 0.0
    used = 0
    history: list[float] = []

    def try_move(dx: np.ndarray) -> float:
        nonlocal best_dx, best_achieved, best_score, best_cos, best_proj, used
        used += 1
        got = measure(dx)
        s, c, p = score_move(got, axis)
        history.append(s if np.isfinite(s) else 0.0)
        if s > best_score:
            best_dx, best_achieved = dx.copy(), (got if got is not None else best_achieved)
            best_score, best_cos, best_proj = s, c, p
        return s

    if live.size == 0 or budget <= 0:
        return RefineResult(best_dx, best_achieved, 0.0, 0.0, 0.0, 0, history)

    # 1) seed direction from the measured Jacobian
    direction = solve_step(J, axis, usable, trust_radius=1e9, max_param_step=1e9)
    dn = float(np.linalg.norm(direction))
    if dn > 1e-12:
        direction /= dn
        # 2) line search: the linear model's direction with a measured magnitude
        for s in LINE_SCALES:
            if used >= budget:
                break
            try_move(_clamp(direction * s * trust_radius, max_param_step, trust_radius))

    # 3) adaptive coordinate refinement around the best move so far
    sigma = 0.05
    stall = 0
    while used < budget:
        cand = best_dx.copy()
        k = int(min(live.size, max(1, rng.integers(1, 4))))
        for i in rng.choice(live, size=k, replace=False):
            cand[i] += rng.normal(0.0, sigma)
        before = best_score
        try_move(_clamp(cand, max_param_step, trust_radius))
        if best_score > before:
            stall = 0
        else:
            stall += 1
            if stall >= 5:            # progress dried up: search finer
                sigma = max(sigma * 0.5, 0.005)
                stall = 0

    return RefineResult(best_dx, best_achieved, best_score, best_cos, best_proj,
                        used, history)


def make_surge_measure(worker, response, avg: int = 1) -> Measure:
    """A `measure` closure that renders through Surge for one anchor.

    The anchor's baseline descriptors are rendered ONCE and cached. The naive
    version re-rendered them on every evaluation, which made each measurement
    cost ~4 renders instead of ~1 and dominated the search's wall time.
    """
    from timbre_graph_lab.descriptors import standardize
    from timbre_graph_lab.probes import get_probe
    from timbre_graph_lab.worker import qc_audio

    probe = get_probe(response.role, "short")
    worker.restore_baseline()
    z0 = worker.render_descriptors(probe, k=avg)

    def measure(dx: np.ndarray):
        delta = {p: float(d) for p, d in zip(response.param_names, dx)
                 if abs(d) > 1e-9}
        if not delta:
            return np.zeros_like(z0)
        worker.apply_delta(delta)
        audio = worker.render(probe)
        if not qc_audio(audio).ok:
            worker.restore_baseline()
            return None
        z1 = worker.render_descriptors(probe, k=avg) if avg > 1 else None
        from timbre_graph_lab.descriptors import extract_descriptors
        if z1 is None:
            z1 = extract_descriptors(audio, worker.cfg.sample_rate)
        worker.restore_baseline()
        return standardize(z1) - standardize(z0)

    return measure


def refine_all(
    worker,
    responses: dict,
    paths: dict[str, str],
    axis_by_role: dict[str, np.ndarray],
    budget: int = 40,
    avg: int = 1,
    locked: set[str] | None = None,
) -> dict[str, RefineResult]:
    """Refine every role along its own axis (see C13c: axes are role-specific)."""
    locked = locked or set()
    out: dict[str, RefineResult] = {}
    for role, resp in responses.items():
        if role in locked or role not in axis_by_role:
            continue
        if not worker.load_preset(paths[role]):
            continue
        out[role] = refine(
            make_surge_measure(worker, resp, avg=avg),
            resp.J, resp.usable, axis_by_role[role], budget=budget,
        )
    return out
