"""Translate one perceptual gesture into parameter moves on every synth.

Pure numpy — no Surge, no torch — so it is fast, testable, and small enough to
re-implement in TypeScript inside the plugin if the panel ever needs to solve
live rather than replay a precomputed graph.

The maths. Around an anchor, a synth's measured Jacobian J maps a small
parameter step dx to a timbre step dz:  dz ~= J^T dx. Asking a follower to
reproduce a target timbre direction t is then a least-squares problem, damped
so the answer prefers small moves and stays inside the trust radius:

    minimise  ||J^T dx - t||^2  +  lam ||dx||^2

which has the closed form  dx = (J J^T + lam I)^-1 J t.

Damping is what keeps this musical: the system is wildly under-determined
(many parameter combinations produce a similar timbre move), and the minimum
norm solution is the one that disturbs the patch least — exactly the locality
principle the design rests on. `usable` masks out controls whose measured
response was below the noise floor, so the solver never leans on a dead knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAMPING = 1e-2
MAX_PARAM_STEP = 0.15   # per-parameter clamp in raw space
TRUST_RADIUS = 0.25     # L2 cap on the whole move, in raw space


def solve_step(
    J: np.ndarray,
    target: np.ndarray,
    usable: np.ndarray | None = None,
    damping: float = DAMPING,
    max_param_step: float = MAX_PARAM_STEP,
    trust_radius: float = TRUST_RADIUS,
) -> np.ndarray:
    """Smallest parameter move whose predicted timbre change approaches `target`.

    J is (n_params, n_features) as measured by the prober.
    """
    n_params = J.shape[0]
    dx = np.zeros(n_params, dtype=np.float64)
    mask = np.ones(n_params, dtype=bool) if usable is None else np.asarray(usable, bool)
    if not mask.any():
        return dx

    Jm = J[mask]                                  # (k, f)
    G = Jm @ Jm.T + damping * np.eye(Jm.shape[0])  # (k, k)
    step = np.linalg.solve(G, Jm @ target)

    step = np.clip(step, -max_param_step, max_param_step)
    norm = float(np.linalg.norm(step))
    if norm > trust_radius:
        step *= trust_radius / norm
    dx[mask] = step
    return dx


def predicted_delta(J: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Timbre change this parameter move is expected to produce."""
    return J.T @ dx


def morph_axes(J: np.ndarray, usable: np.ndarray | None = None, k: int = 2) -> np.ndarray:
    """The k most controllable timbre directions for this patch.

    Right singular vectors of the (masked) Jacobian: directions in descriptor
    space that this synth's own parameters can actually reach. Using them as
    the dial axes means the control surface is expressive by construction
    rather than by guesswork — a dial pointing somewhere the synth cannot go
    would simply do nothing.
    """
    mask = np.ones(J.shape[0], dtype=bool) if usable is None else np.asarray(usable, bool)
    if not mask.any():
        return np.zeros((k, J.shape[1]))
    _, _, Vt = np.linalg.svd(J[mask], full_matrices=False)
    axes = Vt[:k]
    if axes.shape[0] < k:  # pad if the Jacobian is rank-deficient
        axes = np.vstack([axes, np.zeros((k - axes.shape[0], J.shape[1]))])
    return axes


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass
class MorphPoint:
    """One position on the control surface: a parameter snapshot per role."""

    control: tuple[float, ...]
    params: dict[str, np.ndarray]        # role -> absolute raw values
    predicted_dz: dict[str, np.ndarray]  # role -> expected timbre change


def build_trajectory(
    responses: dict,
    axis: np.ndarray,
    coupling: dict[str, float] | None = None,
    span: float = 1.0,
    n_points: int = 11,
    locked: set[str] | None = None,
) -> list[MorphPoint]:
    """Precompute the dial: solve every role at each control position.

    `axis` is a direction in descriptor space (typically from `morph_axes`).
    `coupling` scales how strongly each role follows it — the 6x6 coupling
    policy of the proposal, reduced to a per-role gain for a single axis.
    Locked roles are frozen: their parameters simply do not move, which is the
    per-track unlink the panel exposes.
    """
    axis = unit(np.asarray(axis, dtype=np.float64))
    coupling = coupling or {}
    locked = locked or set()
    points: list[MorphPoint] = []

    for t in np.linspace(-span, span, n_points):
        params: dict[str, np.ndarray] = {}
        preds: dict[str, np.ndarray] = {}
        for role, r in responses.items():
            if role in locked or abs(t) < 1e-12:
                params[role] = r.baseline.copy()
                preds[role] = np.zeros_like(r.z0)
                continue
            gain = coupling.get(role, 1.0)
            # Scale the target by this patch's own reachable magnitude so a
            # gentle synth is not asked for a move it cannot make.
            reach = float(np.linalg.norm(r.J[np.asarray(r.usable, bool)])) or 1.0
            target = axis * (t * gain * reach / max(np.sqrt(r.J.shape[0]), 1))
            dx = solve_step(r.J, target, r.usable)
            params[role] = np.clip(r.baseline + dx, 0.0, 1.0)
            preds[role] = predicted_delta(r.J, dx)
        points.append(MorphPoint(control=(float(t),), params=params, predicted_dz=preds))
    return points


def leader_follower(
    responses: dict,
    leader: str,
    leader_dx: np.ndarray,
    coupling: dict[str, float] | None = None,
    locked: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """A leader's parameter edit -> every follower's own parameter move.

    This is the "turn one knob, the others respond" path. Followers are NOT
    given the leader's parameter change: the leader's *perceptual* delta is
    computed first, then each follower solves for whatever move in its own
    parameter space produces that direction. A cutoff sweep on the lead can
    come out as an envelope move on the pad — translation, not knob-copying.
    """
    coupling = coupling or {}
    locked = locked or set()
    lead = responses[leader]
    dz = predicted_delta(lead.J, np.asarray(leader_dx, dtype=np.float64))

    out: dict[str, np.ndarray] = {leader: np.asarray(leader_dx, dtype=np.float64)}
    for role, r in responses.items():
        if role == leader:
            continue
        if role in locked:
            out[role] = np.zeros(r.J.shape[0])
            continue
        scale = coupling.get(role, 1.0)
        out[role] = solve_step(r.J, dz * scale, r.usable)
    return out
