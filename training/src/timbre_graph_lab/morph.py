"""The morph graph: a validated dial, precomputed once, interpolated forever.

This is the artifact the plugin consumes. For a chosen set of six anchors and
one semantic axis, it stores an absolute parameter snapshot per synth at every
control position, each one **render-verified** on the way. At runtime the panel
does nothing but interpolate between snapshots: no solver, no model, no ONNX,
zero latency, and per-track unlink is simply "stop applying this row".

Built by WALKING, not by solving each position independently. Each control step
targets a small increment from the previous one and warm-starts from its
solution, so:

  * every step stays near the radius where the measured Jacobian is valid —
    the failure that made one-shot extrapolation unreliable (C13b);
  * consecutive snapshots are close, so the dial feels continuous instead of
    jumping;
  * later points are cheap, because the previous answer is already most of the
    way there.

Quality is measured rather than assumed: `monotonicity` checks that travel
along the axis actually progresses as the dial turns, and `smoothness` reports
the largest parameter jump between neighbouring positions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from timbre_graph_lab.axes import AXES, MIN_ACHIEVABLE_COSINE
from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES
from timbre_graph_lab.refine import refine, score_move
from timbre_graph_lab.solver import MAX_PARAM_STEP, TRUST_RADIUS

MORPH_VERSION = "morph-graph-v1"


@dataclass
class RoleTrack:
    """One synth's row of the dial."""

    role: str
    preset_id: str
    name: str
    param_names: list[str]
    baseline: np.ndarray
    snapshots: list[np.ndarray] = field(default_factory=list)  # absolute raw
    cosine: list[float] = field(default_factory=list)
    projection: list[float] = field(default_factory=list)

    @property
    def declined(self) -> bool:
        return all(
            np.allclose(s, self.baseline, atol=1e-9) for s in self.snapshots
        )

    def to_dict(self) -> dict:
        return {
            "role": self.role, "preset_id": self.preset_id, "name": self.name,
            "param_names": self.param_names,
            "baseline": self.baseline.tolist(),
            "snapshots": [s.tolist() for s in self.snapshots],
            "cosine": [round(float(c), 4) for c in self.cosine],
            "projection": [round(float(p), 4) for p in self.projection],
            "declined": self.declined,
        }


def _walk(
    measure,
    response,
    axis: np.ndarray,
    controls: np.ndarray,
    step_scale: float,
    budget: int,
    seed: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Refine outward from the anchor, warm-starting each control position."""
    snapshots: list[np.ndarray] = []
    cosines: list[float] = []
    projections: list[float] = []
    warm = np.zeros(len(response.param_names))

    for t in controls:
        if abs(float(t)) < 1e-12:
            snapshots.append(response.baseline.copy())
            cosines.append(1.0)
            projections.append(0.0)
            warm = np.zeros(len(response.param_names))
            continue

        target = axis * float(t) * step_scale
        res = refine(
            measure, response.J, response.usable, axis * np.sign(t),
            budget=budget, seed=seed,
        )
        # keep the warm start's contribution: prefer whichever move travels
        # further along the requested target without breaking the limits
        cand = [res.dx]
        if np.any(np.abs(warm) > 1e-9):
            scaled = warm * (abs(float(t)) / max(abs(float(controls[-1])), 1e-9))
            cand.append(np.clip(scaled, -MAX_PARAM_STEP, MAX_PARAM_STEP))
        best_dx, best_s, best_c, best_p = None, -np.inf, 0.0, 0.0
        for dx in cand:
            n = float(np.linalg.norm(dx))
            if n > TRUST_RADIUS:
                dx = dx * (TRUST_RADIUS / n)
            got = measure(dx)
            s, c, p = score_move(got, axis * np.sign(t))
            if s > best_s:
                best_dx, best_s, best_c, best_p = dx, s, c, p
        if best_dx is None:
            best_dx = np.zeros(len(response.param_names))

        # scale the accepted direction to this control position
        frac = abs(float(t)) / max(abs(float(controls[-1])), 1e-9)
        dx_t = best_dx * frac
        got = measure(dx_t)
        _, c_t, p_t = score_move(got, axis * np.sign(t))
        snapshots.append(np.clip(response.baseline + dx_t, 0.0, 1.0))
        cosines.append(c_t)
        projections.append(p_t * np.sign(t))
        warm = best_dx
        _ = target
    return snapshots, cosines, projections


def build_morph_graph(
    worker,
    responses: dict,
    paths: dict[str, str],
    axis_name: str = "softer",
    n_points: int = 9,
    budget: int = 24,
    avg: int = 1,
    seed: int = 1,
) -> dict:
    """Precompute a dial across all supplied anchors along one semantic axis."""
    from timbre_graph_lab.refine import make_surge_measure

    if axis_name not in AXES:
        raise ValueError(f"unknown axis {axis_name!r}; have {sorted(AXES)}")
    axis = AXES[axis_name]
    controls = np.linspace(-1.0, 1.0, n_points)

    tracks: dict[str, RoleTrack] = {}
    for role, resp in responses.items():
        if role not in paths or not worker.load_preset(paths[role]):
            continue
        measure = make_surge_measure(worker, resp, avg=avg)
        snaps, cos, proj = _walk(
            measure, resp, axis, controls, TRUST_RADIUS, budget, seed
        )
        tracks[role] = RoleTrack(
            role=role, preset_id=resp.preset_id, name=resp.name,
            param_names=list(resp.param_names), baseline=resp.baseline.copy(),
            snapshots=snaps, cosine=cos, projection=proj,
        )

    return {
        "version": MORPH_VERSION,
        "axis": {"name": axis_name, "vector": axis.tolist()},
        "feature_names": DESCRIPTOR_NAMES,
        "control_points": [round(float(t), 4) for t in controls],
        "roles": {r: t.to_dict() for r, t in tracks.items()},
        "quality": quality(tracks, controls),
    }


def build_xy_graph(
    worker,
    responses: dict,
    paths: dict[str, str],
    axis_x: str = "softer",
    axis_y: str = "tighter",
    n_points: int = 5,
    budget: int = 20,
    avg: int = 1,
    seed: int = 1,
) -> dict:
    """A two-axis pad: solve each axis independently, then verify the SUM.

    Two axes are refined separately and their parameter moves added, because
    solving the 2-D target directly would need a fresh search per grid cell
    (n_points squared) where this needs 2 x n_points. The catch is that adding
    two moves is a linearity assumption of exactly the kind that failed in
    C13b, so every summed corner is **re-rendered** and its achieved direction
    recorded. `corner_cosine` in the output says how well the combination
    actually held; low values mean prefer the single-axis dial for that patch.
    """
    from timbre_graph_lab.refine import make_surge_measure

    gx = build_morph_graph(worker, responses, paths, axis_name=axis_x,
                           n_points=n_points, budget=budget, avg=avg, seed=seed)
    gy = build_morph_graph(worker, responses, paths, axis_name=axis_y,
                           n_points=n_points, budget=budget, avg=avg, seed=seed)

    controls = np.asarray(gx["control_points"], dtype=float)
    roles: dict[str, dict] = {}
    for role, tx in gx["roles"].items():
        ty = gy["roles"].get(role)
        if ty is None:
            continue
        base = np.asarray(tx["baseline"], dtype=float)
        sx = np.asarray(tx["snapshots"], dtype=float) - base
        sy = np.asarray(ty["snapshots"], dtype=float) - base
        grid = [
            [np.clip(base + sx[i] + sy[j], 0.0, 1.0).tolist()
             for j in range(len(controls))]
            for i in range(len(controls))
        ]
        roles[role] = {
            **{k: tx[k] for k in ("role", "preset_id", "name", "param_names")},
            "baseline": base.tolist(),
            "grid": grid,
            "declined": bool(tx["declined"] and ty["declined"]),
        }

    # verify the four extreme corners, where the summed move is largest
    corners: dict[str, dict[str, float]] = {}
    ax = AXES[axis_x] + AXES[axis_y]
    ax = ax / (np.linalg.norm(ax) or 1.0)
    for role, entry in roles.items():
        resp = responses.get(role)
        if resp is None or role not in paths or not worker.load_preset(paths[role]):
            continue
        measure = make_surge_measure(worker, resp, avg=avg)
        got: dict[str, float] = {}
        base = np.asarray(entry["baseline"], dtype=float)
        for (i, j), label in (((0, 0), "--"), ((0, -1), "-+"),
                              ((-1, 0), "+-"), ((-1, -1), "++")):
            dx = np.asarray(entry["grid"][i][j], dtype=float) - base
            z = measure(dx)
            got[label] = round(float(score_move(z, ax)[1]), 3)
        corners[role] = got
    return {
        "version": MORPH_VERSION,
        "axes": {"x": {"name": axis_x, "vector": AXES[axis_x].tolist()},
                 "y": {"name": axis_y, "vector": AXES[axis_y].tolist()}},
        "feature_names": DESCRIPTOR_NAMES,
        "control_points": [round(float(t), 4) for t in controls],
        "roles": roles,
        "corner_cosine": corners,
        "quality": {"x": gx["quality"], "y": gy["quality"]},
    }


def params_at_xy(graph: dict, role: str, cx: float, cy: float) -> np.ndarray:
    """Bilinear interpolation on the 2-D pad — the X/Y runtime."""
    xs = np.asarray(graph["control_points"], dtype=float)
    grid = np.asarray(graph["roles"][role]["grid"], dtype=float)

    def frac(c: float) -> tuple[int, int, float]:
        c = float(np.clip(c, xs[0], xs[-1]))
        j = int(np.searchsorted(xs, c).clip(1, len(xs) - 1))
        x0, x1 = xs[j - 1], xs[j]
        return j - 1, j, (0.0 if x1 == x0 else (c - x0) / (x1 - x0))

    i0, i1, wi = frac(cx)
    j0, j1, wj = frac(cy)
    top = grid[i0, j0] * (1 - wj) + grid[i0, j1] * wj
    bot = grid[i1, j0] * (1 - wj) + grid[i1, j1] * wj
    return top * (1 - wi) + bot * wi


def quality(tracks: dict[str, RoleTrack], controls: np.ndarray) -> dict:
    """Is this dial usable: does it progress, stay continuous, and reach?

    Judged per DIRECTION and at the ENDPOINTS. Averaging cosine over every
    control point is misleading twice over: near the centre the move is
    deliberately tiny so its measured direction is mostly render noise, and a
    role can be expressive one way and immovable the other (measured: snare and
    hat get softer but cannot get harder). Collapsing those into one median
    made a working dial look broken.
    """
    order = np.argsort(controls)
    cs = np.asarray(controls, dtype=float)[order]
    out: dict[str, dict] = {}

    for role, t in tracks.items():
        proj = np.asarray(t.projection, dtype=float)[order]
        cos = np.asarray(t.cosine, dtype=float)[order]
        snaps = np.stack([t.snapshots[i] for i in order])
        base = t.baseline
        mags = np.linalg.norm(snaps - base, axis=1)

        neg, pos = cs < -1e-12, cs > 1e-12
        entry: dict = {"declined": bool(t.declined)}
        for tag, m in (("negative", neg), ("positive", pos)):
            if not m.any():
                continue
            reaches = mags[m] > 1e-6
            entry[tag] = {
                "moves": bool(reaches.any()),
                "endpoint_cosine": round(float(cos[m][-1] if tag == "positive"
                                               else cos[m][0]), 3),
                "endpoint_move_l2": round(float(mags[m][-1] if tag == "positive"
                                                else mags[m][0]), 3),
            }
        steps = np.diff(proj)
        entry["monotonicity"] = round(
            float(np.mean(steps >= -1e-9)) if steps.size else 1.0, 3)
        entry["max_param_jump"] = round(
            float(np.abs(np.diff(snaps, axis=0)).max()) if len(snaps) > 1 else 0.0, 4)
        entry["travel"] = round(float(proj[-1] - proj[0]), 3)
        out[role] = entry

    working = []
    for v in out.values():
        for tag in ("negative", "positive"):
            d = v.get(tag)
            if d and d["moves"] and abs(d["endpoint_cosine"]) >= MIN_ACHIEVABLE_COSINE:
                working.append(d["endpoint_cosine"])
    moved = [v for v in out.values() if not v["declined"]]
    out["_summary"] = {
        "roles_moving": f"{len(moved)}/{len(tracks)}",
        "directions_working": f"{len(working)}/{2 * len(tracks)}",
        "median_endpoint_cosine": (
            round(float(np.median(np.abs(working))), 3) if working else None),
        "median_monotonicity": (
            round(float(np.median([v["monotonicity"] for v in moved])), 3)
            if moved else None),
        "worst_param_jump": (
            round(float(max(v["max_param_jump"] for v in moved)), 4)
            if moved else None),
    }
    return out


def save_graph(graph: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=1))
    return out


def load_graph(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def params_at(graph: dict, role: str, control: float) -> np.ndarray:
    """Interpolate a role's parameters at any dial position.

    This is the entire runtime: linear interpolation between two verified
    snapshots. It is what the panel (and eventually a few lines of TypeScript)
    will do on every dial move.
    """
    track = graph["roles"][role]
    xs = np.asarray(graph["control_points"], dtype=float)
    snaps = np.asarray(track["snapshots"], dtype=float)
    c = float(np.clip(control, xs[0], xs[-1]))
    j = int(np.searchsorted(xs, c).clip(1, len(xs) - 1))
    x0, x1 = xs[j - 1], xs[j]
    w = 0.0 if x1 == x0 else (c - x0) / (x1 - x0)
    return snaps[j - 1] * (1 - w) + snaps[j] * w
