"""End-to-end proof: do solved follower moves actually SOUND like the gesture?

Everything upstream is a prediction from a measured Jacobian. This module
closes the loop the way the proposal's §8.1 "re-rendered follower error"
demands: apply the solved parameters in Surge, render, extract descriptors,
and compare the ACHIEVED perceptual change against the intended direction.

Three arms, so the number means something:

  coupled  - the solver's answer (translation)
  copy     - the leader's raw parameter delta applied verbatim to the follower
             (knob-copying: the thing the design explicitly rejects)
  none     - no movement at all (the floor)

If coupled does not beat both, the instrument does not work and no amount of
UI polish will save it.
"""

from __future__ import annotations

import numpy as np

from timbre_graph_lab.descriptors import standardize
from timbre_graph_lab.prober import AnchorResponse
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.solver import leader_follower, predicted_delta
from timbre_graph_lab.worker import RenderWorker, qc_audio


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


def _render_delta(
    worker: RenderWorker,
    resp: AnchorResponse,
    dx: np.ndarray,
    avg: int,
) -> np.ndarray | None:
    """Apply dx to this anchor, render, and return the achieved descriptor delta."""
    probe = get_probe(resp.role, "short")
    worker.restore_baseline()
    z0 = worker.render_descriptors(probe, k=avg)
    delta = {p: float(d) for p, d in zip(resp.param_names, dx) if abs(d) > 1e-9}
    if not delta:
        return np.zeros_like(z0)
    worker.apply_delta(delta)
    audio = worker.render(probe)
    if not qc_audio(audio).ok:
        worker.restore_baseline()
        return None
    z1 = worker.render_descriptors(probe, k=avg)
    worker.restore_baseline()
    # standardized, to match the units the Jacobians and solver work in
    return standardize(z1 - z0)


def verify_gesture(
    worker: RenderWorker,
    responses: dict[str, AnchorResponse],
    paths: dict[str, str],
    leader: str,
    leader_dx: np.ndarray,
    avg: int = 3,
) -> dict:
    """Render-verify one leader gesture across all followers."""
    lead = responses[leader]
    intended = predicted_delta(lead.J, leader_dx)
    solved = leader_follower(responses, leader, leader_dx)

    rows = []
    for role, resp in responses.items():
        if role == leader:
            continue
        if not worker.load_preset(paths[role]):
            continue

        arms = {
            "coupled": solved[role],
            # knob-copy: same raw delta, matched by parameter NAME where the
            # follower has that control at all
            "copy": np.array([
                float(dict(zip(lead.param_names, leader_dx)).get(p, 0.0))
                for p in resp.param_names
            ]),
            "none": np.zeros(len(resp.param_names)),
        }
        row = {"role": role}
        for arm, dx in arms.items():
            got = _render_delta(worker, resp, dx, avg)
            if got is None:
                row[arm] = None
                continue
            row[arm] = {
                "cosine": round(_cos(got, intended), 4),
                "magnitude": round(float(np.linalg.norm(got)), 4),
                "param_move_l2": round(float(np.linalg.norm(dx)), 4),
            }
        # how well the solver's own prediction matched reality
        pred = predicted_delta(resp.J, solved[role])
        got = row.get("coupled")
        row["prediction_honest"] = (
            None if got is None else round(_cos(pred, intended), 4)
        )
        rows.append(row)

    return {
        "leader": leader,
        "intended_magnitude": round(float(np.linalg.norm(intended)), 4),
        "followers": rows,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate several gestures into the numbers that decide go / no-go."""
    arms = {"coupled": [], "copy": [], "none": []}
    per_role: dict[str, list[float]] = {}
    for res in results:
        for row in res["followers"]:
            for arm in arms:
                v = row.get(arm)
                if isinstance(v, dict):
                    arms[arm].append(v["cosine"])
            v = row.get("coupled")
            if isinstance(v, dict):
                per_role.setdefault(row["role"], []).append(v["cosine"])
    out = {
        f"{arm}_cosine_median": (round(float(np.median(v)), 4) if v else None)
        for arm, v in arms.items()
    }
    out["n_observations"] = len(arms["coupled"])
    out["by_role"] = {
        r: round(float(np.median(v)), 4) for r, v in sorted(per_role.items())
    }
    if arms["coupled"] and arms["copy"]:
        out["beats_knob_copy"] = bool(
            np.median(arms["coupled"]) > np.median(arms["copy"])
        )
    return out
