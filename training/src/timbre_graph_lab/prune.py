"""Final gate: prune anchors that are hot AS THE PANEL LOADS THEM.

Build-time validation cannot fully predict playback here. Surge exposes
oscillator parameters by oscillator type, `_fix_conditional_params`
recalibrates their ranges against whatever patch was loaded before, and the
same preset was measured exposing 97 parameters on one load and 87 on the
next. So the builder — which has loaded hundreds of patches by the time it
sweeps — and the panel — which loads one preset into a fresh instance — can
render the same stored anchor differently. Measured 2026-08-03: a kick anchor
passed every build gate and rendered a dead-consistent 5.08 peak in the panel's
own conditions, against a 4.0 ceiling.

This pass reproduces the panel exactly: load the role's start patch into a
fresh worker, apply `snapshot[c] - snapshot[0]` as the panel does, render, and
drop whatever is over the ceiling. It is the last word because it is the only
measurement taken under the conditions the listener actually gets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_loudness


# How far over the dial's own median an approach may sit. The shipped
# integration gate allows 10 dB; building to 7 leaves headroom for the
# load-history divergence this whole module exists to absorb.
LURCH_TOL_DB = 7.0


def _resolve(cfg: LabConfig, rel: str) -> Path | None:
    for root in (cfg.factory_patches_dir, cfg.third_party_patches_dir):
        p = Path(root) / rel
        if p.exists():
            return p
    return None


def prune_role(
    cfg: LabConfig, graph: dict, role: str, margin: float = 1.0
) -> dict:
    """Drop anchors whose panel-side render exceeds the safety ceiling."""
    entry = graph["roles"][role]
    anchors, snaps = entry["anchors"], entry["snapshots"]
    if entry.get("declined") or len(anchors) < 2:
        return {"role": role, "checked": 0, "dropped": []}

    anchor_path = _resolve(cfg, anchors[0]["fxp_path"])
    if anchor_path is None:
        return {"role": role, "checked": 0, "dropped": [], "reason": "lens missing"}

    # A FRESH worker per role: the panel's Surge instance has no history of
    # this build's other presets, and that history is the whole problem.
    worker = RenderWorker(cfg)
    if not worker.load_preset(anchor_path):
        return {"role": role, "checked": 0, "dropped": [], "reason": "lens unloadable"}

    probe = get_probe(role, "short")
    names = entry["param_names"]
    base = worker.baseline_raw
    start = np.asarray(snaps[0], dtype=float)

    def render_vec(vec: np.ndarray):
        worker.apply_delta(
            {
                n: float(d)
                for n, d in zip(names, vec - start)
                if n in base and abs(d) > 1e-5
            }
        )
        qc = qc_loudness(worker.render(probe), margin)
        worker.restore_baseline()
        return qc

    keep: list[int] = []
    dropped: list[dict] = []
    for i, snap in enumerate(snaps):
        qc = render_vec(np.asarray(snap, dtype=float))
        if qc.ok or i == 0:      # never drop the lens: it IS the structure
            keep.append(i)
        else:
            dropped.append(
                {"preset_id": anchors[i]["preset_id"], "name": anchors[i]["name"],
                 "peak": round(qc.peak, 2), "reason": qc.reason}
            )

    # ...and the APPROACHES. A pair of safe anchors can still be joined by a
    # loud middle, because envelope and filter parameters interact
    # nonlinearly. Kick's anchors passed the ceiling individually while a
    # midpoint sat 11.9 dB over the dial's median level — audible as a lurch,
    # which is what the limiter then has to catch.
    while len(keep) > 2:
        levels: list[tuple[float, int]] = []
        rms_all: list[float] = []
        for a, b in zip(keep, keep[1:]):
            mid = (np.asarray(snaps[a], float) + np.asarray(snaps[b], float)) / 2.0
            qc = render_vec(mid)
            if not qc.ok:
                levels.append((float("inf"), b))
                continue
            rms_all.append(qc.rms)
            levels.append((qc.rms, b))
        finite = [r for r in rms_all if r > 0]
        if not finite:
            break
        med = float(np.median(finite))
        hot = [
            (r, b) for r, b in levels
            if not np.isfinite(r) or 20.0 * np.log10(max(r, 1e-12) / med) > LURCH_TOL_DB
        ]
        if not hot:
            break
        worst = max(hot, key=lambda t: t[0] if np.isfinite(t[0]) else 1e9)[1]
        dropped.append(
            {"preset_id": anchors[worst]["preset_id"], "name": anchors[worst]["name"],
             "peak": None, "reason": "hot-approach"}
        )
        keep = [i for i in keep if i != worst]

    if dropped:
        n = len(keep)
        entry["anchors"] = [anchors[i] for i in keep]
        entry["snapshots"] = [snaps[i] for i in keep]
        entry["control_points"] = (
            [round(j / (n - 1), 6) for j in range(n)] if n > 1 else [0.0]
        )
        entry["declined"] = n < 2
    return {"role": role, "checked": len(snaps), "dropped": dropped}


def prune_graph(path: str | Path, cfg: LabConfig | None = None, margin: float = 1.0) -> dict:
    cfg = cfg or LabConfig()
    p = Path(path)
    graph = json.loads(p.read_text())
    report = {}
    for role in list(graph["roles"]):
        r = prune_role(cfg, graph, role, margin)
        report[role] = r
        if r["dropped"]:
            print(
                f"{role}: dropped {len(r['dropped'])} anchor(s) hot on a fresh "
                f"load -> {len(graph['roles'][role]['anchors'])} remain"
            )
            for d in r["dropped"]:
                print(f"    {d['name']!r} peak {d['peak']} ({d['reason']})")
        else:
            print(f"{role}: all {r['checked']} anchors safe as the panel loads them")
    graph.setdefault("quality", {})["prune"] = {
        role: {"checked": r["checked"], "dropped": len(r["dropped"])}
        for role, r in report.items()
    }
    p.write_text(json.dumps(graph, indent=1))
    return report
