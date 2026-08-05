"""Prune a shipped map until its whole SURFACE is safe to land on.

`validate_surface` gates the map while it is being built, but it can only ever
sample: it checked an 8x8 grid and the shipped gate samples 6x6, which are
mostly different points, and a continuous surface can misbehave between any two
of them. Measured 2026-08-05, several roles passed the build and still had
positions 17-24 dB over their own median.

So this is a separate, denser, LAST pass over the artifact — the same shape as
`prune.py` was for the tour, and for the same reason: it runs under the
listener's conditions (a fresh Surge per lens, the panel's own relative-delta
form) rather than the builder's, and it keeps dropping the anchor responsible
for the worst position until the surface clears the gate the shipped test
applies. The limiter still backs it up for whatever falls between even this
grid's points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.patchmap import grid_points, params_at_xy
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_loudness

PRUNE_GRID = 9        # denser than either the build's or the gate's sampling
MAX_OVER_DB = 8.0     # the gate allows 10; build to 8 for headroom
MIN_POINTS = 6        # below this a map is not worth shipping


def _resolve(cfg: LabConfig, rel: str) -> Path | None:
    for root in (cfg.factory_patches_dir, cfg.third_party_patches_dir):
        p = Path(root) / rel
        if p.exists():
            return p
    return None


def prune_lens(cfg: LabConfig, role: str, lens: dict) -> dict:
    """Drop points until every sampled position is safe. Mutates `lens`."""
    origin = _resolve(cfg, lens["points"][0]["fxp_path"])
    if origin is None:
        return {"dropped": [], "reason": "origin missing"}

    dropped: list[dict] = []

    # ONE fresh worker for the whole lens. Fresh matters (the panel has not
    # loaded the build's hundreds of other patches); fresh PER ITERATION does
    # not, and instantiating a VST host is slow enough that doing it per drop
    # left the process 80% idle.
    worker = RenderWorker(cfg)
    if not worker.load_preset(origin):
        return {"dropped": dropped, "reason": "origin unloadable"}
    probe = get_probe(role, "short")
    base = worker.baseline_raw

    while len(lens["points"]) >= MIN_POINTS:
        pts = np.asarray([p["xy"] for p in lens["points"]], dtype=float)
        # EVERY point's own coordinates, plus a grid.
        #
        # The grid alone was the gap: at sharpness 20 about three quarters of
        # the surface reproduces one point EXACTLY, so those coordinates are
        # most of what a user actually hears — and a 9x9 grid can miss all of
        # them. A random probe found a bass position peaking at 8.0 that was
        # simply an anchor the grid never sat close enough to dominate.
        # Guaranteeing the exact points makes the remaining risk purely
        # hybrid, which the panel's confidence duck then covers.
        grid = [tuple(p) for p in pts.tolist()] + grid_points(PRUNE_GRID)
        snaps = np.asarray(lens["snapshots"], dtype=float)
        start = params_at_xy(pts, snaps, (0.0, 0.0), sharpness=lens["sharpness"])

        levels: list[float] = []
        blame: dict[int, float] = {}
        for xy in grid:
            here = params_at_xy(pts, snaps, xy, sharpness=lens["sharpness"])
            worker.apply_delta(
                {
                    n: float(here[j] - start[j])
                    for j, n in enumerate(lens["param_names"])
                    if n in base and abs(here[j] - start[j]) > 1e-5
                }
            )
            qc = qc_loudness(worker.render(probe))
            worker.restore_baseline()
            # whoever dominates a position owns it
            nearest = int(np.argmin(np.linalg.norm(pts - np.asarray(xy), axis=1)))
            if not qc.ok:
                if nearest != 0:
                    blame[nearest] = blame.get(nearest, 0.0) + 1000.0
                continue
            levels.append(qc.rms)

        if levels:
            db = [20 * np.log10(max(r, 1e-9)) for r in levels]
            med = float(np.median(db))
            for xy, level in zip(grid, levels):
                over = 20 * np.log10(max(level, 1e-9)) - med
                if over > MAX_OVER_DB:
                    nearest = int(
                        np.argmin(np.linalg.norm(pts - np.asarray(xy), axis=1))
                    )
                    if nearest != 0:
                        blame[nearest] = blame.get(nearest, 0.0) + over

        if not blame:
            return {"dropped": dropped, "points": len(lens["points"])}

        worst = max(sorted(blame), key=lambda k: blame[k])
        dropped.append({"name": lens["points"][worst]["name"],
                        "blame": round(blame[worst], 1)})
        lens["points"].pop(worst)
        lens["snapshots"].pop(worst)

    return {"dropped": dropped, "reason": "too few points left"}


def prune_map(path: str | Path, cfg: LabConfig | None = None) -> dict:
    cfg = cfg or LabConfig()
    p = Path(path)
    graph = json.loads(p.read_text())
    report: dict = {}
    for role, entry in graph["roles"].items():
        for li, lens in enumerate(list(entry["lenses"])):
            r = prune_lens(cfg, role, lens)
            report[f"{role}[{li}]"] = r
            n = len(lens["points"])
            print(
                f"{role} lens {li} ({lens['lens']['name']}): "
                f"{n} points, dropped {len(r['dropped'])}"
                + (f" — {r['reason']}" if r.get("reason") else "")
            )
        # a lens pruned below the floor is not worth shipping
        entry["lenses"] = [l for l in entry["lenses"] if len(l["points"]) >= MIN_POINTS]
        entry["declined"] = len(entry["lenses"]) == 0
    graph.setdefault("quality", {})["prune_map"] = {
        k: {"dropped": len(v["dropped"]), **({"reason": v["reason"]} if v.get("reason") else {})}
        for k, v in report.items()
    }
    p.write_text(json.dumps(graph, indent=1))
    return report
