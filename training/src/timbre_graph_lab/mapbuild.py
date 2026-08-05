"""Build the shipped X/Y patch map.

Reuses the tour pipeline's expensive, already-proven stages — screening, lens
selection, measuring every candidate THROUGH the lens, loudness normalization
— and replaces only what comes after. Where the tour then had to find one long
path through a k-nearest-neighbour graph (longest-path is NP-hard, and it
stranded itself on 6-20 of ~40 kept anchors), the map simply lays every kept
anchor out in the plane. Nothing is discarded for failing to lie on a walk.

Validation is the part that matters and the part previous passes got wrong
twice. It renders the actual blend the panel will produce, using the panel's
own relative-delta form (C15f), from a FRESH worker per role (C15f prune) —
because build-time and playback otherwise disagree about what a stored anchor
even sounds like.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import (
    LabConfig,
    POLICY_VERSION,
    PROBE_VERSION,
    ROLES,
    SEED,
)
from timbre_graph_lab.morph import portable_fxp_path, save_graph
from timbre_graph_lab.patchmap import (
    MAP_NEIGHBOURS,
    MAP_SHARPNESS,
    MAP_SNAP,
    embed_2d,
    grid_points,
    params_at_xy,
)
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.tour import (
    DEFAULT_CAP,
    LENS_PROBE,
    LENS_SURVIVAL,
    MAX_LENS_TRIES,
    MIN_PERCUSSIVE_POOL,
    PERCUSSIVE_ROLES,
    Candidate,
    delta_to,
    farthest_point_sample,
    is_percussive,
    measure_effective,
    normalize_loudness,
    probe_subset,
    rank_lenses,
    render_avg,
    screen_role_pool,
    shared_basis,
)
from timbre_graph_lab.worker import BUILD_MARGIN, RenderWorker, qc_loudness

MAP_VERSION = "map-graph-v2"
DEFAULT_LENSES = 3    # alternative worlds per role
DEFAULT_POINTS = 40   # anchors laid out per role
GRID_N = 10           # NxN blend positions rendered to validate the surface
LURCH_TOL_DB = 7.0


def _panel_delta(
    x_t: np.ndarray, start: np.ndarray, base: dict, param_names: list[str]
) -> dict[str, float]:
    """Exactly what the panel sends: position minus the map's origin."""
    return {
        n: float(x_t[j] - start[j])
        for j, n in enumerate(param_names)
        if n in base and abs(x_t[j] - start[j]) > 1e-5
    }


def validate_surface(
    cfg: LabConfig,
    role: str,
    cands: list[Candidate],
    points: np.ndarray,
    param_names: list[str],
    grid_n: int = GRID_N,
    sharpness: float = MAP_SHARPNESS,
) -> tuple[list[int], dict]:
    """Render a grid of BLENDS and drop the anchors that spoil them.

    A map's promise is that anywhere you put the puck sounds good — which is a
    claim about the whole surface, not about the points. So the grid is
    rendered, and when a position fails, the anchor that dominates it is the
    one held responsible and removed.
    """
    if len(cands) < 2:
        return list(range(len(cands))), {"grid": 0, "dropped": []}

    worker = RenderWorker(cfg)          # fresh: the panel has no build history
    lens_path = cands[0].path
    if not worker.load_preset(lens_path):
        return list(range(len(cands))), {"grid": 0, "dropped": [],
                                         "reason": "lens unloadable"}
    probe = get_probe(role, "short")
    base = worker.baseline_raw
    grid = grid_points(grid_n)
    keep = list(range(len(cands)))
    dropped: list[dict] = []

    for _ in range(len(cands)):         # bounded: each pass drops at most one
        snaps = np.stack([cands[i].x for i in keep])
        pts = points[keep]
        start = snaps[0]
        levels: list[float] = []
        blame: dict[int, int] = {}
        for xy in grid:
            x_t = params_at_xy(pts, snaps, xy, sharpness=sharpness)
            worker.apply_delta(_panel_delta(x_t, start, base, param_names))
            qc = qc_loudness(worker.render(probe), BUILD_MARGIN)
            worker.restore_baseline()
            if qc.ok:
                levels.append(qc.rms)
                continue
            # whoever dominates this position owns the failure
            d = np.linalg.norm(pts - np.asarray(xy), axis=1)
            worst = int(np.argmin(d))
            if worst != 0:               # never blame the lens
                blame[worst] = blame.get(worst, 0) + 1

        if not blame and levels:
            med = float(np.median(levels))
            hot = [r for r in levels if 20 * np.log10(max(r, 1e-12) / med) > LURCH_TOL_DB]
            if not hot:
                break
            # a loud-but-valid surface: tighten by dropping the loudest region
            # is not attributable, so accept and report
            break
        if not blame:
            break
        worst = max(sorted(blame), key=lambda k: blame[k])
        dropped.append(
            {"preset_id": cands[keep[worst]].preset_id,
             "name": cands[keep[worst]].name,
             "bad_positions": blame[worst]}
        )
        keep.pop(worst)
        if len(keep) < 2:
            break

    return keep, {"grid": len(grid), "dropped": dropped}


def build_one_lens(
    cfg: LabConfig,
    worker: RenderWorker,
    role: str,
    survivors: list[Candidate],
    lens: int,
    param_names: list[str],
    n_points: int,
    grid_n: int,
) -> tuple[dict | None, dict]:
    """Everything downstream of choosing a start patch, for ONE lens.

    A lens is a whole world: it supplies the oscillators and routing that every
    point on its map is heard through, and the runtime can never write those.
    So exhausting one map is not exhausting the instrument — it is exhausting
    one structure. Building several is how the tool stays open-ended, and it is
    cheap because screening (the expensive part) is per ROLE and already paid.
    """
    name = survivors[lens].name
    live = measure_effective(worker, role, survivors, lens, param_names)
    if len(live) < 2:
        return None, {"lens": name, "reason": "admits nothing"}

    live, loud_q = normalize_loudness(worker, role, live, 0, param_names)
    if len(live) < 2:
        return None, {"lens": name, "reason": "loudness"}

    from timbre_graph_lab.descriptors import standardize

    zs = np.stack([standardize(c.z_eff) for c in live])
    kept_idx = farthest_point_sample(zs, 0, [c.preset_id for c in live], n_points)
    kept = [live[i] for i in kept_idx]
    points = embed_2d(np.stack([standardize(c.z_eff) for c in kept]))

    keep, grid_q = validate_surface(cfg, role, kept, points, param_names, grid_n)
    kept = [kept[i] for i in keep]
    points = points[keep]
    print(
        f"{role}: lens {name!r} -> {len(kept)} points, "
        f"{len(grid_q['dropped'])} dropped by the grid"
    )
    if len(kept) < 2:
        return None, {"lens": name, "reason": "nothing survived the grid"}

    entry = {
        "lens": {"preset_id": kept[0].preset_id, "name": kept[0].name,
                 "category": kept[0].category},
        "param_names": param_names,
        "points": [
            {"preset_id": c.preset_id, "name": c.name,
             "fxp_path": portable_fxp_path(c.path, cfg),
             "xy": [round(float(points[i][0]), 6), round(float(points[i][1]), 6)]}
            for i, c in enumerate(kept)
        ],
        "snapshots": [np.round(c.x, 6).tolist() for c in kept],
        "sharpness": MAP_SHARPNESS,
        "neighbours": MAP_NEIGHBOURS,
        "snap": MAP_SNAP,
    }
    return entry, {"lens": name, "n_points": len(kept),
                   "loudness": loud_q, "grid": grid_q}


def build_role_map(
    cfg: LabConfig,
    worker: RenderWorker,
    role: str,
    allowed: list[str],
    cap: int = DEFAULT_CAP,
    n_points: int = DEFAULT_POINTS,
    grid_n: int = GRID_N,
    n_lenses: int = DEFAULT_LENSES,
    on_progress=None,
) -> tuple[dict, dict]:
    """Screen once, then build a map per lens — several worlds for one role.

    `on_progress(entry, quality)` is called after EVERY lens, not just at the
    end of the role. Surge segfaults occasionally (the documented reason the
    dataset arm runs process-per-job), and a segfault kills the interpreter —
    Python cannot catch it. Measured 2026-08-04: lead built two good lenses and
    died in its third, and because saving happened per ROLE both were lost.
    Saving per lens turns that from an hour of rework into a few minutes.
    """
    survivors, rows = screen_role_pool(cfg, worker, role, allowed, cap=cap)
    quality: dict = {"n_pool": len(rows), "n_passed": len(survivors)}
    print(f"{role}: {len(survivors)}/{len(rows)} pass screening")
    if len(survivors) < 2:
        return {"role": role, "declined": True, "lenses": []}, \
               {**quality, "reason": "too few survivors"}

    if role in PERCUSSIVE_ROLES:
        named = [c for c in survivors if is_percussive(c, role)]
        if len(named) >= MIN_PERCUSSIVE_POOL:
            survivors = named
            print(f"{role}: restricted to {len(named)} percussion patches")

    param_names = shared_basis(survivors, allowed)

    # Rank candidate lenses, probe each cheaply, and keep the ones that admit
    # a usable pool. A lens that collapses the pool is not a world.
    sample = probe_subset(len(survivors), LENS_PROBE)
    viable: list[int] = []
    attempts: list[dict] = []
    for cand in rank_lenses(survivors, role)[: n_lenses + MAX_LENS_TRIES]:
        got = measure_effective(worker, role, survivors, cand, param_names, only=sample)
        rate = len(got) / max(1, len(sample))
        attempts.append({"name": survivors[cand].name, "probe_rate": round(rate, 3)})
        print(f"{role}: lens {survivors[cand].name!r} admits {len(got)}/{len(sample)}")
        if rate >= LENS_SURVIVAL:
            viable.append(cand)
        if len(viable) >= n_lenses:
            break
    if not viable and attempts:
        # nothing cleared the bar: fall back to the best of what we probed
        best = max(range(len(attempts)), key=lambda i: attempts[i]["probe_rate"])
        viable = [rank_lenses(survivors, role)[best]]

    lenses: list[dict] = []
    lens_q: list[dict] = []
    for cand in viable:
        entry, q = build_one_lens(
            cfg, worker, role, survivors, cand, param_names, n_points, grid_n
        )
        lens_q.append(q)
        if entry is not None:
            lenses.append(entry)
            if on_progress is not None:
                on_progress(
                    {"role": role, "lenses": lenses, "declined": False},
                    {**quality, "lens_attempts": attempts, "lenses": lens_q},
                )

    quality.update({"lens_attempts": attempts, "lenses": lens_q})
    print(f"{role}: MAP {len(lenses)} lens(es)")
    return (
        {"role": role, "lenses": lenses, "declined": len(lenses) == 0},
        quality,
    )


def build_map_graph(
    cfg: LabConfig | None = None,
    roles: list[str] | None = None,
    cap: int = DEFAULT_CAP,
    n_points: int = DEFAULT_POINTS,
    grid_n: int = GRID_N,
    n_lenses: int = DEFAULT_LENSES,
    save_to: str | Path | None = None,
) -> dict:
    cfg = cfg or LabConfig()
    worker = RenderWorker(cfg)
    allowed = json.loads(cfg.policy_path.read_text())["allowed"]
    roles_out: dict[str, dict] = {}
    quality: dict[str, dict] = {}
    for role in roles or ROLES:
        def _save(partial: dict, pq: dict, _role=role) -> None:
            if save_to is None:
                return
            roles_out[_role] = partial
            quality[_role] = pq
            save_graph(assemble(roles_out, quality), save_to)
            print(f"{_role}: saved ({len(partial['lenses'])} lens) -> {save_to}")

        entry, q = build_role_map(
            cfg, worker, role, allowed, cap, n_points, grid_n, n_lenses,
            on_progress=_save,
        )
        roles_out[role] = entry
        quality[role] = q
        if save_to is not None:
            save_graph(assemble(roles_out, quality), save_to)
    return assemble(roles_out, quality)


def assemble(roles_out: dict, quality: dict) -> dict:
    return {
        "version": MAP_VERSION,
        "policy_version": POLICY_VERSION,
        "probe_version": PROBE_VERSION,
        "seed": SEED,
        "roles": roles_out,
        "quality": {
            **quality,
            "_summary": {
                "lenses_by_role": {r: len(e["lenses"]) for r, e in roles_out.items()},
                "points_by_role": {
                    r: [len(l["points"]) for l in e["lenses"]]
                    for r, e in roles_out.items()
                },
                "roles_declined": [r for r, e in roles_out.items() if e["declined"]],
            },
        },
    }
