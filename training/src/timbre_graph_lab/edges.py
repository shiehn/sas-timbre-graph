"""Timbre-graph edges: validated morph routes BETWEEN anchors of a role.

The discovery dial must travel between configurations, not only wiggle
around one (user requirement, 2026-08-02). Naive parameter interpolation
between two synth patches routinely passes through audible garbage, so
edges are proposed cheaply and validated by rendering:

1. propose: k-nearest-neighbour per role in (standardized) descriptor space
   — perceptually close patches are the plausible morph partners
2. render:  walk the continuous allow-list interpolation from A toward B,
   probing a few points; discrete/structural params stay at A's values
3. validate: every point passes QC, the path doesn't detour wildly, and the
   endpoint actually lands near B's sound. If A and B differ structurally
   (osc types etc.), the endpoint misses and the edge is rejected — audio
   decides structural compatibility, no fingerprint bookkeeping needed.

Output: workspace/graph/<role>.json — nodes + validated edges with their
descriptor paths. This artifact IS the plugin's namesake graph.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig, PROBE_VERSION, ROLES
from timbre_graph_lab.corpus import load_manifest
from timbre_graph_lab.dataset import load_shards
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_audio

K_NEIGHBOURS = 4
N_POINTS = 5  # interior interpolation points per edge
EDGE_RENDER_AVG = 2
MAX_ENDPOINT_ERR = 0.35  # ||z(t=1) - z_B|| / ||z_A - z_B||
MAX_DETOUR = 3.0  # path length / direct distance


def knn_edges(z: np.ndarray, k: int = K_NEIGHBOURS) -> list[tuple[int, int]]:
    """Undirected kNN edge list over standardized descriptor rows. Pure."""
    n = len(z)
    if n < 2:
        return []
    mu, sd = z.mean(axis=0), z.std(axis=0) + 1e-9
    s = (z - mu) / sd
    d2 = ((s[:, None, :] - s[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argsort(d2[i])[: min(k, n - 1)]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def edge_metrics(
    z_path: np.ndarray, z_a: np.ndarray, z_b: np.ndarray
) -> tuple[float, float]:
    """(endpoint_err, detour) for a rendered path A -> ... -> t=1. Pure.

    z_path rows are the rendered points INCLUDING t=1 as the last row.
    Distances are computed in a per-edge normalized frame so roles/scales
    are comparable.
    """
    direct = float(np.linalg.norm(z_b - z_a)) + 1e-9
    endpoint_err = float(np.linalg.norm(z_path[-1] - z_b)) / direct
    hops = np.vstack([z_a[None, :], z_path])
    path_len = float(np.sum(np.linalg.norm(np.diff(hops, axis=0), axis=1)))
    detour = path_len / direct
    return endpoint_err, detour


def walk_edge(
    worker: RenderWorker,
    param_names: list[str],
    x_a: np.ndarray,
    x_b: np.ndarray,
    z_a: np.ndarray,
    z_b: np.ndarray,
    probe,
    n_points: int = N_POINTS,
    render_avg: int = EDGE_RENDER_AVG,
) -> dict:
    """Render-validate the allow-list interpolation from the LOADED preset A
    toward B. The caller must have loaded A's preset; deltas are taken against
    the worker's live baseline so discrete/structural params stay at A's
    values by construction. Returns the edge verdict without endpoint ids —
    the caller owns naming.
    """
    base = worker.baseline_raw
    # deltas toward B in the shared allow-list frame, applied on A's load
    dx_total = {
        n: float(x_b[i] - base.get(n, x_a[i]))
        for i, n in enumerate(param_names)
        if abs(x_b[i] - x_a[i]) > 1e-6
    }
    z_rows, ok = [], True
    for t in np.linspace(0.0, 1.0, n_points + 2)[1:]:
        worker.apply_delta({n: t * d for n, d in dx_total.items()})
        audio = worker.render(probe)
        if not qc_audio(audio).ok:
            ok = False
            break
        z_rows.append(worker.render_descriptors(probe, k=render_avg))
    worker.restore_baseline()
    if not ok:
        return {"valid": False, "reason": "qc-fail-on-path"}

    z_path = np.stack(z_rows)
    endpoint_err, detour = edge_metrics(z_path, z_a, z_b)
    valid = endpoint_err <= MAX_ENDPOINT_ERR and detour <= MAX_DETOUR
    return {
        "valid": valid,
        "endpoint_err": round(endpoint_err, 4),
        "detour": round(detour, 4),
        # the rendered route itself — kept for diagnostics and for
        # detecting non-monotonic morphs later. The param trajectory
        # needs no storage: it is the linear interpolation A->B over
        # the shared allow-list.
        "z_path": np.round(z_path, 4).tolist(),
        **({} if valid else {"reason": "endpoint-or-detour"}),
    }


def _anchor_table(cfg: LabConfig, role: str) -> list[dict]:
    """QC-passing anchors of a role with their baseline x and z, from shards."""
    out = []
    for s in load_shards(cfg, [role]):
        # first plan rows are singles: X0[0] is the untouched anchor vector
        out.append(
            {
                "preset_id": s["meta"]["preset_id"],
                "param_names": s["meta"]["param_names"],
                "x": s["X0"][0].astype(np.float64),
                "z": s["Z0"][0].astype(np.float64),
            }
        )
    return out


def build_role_graph(
    cfg: LabConfig,
    role: str,
    worker: RenderWorker,
    k: int = K_NEIGHBOURS,
    n_points: int = N_POINTS,
) -> dict:
    anchors = _anchor_table(cfg, role)
    manifest = load_manifest(cfg)
    path_by_id = {e["preset_id"]: e["path"] for e in manifest["entries"]}
    probe = get_probe(role, "short")

    if len(anchors) < 2:
        return {"role": role, "nodes": [a["preset_id"] for a in anchors], "edges": []}

    Z = np.stack([a["z"] for a in anchors])
    proposals = knn_edges(Z, k=k)
    edges = []
    t_start = time.time()

    for ia, ib in proposals:
        a, b = anchors[ia], anchors[ib]
        fxp = path_by_id.get(a["preset_id"])
        if fxp is None or not worker.load_preset(fxp):
            continue
        verdict = walk_edge(
            worker, a["param_names"], a["x"], b["x"], a["z"], b["z"], probe,
            n_points=n_points,
        )
        edges.append({"a": a["preset_id"], "b": b["preset_id"], **verdict})

    return {
        "role": role,
        "probe_version": PROBE_VERSION,
        "nodes": [a["preset_id"] for a in anchors],
        "n_proposed": len(proposals),
        "n_valid": sum(1 for e in edges if e["valid"]),
        "seconds": round(time.time() - t_start, 1),
        "edges": edges,
    }


def build_graphs(
    cfg: LabConfig | None = None,
    roles: list[str] | None = None,
    k: int = K_NEIGHBOURS,
    n_points: int = N_POINTS,
) -> dict[str, Path]:
    cfg = cfg or LabConfig()
    worker = RenderWorker(cfg)
    out: dict[str, Path] = {}
    for role in roles or ROLES:
        g = build_role_graph(cfg, role, worker, k=k, n_points=n_points)
        p = cfg.workspace / "graph" / f"{role}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(g, indent=1))
        print(
            f"{role}: {len(g['nodes'])} nodes, "
            f"{g.get('n_valid', 0)}/{g.get('n_proposed', 0)} edges valid -> {p}"
        )
        out[role] = p
    return out
