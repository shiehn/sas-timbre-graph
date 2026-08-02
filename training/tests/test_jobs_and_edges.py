import numpy as np

from timbre_graph_lab.gen import HAT_EXTRA_AVG, build_jobs
from timbre_graph_lab.edges import edge_metrics, knn_edges

ENTRIES = [
    {"preset_id": "k1", "path": "/k1", "roles": ["kick"]},
    {"preset_id": "k2", "path": "/k2", "roles": ["kick"]},
    {"preset_id": "s1", "path": "/s1", "roles": ["snare"]},
    {"preset_id": "ks", "path": "/ks", "roles": ["kick", "snare"]},
    {"preset_id": "b1", "path": "/b1", "roles": ["bass"]},
    {"preset_id": "p1", "path": "/p1", "roles": ["pad"]},
]
ALL_ROLES = ["kick", "snare", "hat", "bass", "pad", "lead"]


def _jobs(**kw):
    defaults = dict(entries=ENTRIES, roles=ALL_ROLES, per_role=0,
                    singles=10, multis=10, drift=2)
    defaults.update(kw)
    return build_jobs(**defaults)


def test_no_cross_probe_is_role_faithful():
    jobs = _jobs(cross_probe=False)
    assert {(j["role"], j["preset_id"]) for j in jobs} == {
        ("kick", "k1"), ("kick", "k2"), ("kick", "ks"),
        ("snare", "s1"), ("snare", "ks"), ("bass", "b1"), ("pad", "p1"),
    }


def test_cross_probe_expands_percussion_trio_only():
    pairs = {(j["role"], j["preset_id"]) for j in _jobs(cross_probe=True)}
    # kick anchors gain snare+hat probes; snare anchors gain kick+hat
    assert ("snare", "k1") in pairs and ("hat", "k1") in pairs
    assert ("kick", "s1") in pairs and ("hat", "s1") in pairs
    # bass/pad never cross-probe
    assert not any(p == ("lead", "b1") or p[1] == "p1" and p[0] != "pad" for p in pairs)


def test_cross_probe_dedupes_dual_role_presets():
    jobs = _jobs(cross_probe=True)
    ks = [(j["role"], j["preset_id"]) for j in jobs if j["preset_id"] == "ks"]
    assert len(ks) == len(set(ks)) == 3  # kick, snare, hat — each once


def test_per_role_limit_and_hat_render_avg():
    jobs = _jobs(per_role=1, cross_probe=False, render_avg=3)
    kicks = [j for j in jobs if j["role"] == "kick"]
    assert len(kicks) == 1
    hat_jobs = [j for j in _jobs(cross_probe=True, render_avg=3) if j["role"] == "hat"]
    assert hat_jobs and all(j["render_avg"] == 3 + HAT_EXTRA_AVG for j in hat_jobs)
    assert all(j["render_avg"] == 3 for j in jobs if j["role"] != "hat")


def test_knn_edges_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    z = rng.random((10, 6))
    edges = knn_edges(z, k=3)
    assert all(a < b for a, b in edges)
    assert len(edges) == len(set(edges))
    counts = {}
    for a, b in edges:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    assert all(c >= 3 for c in counts.values())  # everyone keeps >= k links


def test_knn_edges_tiny_inputs():
    assert knn_edges(np.zeros((1, 4))) == []
    assert knn_edges(np.array([[0.0, 0], [1, 1]]), k=5) == [(0, 1)]


def test_edge_metrics_straight_line_is_perfect():
    z_a, z_b = np.zeros(4), np.ones(4) * 2
    path = np.stack([z_a + (z_b - z_a) * t for t in (0.25, 0.5, 0.75, 1.0)])
    endpoint_err, detour = edge_metrics(path, z_a, z_b)
    assert endpoint_err < 1e-9
    assert abs(detour - 1.0) < 1e-9


def test_edge_metrics_flags_structural_mismatch():
    z_a, z_b = np.zeros(4), np.ones(4) * 2
    # path wanders and ends nowhere near B (structure differs audibly)
    path = np.stack([np.array([5.0, -3, 2, 0]), np.array([0.1, 0.1, 0.1, 0.1])])
    endpoint_err, detour = edge_metrics(path, z_a, z_b)
    assert endpoint_err > 0.35
    assert detour > 3.0
