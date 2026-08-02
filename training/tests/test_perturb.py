import numpy as np

from timbre_graph_lab.perturb import (
    MULTI_CLIP,
    TRUST_RADIUS,
    build_plan,
    sensitivity_pairs,
)

PARAMS = [f"p{i}" for i in range(12)]


def test_sensitivity_pairs_are_central():
    edits = sensitivity_pairs(PARAMS[:3])
    assert len(edits) == 6
    assert edits[0].delta == {"p0": +0.04}
    assert edits[1].delta == {"p0": -0.04}


def test_plan_deterministic_per_anchor():
    a = build_plan("abc", "bass", PARAMS, seed=1)
    b = build_plan("abc", "bass", PARAMS, seed=1)
    assert [(e.kind, e.delta, e.base_offset) for e in a] == [
        (e.kind, e.delta, e.base_offset) for e in b
    ]


def test_plan_differs_across_anchors_and_seeds():
    a = build_plan("abc", "bass", PARAMS, seed=1)
    b = build_plan("xyz", "bass", PARAMS, seed=1)
    c = build_plan("abc", "bass", PARAMS, seed=2)
    sig = lambda plan: [(e.kind, tuple(sorted(e.delta.items()))) for e in plan]
    assert sig(a) != sig(b)
    assert sig(a) != sig(c)


def test_magnitudes_bounded():
    plan = build_plan("abc", "lead", PARAMS, seed=3)
    for e in plan:
        for d in e.delta.values():
            assert abs(d) <= max(MULTI_CLIP, 0.15) + 1e-9


def test_drift_stays_inside_trust_radius():
    plan = build_plan("abc", "pad", PARAMS, n_drift_chains=8, seed=4)
    for e in plan:
        if e.kind == "drift":
            l2 = np.sqrt(sum(v * v for v in e.base_offset.values()))
            assert l2 <= TRUST_RADIUS + 1e-9


def test_empty_params_empty_plan():
    assert build_plan("abc", "kick", [], seed=1) == []
