"""The semantic axis library and measured achievability.

Codifies what render-verification found: roles differ in what they can express
(a kick has 0.000 reach in band_high — "brighter" is impossible, not merely
hard), and sign coupling matters (an axis pairing decay_slope+ with
attack_time+ is unreachable for a patch whose reachable direction pairs
decay_slope+ with attack_time-).
"""

import numpy as np

from timbre_graph_lab.axes import (
    AXES,
    MIN_ACHIEVABLE_COSINE,
    achievability,
    best_axes,
    make_axis,
    shared_axes,
)
from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES, N_DESCRIPTORS
from timbre_graph_lab.prober import AnchorResponse


def test_every_axis_is_a_unit_vector_of_the_right_size():
    for name, ax in AXES.items():
        assert ax.shape == (N_DESCRIPTORS,), name
        assert abs(float(np.linalg.norm(ax)) - 1.0) < 1e-9, name


def test_axes_are_distinct_directions():
    names = list(AXES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert abs(float(AXES[a] @ AXES[b])) < 0.999, (a, b)


def test_make_axis_places_weights_on_named_descriptors():
    ax = make_axis(crest_db=1.0, attack_time=-1.0)
    i, j = DESCRIPTOR_NAMES.index("crest_db"), DESCRIPTOR_NAMES.index("attack_time")
    assert ax[i] > 0 and ax[j] < 0
    assert np.count_nonzero(ax) == 2


def test_envelope_axes_encode_the_sign_coupling_that_mattered():
    """`longer` and `boomier` must differ in attack_time sign.

    The original single "longer" axis paired decay_slope+ with attack_time+ and
    was unreachable for the kick, whose own direction pairs decay_slope+ with
    attack_time-. Both variants exist so measurement can pick.
    """
    a = DESCRIPTOR_NAMES.index("attack_time")
    d = DESCRIPTOR_NAMES.index("decay_slope")
    assert AXES["longer"][d] > 0 and AXES["boomier"][d] > 0
    assert AXES["longer"][a] > 0 > AXES["boomier"][a]


def test_tighter_opposes_longer_on_decay():
    d = DESCRIPTOR_NAMES.index("decay_slope")
    assert AXES["tighter"][d] < 0 < AXES["longer"][d]


def _resp(n_params=8, seed=0):
    rng = np.random.default_rng(seed)
    return AnchorResponse(
        role="bass", preset_id="x", name="x",
        param_names=[f"p{i}" for i in range(n_params)],
        baseline=np.full(n_params, 0.5), z0=np.zeros(N_DESCRIPTORS),
        J=rng.normal(size=(n_params, N_DESCRIPTORS)),
        sigma=np.full(N_DESCRIPTORS, 1e-6),
        usable=np.ones(n_params, bool),
    )


def test_achievability_scores_every_axis():
    r = _resp()
    table = achievability(lambda dx: r.J.T @ dx, r, budget=12)
    assert set(table) == set(AXES)
    assert all(isinstance(v, float) for v in table.values())


def test_achievability_records_a_decline_as_zero():
    """A synth that always moves the wrong way must score 0.0, not negative."""
    r = _resp(seed=2)
    table = achievability(lambda dx: -np.abs(dx).sum() * np.ones(N_DESCRIPTORS),
                          r, budget=10)
    assert set(table.values()) == {0.0}


def test_best_axes_filters_and_orders():
    table = {"a": 0.9, "b": 0.1, "c": 0.55, "d": 0.0, "e": 0.7}
    got = best_axes(table, k=3)
    assert [n for n, _ in got] == ["a", "e", "c"]
    assert all(c >= MIN_ACHIEVABLE_COSINE for _, c in got)


def test_best_axes_can_be_empty():
    assert best_axes({"a": 0.1, "b": 0.0}) == []


def test_shared_axes_ranks_by_coverage_then_strength():
    tables = {
        "kick":  {"soft": 0.75, "bright": 0.00, "tight": 0.81},
        "snare": {"soft": 0.82, "bright": 0.00, "tight": 0.86},
        "pad":   {"soft": 0.76, "bright": 0.46, "tight": 0.83},
    }
    ranked = shared_axes(tables)
    assert [n for n, _, _ in ranked][:2] == ["tight", "soft"]  # 3/3 each, tight stronger
    names = {n: (c, m) for n, c, m in ranked}
    assert names["soft"][0] == 3
    assert names["bright"][0] == 1        # only the pad can express it
