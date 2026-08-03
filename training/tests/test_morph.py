"""The morph graph: the artifact the panel consumes.

Properties that matter to a player: the dial's centre is the untouched patch,
turning it travels along the requested axis, neighbouring positions are close
(no jumps), a role that cannot follow holds still, and interpolation between
snapshots is exact at the stored points.
"""

import numpy as np
import pytest

from timbre_graph_lab.axes import AXES
from timbre_graph_lab.morph import (
    MORPH_VERSION,
    RoleTrack,
    load_graph,
    params_at,
    quality,
    save_graph,
)
from timbre_graph_lab.descriptors import N_DESCRIPTORS


def _track(role="pad", n=6, n_points=5, move=True):
    base = np.full(n, 0.5)
    controls = np.linspace(-1, 1, n_points)
    snaps, cos, proj = [], [], []
    for t in controls:
        snaps.append(base + (0.1 * t if move else 0.0))
        cos.append(1.0 if move else 0.0)
        proj.append(float(t) if move else 0.0)
    return RoleTrack(role=role, preset_id="p", name=role,
                     param_names=[f"p{i}" for i in range(n)],
                     baseline=base, snapshots=snaps, cosine=cos,
                     projection=proj), controls


def test_declined_track_is_detected():
    t, _ = _track(move=False)
    assert t.declined
    moved, _ = _track(move=True)
    assert not moved.declined


def test_quality_reports_monotonic_smooth_travel():
    t, controls = _track()
    q = quality({"pad": t}, controls)
    assert q["pad"]["monotonicity"] == 1.0
    assert q["pad"]["travel"] == pytest.approx(2.0)
    assert q["pad"]["max_param_jump"] == pytest.approx(0.05, abs=1e-6)
    assert q["_summary"]["roles_moving"] == "1/1"
    assert q["pad"]["positive"]["moves"] and q["pad"]["negative"]["moves"]
    assert q["_summary"]["directions_working"] == "2/2"


def test_quality_scores_directions_independently():
    """A role can be expressive one way and immovable the other.

    Measured: snare and hat get softer but cannot get harder. Averaging the two
    halves together made a working dial look broken.
    """
    t, controls = _track(n_points=5)
    base = t.baseline
    # positive half moves and tracks the axis; negative half holds still
    for i, c in enumerate(controls):
        if c < 0:
            t.snapshots[i] = base.copy()
            t.cosine[i] = 0.0
            t.projection[i] = 0.0
    q = quality({"pad": t}, controls)
    assert q["pad"]["positive"]["moves"] is True
    assert q["pad"]["negative"]["moves"] is False
    assert q["_summary"]["directions_working"] == "1/2"
    # the working direction is not dragged down by the dead one
    assert q["_summary"]["median_endpoint_cosine"] >= 0.9


def test_quality_flags_non_monotonic_travel():
    t, controls = _track()
    t.projection = [0.0, 1.0, -1.0, 0.5, 2.0]      # wanders
    q = quality({"pad": t}, controls)
    assert q["pad"]["monotonicity"] < 1.0


def test_quality_flags_a_parameter_jump():
    t, controls = _track()
    t.snapshots[2] = t.snapshots[2] + 0.4          # sudden lurch
    q = quality({"pad": t}, controls)
    assert q["pad"]["max_param_jump"] > 0.3


def test_quality_handles_all_roles_declining():
    t, controls = _track(move=False)
    q = quality({"kick": t}, controls)
    assert q["kick"]["declined"]
    assert q["_summary"]["roles_moving"] == "0/1"
    assert q["_summary"]["median_monotonicity"] is None
    assert q["_summary"]["directions_working"] == "0/2"
    assert q["_summary"]["median_endpoint_cosine"] is None


# --- artifact round trip / runtime interpolation ----------------------------

def _graph(n_points=5):
    t, controls = _track(n_points=n_points)
    return {
        "version": MORPH_VERSION,
        "axis": {"name": "softer", "vector": AXES["softer"].tolist()},
        "feature_names": [f"f{i}" for i in range(N_DESCRIPTORS)],
        "control_points": [float(c) for c in controls],
        "roles": {"pad": t.to_dict()},
        "quality": quality({"pad": t}, controls),
    }


def test_graph_round_trips_through_disk(tmp_path):
    g = _graph()
    p = save_graph(g, tmp_path / "graph.json")
    back = load_graph(p)
    assert back["version"] == MORPH_VERSION
    assert back["axis"]["name"] == "softer"
    np.testing.assert_allclose(back["roles"]["pad"]["snapshots"],
                               g["roles"]["pad"]["snapshots"])


def test_interpolation_is_exact_at_stored_points():
    g = _graph()
    for i, c in enumerate(g["control_points"]):
        np.testing.assert_allclose(
            params_at(g, "pad", c),
            np.asarray(g["roles"]["pad"]["snapshots"][i]), atol=1e-12)


def test_interpolation_is_between_neighbours():
    g = _graph()
    xs = g["control_points"]
    mid = (xs[1] + xs[2]) / 2
    got = params_at(g, "pad", mid)
    a = np.asarray(g["roles"]["pad"]["snapshots"][1])
    b = np.asarray(g["roles"]["pad"]["snapshots"][2])
    np.testing.assert_allclose(got, (a + b) / 2, atol=1e-12)


def test_interpolation_clamps_outside_the_dial_range():
    g = _graph()
    lo = np.asarray(g["roles"]["pad"]["snapshots"][0])
    hi = np.asarray(g["roles"]["pad"]["snapshots"][-1])
    np.testing.assert_allclose(params_at(g, "pad", -99.0), lo, atol=1e-12)
    np.testing.assert_allclose(params_at(g, "pad", 99.0), hi, atol=1e-12)


def test_interpolated_params_stay_in_raw_range():
    g = _graph(n_points=9)
    for c in np.linspace(-1.2, 1.2, 41):
        p = params_at(g, "pad", c)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_dial_centre_is_the_untouched_patch():
    g = _graph()
    base = np.asarray(g["roles"]["pad"]["baseline"])
    np.testing.assert_allclose(params_at(g, "pad", 0.0), base, atol=1e-12)


# --- X/Y pad ----------------------------------------------------------------

def _xy_graph(n=3):
    """Two-axis grid: role moves +0.1 per x step and +0.01 per y step."""
    base = np.full(4, 0.5)
    xs = list(np.linspace(-1, 1, n))
    grid = [
        [(base + 0.1 * xs[i] + 0.01 * xs[j]).tolist() for j in range(n)]
        for i in range(n)
    ]
    return {
        "version": MORPH_VERSION,
        "axes": {"x": {"name": "softer", "vector": []},
                 "y": {"name": "tighter", "vector": []}},
        "control_points": [float(x) for x in xs],
        "roles": {"pad": {"role": "pad", "preset_id": "p", "name": "pad",
                          "param_names": [f"p{i}" for i in range(4)],
                          "baseline": base.tolist(), "grid": grid,
                          "declined": False}},
        "corner_cosine": {"pad": {"--": 0.7, "-+": 0.6, "+-": 0.6, "++": 0.5}},
    }


def test_xy_interpolation_is_exact_at_grid_nodes():
    from timbre_graph_lab.morph import params_at_xy
    g = _xy_graph()
    xs = g["control_points"]
    for i, cx in enumerate(xs):
        for j, cy in enumerate(xs):
            np.testing.assert_allclose(
                params_at_xy(g, "pad", cx, cy),
                np.asarray(g["roles"]["pad"]["grid"][i][j]), atol=1e-12)


def test_xy_centre_is_the_untouched_patch():
    from timbre_graph_lab.morph import params_at_xy
    g = _xy_graph()
    base = np.asarray(g["roles"]["pad"]["baseline"])
    np.testing.assert_allclose(params_at_xy(g, "pad", 0.0, 0.0), base, atol=1e-12)


def test_xy_is_bilinear_between_nodes():
    from timbre_graph_lab.morph import params_at_xy
    g = _xy_graph()
    xs = g["control_points"]
    mx, my = (xs[0] + xs[1]) / 2, (xs[0] + xs[1]) / 2
    grid = np.asarray(g["roles"]["pad"]["grid"])
    expect = (grid[0, 0] + grid[0, 1] + grid[1, 0] + grid[1, 1]) / 4
    np.testing.assert_allclose(params_at_xy(g, "pad", mx, my), expect, atol=1e-12)


def test_xy_clamps_outside_the_pad():
    from timbre_graph_lab.morph import params_at_xy
    g = _xy_graph()
    grid = np.asarray(g["roles"]["pad"]["grid"])
    np.testing.assert_allclose(params_at_xy(g, "pad", -9, -9), grid[0, 0], atol=1e-12)
    np.testing.assert_allclose(params_at_xy(g, "pad", 9, 9), grid[-1, -1], atol=1e-12)


def test_xy_corner_cosine_is_reported_for_the_summed_move():
    """Summing two axes assumes linearity, so corners must be re-verified."""
    g = _xy_graph()
    assert set(g["corner_cosine"]["pad"]) == {"--", "-+", "+-", "++"}
