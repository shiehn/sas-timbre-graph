"""Follower solver: the maths that turns one gesture into six.

Guards the properties the instrument depends on — the move approaches the
requested timbre direction, stays small and inside the trust radius, never
leans on a dead control, respects locks, and translates rather than copies
knobs.
"""

import numpy as np
import pytest

from timbre_graph_lab.prober import AnchorResponse
from timbre_graph_lab.solver import (
    MAX_PARAM_STEP,
    TRUST_RADIUS,
    build_trajectory,
    leader_follower,
    morph_axes,
    predicted_delta,
    solve_step,
    unit,
)

F = 20  # descriptor count


def _resp(role, J, usable=None, baseline=None):
    n = J.shape[0]
    return AnchorResponse(
        role=role, preset_id=f"id-{role}", name=role,
        param_names=[f"p{i}" for i in range(n)],
        baseline=np.full(n, 0.5) if baseline is None else baseline,
        z0=np.zeros(F), J=J,
        sigma=np.full(F, 1e-6),
        usable=np.ones(n, bool) if usable is None else usable,
    )


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _reachable(J, v):
    """Component of v inside the timbre subspace this Jacobian can reach.

    A synth with k parameters can only move within the span of its k Jacobian
    rows. With fewer parameters than descriptors that span is a strict
    subspace, so a target chosen at random is largely unreachable and the best
    possible cosine is bounded by rank, not by solver quality.
    """
    B = np.linalg.svd(J, full_matrices=False)[2]  # orthonormal row-space basis
    return B.T @ (B @ v)


def test_solve_reaches_an_achievable_direction():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(8, F))
    # a target the synth can actually produce
    target = predicted_delta(J, rng.normal(size=8) * 0.05)
    got = predicted_delta(J, solve_step(J, target))
    assert _cos(got, target) > 0.95, _cos(got, target)


def test_solve_is_optimal_for_an_unreachable_target():
    """For an arbitrary target, the move must align with its reachable part."""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(8, F))
    target = unit(rng.normal(size=F)) * 0.4
    got = predicted_delta(J, solve_step(J, target))
    proj = _reachable(J, target)
    assert _cos(got, proj) > 0.95, _cos(got, proj)
    # and the raw cosine is capped by the rank ratio, not by the solver
    assert _cos(got, target) == pytest.approx(_cos(proj, target), abs=0.05)


def test_opposite_targets_give_opposite_moves():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(8, F))
    t = unit(rng.normal(size=F)) * 0.3
    np.testing.assert_allclose(solve_step(J, t), -solve_step(J, -t), atol=1e-9)


def test_respects_per_param_and_trust_radius_limits():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(12, F)) * 0.01      # weak response -> wants a huge step
    dx = solve_step(J, unit(rng.normal(size=F)) * 50.0)
    assert np.all(np.abs(dx) <= MAX_PARAM_STEP + 1e-9)
    assert np.linalg.norm(dx) <= TRUST_RADIUS + 1e-9


def test_never_moves_a_dead_control():
    rng = np.random.default_rng(3)
    J = rng.normal(size=(6, F))
    usable = np.array([True, False, True, False, True, True])
    dx = solve_step(J, unit(rng.normal(size=F)) * 0.3, usable)
    assert np.all(dx[~usable] == 0.0)
    assert np.any(dx[usable] != 0.0)


def test_all_dead_controls_yields_no_move():
    J = np.random.default_rng(4).normal(size=(5, F))
    dx = solve_step(J, np.ones(F) * 0.2, np.zeros(5, bool))
    assert np.all(dx == 0.0)


def test_damping_prefers_smaller_moves():
    rng = np.random.default_rng(5)
    J = rng.normal(size=(10, F))
    t = unit(rng.normal(size=F)) * 0.3
    small = solve_step(J, t, damping=1.0)
    large = solve_step(J, t, damping=1e-6)
    assert np.linalg.norm(small) < np.linalg.norm(large)


def test_morph_axes_are_orthonormal_and_reachable():
    rng = np.random.default_rng(6)
    J = rng.normal(size=(9, F))
    axes = morph_axes(J, k=2)
    assert axes.shape == (2, F)
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), [1, 1], atol=1e-9)
    assert abs(float(axes[0] @ axes[1])) < 1e-9
    # the top axis must be more reachable than a random direction
    top = np.linalg.norm(predicted_delta(J, solve_step(J, axes[0] * 0.3)))
    rnd = np.linalg.norm(predicted_delta(J, solve_step(J, unit(rng.normal(size=F)) * 0.3)))
    assert top > rnd


def test_morph_axes_handles_rank_deficient_jacobian():
    J = np.zeros((4, F))
    J[0, 0] = 1.0
    axes = morph_axes(J, k=3)
    assert axes.shape == (3, F)
    assert np.all(np.isfinite(axes))


# --- trajectory / dial ------------------------------------------------------

def _six(rng):
    return {r: _resp(r, rng.normal(size=(8, F)))
            for r in ("kick", "snare", "hat", "bass", "pad", "lead")}


def test_trajectory_centre_is_the_untouched_anchor():
    rng = np.random.default_rng(7)
    resp = _six(rng)
    pts = build_trajectory(resp, morph_axes(resp["lead"].J)[0], n_points=5)
    mid = pts[len(pts) // 2]
    assert abs(mid.control[0]) < 1e-9
    for role, r in resp.items():
        np.testing.assert_allclose(mid.params[role], r.baseline, atol=1e-12)


def test_trajectory_moves_every_unlocked_role():
    rng = np.random.default_rng(8)
    resp = _six(rng)
    pts = build_trajectory(resp, morph_axes(resp["lead"].J)[0], n_points=5)
    end = pts[-1]
    for role, r in resp.items():
        assert np.any(np.abs(end.params[role] - r.baseline) > 1e-6), role


def test_locked_role_never_moves_but_others_still_do():
    rng = np.random.default_rng(9)
    resp = _six(rng)
    pts = build_trajectory(resp, morph_axes(resp["lead"].J)[0],
                           n_points=5, locked={"pad"})
    end = pts[-1]
    np.testing.assert_allclose(end.params["pad"], resp["pad"].baseline, atol=1e-12)
    assert np.any(np.abs(end.params["bass"] - resp["bass"].baseline) > 1e-6)


def test_trajectory_stays_in_raw_range():
    rng = np.random.default_rng(10)
    resp = {r: _resp(r, rng.normal(size=(8, F)) * 0.01,
                     baseline=np.full(8, 0.98)) for r in ("kick", "bass")}
    for p in build_trajectory(resp, unit(rng.normal(size=F)), n_points=7, span=3.0):
        for role in resp:
            assert np.all(p.params[role] >= 0.0) and np.all(p.params[role] <= 1.0)


def test_zero_coupling_freezes_a_role():
    rng = np.random.default_rng(11)
    resp = _six(rng)
    pts = build_trajectory(resp, morph_axes(resp["lead"].J)[0],
                           n_points=5, coupling={"hat": 0.0})
    np.testing.assert_allclose(pts[-1].params["hat"], resp["hat"].baseline, atol=1e-12)


# --- leader / follower ------------------------------------------------------

def test_followers_translate_rather_than_copy_knobs():
    """The whole point: a follower must not mirror the leader's parameter."""
    rng = np.random.default_rng(12)
    resp = _six(rng)
    dx_lead = np.zeros(8)
    dx_lead[2] = 0.08                      # user turns leader's param #2
    out = leader_follower(resp, "lead", dx_lead)

    np.testing.assert_allclose(out["lead"], dx_lead)
    for role in ("kick", "bass", "pad"):
        f = out[role]
        assert np.any(np.abs(f) > 1e-6), f"{role} did not respond"
        # not a copy: the follower's move is not concentrated on param #2
        assert np.argmax(np.abs(f)) != 2 or np.count_nonzero(np.abs(f) > 1e-6) > 1


def test_follower_direction_matches_the_leader_gesture():
    """Each follower must reproduce as much of the leader's gesture as its own
    parameters allow — judged against its reachable subspace, since a follower
    cannot be blamed for a direction its synth simply cannot produce."""
    rng = np.random.default_rng(13)
    resp = _six(rng)
    dx_lead = rng.normal(size=8) * 0.05
    dz_lead = predicted_delta(resp["lead"].J, dx_lead)
    out = leader_follower(resp, "lead", dx_lead)
    for role in ("bass", "pad", "hat"):
        dz_f = predicted_delta(resp[role].J, out[role])
        proj = _reachable(resp[role].J, dz_lead)
        assert _cos(dz_f, proj) > 0.9, (role, _cos(dz_f, proj))
        assert _cos(dz_f, dz_lead) > 0.3, (role, _cos(dz_f, dz_lead))


def test_follower_beats_a_random_move_at_matching_the_gesture():
    rng = np.random.default_rng(23)
    resp = _six(rng)
    dx_lead = rng.normal(size=8) * 0.05
    dz_lead = predicted_delta(resp["lead"].J, dx_lead)
    out = leader_follower(resp, "lead", dx_lead)
    for role in ("bass", "pad", "hat"):
        solved = _cos(predicted_delta(resp[role].J, out[role]), dz_lead)
        rand = np.median([
            _cos(predicted_delta(resp[role].J, rng.normal(size=8) * 0.05), dz_lead)
            for _ in range(200)
        ])
        assert solved > rand + 0.2, (role, solved, rand)


def test_locked_follower_is_untouched_by_leader_moves():
    rng = np.random.default_rng(14)
    resp = _six(rng)
    out = leader_follower(resp, "lead", rng.normal(size=8) * 0.05, locked={"kick"})
    assert np.all(out["kick"] == 0.0)


def test_anchor_response_json_round_trip():
    rng = np.random.default_rng(15)
    r = _resp("bass", rng.normal(size=(6, F)),
              usable=np.array([1, 0, 1, 1, 0, 1], bool))
    back = AnchorResponse.from_dict(r.to_dict())
    np.testing.assert_allclose(back.J, r.J)
    np.testing.assert_allclose(back.baseline, r.baseline)
    assert list(back.usable) == list(r.usable)
    assert back.live_params == r.live_params
    assert back.role == "bass"


# --- descriptor standardisation ---------------------------------------------

def test_feature_scale_covers_every_descriptor():
    from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES, FEATURE_SCALE
    assert FEATURE_SCALE.shape == (len(DESCRIPTOR_NAMES),)
    assert np.all(FEATURE_SCALE > 0)


def test_standardisation_equalises_dimension_influence():
    """Raw deltas are dominated by the Hz-scale dims; standardised ones are not.

    Measured on the corpus: rolloff85_mean + bandwidth_mean carried 85% of all
    raw delta energy, so a cosine on raw vectors ignored 15 of 20 descriptors.
    """
    from timbre_graph_lab.descriptors import (
        DESCRIPTOR_NAMES, FEATURE_SCALE, standardize,
    )
    # a delta of one corpus standard deviation in every dimension
    raw = FEATURE_SCALE.copy()
    raw_share = raw**2 / (raw**2).sum()
    std_share = standardize(raw) ** 2 / (standardize(raw) ** 2).sum()
    roll = DESCRIPTOR_NAMES.index("rolloff85_mean")
    band = DESCRIPTOR_NAMES.index("bandwidth_mean")
    assert raw_share[roll] + raw_share[band] > 0.7      # raw: two dims dominate
    np.testing.assert_allclose(std_share, np.full(len(std_share),
                                                  1 / len(std_share)), atol=1e-9)


def test_standardise_is_linear():
    from timbre_graph_lab.descriptors import standardize
    v = np.arange(1, 21, dtype=float)
    np.testing.assert_allclose(standardize(3 * v), 3 * standardize(v))
