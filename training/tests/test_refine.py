"""Closed-loop refiner, tested against a synthetic NONLINEAR synth.

The whole reason refinement exists (C13b) is that Surge's parameters interact
multiplicatively, so a linear Jacobian mispredicts combined moves. The fake
synth here reproduces exactly that: a linear term the Jacobian can see, plus a
cross-term it cannot. A refiner that helps on this must beat its own seed.
"""

import numpy as np

from timbre_graph_lab.refine import (
    OFF_AXIS_PENALTY,
    refine,
    score_move,
)
from timbre_graph_lab.solver import MAX_PARAM_STEP, TRUST_RADIUS, solve_step, unit

F, P = 20, 10


class FakeSynth:
    """dz = J^T dx  +  cross-terms  (+ optional saturation)."""

    def __init__(self, seed=0, cross=1.5, saturate=False):
        rng = np.random.default_rng(seed)
        self.J = rng.normal(size=(P, F))
        self.C = rng.normal(size=(P, P, F)) * cross / P
        self.saturate = saturate
        self.calls = 0

    def __call__(self, dx):
        self.calls += 1
        dz = self.J.T @ dx + np.einsum("i,j,ijf->f", dx, dx, self.C)
        if self.saturate:
            dz = np.tanh(dz / 3.0) * 3.0
        return dz

    def jacobian_at_origin(self, eps=0.04):
        """What the prober would measure: per-parameter central differences."""
        J = np.zeros((P, F))
        for i in range(P):
            e = np.zeros(P)
            e[i] = eps
            J[i] = (self(e) - self(-e)) / (2 * eps)
        self.calls = 0
        return J


def _axis(seed=1):
    return unit(np.random.default_rng(seed).normal(size=F))


# --- scoring ---------------------------------------------------------------

def test_score_rewards_on_axis_and_penalises_drift():
    ax = np.zeros(F); ax[0] = 1.0
    on = np.zeros(F); on[0] = 1.0
    off = np.zeros(F); off[0] = 1.0; off[1] = 1.0
    s_on, c_on, p_on = score_move(on, ax)
    s_off, c_off, p_off = score_move(off, ax)
    assert p_on == p_off == 1.0          # same distance along the axis
    assert s_on > s_off                  # but drift is penalised
    assert c_on > c_off
    assert s_off == 1.0 - OFF_AXIS_PENALTY * 1.0


def test_score_handles_none_and_silence():
    ax = _axis()
    assert score_move(None, ax)[0] == -np.inf
    assert score_move(np.zeros(F), ax) == (0.0, 0.0, 0.0)


# --- refinement ------------------------------------------------------------

def test_refine_beats_its_own_open_loop_seed():
    synth = FakeSynth(seed=3)
    J = synth.jacobian_at_origin()
    ax = _axis(4)
    usable = np.ones(P, bool)

    seed_dx = solve_step(J, ax, usable)
    seed_score = score_move(synth(seed_dx), ax)[0]

    res = refine(synth, J, usable, ax, budget=60, seed=5)
    assert res.score > seed_score, (res.score, seed_score)
    assert res.n_renders <= 60


def test_refine_respects_limits_and_dead_controls():
    synth = FakeSynth(seed=6)
    J = synth.jacobian_at_origin()
    usable = np.zeros(P, bool)
    usable[[0, 2, 5]] = True
    res = refine(synth, J, usable, _axis(7), budget=40, seed=8)
    assert np.all(np.abs(res.dx) <= MAX_PARAM_STEP + 1e-9)
    assert np.linalg.norm(res.dx) <= TRUST_RADIUS + 1e-9
    assert np.all(res.dx[~usable] == 0.0)


def test_refine_with_no_live_controls_does_nothing():
    synth = FakeSynth(seed=9)
    res = refine(synth, synth.jacobian_at_origin(), np.zeros(P, bool),
                 _axis(), budget=30)
    assert res.n_renders == 0
    assert np.all(res.dx == 0.0)
    assert res.score == 0.0


def test_refine_tolerates_failed_renders():
    """QC rejections mid-search must not crash or poison the best-so-far."""
    synth = FakeSynth(seed=10)
    J = synth.jacobian_at_origin()
    calls = {"n": 0}

    def flaky(dx):
        calls["n"] += 1
        return None if calls["n"] % 3 == 0 else synth(dx)

    res = refine(flaky, J, np.ones(P, bool), _axis(11), budget=40, seed=12)
    assert np.all(np.isfinite(res.dx))
    assert res.score >= 0.0


def test_refine_is_deterministic_for_a_given_seed():
    synth = FakeSynth(seed=13)
    J = synth.jacobian_at_origin()
    ax = _axis(14)
    a = refine(synth, J, np.ones(P, bool), ax, budget=35, seed=99)
    b = refine(FakeSynth(seed=13), J, np.ones(P, bool), ax, budget=35, seed=99)
    np.testing.assert_allclose(a.dx, b.dx)
    assert a.score == b.score


def test_refine_never_exceeds_its_render_budget():
    synth = FakeSynth(seed=15)
    for budget in (1, 5, 13, 50):
        res = refine(synth, synth.jacobian_at_origin(), np.ones(P, bool),
                     _axis(16), budget=budget, seed=17)
        assert res.n_renders <= budget, (budget, res.n_renders)


def test_refine_helps_more_when_nonlinearity_is_stronger():
    """Refinement should earn its keep precisely where prediction fails."""
    ax = _axis(18)
    gains = []
    for cross in (0.0, 3.0):
        synth = FakeSynth(seed=19, cross=cross)
        J = synth.jacobian_at_origin()
        seed_score = score_move(synth(solve_step(J, ax, np.ones(P, bool))), ax)[0]
        res = refine(synth, J, np.ones(P, bool), ax, budget=60, seed=20)
        gains.append(res.score - seed_score)
    assert gains[1] > gains[0]


def test_refine_handles_saturating_synth():
    synth = FakeSynth(seed=21, cross=1.0, saturate=True)
    res = refine(synth, synth.jacobian_at_origin(), np.ones(P, bool),
                 _axis(22), budget=45, seed=23)
    assert np.isfinite(res.score) and res.score > 0
    assert len(res.history) == res.n_renders


def test_moved_flag_separates_declining_from_failing():
    """A refiner that cannot help must decline, and say so."""
    synth = FakeSynth(seed=30)
    J = synth.jacobian_at_origin()
    ax = _axis(31)

    # a synth that always moves the WRONG way: every move scores negative,
    # so standing still is genuinely the best answer
    res = refine(lambda dx: -np.abs(dx).sum() * ax * 10.0, J, np.ones(P, bool),
                 ax, budget=30, seed=32)
    assert not res.moved
    assert np.all(res.dx == 0.0)

    # and a normal synth does move
    ok = refine(synth, J, np.ones(P, bool), ax, budget=40, seed=33)
    assert ok.moved
