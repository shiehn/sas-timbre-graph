"""The patch map's pure core: embedding, blending, and the "cheat".

No Surge needed. The rendering-side gates (every anchor and a grid of blends
between them) live in the shipped-artifact integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from timbre_graph_lab.patchmap import (
    MAP_SHARPNESS,
    blend_weights,
    embed_2d,
    grid_points,
    params_at_xy,
)


def _pool(n: int = 40, dims: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dims))


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


def test_embedding_fills_the_unit_square():
    pts = embed_2d(_pool())
    assert pts.shape == (40, 2)
    assert pts.min() >= 0.0 and pts.max() <= 1.0
    # both axes actually used, not collapsed
    assert pts[:, 0].ptp() > 0.9 and pts[:, 1].ptp() > 0.9


def test_embedding_is_deterministic_including_component_sign():
    """A principal component's sign is arbitrary; unpinned, a rebuild could
    mirror the map and every stored coordinate would change meaning."""
    z = _pool()
    assert np.allclose(embed_2d(z), embed_2d(z.copy()))


def test_embedding_puts_similar_sounds_near_each_other():
    # two tight clusters far apart in descriptor space
    rng = np.random.default_rng(1)
    a = rng.normal(size=(10, 20)) * 0.05
    b = rng.normal(size=(10, 20)) * 0.05 + 12.0
    pts = embed_2d(np.vstack([a, b]))
    within = np.linalg.norm(pts[:10] - pts[:10].mean(axis=0), axis=1).mean()
    between = np.linalg.norm(pts[:10].mean(axis=0) - pts[10:].mean(axis=0))
    assert between > 5 * within


def test_embedding_handles_degenerate_pools():
    assert embed_2d(np.zeros((0, 20))).shape == (0, 2)
    assert embed_2d(np.zeros((1, 20))).tolist() == [[0.5, 0.5]]
    # identical rows: no variance anywhere, everything lands at the centre
    same = embed_2d(np.ones((5, 20)))
    assert np.allclose(same, 0.5)


# --------------------------------------------------------------------------
# blending — travelling THROUGH the patches
# --------------------------------------------------------------------------


def test_standing_on_an_anchor_reproduces_it_exactly():
    """'Travel through them, not near them' — at an anchor's own coordinates
    the blend must be that anchor and nothing else."""
    pts = embed_2d(_pool())
    snaps = _pool(dims=7)
    for i in (0, 5, 17, 39):
        idx, w = blend_weights(pts, pts[i])
        assert idx[0] == i and w[0] == pytest.approx(1.0)
        assert params_at_xy(pts, snaps, pts[i]) == pytest.approx(snaps[i])


def test_weights_always_form_a_convex_blend():
    """Never extrapolate: every point is a mixture of real patches, so no
    parameter can leave the range its anchors span."""
    pts = embed_2d(_pool())
    for xy in grid_points(12):
        _, w = blend_weights(pts, xy)
        assert w.min() >= 0.0
        assert w.sum() == pytest.approx(1.0)


def test_blended_parameters_stay_inside_the_anchors_that_made_them():
    pts = embed_2d(_pool())
    snaps = np.clip(_pool(dims=5) * 0.2 + 0.5, 0, 1)
    for xy in grid_points(8):
        idx, _ = blend_weights(pts, xy)
        got = params_at_xy(pts, snaps, xy)
        lo = snaps[idx].min(axis=0) - 1e-9
        hi = snaps[idx].max(axis=0) + 1e-9
        assert np.all(got >= lo) and np.all(got <= hi)


# --------------------------------------------------------------------------
# the "cheat" — most of the surface must BE a validated patch
# --------------------------------------------------------------------------


def test_most_of_the_map_is_a_definite_patch():
    """The user's constraint: 'I don't want the user to be able to go too far
    from a patch'. At the shipped sharpness the majority of the surface must
    resolve to one anchor outright."""
    pts = embed_2d(_pool())
    exact = sum(
        1 for xy in grid_points(40) if blend_weights(pts, xy)[1][0] >= 0.999
    )
    assert exact / len(grid_points(40)) > 0.5


def test_sharpness_monotonically_tightens_the_map():
    pts = embed_2d(_pool())
    grid = grid_points(24)

    def dominance(sh: float) -> float:
        return float(np.mean([blend_weights(pts, xy, sharpness=sh)[1][0] for xy in grid]))

    loose, shipped, tight = dominance(2.0), dominance(MAP_SHARPNESS), dominance(24.0)
    assert loose < shipped < tight
    assert loose < 0.7, "a low exponent should genuinely blend"
    assert tight > 0.9, "a high exponent should be nearly discrete"


def test_a_single_anchor_map_is_that_anchor_everywhere():
    pts = np.array([[0.5, 0.5]])
    snaps = np.array([[0.1, 0.9]])
    for xy in grid_points(5):
        assert params_at_xy(pts, snaps, xy) == pytest.approx(snaps[0])


def test_empty_map_does_not_explode():
    idx, w = blend_weights(np.zeros((0, 2)), (0.5, 0.5))
    assert len(idx) == 0 and len(w) == 0
    assert len(params_at_xy(np.zeros((0, 2)), np.zeros((0, 3)), (0.5, 0.5))) == 0
