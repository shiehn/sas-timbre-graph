"""The patch MAP: a 2-D surface whose every point is a good sound.

The 1-D tour has a structural flaw the numbers made plain: it must find one
long path through a k-nearest-neighbour graph, and longest-path is NP-hard, so
it strands itself. Measured 2026-08-03, bass kept 40 spread anchors inside a
37-node connected component and the tour visited **13** of them. Two thirds of
a screened, loudness-normalized, lens-validated pool went unused because it did
not happen to lie on one walk.

A map needs no path. Every kept anchor becomes a point in the plane, and the
control is a position rather than a distance along a route — so all 40 are
reachable, and there are infinitely many blends between them instead of 19
fixed hops.

## Why it does not just become mush

"Can we cheat a bit — I don't want the user to be able to go too far from a
patch, we want the sounds as predictably good as possible" (user, 2026-08-03).

So the blend is deliberately NOT a smooth average over the whole plane. Weights
are inverse distance raised to a high power, which makes the surface a **soft
Voronoi**: across most of its area one anchor dominates so completely that you
are, to the ear, standing on that validated patch. The blending happens in
narrow bands near the boundaries between neighbours. Turn `sharpness` up and
the map approaches "twenty discrete patches with short crossfades"; turn it
down and it approaches a continuous wash. The default sits near the former.

That also answers "travel through the patches, not near them": at an anchor's
own coordinates its weight is 1 and the reproduction is exact.
"""

from __future__ import annotations

import numpy as np

# Neighbours considered at any point. Beyond ~4 the extra weights are
# vanishingly small at usable sharpness, and each one costs a parameter blend.
MAP_NEIGHBOURS = 4
# Inverse-distance exponent — the "cheat" knob. Higher = more Voronoi-like.
#
# Measured over a 40-anchor map, fraction of the surface where one anchor wins
# outright (so the sound IS that validated patch), and the mean dominance of
# the nearest anchor elsewhere:
#
#     sharpness   exact patch   mean top weight
#         2.0          5%            0.53        a wash — everything is a blend
#         4.0         24%            0.70
#         8.0         52%            0.84
#        12.0         66%            0.89
#        20.0         77%            0.93        <- default
#
# Raised from 12 to 20 on 2026-08-05 for SAFETY, not taste. The lurches that
# survived pruning all lived in hybrid territory — a blend of two safe filter
# settings can resonate where neither does — so shrinking that territory is a
# more honest fix than sampling it more densely and hoping.
#
# 20 keeps ~three quarters of the map standing on a checked patch while leaving
# real hybrid territory in the seams, which is the stated goal: "I don't want
# the user to be able to go too far from a patch". Shipped IN the artifact so it
# can be retuned by ear without a code change.
MAP_SHARPNESS = 20.0
# Above this dominance the nearest anchor simply wins, so most of the surface
# reproduces a validated patch EXACTLY rather than to within rounding.
MAP_SNAP = 0.9
_EPS = 1e-9


def embed_2d(z: np.ndarray) -> np.ndarray:
    """Lay descriptor rows out in the unit square. Deterministic. Pure.

    PCA keeps the two directions the pool actually varies along, so patches
    that sound alike land near each other and the map reads as a space rather
    than a scatter. The sign of a principal component is arbitrary, so it is
    pinned (largest-|loading| entry made positive) — otherwise the same corpus
    could produce a mirrored map on a re-run and every stored coordinate would
    silently change meaning.
    """
    n = len(z)
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.array([[0.5, 0.5]])
    x = np.asarray(z, dtype=np.float64)
    x = x - x.mean(axis=0)
    # economy SVD: rows are observations, so V rows are the components
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2] if len(vt) >= 2 else np.vstack([vt, np.zeros_like(vt[0])])
    for i in range(comps.shape[0]):
        if comps[i][np.argmax(np.abs(comps[i]))] < 0:
            comps[i] = -comps[i]
    xy = x @ comps.T

    # normalise each axis into [0,1]; a degenerate axis collapses to centre
    out = np.zeros_like(xy)
    for j in range(2):
        lo, hi = float(xy[:, j].min()), float(xy[:, j].max())
        out[:, j] = 0.5 if hi - lo < _EPS else (xy[:, j] - lo) / (hi - lo)
    return out


def blend_weights(
    points: np.ndarray,
    xy: tuple[float, float] | np.ndarray,
    k: int = MAP_NEIGHBOURS,
    sharpness: float = MAP_SHARPNESS,
    snap: float = MAP_SNAP,
) -> tuple[np.ndarray, np.ndarray]:
    """(indices, weights) of the anchors contributing at `xy`. Pure.

    This is the whole runtime of the map, and the TypeScript panel mirrors it
    exactly — any change here must be mirrored there or the artifact stops
    describing what the user hears.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    p = np.asarray(xy, dtype=np.float64)
    d = np.linalg.norm(np.asarray(points, dtype=np.float64) - p, axis=1)

    idx = np.argsort(d, kind="stable")[: min(k, n)]
    dn = d[idx]
    if dn[0] < _EPS:                      # standing exactly on an anchor
        w = np.zeros(len(idx))
        w[0] = 1.0
        return idx, w

    w = 1.0 / np.power(dn, sharpness)
    w = w / w.sum()
    if w[0] >= snap:                      # one anchor dominates — let it win
        w = np.zeros(len(idx))
        w[0] = 1.0
    return idx, w


def params_at_xy(
    points: np.ndarray,
    snapshots: np.ndarray,
    xy: tuple[float, float] | np.ndarray,
    k: int = MAP_NEIGHBOURS,
    sharpness: float = MAP_SHARPNESS,
    snap: float = MAP_SNAP,
) -> np.ndarray:
    """The parameter vector at a point on the map. Pure."""
    idx, w = blend_weights(points, xy, k=k, sharpness=sharpness, snap=snap)
    if len(idx) == 0:
        return np.zeros(0)
    return (np.asarray(snapshots, dtype=np.float64)[idx] * w[:, None]).sum(axis=0)


# Mirrors CONFIDENCE_FLOOR in TimbreGraphPanel.tsx. The panel plays a position
# at this fraction of full when it has no confidence, scaling to full when one
# validated patch dominates — so a safety gate must measure `rms * this` to
# hear what the user hears.
CONFIDENCE_FLOOR = 0.15


def level_for_confidence(confidence: float) -> float:
    c = min(1.0, max(0.0, float(confidence)))
    return CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * c


def grid_points(n: int) -> list[tuple[float, float]]:
    """An n x n sample of the unit square, for build-time validation."""
    if n <= 1:
        return [(0.5, 0.5)]
    step = 1.0 / (n - 1)
    return [(i * step, j * step) for i in range(n) for j in range(n)]
