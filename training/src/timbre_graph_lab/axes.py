"""Semantic timbre axes — the vocabulary the dial speaks.

An axis is a direction in STANDARDIZED descriptor space with a name a musician
would recognise. The same axis means the same thing for every role, but each
synth realises it with whatever parameters it happens to have: "longer" is a
release change on a pad and an amp-envelope change on a kick.

The library is deliberately redundant, including near-opposites and several
envelope variants, because which axis a given patch can express is decided by
MEASUREMENT (`achievability`), not by taste. Two findings drove the design:

- Roles differ enormously in what they can reach. A kick measured **0.000**
  reach in `band_high` and `band_air` — "brighter" is not a hard target for
  it, it is a physically impossible one. Its expressive dimensions are
  envelope and dynamics, where it reaches 0.70-0.86.
- Sign coupling matters. An early "longer" axis paired `decay_slope+` with
  `attack_time+`; the kick's own reachable direction couples `decay_slope+`
  with `attack_time-`, so the request was unreachable even though each
  component was individually movable. Hence `longer` and `boomier` both exist.
"""

from __future__ import annotations

import numpy as np

from timbre_graph_lab.config import POLICY_VERSION
from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES
from timbre_graph_lab.solver import unit


def make_axis(**weights: float) -> np.ndarray:
    """Build a unit direction from named descriptor weights."""
    v = np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float64)
    for name, w in weights.items():
        v[DESCRIPTOR_NAMES.index(name)] = w
    return unit(v)


# Each entry is one direction; its negative is the opposite gesture, so the
# dial can travel both ways along any axis.
AXES: dict[str, np.ndarray] = {
    # --- spectral ---------------------------------------------------------
    "brighter": make_axis(centroid_mean=1.0, rolloff85_mean=1.0,
                          band_high=1.0, band_air=0.5, band_low=-0.5),
    "fuller": make_axis(band_sub=1.0, band_low=1.0,
                        bandwidth_mean=0.5, band_high=-0.5),
    "wider": make_axis(bandwidth_mean=1.0, centroid_std=1.0,
                       contrast_mean=-0.3),
    # --- envelope ---------------------------------------------------------
    # decay_slope is dB/s and negative while a sound decays, so a larger
    # (less negative) slope means a longer tail.
    "longer": make_axis(decay_slope=1.0, env_sparsity=-1.0, attack_time=0.3),
    "boomier": make_axis(decay_slope=1.0, attack_time=-0.5, env_sparsity=-0.5),
    "tighter": make_axis(decay_slope=-1.0, attack_time=-0.3, env_sparsity=0.5),
    # --- dynamics ---------------------------------------------------------
    "punchier": make_axis(crest_db=1.0, loud_peak_db=0.5, attack_time=-0.5),
    "softer": make_axis(crest_db=-1.0, attack_time=1.0, loud_peak_db=-0.3),
    # --- texture ----------------------------------------------------------
    "rougher": make_axis(flatness_mean=1.0, zcr_mean=1.0, contrast_mean=-0.5),
}

MIN_ACHIEVABLE_COSINE = 0.40


def achievability(
    measure,
    response,
    budget: int = 60,
    axes: dict[str, np.ndarray] | None = None,
    seed: int = 1,
) -> dict[str, float]:
    """Which axes can THIS anchor actually express, and how well.

    Render-verified per axis, so the resulting table is the coupling policy —
    measured rather than hand-tuned. Axes the refiner declines are recorded as
    0.0, which is meaningfully different from a small positive score: it means
    standing still beat every move tried.
    """
    from timbre_graph_lab.refine import refine

    axes = axes or AXES
    out: dict[str, float] = {}
    for name, axis in axes.items():
        res = refine(measure, response.J, response.usable, axis,
                     budget=budget, seed=seed)
        out[name] = round(float(res.cosine), 4) if res.moved else 0.0
    return out


def best_axes(table: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    """The axes this anchor expresses best, strongest first."""
    good = [(n, c) for n, c in table.items() if c >= MIN_ACHIEVABLE_COSINE]
    return sorted(good, key=lambda kv: -kv[1])[:k]


def shared_axes(tables: dict[str, dict[str, float]]) -> list[tuple[str, int, float]]:
    """Axes that the most roles can express — candidates for the global dial.

    Returns (axis, n_roles_that_can_express_it, median_cosine_among_those),
    ordered by coverage then strength. A dial built on a widely-expressible
    axis moves more of the ensemble at once; a role that cannot express it
    simply holds still, which is the honest fallback.
    """
    rows = []
    for name in next(iter(tables.values())):
        vals = [t[name] for t in tables.values() if t.get(name, 0.0) >= MIN_ACHIEVABLE_COSINE]
        if vals:
            rows.append((name, len(vals), round(float(np.median(vals)), 4)))
    return sorted(rows, key=lambda r: (-r[1], -r[2]))


def _cache_path(cfg) -> "Path":
    from pathlib import Path

    return Path(cfg.workspace) / "achievability_cache.json"


def achievability_cached(
    measure,
    response,
    cfg,
    budget: int = 60,
    axes: dict[str, np.ndarray] | None = None,
    seed: int = 1,
) -> dict[str, float]:
    """`achievability` memoised per preset on disk.

    Probing costs ~5.6 s per (role, axis) pair, so a six-anchor set against the
    full library is minutes — but a preset's reachable axes depend only on the
    patch, the probe and the policy, never on the other five anchors. Keying on
    (preset_id, role, policy version, axis set) makes re-picking a previously
    seen anchor free.
    """
    import hashlib
    import json

    axes = axes or AXES
    tag = hashlib.sha1(
        "|".join(sorted(axes)).encode() + str(budget).encode()
    ).hexdigest()[:8]
    key = f"{response.preset_id or response.name}/{response.role}/{POLICY_VERSION}/{tag}"

    path = _cache_path(cfg)
    cache = json.loads(path.read_text()) if path.exists() else {}
    if key in cache:
        return cache[key]

    table = achievability(measure, response, budget=budget, axes=axes, seed=seed)
    cache[key] = table
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return table
