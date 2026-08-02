"""Anchor-centered perturbation plans (seeded, reproducible).

All edits live in pedalboard raw [0,1] space — the exact space the runtime
can set — on top of a trusted preset anchor. Plan types:

- sensitivity screen: central finite differences to find which params are
  audible for THIS anchor under THIS probe (dead-control filter).
- singles: multiple magnitudes/directions of one sensitive param.
- sparse multis: 2-4 param gestures — realistic sound-design moves.
- drift: short chains of small multi-moves inside a trust radius, so data
  covers a neighbourhood, not just the anchor point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

FD_EPS = 0.04
SINGLE_MAGNITUDES = [0.03, 0.08, 0.15]
MULTI_SIGMA = 0.06
MULTI_CLIP = 0.15
DRIFT_STEP_SIGMA = 0.03
TRUST_RADIUS = 0.25  # L2 in raw space, per-chain


@dataclass
class Edit:
    """One dataset item: a raw-space delta applied to a (possibly drifted) base."""

    kind: str  # "fd" | "single" | "multi" | "drift"
    base_offset: dict[str, float] = field(default_factory=dict)  # from anchor
    delta: dict[str, float] = field(default_factory=dict)


def _rng_for(preset_id: str, role: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{preset_id}:{role}:{seed}".encode()).digest()
    return np.random.default_rng(np.frombuffer(h[:8], dtype=np.uint64)[0])


def sensitivity_pairs(params: list[str]) -> list[Edit]:
    """Central FD pair per param; caller diffs the rendered features."""
    edits: list[Edit] = []
    for p in params:
        edits.append(Edit(kind="fd", delta={p: +FD_EPS}))
        edits.append(Edit(kind="fd", delta={p: -FD_EPS}))
    return edits


def build_plan(
    preset_id: str,
    role: str,
    sensitive_params: list[str],
    n_singles: int = 60,
    n_multis: int = 80,
    n_drift_chains: int = 4,
    drift_len: int = 6,
    seed: int = 0,
) -> list[Edit]:
    """Deterministic gesture plan for one (preset, role) anchor."""
    rng = _rng_for(preset_id, role, seed)
    params = list(sensitive_params)
    if not params:
        return []
    edits: list[Edit] = []

    # singles: cycle sensitive params, alternate direction, vary magnitude
    for i in range(n_singles):
        p = params[i % len(params)]
        mag = SINGLE_MAGNITUDES[i % len(SINGLE_MAGNITUDES)]
        sign = 1.0 if (i // len(params)) % 2 == 0 else -1.0
        edits.append(Edit(kind="single", delta={p: sign * mag}))

    # sparse multis
    for _ in range(n_multis):
        k = int(rng.integers(2, 5))
        chosen = rng.choice(len(params), size=min(k, len(params)), replace=False)
        delta = {}
        for idx in chosen:
            d = float(np.clip(rng.normal(0.0, MULTI_SIGMA), -MULTI_CLIP, MULTI_CLIP))
            if abs(d) > 1e-4:
                delta[params[int(idx)]] = d
        if delta:
            edits.append(Edit(kind="multi", delta=delta))

    # drift chains: pairs measured from progressively drifted bases
    for _ in range(n_drift_chains):
        offset: dict[str, float] = {}
        for _ in range(drift_len):
            k = int(rng.integers(1, 4))
            chosen = rng.choice(len(params), size=min(k, len(params)), replace=False)
            step = {
                params[int(idx)]: float(
                    np.clip(rng.normal(0.0, DRIFT_STEP_SIGMA), -0.1, 0.1)
                )
                for idx in chosen
            }
            # stop the chain at the trust radius
            trial = dict(offset)
            for name, d in step.items():
                trial[name] = trial.get(name, 0.0) + d
            if np.sqrt(sum(v * v for v in trial.values())) > TRUST_RADIUS:
                break
            edits.append(Edit(kind="drift", base_offset=dict(offset), delta=step))
            offset = trial

    return edits
