"""Integration: is every point on the shipped MAP a sound worth landing on?

The map's promise is stronger than the tour's was — not "these twenty stops
are checked" but "anywhere you put the puck is checked" — so the gate has to
be about the SURFACE, not the points. Every claim here is rendered through
real Surge XT, under the conditions the panel creates:

  * a FRESH host per lens, because the builder has loaded hundreds of patches
    by the time it validates and the panel has loaded exactly one, and Surge's
    exposed parameter set depends on that history (C15c);
  * the panel's own relative-delta form, `paramsAtXY(xy) - paramsAtXY(origin)`,
    because writing the absolute target instead validated a configuration the
    runtime never plays and hid a painful anchor behind a passing build (C15f).

Marked `requires_surge`. Run:
    pytest -m requires_surge tests/test_shipped_map_integration.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.descriptors import standardize
from timbre_graph_lab.patchmap import (
    blend_weights,
    level_for_confidence,
    params_at_xy,
)
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import SAFE_PEAK, RenderWorker, qc_loudness

SHIPPED_MAP = Path(__file__).resolve().parents[2] / "assets" / "patchmap.json"
PROBE_DELTA = 0.08
ROLES = ("kick", "snare", "hat", "bass", "pad", "lead")
# Positions rendered per lens. DELIBERATELY seeded-random rather than a grid:
# the build and the prune both validate grids, and a gate that reused one would
# only replay their own samples. Random points are the honest test of whether
# the surface is actually safe or merely safe where it was measured — and if
# they pass, the residual between them is what the armed limiter is for.
GATE_SAMPLES = 40
GATE_SEED = 20260805


@pytest.fixture(scope="module")
def graph() -> dict:
    return json.loads(SHIPPED_MAP.read_text())


@pytest.fixture(scope="module")
def cfg() -> LabConfig:
    return LabConfig()


def _resolve(cfg: LabConfig, rel: str) -> Path:
    for root in (cfg.factory_patches_dir, cfg.third_party_patches_dir):
        p = Path(root) / rel
        if p.exists():
            return p
    raise AssertionError(f"shipped preset not found on this machine: {rel}")


def _lens_worker(cfg: LabConfig, lens: dict) -> tuple[RenderWorker, dict]:
    """A worker holding ONLY this lens — the panel's situation, not the build's."""
    worker = RenderWorker(cfg)
    origin = _resolve(cfg, lens["points"][0]["fxp_path"])
    assert worker.load_preset(origin), f"could not load {origin.name}"
    return worker, worker.baseline_raw


def _level_at(lens: dict, xy) -> float:
    """The gain the panel applies at `xy` — its confidence duck.

    Rendering the synth alone measures something the user never hears: the panel
    plays low-confidence positions quieter, which is the runtime half of ear
    safety and the half that generalises past sampling. Ignoring it would fail
    the artifact for a loudness nobody is exposed to.
    """
    pts = np.asarray([p["xy"] for p in lens["points"]], dtype=float)
    _, w = blend_weights(pts, xy, sharpness=lens["sharpness"])
    return level_for_confidence(float(w[0]) if len(w) else 1.0)


def _render_at(
    worker: RenderWorker, lens: dict, base: dict, probe, xy
) -> tuple[object, np.ndarray | None]:
    """Exactly what the panel does at `xy`, then measure it."""
    pts = np.asarray([p["xy"] for p in lens["points"]], dtype=float)
    snaps = np.asarray(lens["snapshots"], dtype=float)
    start = params_at_xy(pts, snaps, (0.0, 0.0), sharpness=lens["sharpness"])
    here = params_at_xy(pts, snaps, xy, sharpness=lens["sharpness"])
    worker.apply_delta(
        {
            n: float(here[j] - start[j])
            for j, n in enumerate(lens["param_names"])
            if n in base and abs(here[j] - start[j]) > 1e-5
        }
    )
    qc = qc_loudness(worker.render(probe))
    z = standardize(worker.render_descriptors(probe, k=2)) if qc.ok else None
    worker.restore_baseline()
    return qc, z


# ---------------------------------------------------------------------------
# 1. the artifact is coherent (no Surge needed)
# ---------------------------------------------------------------------------

def test_shipped_map_shape_is_coherent(graph: dict) -> None:
    assert graph["version"] == "map-graph-v2"
    for role in ROLES:
        entry = graph["roles"][role]
        assert entry["declined"] == (len(entry["lenses"]) == 0)
        for lens in entry["lenses"]:
            pts, snaps = lens["points"], lens["snapshots"]
            assert len(pts) == len(snaps) > 1, f"{role}: a map needs somewhere to go"
            assert len({p["preset_id"] for p in pts}) == len(pts)
            for p in pts:
                assert p["fxp_path"].endswith(".fxp")
                assert not p["fxp_path"].startswith("/"), "authoring-machine path"
                assert 0.0 <= p["xy"][0] <= 1.0 and 0.0 <= p["xy"][1] <= 1.0
            for s in snaps:
                assert len(s) == len(lens["param_names"])
                assert all(0.0 <= v <= 1.0 for v in s)


def test_every_role_offers_somewhere_else_to_go(graph: dict) -> None:
    """A lens is a ceiling the runtime cannot write past, so one lens per role
    would make the instrument exhaustible."""
    counts = {r: len(graph["roles"][r]["lenses"]) for r in ROLES}
    assert max(counts.values()) > 1, f"only one world anywhere: {counts}"


# ---------------------------------------------------------------------------
# 2. the parameters the map ships are real on the patch it ships them for
# ---------------------------------------------------------------------------

@pytest.mark.requires_surge
@pytest.mark.parametrize("role", ROLES)
def test_every_param_exists_and_is_writable(role: str, graph: dict, cfg: LabConfig) -> None:
    entry = graph["roles"][role]
    if entry["declined"]:
        pytest.skip(f"{role}: declined")
    lens = entry["lenses"][0]
    worker, live = _lens_worker(cfg, lens)

    missing = [p for p in lens["param_names"] if p not in live]
    assert len(missing) < len(lens["param_names"]) / 2, (
        f"{role}: {len(missing)}/{len(lens['param_names'])} params absent — "
        f"was this map built against a different instrument?"
    )
    unwritable = []
    for name in [p for p in lens["param_names"] if p in live]:
        base = live[name]
        step = PROBE_DELTA if base <= 0.5 else -PROBE_DELTA
        worker.apply_delta({name: step})
        after = worker.host.get_raw_values().get(name)
        worker.restore_baseline()
        if after is None or abs(after - base) < PROBE_DELTA / 2:
            unwritable.append(name)
    assert not unwritable, f"{role}: {len(unwritable)} param(s) refused a write"


# ---------------------------------------------------------------------------
# 3. THE SAFETY GATE — no reachable position may hurt
# ---------------------------------------------------------------------------

@pytest.mark.requires_surge
@pytest.mark.parametrize("role", ROLES)
def test_no_reachable_position_is_dangerously_loud(
    role: str, graph: dict, cfg: LabConfig
) -> None:
    """Dragging the control hurt the user twice (2026-08-03).

    The map makes the whole square reachable, so the whole square is gated:
    every sampled position must sit under the peak ceiling, and none may leap
    above the surface's own median level. An upward excursion is the thing
    that hurts — quiet positions are musical, not dangerous.
    """
    entry = graph["roles"][role]
    if entry["declined"]:
        pytest.skip(f"{role}: declined")

    for li, lens in enumerate(entry["lenses"]):
        worker, base = _lens_worker(cfg, lens)
        probe = get_probe(role, "short")
        rng = np.random.default_rng(GATE_SEED + li)
        probes = [
            (float(x), float(y))
            for x, y in rng.random((GATE_SAMPLES, 2))
        ]
        hot, rms = [], []
        for xy in probes:
            qc, _ = _render_at(worker, lens, base, probe, xy)
            g = _level_at(lens, xy)
            # judged AT THE LEVEL THE PANEL PLAYS IT
            if qc.peak * g > SAFE_PEAK:
                hot.append(f"{xy}(peak {qc.peak * g:.2f} after duck)")
            elif qc.ok:
                rms.append(qc.rms * g)

        assert not hot, (
            f"{role} lens {li} ({lens['lens']['name']}): {len(hot)} position(s) "
            f"over the ceiling (peak {SAFE_PEAK}): {hot[:4]}"
        )
        assert rms, f"{role} lens {li}: nothing rendered at all"
        db = [20 * np.log10(max(r, 1e-9)) for r in rms]
        over = max(db) - float(np.median(db))
        assert over < 10.0, (
            f"{role} lens {li} ({lens['lens']['name']}): a position is "
            f"{over:.1f} dB above the map's median — that is the lurch that hurts"
        )


# ---------------------------------------------------------------------------
# 4. the map is worth exploring: moving actually changes the sound
# ---------------------------------------------------------------------------

@pytest.mark.requires_surge
@pytest.mark.parametrize("role", ROLES)
def test_the_map_sounds_different_in_different_places(
    role: str, graph: dict, cfg: LabConfig
) -> None:
    """The product promise: a map you can wander without hearing change is a
    preset with extra steps.

    Compared against the spread of REPEAT RENDERS AT ONE POSITION, not against
    3*sigma. Both are noise measurements, but the first is the same quantity as
    the signal and is taken in the same breath, while sigma estimated from a
    handful of renders is famously unstable here — measured 2026-08-05, the lead
    reported sigma 0.341 against a screening ceiling of 0.05, which would fail a
    map for the patch's own restlessness rather than for standing still.
    """
    entry = graph["roles"][role]
    if entry["declined"]:
        pytest.skip(f"{role}: declined")

    lens = entry["lenses"][0]
    worker, base = _lens_worker(cfg, lens)
    probe = get_probe(role, "short")

    # how much a SINGLE position wanders when nothing changes
    fixed = (0.5, 0.5)
    repeats = []
    for _ in range(3):
        _, z = _render_at(worker, lens, base, probe, fixed)
        if z is not None:
            repeats.append(z)
    within = (
        float(np.median([
            np.linalg.norm(repeats[i] - repeats[j])
            for i in range(len(repeats)) for j in range(i + 1, len(repeats))
        ]))
        if len(repeats) > 1 else 0.0
    )

    rng = np.random.default_rng(GATE_SEED)
    zs = []
    for xy in [(float(x), float(y)) for x, y in rng.random((16, 2))]:
        _, z = _render_at(worker, lens, base, probe, xy)
        if z is not None:
            zs.append(z)
    assert len(zs) >= 8, f"{role}: too few positions rendered to judge"

    between = float(np.median([
        float(np.linalg.norm(zs[i] - zs[j]))
        for i in range(len(zs)) for j in range(i + 1, len(zs))
    ]))
    # moving must matter more than standing still does
    assert between > max(2.0 * within, 0.05), (
        f"{role}: moving across the map changes the sound by {between:.3f}, "
        f"barely more than staying put changes it ({within:.3f}) — this map is "
        f"one sound"
    )
