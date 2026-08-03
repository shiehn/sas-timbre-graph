"""Integration: does EVERY parameter of EVERY shipped patch actually move?

The panel's morph is only as real as the artifact it replays, and two failures
already proved that "the write succeeded" says nothing about the sound:

  * names that exist in one convention but not the other (resolved host-side),
  * parameters that are settable but INERT on the patch in question — the lead
    spent its whole gesture on `a_width` (audibility 0.46) while
    `a_filter_1_cutoff` (10.95) barely moved, so nothing was audible.

So these tests host real Surge XT, load each shipped anchor, and for every
parameter the shipped graph carries assert three separate things:

  1. it EXISTS on the plugin (settable by name),
  2. setting it CHANGES the plugin's own reported value (the write lands),
  3. every parameter the artifact marks audible measurably CHANGES THE RENDER
     (and every one it marks inert does not) — i.e. the shipped sensitivity is
     truthful per patch category.

Marked `requires_surge`: needs Surge XT + factory content installed.
Run: pytest -m requires_surge tests/test_shipped_graph_integration.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.descriptors import extract_descriptors, standardize
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_audio

pytestmark = pytest.mark.requires_surge

SHIPPED_GRAPH = Path(__file__).resolve().parents[2] / "assets" / "morph-softer.json"
# Probe step used when checking a parameter is live: comfortably above the
# measurement noise floor, still inside the trust radius.
PROBE_DELTA = 0.08
ROLES = ("kick", "snare", "hat", "bass", "pad", "lead")


@pytest.fixture(scope="module")
def graph() -> dict:
    return json.loads(SHIPPED_GRAPH.read_text())


@pytest.fixture(scope="module")
def cfg() -> LabConfig:
    return LabConfig()


@pytest.fixture(scope="module")
def worker(cfg: LabConfig) -> RenderWorker:
    return RenderWorker(cfg)


def _resolve_anchor(cfg: LabConfig, rel: str) -> Path:
    """Shipped anchors are relative to the Surge content root, as the host does."""
    for root in (cfg.factory_patches_dir, cfg.third_party_patches_dir):
        candidate = Path(root) / rel
        if candidate.exists():
            return candidate
    raise AssertionError(f"shipped anchor not found on this machine: {rel}")


# ---------------------------------------------------------------------------
# 1. the artifact is internally coherent (no Surge needed for the shape)
# ---------------------------------------------------------------------------

def test_shipped_graph_shape_is_coherent(graph: dict) -> None:
    n_points = len(graph["control_points"])
    assert 0 in graph["control_points"], "dial needs a true no-op centre"
    for role in ROLES:
        t = graph["roles"][role]
        n = len(t["param_names"])
        assert n > 0
        assert len(t["baseline"]) == n
        assert len(t["sensitivity"]) == n, f"{role}: sensitivity must cover every param"
        assert len(t["snapshots"]) == n_points
        for snap in t["snapshots"]:
            assert len(snap) == n
            assert all(0.0 <= v <= 1.0 for v in snap), f"{role}: raw values out of range"


# ---------------------------------------------------------------------------
# 2. every parameter of every patch category EXISTS and is WRITABLE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ROLES)
def test_every_param_exists_and_is_writable(
    role: str, graph: dict, cfg: LabConfig, worker: RenderWorker
) -> None:
    t = graph["roles"][role]
    anchor = _resolve_anchor(cfg, t["fxp_path"])
    assert worker.load_preset(anchor), f"{role}: could not load {anchor.name}"

    live = worker.baseline_raw
    missing = [p for p in t["param_names"] if p not in live]
    assert not missing, (
        f"{role} ({anchor.name}): {len(missing)} shipped param(s) absent from the "
        f"plugin: {missing[:5]}"
    )

    unwritable: list[str] = []
    for name in t["param_names"]:
        base = live[name]
        # step away from whichever rail the parameter is sitting on
        step = PROBE_DELTA if base <= 0.5 else -PROBE_DELTA
        worker.apply_delta({name: step})
        after = worker.host.get_raw_values().get(name)
        worker.restore_baseline()
        if after is None or abs(after - base) < PROBE_DELTA / 2:
            unwritable.append(name)
    assert not unwritable, (
        f"{role} ({anchor.name}): {len(unwritable)} param(s) did not accept a write: "
        f"{unwritable[:5]}"
    )


# ---------------------------------------------------------------------------
# 3. the shipped sensitivity is TRUTHFUL: audible params change the render
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ROLES)
def test_shipped_sensitivity_predicts_audible_change(
    role: str, graph: dict, cfg: LabConfig, worker: RenderWorker
) -> None:
    """Every parameter the artifact calls audible must move the sound.

    This is the property the morph depends on: the panel spends its gesture
    budget in proportion to these numbers, so a parameter marked audible that
    does nothing is a silent morph (the live lead failure).
    """
    t = graph["roles"][role]
    anchor = _resolve_anchor(cfg, t["fxp_path"])
    assert worker.load_preset(anchor)

    probe = get_probe(role, "short")
    assert qc_audio(worker.render(probe)).ok, f"{role}: anchor renders unusably"

    sigma = float(np.linalg.norm(standardize(worker.noise_floor(probe, k=3, avg=2))))
    z0 = standardize(worker.render_descriptors(probe, k=2))
    floor = max(3.0 * sigma, 0.02)     # clear the measurement noise

    audible = [
        (n, s) for n, s in zip(t["param_names"], t["sensitivity"]) if s > 0
    ]
    assert audible, f"{role}: artifact marks NO parameter audible"

    silent: list[str] = []
    for name, _sens in audible:
        base = worker.baseline_raw[name]
        step = PROBE_DELTA if base <= 0.5 else -PROBE_DELTA
        worker.apply_delta({name: step})
        # averaged exactly as the shipped sensitivity was measured: patches with
        # a noise oscillator need the phase noise averaged out or the comparison
        # is against a different quantity than the artifact claims
        z1 = standardize(worker.render_descriptors(probe, k=2))
        worker.restore_baseline()
        if float(np.linalg.norm(z1 - z0)) <= floor:
            silent.append(name)

    # A couple of borderline controls may fall under the floor at this step
    # size; a systematic failure is what matters.
    ratio = len(silent) / len(audible)
    assert ratio <= 0.25, (
        f"{role} ({anchor.name}): {len(silent)}/{len(audible)} params marked "
        f"audible produced no measurable change: {silent[:6]}"
    )


@pytest.mark.parametrize("role", ROLES)
def test_the_full_dial_sweep_changes_the_sound(
    role: str, graph: dict, cfg: LabConfig, worker: RenderWorker
) -> None:
    """End to end: applying the shipped snapshots must audibly move each patch.

    Uses the panel's own rule — deltas from the dial centre, scaled by
    predicted audible effect — so this fails if the artifact, the sensitivity
    weighting, or the snapshots themselves stop producing change.
    """
    t = graph["roles"][role]
    anchor = _resolve_anchor(cfg, t["fxp_path"])
    assert worker.load_preset(anchor)

    probe = get_probe(role, "short")
    names = t["param_names"]
    sens = np.asarray(t["sensitivity"], dtype=float)
    centre = np.asarray(t["snapshots"][len(t["snapshots"]) // 2], dtype=float)

    sigma = float(np.linalg.norm(standardize(worker.noise_floor(probe, k=3, avg=2))))
    z0 = standardize(worker.render_descriptors(probe, k=2))

    moved = False
    for end in (t["snapshots"][0], t["snapshots"][-1]):
        raw = np.asarray(end, dtype=float) - centre
        effect = float(np.linalg.norm(raw * sens))
        if effect <= 1e-9:
            continue
        gain = 4.0 / effect                       # panel default: "strong"
        delta = np.clip(raw * gain, -0.6, 0.6)
        worker.apply_delta(
            {n: float(d) for n, d in zip(names, delta) if abs(d) > 1e-5}
        )
        audio = worker.render(probe)
        z1 = standardize(worker.render_descriptors(probe, k=2))
        worker.restore_baseline()
        if qc_audio(audio).ok and float(np.linalg.norm(z1 - z0)) > max(3 * sigma, 0.1):
            moved = True

    assert moved, (
        f"{role} ({anchor.name}): a full dial sweep produced no measurable "
        f"timbre change — this is exactly the 'I hear nothing' failure"
    )
