"""Render worker: one Surge host with settle-render discipline and QC.

Benchmark findings this encodes (docs/TRAINING.md §benchmarks):
- Offline rendering is ~100x real-time, so renders are cheap.
- The FIRST render after any parameter change carries parameter-smoothing
  bleed (observed maxdiff 0.65 vs 6e-8 once settled). Every measured render
  is therefore preceded by a throwaway settle render.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synth2surge.audio.engine import PluginHost
from synth2surge.surge.patch import SurgePatch
from synth2surge.surge.preset_loader import (
    _ensure_mapping,
    _get_xml_params,
    load_fxp_into_host,
)

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.probes import Probe

SILENCE_RMS = 1e-4
# True hard clipping is a FLAT-TOPPED waveform: many consecutive samples
# pinned at the same extreme value, with the peak at or below unity.
CLIP_PINNED_RUN = 8
CLIP_CEILING = 1.0001
# Absurd level (~+30 dBFS) means self-oscillation or instability, not music.
RUNAWAY_PEAK = 32.0


@dataclass
class RenderQC:
    rms: float
    peak: float
    pinned_run: int
    ok: bool
    reason: str = ""


def _longest_pinned_run(mag: np.ndarray, peak: float) -> int:
    """Longest run of consecutive samples sitting at the signal's maximum."""
    at_peak = mag >= peak - 1e-6
    if not at_peak.any():
        return 0
    edges = np.flatnonzero(
        np.diff(np.concatenate(([0], at_peak.view(np.int8), [0])))
    )
    return int((edges[1::2] - edges[0::2]).max())


def qc_audio(audio: np.ndarray) -> RenderQC:
    """Reject renders that cannot yield a meaningful timbre measurement.

    Loud is NOT broken. pedalboard returns float32 and never truncates at
    1.0, and the descriptors compute spectral/temporal features on
    RMS-normalized audio, so a hot preset measures the same as a quiet one.
    An earlier peak-fraction test rejected 13% of anchors whose renders were
    merely loud — measured peaks of 1.4-3.9 with pinned-runs of 1, i.e. not a
    single flat-topped sample pair (2026-08-02). Judge the waveform shape
    instead of its level.
    """
    if len(audio) == 0 or not np.all(np.isfinite(audio)):
        return RenderQC(0.0, 0.0, 0, False, "empty-or-nonfinite")
    rms = float(np.sqrt(np.mean(audio**2)))
    mag = np.abs(audio)
    peak = float(np.max(mag))
    if rms < SILENCE_RMS:
        return RenderQC(rms, peak, 0, False, "silent")
    if peak > RUNAWAY_PEAK:
        return RenderQC(rms, peak, 0, False, "runaway-level")
    pinned = _longest_pinned_run(mag, peak)
    if peak <= CLIP_CEILING and pinned >= CLIP_PINNED_RUN:
        return RenderQC(rms, peak, pinned, False, "clipping")
    return RenderQC(rms, peak, pinned, True)


class RenderWorker:
    """Owns one PluginHost; loads presets, applies raw deltas, renders probes."""

    def __init__(self, cfg: LabConfig | None = None) -> None:
        self.cfg = cfg or LabConfig()
        self.host = PluginHost(self.cfg.surge_vst3, sample_rate=self.cfg.sample_rate)
        self._baseline_raw: dict[str, float] = {}
        # params currently deviating from baseline — restoring only these
        # keeps apply_delta at ~ms instead of rewriting all ~750 params
        self._dirty: set[str] = set()

    def load_preset(self, fxp_path: str | Path, freeze: bool = True) -> bool:
        result = load_fxp_into_host(Path(fxp_path), self.host)
        if not result.success:
            return False
        self._fix_conditional_params(Path(fxp_path))
        if freeze:
            self.freeze_stochastic()
        self._baseline_raw = self.host.get_raw_values()
        self._dirty = set()
        return True

    def freeze_stochastic(self) -> None:
        """Measurement convention: make renders repeatable.

        Surge randomizes oscillator start phase (when retrigger is off) and
        applies per-voice drift, so two identical renders differ — measured
        day one, the run-to-run descriptor noise EXCEEDED the finite-
        difference signal for most patches (bass SNR 0.16). Forcing
        retrigger on and drift to zero collapses that noise to ~0 for
        patches without a noise oscillator (snare 24.2 -> 0.0009) and is
        harmless for timbre statistics, which barely depend on start phase.

        Patches that genuinely use the noise oscillator (hats especially)
        stay stochastic; `render_descriptors(k>1)` averages those down and
        `noise_floor()` measures what is left so the caller can gate on it.
        """
        forced = {}
        for name in self.host.parameter_names():
            lname = name.lower()
            if "retrigger" in lname:
                forced[name] = 1.0
            elif "drift" in lname:
                forced[name] = 0.0
        if forced:
            self.host.set_raw_values(forced)

    def _fix_conditional_params(self, fxp_path: Path) -> None:
        """Re-apply oscillator params under the *loaded* oscillator types.

        The session-wide parameter calibration runs under the default patch,
        but Surge osc param ranges depend on osc type (an FM ratio can span
        0..32 while the same slot spans 0..1 for another type). After the
        first load pass has set the types, recalibrate just the osc param
        ranges in the current structure and re-apply the target values —
        otherwise FM/S&H patches load clamped and often silent.
        """
        fxp_params = SurgePatch.from_file(fxp_path).get_all_parameters()
        mapping, _ = _ensure_mapping(self.host)
        xml_to_pb = {v: k for k, v in mapping.items()}

        cond = {
            n: v
            for n, v in fxp_params.items()
            if "_osc" in n and "_param" in n and n in xml_to_pb
        }
        if not cond:
            return
        pb_names = [xml_to_pb[n] for n in cond]
        saved = {
            n: v for n, v in self.host.get_raw_values().items() if n in pb_names
        }

        self.host.set_raw_values({n: 0.0 for n in pb_names})
        xml_at_0 = _get_xml_params(self.host)
        self.host.set_raw_values({n: 1.0 for n in pb_names})
        xml_at_1 = _get_xml_params(self.host)

        target: dict[str, float] = {}
        for xml_name, want in cond.items():
            lo = xml_at_0.get(xml_name)
            hi = xml_at_1.get(xml_name)
            pb = xml_to_pb[xml_name]
            if lo is None or hi is None or abs(hi - lo) < 1e-10:
                target[pb] = saved.get(pb, 0.5)
                continue
            target[pb] = float(np.clip((want - lo) / (hi - lo), 0.0, 1.0))
        self.host.set_raw_values(target)

    @property
    def baseline_raw(self) -> dict[str, float]:
        return dict(self._baseline_raw)

    def restore_baseline(self) -> None:
        if self._dirty:
            self.host.set_raw_values(
                {n: self._baseline_raw[n] for n in self._dirty if n in self._baseline_raw}
            )
            self._dirty = set()

    def apply_delta(self, delta: dict[str, float]) -> dict[str, float]:
        """Apply a raw-space delta on top of the baseline; returns the clamped
        absolute values that were actually set."""
        target = {}
        for name, d in delta.items():
            base = self._baseline_raw.get(name)
            if base is None:
                continue
            target[name] = float(np.clip(base + d, 0.0, 1.0))
        self.restore_baseline()
        if target:
            self.host.set_raw_values(target)
            self._dirty = set(target)
        return target

    def render(self, probe: Probe, settle: bool = True) -> np.ndarray:
        """Measured render with settle discipline."""
        if settle:
            # throwaway pass flushes parameter smoothing
            self.host.reset()
            self.host.render_midi_mono(
                midi_messages=probe.messages[:2], duration=0.25
            )
        self.host.reset()
        return self.host.render_midi_mono(
            midi_messages=probe.messages, duration=probe.duration
        )

    def render_descriptors(self, probe: Probe, k: int = 1) -> np.ndarray:
        """Descriptor vector, averaged over k renders.

        Averaging descriptors (not audio) is the right estimator for the
        residual stochasticity of noise-oscillator patches: it cuts the
        run-to-run spread by sqrt(k) without altering the expected timbre.
        """
        from timbre_graph_lab.descriptors import extract_descriptors

        acc = None
        for _ in range(max(1, k)):
            z = extract_descriptors(self.render(probe), self.cfg.sample_rate)
            acc = z if acc is None else acc + z
        return acc / max(1, k)

    def noise_floor(self, probe: Probe, k: int = 4, avg: int = 1) -> np.ndarray:
        """Per-descriptor standard deviation of repeated identical renders.

        This is the measurement's own uncertainty; any finite-difference
        response smaller than a few times this is unmeasurable, not absent.
        """
        Z = np.stack([self.render_descriptors(probe, k=avg) for _ in range(k)])
        return Z.std(axis=0)
