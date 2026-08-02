"""The QC gate must judge waveform shape, not loudness.

Measured 2026-08-02: the original peak-fraction test rejected 13% of anchors
whose renders were merely hot — peaks of 1.4-3.9 with a longest pinned-run of
1 sample, i.e. no flat-topping anywhere. pedalboard returns float32 and never
truncates at 1.0, and descriptors are computed on RMS-normalized audio, so
those anchors were perfectly measurable. These tests pin the distinction.
"""

import numpy as np

from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES, extract_descriptors
from timbre_graph_lab.worker import RUNAWAY_PEAK, qc_audio

SR = 44100


def _tone(amp=0.3, f=220.0, n=SR):
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_loud_float_audio_passes():
    """Peak far above 1.0 with no flat-topping is valid, just hot."""
    loud = _tone(amp=3.5)
    qc = qc_audio(loud)
    assert qc.ok, qc.reason
    assert qc.peak > 3.0
    assert qc.pinned_run < 8


def test_genuinely_hard_clipped_audio_is_rejected():
    """Flat-topped at unity => truncated waveform, unmeasurable."""
    clipped = np.clip(_tone(amp=3.5), -1.0, 1.0).astype(np.float32)
    qc = qc_audio(clipped)
    assert not qc.ok
    assert qc.reason == "clipping"
    assert qc.pinned_run >= 8


def test_silence_still_rejected():
    assert qc_audio(np.zeros(SR, dtype=np.float32)).reason == "silent"


def test_nonfinite_rejected():
    bad = _tone()
    bad[100] = np.nan
    assert qc_audio(bad).reason == "empty-or-nonfinite"


def test_runaway_level_rejected():
    qc = qc_audio(_tone(amp=RUNAWAY_PEAK * 2))
    assert not qc.ok
    assert qc.reason == "runaway-level"


def test_quiet_but_audible_passes():
    qc = qc_audio(_tone(amp=0.002))
    assert qc.ok, qc.reason


def _decaying_tone(amp: float) -> np.ndarray:
    """Percussive-ish signal: a steady sine makes decay_slope ill-conditioned."""
    t = np.arange(SR) / SR
    env = np.exp(-6.0 * t)
    sig = np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 1330 * t)
    return (amp * env * sig).astype(np.float32)


def test_descriptors_are_level_invariant():
    """Why re-accepting loud anchors does not invalidate existing shards.

    Spectral/temporal descriptors are computed on RMS-normalized audio, so a
    hot render and a quiet one yield the same timbre measurement; only the
    explicit loudness dims move. This is what makes the QC-gate change a
    pure accept/reject change: shards already on disk stay comparable.
    """
    zq = extract_descriptors(_decaying_tone(0.05), SR)
    zh = extract_descriptors(_decaying_tone(3.5), SR)

    loud_dims = {"loud_rms_db", "loud_peak_db"}
    for i, name in enumerate(DESCRIPTOR_NAMES):
        diff = abs(float(zh[i]) - float(zq[i]))
        if name in loud_dims:
            assert diff > 10, f"{name} should track level"
        else:
            scale = max(abs(float(zq[i])), abs(float(zh[i])), 1e-3)
            assert diff / scale < 0.02, f"{name} drifted with level ({diff=})"


def test_crest_factor_is_level_invariant():
    """crest_db is a ratio, so it must not move with absolute level."""
    zq = extract_descriptors(_decaying_tone(0.05), SR)
    zh = extract_descriptors(_decaying_tone(3.5), SR)
    i = DESCRIPTOR_NAMES.index("crest_db")
    assert abs(float(zh[i]) - float(zq[i])) < 0.1
