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


# --- parameter policy: switches must not be treated as continuous ----------

class _FakeParam:
    """Mimics pedalboard: num_steps/is_boolean lie, range is often unreported."""

    def __init__(self, rng, displays):
        self.range = rng
        self.num_steps = 2147483647
        self.is_boolean = False
        self._displays = displays
        self.raw_value = 0.0

    @property
    def string_value(self):
        idx = min(int(self.raw_value * (len(self._displays) - 1) + 0.5),
                  len(self._displays) - 1)
        return self._displays[idx]


def test_declared_boolean_is_discrete():
    from timbre_graph_lab.policy import _is_discrete
    assert _is_discrete(_FakeParam((False, True, 1), ["Off", "On"]))


def test_continuous_with_unreported_range_is_allowed():
    """Envelope stages report range (None,None,None) but sweep numerically."""
    from timbre_graph_lab.policy import _is_discrete
    eg = _FakeParam((None, None, None),
                    ["0.0 ms", "12.3 ms", "37.2 ms", "353.6 ms", "8.00 s"])
    assert not _is_discrete(eg)


def test_continuous_with_reported_range_is_allowed():
    from timbre_graph_lab.policy import _is_discrete
    cutoff = _FakeParam((13.75, 25087.71, None),
                        ["13.75 Hz", "89.87 Hz", "587.33 Hz",
                         "3.84 kHz", "25.09 kHz"])
    assert not _is_discrete(cutoff)


def test_negative_numeric_display_is_continuous():
    from timbre_graph_lab.policy import _is_discrete
    feg = _FakeParam((None, None, None),
                     ["-96.00 semitones", "-48.00 semitones", "0.00 semitones",
                      "48.00 semitones", "96.00 semitones"])
    assert not _is_discrete(feg)


def test_enum_with_many_distinct_text_values_is_discrete():
    """filter_configuration sweeps 5 DISTINCT values — but they are labels."""
    from timbre_graph_lab.policy import _is_discrete
    cfg = _FakeParam((None, None, None),
                     ["Serial 1", "Serial 2", "Serial 3", "Dual 1", "Dual 2"])
    assert _is_discrete(cfg)


def test_routing_enum_is_discrete():
    from timbre_graph_lab.policy import _is_discrete
    route = _FakeParam((None, None, None),
                       ["Filter 1", "Filter 1", "Both", "Filter 2", "Filter 2"])
    assert _is_discrete(route)


def test_switch_names_are_denied():
    from timbre_graph_lab.policy import _denied
    for n in ("a_osc_1_mute", "a_noise_solo", "a_osc_2_route",
              "a_osc_1_retrigger", "a_master_volume"):
        assert _denied(n), n
    for n in ("a_filter_1_cutoff", "a_amp_eg_release", "a_highpass"):
        assert not _denied(n), n


def test_core_params_resolve_and_are_shared_across_anchors():
    """Every anchor must exercise the same core columns (Jaccard was 0.205)."""
    from timbre_graph_lab.gen import core_params
    allowed = ["a_filter_1_cutoff", "a_amp_eg_release", "a_highpass",
               "a_noise_color", "a_some_obscure_thing"]
    base_a = {p: 0.5 for p in allowed}
    base_b = {p: 0.4 for p in allowed}
    ca, cb = core_params(allowed, base_a), core_params(allowed, base_b)
    assert ca == cb                       # identical across anchors
    assert "a_filter_1_cutoff" in ca and "a_highpass" in ca
    assert "a_some_obscure_thing" not in ca
