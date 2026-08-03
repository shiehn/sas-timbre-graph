"""Explicit perceptual descriptors — the PRIMARY feature basis for v0/v1.

CLAP embeddings are noisy under small parameter perturbations; explicit DSP
descriptors give a stable, interpretable local basis (docs/TRAINING.md,
challenge #2). CLAP can be added later as a separate, separately-weighted
block without invalidating shards (blocks are stored independently).

Loudness policy: spectral/temporal descriptors are computed on peak-RMS
normalized audio; loudness itself is kept as explicit separate dimensions so
"turn it up" can never masquerade as a timbre move.
"""

from __future__ import annotations

import librosa
import numpy as np

DESCRIPTOR_NAMES = [
    "loud_rms_db",
    "loud_peak_db",
    "crest_db",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "rolloff85_mean",
    "flatness_mean",
    "zcr_mean",
    "band_sub",       # <60 Hz
    "band_low",       # 60-250
    "band_lowmid",    # 250-1k
    "band_mid",       # 1k-4k
    "band_high",      # 4k-10k
    "band_air",       # >10k
    "attack_time",
    "decay_slope",
    "env_sparsity",
    "env_flux",
    "contrast_mean",
]

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

# Per-descriptor scale for comparing timbre CHANGES.
#
# The raw descriptors live in wildly different units — rolloff and bandwidth
# in Hz (thousands), band fractions in [0,1], loudness in dB — so any distance
# or cosine taken on raw vectors is decided almost entirely by the Hz-scale
# dimensions. Measured over the full corpus: rolloff85_mean and
# bandwidth_mean alone carried 85% of all delta energy while 15 of the 20
# descriptors contributed ~0%. A solver working in that space is optimising
# spectral rolloff and ignoring band balance, envelope shape and loudness
# entirely, which produced erratic and even anti-correlated coupling.
#
# These are the per-dimension standard deviations across the v1 corpus
# (2,111 anchors / 775,654 observations), baked in as constants so the runtime
# never needs the corpus to interpret a timbre delta.
FEATURE_SCALE = np.array([
    13.96067,      # loud_rms_db
    12.72466,      # loud_peak_db
    4.86431,       # crest_db
    1554.48010,    # centroid_mean
    1077.25525,    # centroid_std
    1283.77832,    # bandwidth_mean
    2823.85596,    # rolloff85_mean
    0.18004,       # flatness_mean
    0.05579,       # zcr_mean
    0.21499,       # band_sub
    0.34542,       # band_low
    0.35641,       # band_lowmid
    0.19408,       # band_mid
    0.07176,       # band_high
    0.06303,       # band_air
    0.59352,       # attack_time
    26.42802,      # decay_slope
    0.23731,       # env_sparsity
    0.40359,       # env_flux
    6.13604,       # contrast_mean
], dtype=np.float64)


def standardize(z: np.ndarray) -> np.ndarray:
    """Put a descriptor vector (or delta) into comparable per-dimension units."""
    return np.asarray(z, dtype=np.float64) / FEATURE_SCALE

_BANDS = [(20, 60), (60, 250), (250, 1000), (1000, 4000), (4000, 10000), (10000, 20000)]


def extract_descriptors(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Fixed-length descriptor vector; finite for silent input."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    n = len(audio)
    if n == 0:
        return np.zeros(N_DESCRIPTORS, dtype=np.float32)

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio))) if n else 0.0
    loud_rms_db = 20 * np.log10(max(rms, 1e-6))
    loud_peak_db = 20 * np.log10(max(peak, 1e-6))
    crest_db = loud_peak_db - loud_rms_db

    if rms < 1e-5:
        out = np.zeros(N_DESCRIPTORS, dtype=np.float32)
        out[0], out[1], out[2] = loud_rms_db, loud_peak_db, crest_db
        return out

    # normalize for all spectral/temporal work
    x = audio / rms * 0.1

    n_fft, hop = 2048, 512
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(x, hop_length=hop)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    total_energy = float(np.sum(S**2)) + 1e-12
    band_fracs = []
    for lo, hi in _BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band_fracs.append(float(np.sum(S[mask] ** 2)) / total_energy)

    # envelope features
    env = librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr)
    frame_rms = librosa.feature.rms(S=S)[0]
    peak_idx = int(np.argmax(frame_rms))
    frame_t = hop / sr
    # attack: time from 10% to 90% of peak frame-rms before the peak
    pre = frame_rms[: peak_idx + 1]
    if len(pre) > 1 and pre.max() > 0:
        t10 = np.argmax(pre >= 0.1 * pre.max())
        t90 = np.argmax(pre >= 0.9 * pre.max())
        attack_time = max(t90 - t10, 0) * frame_t
    else:
        attack_time = 0.0
    # decay: dB/s slope over the post-peak tail
    post = frame_rms[peak_idx:]
    if len(post) > 4:
        db = 20 * np.log10(np.maximum(post, 1e-6))
        t = np.arange(len(db)) * frame_t
        decay_slope = float(np.polyfit(t, db, 1)[0])
    else:
        decay_slope = 0.0
    env_sparsity = float(np.mean(frame_rms < 0.1 * (frame_rms.max() + 1e-9)))
    env_flux = float(np.mean(np.abs(np.diff(env)))) if len(env) > 1 else 0.0

    out = np.array(
        [
            loud_rms_db,
            loud_peak_db,
            crest_db,
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(bandwidth)),
            float(np.mean(rolloff)),
            float(np.mean(flatness)),
            float(np.mean(zcr)),
            *band_fracs,
            attack_time,
            decay_slope,
            env_sparsity,
            env_flux,
            float(np.mean(contrast)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
