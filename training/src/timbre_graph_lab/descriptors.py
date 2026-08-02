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
