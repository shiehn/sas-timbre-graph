"""Guards for the render-reproducibility fix (the day-one data-quality bug).

Renders were not repeatable (free-running oscillator phase + per-voice
drift), so finite-difference targets were noise-dominated — within-anchor
predictability was 0.33 and pad was pure noise at 0.006. Freezing phase and
averaging lifted that to 0.74. These tests pin the pieces that must not
silently regress; the Surge-dependent half lives behind `requires_surge`.
"""

import numpy as np

from timbre_graph_lab.dataset import load_shards, write_shard
from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.gen import MIN_SNR, RENDER_AVG


def test_render_averaging_is_enabled():
    # k=1 would restore the noise-dominated regime
    assert RENDER_AVG >= 2
    assert MIN_SNR >= 2.0


def test_shard_round_trips_noise_sigma(tmp_path):
    cfg = LabConfig(workspace=tmp_path)
    n, d, f = 25, 4, 20
    rng = np.random.default_rng(0)
    sigma = rng.random(f).astype(np.float32)
    write_shard(
        cfg, "pad", "sig123", [f"p{i}" for i in range(d)],
        rng.random((n, d)), rng.random((n, d)) * 0.1,
        rng.random((n, f)), rng.random((n, f)),
        ["single"] * n, noise_sigma=sigma,
    )
    s = load_shards(cfg, ["pad"])[0]
    assert s["SIGMA"].shape == (f,)
    np.testing.assert_allclose(s["SIGMA"], sigma, rtol=1e-6)


def test_legacy_shard_without_sigma_still_loads(tmp_path):
    cfg = LabConfig(workspace=tmp_path)
    n, d, f = 25, 4, 20
    rng = np.random.default_rng(1)
    write_shard(
        cfg, "bass", "old456", [f"p{i}" for i in range(d)],
        rng.random((n, d)), rng.random((n, d)) * 0.1,
        rng.random((n, f)), rng.random((n, f)),
        ["single"] * n,  # no noise_sigma -> zeros
    )
    s = load_shards(cfg, ["bass"])[0]
    assert s["SIGMA"].shape == (f,)
    assert np.all(s["SIGMA"] == 0)


def test_parity_gate_rejects_rotated_deltas():
    """A platform that flips gesture direction must FAIL the gate."""
    from timbre_graph_lab.parity import compare

    z = np.arange(20, dtype=float).tolist()
    d = (np.ones(20) * 0.5).tolist()
    base = {
        "platform": "A",
        "samples": [
            {"preset_id": "p1", "role": "bass", "status": "ok", "z0": z,
             "deltas": {"a_filter_1_cutoff": d}}
        ],
    }
    same = {**base, "platform": "B"}
    assert compare(base, same)["PASS"]

    flipped = {
        "platform": "B",
        "samples": [
            {"preset_id": "p1", "role": "bass", "status": "ok", "z0": z,
             "deltas": {"a_filter_1_cutoff": (-np.ones(20) * 0.5).tolist()}}
        ],
    }
    verdict = compare(base, flipped)
    assert not verdict["PASS"]
    assert verdict["delta_cosine_median"] < 0
