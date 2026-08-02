import numpy as np

from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES, N_DESCRIPTORS, extract_descriptors


def test_descriptor_names_match_dim():
    assert len(DESCRIPTOR_NAMES) == N_DESCRIPTORS


def test_descriptors_on_sine_and_silence():
    sr = 44100
    t = np.arange(sr * 2) / sr
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    z = extract_descriptors(sine, sr)
    assert z.shape == (N_DESCRIPTORS,)
    assert np.all(np.isfinite(z))
    centroid = z[DESCRIPTOR_NAMES.index("centroid_mean")]
    assert 200 < centroid < 2000  # centroid near the fundamental

    silent = np.zeros(sr, dtype=np.float32)
    zs = extract_descriptors(silent, sr)
    assert np.all(np.isfinite(zs))


def test_brighter_audio_moves_centroid():
    sr = 44100
    t = np.arange(sr) / sr
    dark = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    rng = np.random.default_rng(0)
    bright = (0.3 * rng.standard_normal(sr)).astype(np.float32)
    zd = extract_descriptors(dark, sr)
    zb = extract_descriptors(bright, sr)
    i = DESCRIPTOR_NAMES.index("centroid_mean")
    assert zb[i] > zd[i]


def test_forward_proxy_shapes_and_delta():
    import torch

    from timbre_graph_lab.model import ForwardProxy

    model = ForwardProxy(n_params=10, n_features=N_DESCRIPTORS, hidden=32, n_blocks=2)
    x = torch.rand(7, 10)
    r = torch.randint(0, 6, (7,))
    z = model(x, r)
    assert z.shape == (7, N_DESCRIPTORS)
    dz = model(x + 0.05, r) - z
    assert torch.all(torch.isfinite(dz))
