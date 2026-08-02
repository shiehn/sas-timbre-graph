import numpy as np

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.dataset import load_shards, split_for, write_shard


def test_split_stable_and_by_preset():
    assert split_for("deadbeef01234567") == split_for("deadbeef01234567")
    splits = {split_for(f"preset{i}") for i in range(200)}
    assert splits == {"train", "val", "test"}


def test_shard_round_trip(tmp_path):
    cfg = LabConfig(workspace=tmp_path)
    n, d, f = 30, 5, 20
    rng = np.random.default_rng(0)
    write_shard(
        cfg, "bass", "abc123", [f"p{i}" for i in range(d)],
        rng.random((n, d)), rng.random((n, d)) * 0.1,
        rng.random((n, f)), rng.random((n, f)),
        ["single"] * n,
    )
    shards = load_shards(cfg, ["bass"])
    assert len(shards) == 1
    s = shards[0]
    assert s["meta"]["preset_id"] == "abc123"
    assert s["meta"]["role"] == "bass"
    assert s["X0"].shape == (n, d)
    assert s["Z1"].shape == (n, f)
    assert s["meta"]["split"] in ("train", "val", "test")
