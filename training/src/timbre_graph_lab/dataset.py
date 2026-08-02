"""Dataset shards: one .npz per (preset, role) anchor + a manifest.

Shard arrays (n samples, aligned):
  X0   (n, d)  absolute raw values of the allow-list vector BEFORE the edit
  DX   (n, d)  the raw-space edit
  Z0   (n, f)  descriptors before
  Z1   (n, f)  descriptors after
  KIND (n,)    edit kind id

Splits are assigned BY PRESET via stable hash so no anchor leaks across
train/val/test. Feature standardization happens at train time from train
shards only; raw descriptors are stored un-standardized so the basis can be
replaced without re-rendering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import (
    CORPUS_VERSION,
    POLICY_VERSION,
    PROBE_VERSION,
    LabConfig,
)

KIND_IDS = {"fd": 0, "single": 1, "multi": 2, "drift": 3}


def split_for(preset_id: str) -> str:
    h = int(hashlib.sha256(preset_id.encode()).hexdigest()[:8], 16) % 10
    if h < 8:
        return "train"
    return "val" if h == 8 else "test"


def shard_path(cfg: LabConfig, role: str, preset_id: str) -> Path:
    return cfg.shards_dir / role / f"{preset_id}.npz"


def write_shard(
    cfg: LabConfig,
    role: str,
    preset_id: str,
    param_names: list[str],
    x0: np.ndarray,
    dx: np.ndarray,
    z0: np.ndarray,
    z1: np.ndarray,
    kinds: list[str],
    noise_sigma: np.ndarray | None = None,
) -> Path:
    out = shard_path(cfg, role, preset_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "preset_id": preset_id,
        "role": role,
        "split": split_for(preset_id),
        "param_names": param_names,
        "probe_version": PROBE_VERSION,
        "policy_version": POLICY_VERSION,
        "corpus_version": CORPUS_VERSION,
    }
    if noise_sigma is None:
        noise_sigma = np.zeros(z0.shape[1], dtype=np.float32)
    np.savez_compressed(
        out,
        X0=x0.astype(np.float32),
        DX=dx.astype(np.float32),
        Z0=z0.astype(np.float32),
        Z1=z1.astype(np.float32),
        KIND=np.array([KIND_IDS[k] for k in kinds], dtype=np.int8),
        SIGMA=noise_sigma.astype(np.float32),
        meta=json.dumps(meta),
    )
    return out


def load_shards(cfg: LabConfig, roles: list[str]) -> list[dict]:
    shards = []
    for role in roles:
        d = cfg.shards_dir / role
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                shards.append(
                    {
                        "meta": meta,
                        "X0": z["X0"],
                        "DX": z["DX"],
                        "Z0": z["Z0"],
                        "Z1": z["Z1"],
                        "KIND": z["KIND"],
                        "SIGMA": z["SIGMA"]
                        if "SIGMA" in z.files
                        else np.zeros(z["Z0"].shape[1], dtype=np.float32),
                    }
                )
    return shards
