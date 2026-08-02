"""Forward timbre proxy: F(x_raw, role) -> z (standardized descriptors).

Role-conditioned residual MLP with FiLM. The delta objective is trained
siamese-style: F(x0+dx) - F(x0) vs the measured (standardized) delta z1-z0.
Small on purpose — it must run at control rate on CPU inside the plugin
(or be hand-rolled in TS from the exported weights).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from timbre_graph_lab.config import ROLES


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        y = self.norm(h)
        y = y * (1 + scale) + shift
        y = self.fc2(self.act(self.fc1(y)))
        return h + y


class ForwardProxy(nn.Module):
    def __init__(
        self,
        n_params: int,
        n_features: int,
        hidden: int = 384,
        n_blocks: int = 3,
        role_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_params = n_params
        self.n_features = n_features
        self.role_embed = nn.Embedding(len(ROLES), role_dim)
        self.inp = nn.Linear(n_params, hidden)
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden, role_dim) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor, role_idx: torch.Tensor) -> torch.Tensor:
        cond = self.role_embed(role_idx)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)


def export_bundle(
    model: ForwardProxy,
    out_dir: Path,
    param_names: list[str],
    feature_names: list[str],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    metrics: dict,
    versions: dict,
) -> None:
    """model.onnx + manifest.json + torch checkpoint — the runtime contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), out_dir / "model.pt")

    dummy_x = torch.zeros(1, model.n_params)
    dummy_r = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_x, dummy_r),
        str(out_dir / "model.onnx"),
        input_names=["x_raw", "role_idx"],
        output_names=["z"],
        dynamic_axes={"x_raw": {0: "batch"}, "role_idx": {0: "batch"}},
        opset_version=17,
    )
    manifest = {
        "model_version": "tg-v0",
        "ordered_parameter_ids": param_names,
        "feature_names": feature_names,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "role_ids": ROLES,
        "metrics": metrics,
        **versions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
