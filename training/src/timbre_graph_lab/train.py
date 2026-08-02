"""Training loop for the forward/delta proxy + linear baseline gate.

Metrics that matter (docs/TRAINING.md §gates):
- delta cosine (median, val): does predicted movement point the right way?
- delta magnitude relative error (median, val)
- the model must beat a per-role ridge-regression delta baseline, else the
  extra capacity isn't paying for itself yet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from timbre_graph_lab.config import (
    CORPUS_VERSION,
    POLICY_VERSION,
    PROBE_VERSION,
    LabConfig,
    ROLES,
    SEED,
)
from timbre_graph_lab.dataset import load_shards
from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES
from timbre_graph_lab.model import ForwardProxy, export_bundle


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _stack(shards: list[dict], split: str):
    xs, dxs, z0s, z1s, roles = [], [], [], [], []
    for s in shards:
        if s["meta"]["split"] != split:
            continue
        n = len(s["X0"])
        if n == 0:
            continue
        xs.append(s["X0"])
        dxs.append(s["DX"])
        z0s.append(s["Z0"])
        z1s.append(s["Z1"])
        roles.append(np.full(n, ROLES.index(s["meta"]["role"]), dtype=np.int64))
    if not xs:
        return None
    return (
        np.concatenate(xs),
        np.concatenate(dxs),
        np.concatenate(z0s),
        np.concatenate(z1s),
        np.concatenate(roles),
    )


def _delta_metrics(pred_dz: np.ndarray, true_dz: np.ndarray) -> dict:
    eps = 1e-8
    true_norm = np.linalg.norm(true_dz, axis=1)
    meaningful = true_norm > np.percentile(true_norm, 25)  # drop near-null edits
    p, t = pred_dz[meaningful], true_dz[meaningful]
    cos = np.sum(p * t, axis=1) / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + eps
    )
    mag_rel = np.abs(np.linalg.norm(p, axis=1) - np.linalg.norm(t, axis=1)) / (
        np.linalg.norm(t, axis=1) + eps
    )
    return {
        "delta_cosine_median": float(np.median(cos)),
        "delta_cosine_p25": float(np.percentile(cos, 25)),
        "delta_mag_rel_err_median": float(np.median(mag_rel)),
        "n_eval": int(meaningful.sum()),
    }


def _ridge_baseline(train, val, lam: float = 1e-2) -> dict:
    """Global linear map dx->dz (ridge, closed form) as the floor to beat."""
    _, dx_tr, z0_tr, z1_tr, _ = train
    _, dx_va, z0_va, z1_va, _ = val
    dz_tr, dz_va = z1_tr - z0_tr, z1_va - z0_va
    A = dx_tr.astype(np.float64)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ dz_tr)
    return _delta_metrics(dx_va @ W, dz_va)


def train_model(
    cfg: LabConfig | None = None,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 3e-4,
    delta_weight: float = 2.0,
    hidden: int = 384,
    out_name: str = "tg-v0",
) -> dict:
    cfg = cfg or LabConfig()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    shards = load_shards(cfg, ROLES)
    if not shards:
        raise SystemExit("no shards found — run `tglab gen` first")

    train = _stack(shards, "train")
    val = _stack(shards, "val") or _stack(shards, "test")
    if train is None or val is None:
        raise SystemExit("not enough shards for a train/val split yet")

    x0, dx, z0, z1, roles = train
    # standardize features on TRAIN stats
    feat_mean = z0.mean(axis=0)
    feat_std = z0.std(axis=0) + 1e-6

    def norm_z(z):
        return (z - feat_mean) / feat_std

    device = _device()
    model = ForwardProxy(
        n_params=x0.shape[1], n_features=z0.shape[1], hidden=hidden
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def to_t(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    tx0, tdx = to_t(x0), to_t(dx)
    tz0, tz1 = to_t(norm_z(z0)), to_t(norm_z(z1))
    troles = to_t(roles, torch.long)

    vx0, vdx, vz0, vz1, vroles = val
    tvx0, tvdx = to_t(vx0), to_t(vdx)
    tvz0, tvz1 = to_t(norm_z(vz0)), to_t(norm_z(vz1))
    tvroles = to_t(vroles, torch.long)

    n = len(tx0)
    best = {"delta_cosine_median": -1.0}
    best_state = None
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, dxb = tx0[idx], tdx[idx]
            zb0, zb1, rb = tz0[idx], tz1[idx], troles[idx]
            pred0 = model(xb, rb)
            pred1 = model(xb + dxb, rb)
            loss_abs = torch.nn.functional.mse_loss(pred0, zb0) + (
                torch.nn.functional.mse_loss(pred1, zb1)
            )
            loss_delta = torch.nn.functional.mse_loss(pred1 - pred0, zb1 - zb0)
            loss = loss_abs + delta_weight * loss_delta
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            p0 = model(tvx0, tvroles)
            p1 = model(tvx0 + tvdx, tvroles)
            m = _delta_metrics(
                (p1 - p0).cpu().numpy(), (tvz1 - tvz0).cpu().numpy()
            )
        if m["delta_cosine_median"] > best["delta_cosine_median"]:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:3d} loss {total/n:.4f} "
                f"val_cos {m['delta_cosine_median']:.3f} "
                f"val_magerr {m['delta_mag_rel_err_median']:.3f}"
            )

    baseline = _ridge_baseline(
        (x0, dx, norm_z(z0), norm_z(z1), roles),
        (vx0, vdx, norm_z(vz0), norm_z(vz1), vroles),
    )
    metrics = {
        "val": best,
        "ridge_baseline_val": baseline,
        "beats_baseline": best["delta_cosine_median"]
        > baseline["delta_cosine_median"],
        "n_train": int(n),
        "n_val": int(len(vx0)),
        "train_seconds": round(time.time() - t_start, 1),
        "device": str(device),
    }
    print(json.dumps(metrics, indent=1))

    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = cfg.models_dir / out_name
    param_names = shards[0]["meta"]["param_names"]
    export_bundle(
        model.cpu(),
        out_dir,
        param_names=param_names,
        feature_names=DESCRIPTOR_NAMES,
        feat_mean=feat_mean,
        feat_std=feat_std,
        metrics=metrics,
        versions={
            "probe_version": PROBE_VERSION,
            "policy_version": POLICY_VERSION,
            "corpus_version": CORPUS_VERSION,
        },
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"exported -> {out_dir}")
    return metrics
