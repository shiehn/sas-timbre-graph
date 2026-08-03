"""Measure an anchor's local response matrix (its Jacobian) by rendering.

This replaces the learned forward proxy on the critical path. C12 measured
that predicting a preset's local parameter->timbre response from any other
preset plateaus at ~0.14 delta-cosine, while measuring the preset itself
scores 0.534 — four times better. Rendering is ~100x real-time, so measuring
is also fast enough to do on demand:

    34 core params x 2 (central difference) x AVG renders ~= 200-280 renders
    ~= 4-8 s per anchor, ~10 s for six anchors across six processes.

The result is a per-anchor `AnchorResponse` holding J = dz/dx around that
patch, plus the measurement noise floor so callers can tell a real response
from an unmeasurable one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig, POLICY_VERSION, PROBE_VERSION
from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES, standardize
from timbre_graph_lab.gen import MIN_SNR, RENDER_AVG, core_params
from timbre_graph_lab.policy import load_policy
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_audio

EPS = 0.04  # raw-space step for the central difference


@dataclass
class AnchorResponse:
    """Measured local behaviour of one patch under one role probe."""

    role: str
    preset_id: str
    name: str
    param_names: list[str]
    baseline: np.ndarray          # raw [0,1] values at the anchor
    z0: np.ndarray                # descriptors at the anchor
    J: np.ndarray                 # (n_params, n_features) = dz/dx
    sigma: np.ndarray             # per-descriptor measurement noise
    usable: np.ndarray            # bool per param: response exceeds the noise
    n_renders: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def live_params(self) -> list[str]:
        return [p for p, u in zip(self.param_names, self.usable) if u]

    def to_dict(self) -> dict:
        return {
            "role": self.role, "preset_id": self.preset_id, "name": self.name,
            "param_names": self.param_names,
            "baseline": self.baseline.tolist(),
            "z0": self.z0.tolist(),
            "J": self.J.tolist(),
            "sigma": self.sigma.tolist(),
            "usable": [bool(u) for u in self.usable],
            "n_renders": self.n_renders,
            "feature_names": DESCRIPTOR_NAMES,
            "probe_version": PROBE_VERSION,
            "policy_version": POLICY_VERSION,
            **self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnchorResponse:
        return cls(
            role=d["role"], preset_id=d["preset_id"], name=d.get("name", ""),
            param_names=list(d["param_names"]),
            baseline=np.asarray(d["baseline"], dtype=np.float64),
            z0=np.asarray(d["z0"], dtype=np.float64),
            J=np.asarray(d["J"], dtype=np.float64),
            sigma=np.asarray(d["sigma"], dtype=np.float64),
            usable=np.asarray(d["usable"], dtype=bool),
            n_renders=int(d.get("n_renders", 0)),
        )


def probe_anchor(
    worker: RenderWorker,
    role: str,
    fxp_path: str | Path,
    params: list[str] | None = None,
    eps: float = EPS,
    avg: int = RENDER_AVG,
    name: str = "",
) -> AnchorResponse | None:
    """Measure dz/dx around one preset. Returns None if the patch is unusable."""
    cfg = worker.cfg
    if not worker.load_preset(fxp_path):
        return None

    probe = get_probe(role, "short")
    audio = worker.render(probe)
    if not qc_audio(audio).ok:
        return None

    baseline_raw = worker.baseline_raw
    if params is None:
        params = core_params(load_policy(cfg)["allowed"], baseline_raw)
    params = [p for p in params if p in baseline_raw]
    if not params:
        return None

    sigma = worker.noise_floor(probe, k=4, avg=avg)
    z0 = worker.render_descriptors(probe, k=avg)
    n_renders = 4 * avg + avg

    J = np.zeros((len(params), len(z0)), dtype=np.float64)
    usable = np.zeros(len(params), dtype=bool)
    for i, p in enumerate(params):
        base = baseline_raw[p]
        # Keep the central difference inside [0,1]; near a rail fall back to a
        # one-sided step so rail-pinned controls are still measured.
        hi, lo = min(1.0, base + eps), max(0.0, base - eps)
        span = hi - lo
        if span < 1e-6:
            continue
        worker.apply_delta({p: hi - base})
        z_hi = worker.render_descriptors(probe, k=avg)
        worker.apply_delta({p: lo - base})
        z_lo = worker.render_descriptors(probe, k=avg)
        n_renders += 2 * avg
        # J is stored in STANDARDIZED descriptor units so every dimension
        # counts comparably. On raw units, rolloff and bandwidth (Hz, in the
        # thousands) carry ~85% of all delta energy and the solver silently
        # optimises spectral rolloff alone.
        J[i] = standardize(z_hi - z_lo) / span
        # a response counts only if the measured change clears the noise floor
        usable[i] = float(np.linalg.norm(standardize(z_hi - z_lo))) > (
            MIN_SNR * float(np.linalg.norm(standardize(sigma))) / np.sqrt(avg)
        )
    worker.restore_baseline()

    return AnchorResponse(
        role=role, preset_id="", name=name or Path(fxp_path).stem,
        param_names=params,
        baseline=np.array([baseline_raw[p] for p in params], dtype=np.float64),
        z0=z0, J=J, sigma=sigma, usable=usable, n_renders=n_renders,
        meta={"path": str(fxp_path), "eps": eps, "render_avg": avg},
    )


def write_responses(responses: dict[str, AnchorResponse], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {r: resp.to_dict() for r, resp in responses.items()}, indent=1))
    return out


def read_responses(path: str | Path) -> dict[str, AnchorResponse]:
    raw = json.loads(Path(path).read_text())
    return {r: AnchorResponse.from_dict(d) for r, d in raw.items()}


def default_config() -> LabConfig:
    return LabConfig()
