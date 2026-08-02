"""Linux-vs-macOS parity gate.

The proposal's §9.5 rule, made executable: never trust cloud-rendered corpus
data until the same preset + probe produces the same *perceptual* answer on
both platforms. Absolute float equality is not expected; what must hold is:

1. **Same content** — identical .fxp bytes (preset_id sha1s match), so shards
   from either platform describe the same presets.
2. **Same baseline descriptors** — within a loose relative tolerance.
3. **Same gesture direction** — the thing the model actually learns. For a
   fixed FD perturbation, the descriptor delta vectors must point the same
   way on both platforms (cosine ≈ 1). This is the gate that matters; a
   constant offset in absolute descriptors is survivable, a rotated delta
   space is not.

Usage:
    # on the Mac
    tglab parity --write workspace/reports/parity-macos.json
    # on the pod (after bootstrap)
    tglab parity --write /tmp/parity-linux.json --compare parity-macos.json
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig, ROLES
from timbre_graph_lab.corpus import load_manifest
from timbre_graph_lab.descriptors import DESCRIPTOR_NAMES
from timbre_graph_lab.gen import RENDER_AVG
from timbre_graph_lab.perturb import FD_EPS
from timbre_graph_lab.policy import load_policy
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_audio

# Gate thresholds
MIN_DELTA_COSINE = 0.95
MAX_BASELINE_REL_ERR = 0.05


def _probe_params(policy: dict, baseline: dict[str, float], n: int) -> list[str]:
    """Deterministic, structurally-chosen perturbation params for the check."""
    want = ("cutoff", "resonance", "decay", "release", "shape", "level")
    picked: list[str] = []
    for tok in want:
        for p in policy["allowed"]:
            if tok in p.lower() and p in baseline and p not in picked:
                picked.append(p)
                break
        if len(picked) >= n:
            break
    return picked[:n]


def measure(cfg: LabConfig | None = None, per_role: int = 3) -> dict:
    """Render a deterministic sample of anchors + FD deltas on this platform."""
    cfg = cfg or LabConfig()
    manifest = load_manifest(cfg)
    policy = load_policy(cfg)
    worker = RenderWorker(cfg)

    samples = []
    for role in ROLES:
        # sort by preset_id so both platforms pick the SAME presets even if
        # directory iteration order differs
        pool = sorted(
            (e for e in manifest["entries"] if role in e["roles"]),
            key=lambda e: e["preset_id"],
        )[:per_role]
        probe = get_probe(role, "short")
        for e in pool:
            if not worker.load_preset(e["path"]):
                samples.append(
                    {"preset_id": e["preset_id"], "role": role, "status": "load-failed"}
                )
                continue
            audio = worker.render(probe)
            qc = qc_audio(audio)
            if not qc.ok:
                samples.append(
                    {
                        "preset_id": e["preset_id"],
                        "role": role,
                        "status": f"qc-{qc.reason}",
                    }
                )
                continue
            # same averaged measurement path the dataset generator uses
            z0 = worker.render_descriptors(probe, k=RENDER_AVG)

            deltas = {}
            for p in _probe_params(policy, worker.baseline_raw, 4):
                worker.apply_delta({p: FD_EPS})
                z1 = worker.render_descriptors(probe, k=RENDER_AVG)
                deltas[p] = (z1 - z0).tolist()
            worker.restore_baseline()

            samples.append(
                {
                    "preset_id": e["preset_id"],
                    "role": role,
                    "name": e["name"],
                    "status": "ok",
                    "z0": z0.tolist(),
                    "deltas": deltas,
                }
            )

    return {
        "platform": f"{platform.system()}-{platform.machine()}",
        "descriptor_names": DESCRIPTOR_NAMES,
        "n_samples": len(samples),
        "samples": samples,
    }


def compare(local: dict, other: dict) -> dict:
    """Diff two parity measurements; returns a verdict dict."""
    by_id_other = {
        (s["preset_id"], s["role"]): s for s in other["samples"] if s["status"] == "ok"
    }
    rows = []
    for s in local["samples"]:
        if s["status"] != "ok":
            continue
        o = by_id_other.get((s["preset_id"], s["role"]))
        if o is None:
            rows.append(
                {"preset_id": s["preset_id"], "role": s["role"], "issue": "missing-on-other"}
            )
            continue
        z_a, z_b = np.array(s["z0"]), np.array(o["z0"])
        denom = np.maximum(np.abs(z_a), 1e-3)
        base_rel = float(np.median(np.abs(z_a - z_b) / denom))

        cosines = []
        for p, d_a in s["deltas"].items():
            d_b = o["deltas"].get(p)
            if d_b is None:
                continue
            a, b = np.array(d_a), np.array(d_b)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                continue
            cosines.append(float(a @ b / (na * nb)))
        rows.append(
            {
                "preset_id": s["preset_id"],
                "role": s["role"],
                "baseline_rel_err_median": base_rel,
                "delta_cosine_median": float(np.median(cosines)) if cosines else None,
                "n_deltas": len(cosines),
            }
        )

    ok_rows = [r for r in rows if r.get("delta_cosine_median") is not None]
    matched_ids = len(ok_rows)
    delta_cos = (
        float(np.median([r["delta_cosine_median"] for r in ok_rows])) if ok_rows else 0.0
    )
    base_err = (
        float(np.median([r["baseline_rel_err_median"] for r in ok_rows]))
        if ok_rows
        else 1.0
    )
    ids_local = {s["preset_id"] for s in local["samples"]}
    ids_other = {s["preset_id"] for s in other["samples"]}

    passed = (
        matched_ids > 0
        and delta_cos >= MIN_DELTA_COSINE
        and base_err <= MAX_BASELINE_REL_ERR
    )
    return {
        "platforms": [local["platform"], other["platform"]],
        "content_identical": ids_local == ids_other,
        "n_matched": matched_ids,
        "delta_cosine_median": delta_cos,
        "baseline_rel_err_median": base_err,
        "thresholds": {
            "min_delta_cosine": MIN_DELTA_COSINE,
            "max_baseline_rel_err": MAX_BASELINE_REL_ERR,
        },
        "PASS": passed,
        "rows": rows,
    }


def write(data: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    return out
