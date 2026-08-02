"""Dataset generation: sensitivity screen -> gesture plan -> shards.

Per (preset, role) anchor:
1. load preset, render baseline, QC-gate it
2. single-sided FD screen over the allow-list -> top-K audibly sensitive params
3. build the seeded gesture plan (singles / sparse multis / drift chains)
4. render before/after for every edit, extract descriptors, write one shard

Resumable: existing shard files are skipped. `workers=1` runs in-process
(safe everywhere); `workers>1` uses spawn-context processes, one Surge host
each (mapping build ~16s amortized per process).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from timbre_graph_lab.config import LabConfig, SEED
from timbre_graph_lab.corpus import load_manifest
from timbre_graph_lab.dataset import shard_path, write_shard
from timbre_graph_lab.descriptors import extract_descriptors
from timbre_graph_lab.perturb import Edit, build_plan, sensitivity_pairs
from timbre_graph_lab.policy import load_policy
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_audio

# Screen at most this many params per anchor, name-prioritized so the
# classically audible groups go first.
SCREEN_CAP = 120
TOP_K_SENSITIVE = 24
_PRIORITY = ("cutoff", "resonance", "env", "attack", "decay", "sustain", "release",
             "shape", "width", "level", "balance", "feedback", "drive", "lfo")

# Renders of noise-oscillator patches are not bit-identical even with phase
# frozen, so every measurement is averaged and every response is judged
# against the anchor's own measured noise floor rather than a fixed epsilon.
RENDER_AVG = 3
# Hats are the one role built on noise oscillators; extra averaging buys back
# the SNR the frozen-phase fix cannot reach there.
HAT_EXTRA_AVG = 2
MIN_SNR = 3.0
# An anchor whose noise swamps everything cannot produce learnable targets.
MAX_ANCHOR_NOISE_FRAC = 0.5

# Cross-probing (C4): a preset probed under a sibling role's MIDI is valid
# training data for that role. Restricted to the percussion trio, where the
# anchor pools are thin (51-61) and the registers genuinely overlap; bass/pad/
# lead are already data-rich.
CROSS_PROBE_MAP = {
    "kick": ["snare", "hat"],
    "snare": ["kick", "hat"],
    "hat": ["kick", "snare"],
}


_worker: RenderWorker | None = None
_cfg: LabConfig | None = None


def _init_worker(cfg: LabConfig) -> None:
    global _worker, _cfg
    _cfg = cfg
    _worker = RenderWorker(cfg)


def _prioritize(params: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        lname = name.lower()
        for i, tok in enumerate(_PRIORITY):
            if tok in lname:
                return (i, lname)
        return (len(_PRIORITY), lname)

    return sorted(params, key=key)[:SCREEN_CAP]


def _render_z(worker: RenderWorker, probe, k: int = RENDER_AVG) -> tuple[np.ndarray, object]:
    """Averaged descriptors + QC on a single representative render."""
    audio = worker.render(probe)
    qc = qc_audio(audio)
    if k <= 1:
        return extract_descriptors(audio, worker.cfg.sample_rate), qc
    return worker.render_descriptors(probe, k=k), qc


def build_jobs(
    entries: list[dict],
    roles: list[str],
    per_role: int,
    singles: int,
    multis: int,
    drift: int,
    cross_probe: bool = False,
    render_avg: int = RENDER_AVG,
) -> list[dict]:
    """Expand corpus entries into (probe-role, preset) jobs. Pure — testable.

    per_role <= 0 means ALL anchors for that role. Cross-probing adds
    percussion-trio sibling jobs; (role, preset) pairs are deduped so a
    preset that already carries the sibling role is not queued twice.
    """
    jobs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(role: str, e: dict) -> None:
        key = (role, e["preset_id"])
        if key in seen:
            return
        seen.add(key)
        avg = render_avg + (HAT_EXTRA_AVG if role == "hat" else 0)
        jobs.append(
            {
                "role": role, "path": e["path"], "preset_id": e["preset_id"],
                "singles": singles, "multis": multis, "drift": drift,
                "render_avg": avg,
            }
        )

    for role in roles:
        picked = [e for e in entries if role in e["roles"]]
        if per_role > 0:
            picked = picked[:per_role]
        for e in picked:
            add(role, e)
            if cross_probe:
                for sibling in CROSS_PROBE_MAP.get(role, []):
                    if sibling in roles:
                        add(sibling, e)
    return jobs


def process_anchor(job: dict) -> dict:
    """Runs inside a worker process. Returns a stats dict."""
    assert _worker is not None and _cfg is not None
    t0 = time.time()
    role, path, preset_id = job["role"], job["path"], job["preset_id"]
    n_singles, n_multis, n_drift = job["singles"], job["multis"], job["drift"]
    avg = job.get("render_avg", RENDER_AVG)

    out = shard_path(_cfg, role, preset_id)
    if out.exists():
        return {"preset_id": preset_id, "role": role, "status": "exists"}

    if not _worker.load_preset(path):
        return {"preset_id": preset_id, "role": role, "status": "load-failed"}

    probe = get_probe(role, "short")
    z_anchor, qc = _render_z(_worker, probe, k=avg)
    if not qc.ok:
        return {"preset_id": preset_id, "role": role, "status": f"qc-{qc.reason}"}

    policy = load_policy(_cfg)
    baseline = _worker.baseline_raw
    candidates = _prioritize([p for p in policy["allowed"] if p in baseline])

    # --- measurement noise floor for THIS anchor ---
    sigma = _worker.noise_floor(probe, k=4, avg=avg)
    sigma_norm = float(np.linalg.norm(sigma))
    anchor_norm = float(np.linalg.norm(z_anchor)) + 1e-9
    if sigma_norm / anchor_norm > MAX_ANCHOR_NOISE_FRAC:
        return {
            "preset_id": preset_id, "role": role, "status": "too-noisy",
            "noise_frac": round(sigma_norm / anchor_norm, 4),
        }

    # --- sensitivity screen (single-sided FD), gated on SNR not a bare epsilon ---
    deltas = []
    for edit in sensitivity_pairs(candidates)[::2]:  # +eps only
        _worker.apply_delta(edit.delta)
        z, q = _render_z(_worker, probe, k=avg)
        deltas.append(np.linalg.norm(z - z_anchor) if q.ok else 0.0)
    _worker.restore_baseline()
    order = np.argsort(deltas)[::-1]
    snr_floor = MIN_SNR * sigma_norm / np.sqrt(avg)
    sensitive = [
        candidates[i] for i in order[:TOP_K_SENSITIVE] if deltas[i] > snr_floor
    ]
    if len(sensitive) < 4:
        return {
            "preset_id": preset_id, "role": role, "status": "insensitive",
            "snr_floor": round(float(snr_floor), 4),
            "best_response": round(float(max(deltas)) if deltas else 0.0, 4),
        }

    # --- gesture plan ---
    plan = build_plan(
        preset_id, role, sensitive,
        n_singles=n_singles, n_multis=n_multis, n_drift_chains=n_drift, seed=SEED,
    )

    # X0/DX live in the FIXED policy allow-list vector so column j means the
    # same parameter in every shard; sensitivity only chose what to perturb.
    # Params whose raw value is unreadable on this preset get a neutral 0.5.
    param_names = list(policy["allowed"])
    p_index = {p: i for i, p in enumerate(param_names)}
    x_anchor = np.array(
        [baseline.get(p, 0.5) for p in param_names], dtype=np.float32
    )

    X0, DX, Z0, Z1, KINDS = [], [], [], [], []
    z0_cache: dict[str, np.ndarray] = {"": z_anchor}

    def base_key(offset: dict[str, float]) -> str:
        return json.dumps(offset, sort_keys=True) if offset else ""

    n_bad = 0
    for edit in plan:
        key = base_key(edit.base_offset)
        if key not in z0_cache:
            _worker.apply_delta(edit.base_offset)
            z, q = _render_z(_worker, probe, k=avg)
            if not q.ok:
                n_bad += 1
                continue
            z0_cache[key] = z
        z0 = z0_cache[key]

        combined = dict(edit.base_offset)
        for name, d in edit.delta.items():
            combined[name] = combined.get(name, 0.0) + d
        _worker.apply_delta(combined)
        z1, q = _render_z(_worker, probe, k=avg)
        if not q.ok:
            n_bad += 1
            continue

        x0 = x_anchor.copy()
        dx = np.zeros(len(param_names), dtype=np.float32)
        for name, d in edit.base_offset.items():
            if name in p_index:
                x0[p_index[name]] = np.clip(x0[p_index[name]] + d, 0.0, 1.0)
        for name, d in edit.delta.items():
            if name in p_index:
                dx[p_index[name]] = d
        X0.append(x0)
        DX.append(dx)
        Z0.append(z0)
        Z1.append(z1)
        KINDS.append(edit.kind)

    _worker.restore_baseline()
    if len(X0) < 20:
        return {"preset_id": preset_id, "role": role, "status": "too-few-samples"}

    write_shard(
        _cfg, role, preset_id, param_names,
        np.stack(X0), np.stack(DX), np.stack(Z0), np.stack(Z1), KINDS,
        noise_sigma=sigma,
    )
    dz = np.stack(Z1) - np.stack(Z0)
    med_snr = float(
        np.median(np.linalg.norm(dz, axis=1)) / (sigma_norm / np.sqrt(avg) + 1e-9)
    )
    return {
        "preset_id": preset_id, "role": role, "status": "ok",
        "n_samples": len(X0), "n_rejected": n_bad,
        "median_snr": round(med_snr, 2),
        "seconds": round(time.time() - t0, 1),
    }


def generate(
    cfg: LabConfig | None = None,
    per_role: int = 10,
    singles: int = 60,
    multis: int = 80,
    drift: int = 4,
    workers: int = 1,
    roles: list[str] | None = None,
    cross_probe: bool = False,
    render_avg: int = RENDER_AVG,
) -> list[dict]:
    cfg = cfg or LabConfig()
    manifest = load_manifest(cfg)
    from timbre_graph_lab.config import ROLES

    jobs = build_jobs(
        manifest["entries"], roles or ROLES, per_role,
        singles, multis, drift, cross_probe=cross_probe, render_avg=render_avg,
    )

    results: list[dict] = []
    if workers <= 1:
        _init_worker(cfg)
        for i, job in enumerate(jobs):
            r = process_anchor(job)
            results.append(r)
            print(f"[{i+1}/{len(jobs)}] {r}")
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=_init_worker, initargs=(cfg,),
        ) as pool:
            for i, r in enumerate(pool.map(process_anchor, jobs)):
                results.append(r)
                print(f"[{i+1}/{len(jobs)}] {r}")

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "gen_report.json").write_text(json.dumps(results, indent=1))
    return results
