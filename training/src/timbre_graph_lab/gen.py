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


def _render_z(worker: RenderWorker, probe) -> tuple[np.ndarray, object]:
    audio = worker.render(probe)
    return extract_descriptors(audio, worker.cfg.sample_rate), qc_audio(audio)


def process_anchor(job: dict) -> dict:
    """Runs inside a worker process. Returns a stats dict."""
    assert _worker is not None and _cfg is not None
    t0 = time.time()
    role, path, preset_id = job["role"], job["path"], job["preset_id"]
    n_singles, n_multis, n_drift = job["singles"], job["multis"], job["drift"]

    out = shard_path(_cfg, role, preset_id)
    if out.exists():
        return {"preset_id": preset_id, "role": role, "status": "exists"}

    if not _worker.load_preset(path):
        return {"preset_id": preset_id, "role": role, "status": "load-failed"}

    probe = get_probe(role, "short")
    z_anchor, qc = _render_z(_worker, probe)
    if not qc.ok:
        return {"preset_id": preset_id, "role": role, "status": f"qc-{qc.reason}"}

    policy = load_policy(_cfg)
    baseline = _worker.baseline_raw
    candidates = _prioritize([p for p in policy["allowed"] if p in baseline])

    # --- sensitivity screen (single-sided FD) ---
    deltas = []
    for edit in sensitivity_pairs(candidates)[::2]:  # +eps only
        _worker.apply_delta(edit.delta)
        z, q = _render_z(_worker, probe)
        deltas.append(np.linalg.norm(z - z_anchor) if q.ok else 0.0)
    _worker.restore_baseline()
    order = np.argsort(deltas)[::-1]
    sensitive = [candidates[i] for i in order[:TOP_K_SENSITIVE] if deltas[i] > 1e-3]
    if len(sensitive) < 4:
        return {"preset_id": preset_id, "role": role, "status": "insensitive"}

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
            z, q = _render_z(_worker, probe)
            if not q.ok:
                n_bad += 1
                continue
            z0_cache[key] = z
        z0 = z0_cache[key]

        combined = dict(edit.base_offset)
        for name, d in edit.delta.items():
            combined[name] = combined.get(name, 0.0) + d
        _worker.apply_delta(combined)
        z1, q = _render_z(_worker, probe)
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
    )
    return {
        "preset_id": preset_id, "role": role, "status": "ok",
        "n_samples": len(X0), "n_rejected": n_bad,
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
) -> list[dict]:
    cfg = cfg or LabConfig()
    manifest = load_manifest(cfg)
    from timbre_graph_lab.config import ROLES

    jobs = []
    for role in roles or ROLES:
        picked = [e for e in manifest["entries"] if role in e["roles"]][:per_role]
        for e in picked:
            jobs.append(
                {
                    "role": role, "path": e["path"], "preset_id": e["preset_id"],
                    "singles": singles, "multis": multis, "drift": drift,
                }
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
