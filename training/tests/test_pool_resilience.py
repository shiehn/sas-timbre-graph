"""The crash-tolerant pool.

Live finding 2026-08-02: a Surge worker segfaulted under 30-way concurrency
and `pool.map` propagated BrokenProcessPool, aborting a ~10 h unattended run
52 s in. Not OOM (cgroup peak 13.8 GB of 64 GB, oom_kill 0) and not a poison
preset (all loaded fine sequentially) — so the pool must survive dead
children, and crash attribution must not convict bystanders.
"""

import os
import signal

import timbre_graph_lab.gen as G
from timbre_graph_lab.config import LabConfig

# Module-level so spawn-based workers can import it.
POISON = "poison-preset"


def _fake_process_anchor(job: dict) -> dict:
    """Segfault-alike for the poison job; succeed otherwise."""
    if job["preset_id"] == POISON:
        os.kill(os.getpid(), signal.SIGKILL)
    return {"preset_id": job["preset_id"], "role": job["role"], "status": "ok"}


def _fake_init(cfg) -> None:  # no Surge host in tests
    G._cfg = cfg


def _job(pid: str, role: str = "bass") -> dict:
    return {"role": role, "path": f"/{pid}", "preset_id": pid,
            "singles": 1, "multis": 1, "drift": 1, "render_avg": 1}


def test_pool_survives_a_dying_worker_and_quarantines_only_the_culprit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(G, "process_anchor", _fake_process_anchor)
    monkeypatch.setattr(G, "_init_worker", _fake_init)
    cfg = LabConfig(workspace=tmp_path)

    jobs = [_job(f"good{i}") for i in range(6)]
    jobs.insert(3, _job(POISON))

    results = G._run_pool(cfg, jobs, workers=4)

    ok = {r["preset_id"] for r in results if r["status"] == "ok"}
    quarantined = [r for r in results if r["status"] == "quarantined"]

    # every healthy job still completed despite the pool dying mid-run
    assert ok == {f"good{i}" for i in range(6)}
    # and the culprit — not a bystander — is the one convicted
    assert len(quarantined) == 1
    assert quarantined[0]["preset_id"] == POISON

    saved = G.load_quarantine(cfg)
    assert list(saved) == [f"bass/{POISON}"]


def test_quarantined_jobs_are_skipped_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "process_anchor", _fake_process_anchor)
    monkeypatch.setattr(G, "_init_worker", _fake_init)
    cfg = LabConfig(workspace=tmp_path)
    G._quarantine(cfg, _job(POISON), "BrokenProcessPool")

    results = G._run_pool(cfg, [_job(POISON), _job("good0")], workers=2)

    # the known-fatal job is never re-run, so the healthy one still lands
    assert [r["preset_id"] for r in results] == ["good0"]


def test_clean_run_has_no_quarantine_file(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "process_anchor", _fake_process_anchor)
    monkeypatch.setattr(G, "_init_worker", _fake_init)
    cfg = LabConfig(workspace=tmp_path)

    results = G._run_pool(cfg, [_job(f"good{i}") for i in range(4)], workers=3)

    assert len(results) == 4
    assert all(r["status"] == "ok" for r in results)
    assert G.load_quarantine(cfg) == {}
