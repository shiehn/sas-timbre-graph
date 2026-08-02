"""Process-per-job isolation.

Live failure 2026-08-02: inside a ProcessPoolExecutor a segfaulting Surge
child broke the pool and killed every in-flight job. Crashes arrived every
~3.5 min while jobs needed ~7 min, so long jobs were perpetually restarted
and throughput collapsed from 4.16 to 0.29 shards/min — a livelock, not a
slowdown. Not OOM (cgroup peak 13.8 GB of 64 GB, oom_kill 0) and no single
preset was reproducibly fatal.

Fix: each job runs in its own OS process, so a crash is a mere exit code
that affects nothing else. These tests pin that contract by making a job's
subprocess die outright.
"""

import json

import pytest

import timbre_graph_lab.gen as G
from timbre_graph_lab.config import LabConfig

POISON = "poison-preset"
SLOWPOKE = "slow-preset"


def _job(pid: str, role: str = "bass") -> dict:
    return {"role": role, "path": f"/{pid}", "preset_id": pid,
            "singles": 1, "multis": 1, "drift": 1, "render_avg": 1}


@pytest.fixture
def fake_run_one(monkeypatch):
    """Stand in for the subprocess: POISON 'segfaults', others succeed."""
    calls = {"n": 0}

    def _fake(cfg, job):
        calls["n"] += 1
        if job["preset_id"] == POISON:
            return {"preset_id": job["preset_id"], "role": job["role"],
                    "status": "failed", "error": "exit=-11"}
        return {"preset_id": job["preset_id"], "role": job["role"],
                "status": "ok"}

    monkeypatch.setattr(G, "_run_one", _fake)
    return calls


def test_one_crashing_job_does_not_affect_the_others(tmp_path, fake_run_one):
    cfg = LabConfig(workspace=tmp_path)
    jobs = [_job(f"good{i}") for i in range(6)]
    jobs.insert(3, _job(POISON))

    results = G._run_pool(cfg, jobs, workers=4)

    ok = {r["preset_id"] for r in results if r["status"] == "ok"}
    assert ok == {f"good{i}" for i in range(6)}

    quarantined = [r for r in results if r["status"] == "quarantined"]
    assert [r["preset_id"] for r in quarantined] == [POISON]
    assert list(G.load_quarantine(cfg)) == [f"bass/{POISON}"]


def test_crashing_job_is_retried_before_being_convicted(tmp_path, fake_run_one):
    cfg = LabConfig(workspace=tmp_path)
    G._run_pool(cfg, [_job(POISON)], workers=2)
    # attempted MAX_ATTEMPTS times, not given up on after one crash
    assert fake_run_one["n"] == G.MAX_ATTEMPTS


def test_quarantined_jobs_are_skipped_on_resume(tmp_path, fake_run_one):
    cfg = LabConfig(workspace=tmp_path)
    G._quarantine(cfg, _job(POISON), "exit=-11")
    results = G._run_pool(cfg, [_job(POISON), _job("good0")], workers=2)
    assert [r["preset_id"] for r in results] == ["good0"]
    assert fake_run_one["n"] == 1  # the fatal job was never launched again


def test_clean_run_leaves_no_quarantine(tmp_path, fake_run_one):
    cfg = LabConfig(workspace=tmp_path)
    results = G._run_pool(cfg, [_job(f"good{i}") for i in range(4)], workers=3)
    assert len(results) == 4 and all(r["status"] == "ok" for r in results)
    assert G.load_quarantine(cfg) == {}


def test_run_one_reports_nonzero_exit_as_failure(tmp_path, monkeypatch):
    """A real subprocess that dies must surface as status=failed, not raise."""
    import subprocess

    def boom(*a, **kw):
        return subprocess.CompletedProcess(a[0], -11, stdout="", stderr="Segfault")

    monkeypatch.setattr(subprocess, "run", boom)
    r = G._run_one(LabConfig(workspace=tmp_path), _job("x"))
    assert r["status"] == "failed"
    assert "exit=-11" in r["error"]


def test_run_one_parses_tagged_result(tmp_path, monkeypatch):
    import subprocess

    from timbre_graph_lab.onejob import RESULT_TAG

    payload = {"preset_id": "x", "role": "bass", "status": "ok", "n_samples": 7}

    def fine(*a, **kw):
        return subprocess.CompletedProcess(
            a[0], 0,
            stdout=f"chatter\n{RESULT_TAG}{json.dumps(payload)}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fine)
    assert G._run_one(LabConfig(workspace=tmp_path), _job("x")) == payload


def test_run_one_treats_timeout_as_failure(tmp_path, monkeypatch):
    import subprocess

    def hang(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=G.JOB_TIMEOUT_S)

    monkeypatch.setattr(subprocess, "run", hang)
    r = G._run_one(LabConfig(workspace=tmp_path), _job("x"))
    assert r["status"] == "failed" and r["error"] == "timeout"
