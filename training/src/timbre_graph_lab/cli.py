"""tglab — Timbre Graph training lab CLI.

Typical pilot session (see docs/TRAINING.md for the full runbook):

    tglab inventory            # scan + role-assign installed Surge patches
    tglab policy               # build the live-safe parameter allow-list
    tglab pilot --per-role 10  # baseline render + QC gate per role
    tglab gen --per-role 10    # perturbation dataset shards
    tglab train --epochs 60    # forward/delta proxy + ONNX export
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from timbre_graph_lab.config import LabConfig, ROLES

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()


@app.command()
def inventory() -> None:
    """Scan factory + 3rd-party patches and write the corpus manifest."""
    from timbre_graph_lab.corpus import scan, write_manifest

    cfg = LabConfig()
    entries = scan(cfg)
    out = write_manifest(entries, cfg)
    table = Table(title="corpus by role")
    table.add_column("role")
    table.add_column("presets", justify="right")
    for role in ROLES:
        table.add_row(role, str(sum(1 for e in entries if role in e.roles)))
    console.print(table)
    console.print(f"manifest -> {out}")


@app.command()
def policy() -> None:
    """Introspect the live Surge host and write the parameter allow-list."""
    from timbre_graph_lab.policy import build_policy, write_policy
    from timbre_graph_lab.worker import RenderWorker

    cfg = LabConfig()
    worker = RenderWorker(cfg)
    pol = build_policy(worker)
    out = write_policy(pol, cfg)
    console.print(
        f"allowed={pol['n_allowed']} excluded={len(pol['excluded'])} -> {out}"
    )


@app.command()
def pilot(per_role: int = 10) -> None:
    """Baseline render + QC gate for the first N corpus anchors per role."""
    from timbre_graph_lab.corpus import load_manifest
    from timbre_graph_lab.descriptors import extract_descriptors
    from timbre_graph_lab.probes import get_probe
    from timbre_graph_lab.worker import RenderWorker, qc_audio

    cfg = LabConfig()
    manifest = load_manifest(cfg)
    worker = RenderWorker(cfg)
    report = []
    for role in ROLES:
        picked = [e for e in manifest["entries"] if role in e["roles"]][:per_role]
        probe = get_probe(role, "short")
        n_ok = 0
        for e in picked:
            if not worker.load_preset(e["path"]):
                report.append({"role": role, "name": e["name"], "status": "load-failed"})
                continue
            audio = worker.render(probe)
            qc = qc_audio(audio)
            z = extract_descriptors(audio, cfg.sample_rate)
            status = "ok" if qc.ok else f"qc-{qc.reason}"
            n_ok += int(qc.ok)
            report.append(
                {
                    "role": role, "name": e["name"], "status": status,
                    "rms": round(qc.rms, 5),
                    "centroid": round(float(z[3]), 1),
                }
            )
        console.print(f"{role}: {n_ok}/{len(picked)} pass QC")
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / "pilot.json"
    out.write_text(json.dumps(report, indent=1))
    console.print(f"report -> {out}")


@app.command()
def gen(
    per_role: int = 10,
    singles: int = 60,
    multis: int = 80,
    drift: int = 4,
    workers: int = 1,
    role: list[str] = typer.Option(None, help="restrict to specific roles"),
) -> None:
    """Generate perturbation dataset shards (resumable)."""
    from timbre_graph_lab.gen import generate

    results = generate(
        LabConfig(), per_role=per_role, singles=singles, multis=multis,
        drift=drift, workers=workers, roles=list(role) if role else None,
    )
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "exists")
    console.print(f"done: {ok} new shards, {skipped} existing, {len(results)} total jobs")


@app.command()
def train(
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 3e-4,
    hidden: int = 384,
    out_name: str = "tg-v0",
) -> None:
    """Train the forward/delta proxy and export the ONNX bundle."""
    from timbre_graph_lab.train import train_model

    train_model(
        LabConfig(), epochs=epochs, batch_size=batch_size, lr=lr,
        hidden=hidden, out_name=out_name,
    )


@app.command()
def bench() -> None:
    """Render-throughput benchmark on this machine."""
    import time

    from timbre_graph_lab.corpus import load_manifest
    from timbre_graph_lab.probes import get_probe
    from timbre_graph_lab.worker import RenderWorker

    cfg = LabConfig()
    worker = RenderWorker(cfg)
    manifest = load_manifest(cfg)
    entry = next(e for e in manifest["entries"] if "bass" in e["roles"])
    worker.load_preset(entry["path"])
    probe = get_probe("bass", "short")
    worker.render(probe)  # warm
    t = time.perf_counter()
    n = 20
    for _ in range(n):
        worker.render(probe)
    dt = (time.perf_counter() - t) / n
    console.print(
        f"{dt*1000:.0f} ms/render ({probe.duration:.1f}s audio) "
        f"-> {3600/dt:.0f} renders/hour/process"
    )


if __name__ == "__main__":
    app()
