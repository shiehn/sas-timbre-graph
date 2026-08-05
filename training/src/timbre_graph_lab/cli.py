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
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from timbre_graph_lab.config import LabConfig, ROLES, SEED

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
    per_role: int = typer.Option(10, help="anchors per role; 0 = ALL"),
    singles: int = 60,
    multis: int = 80,
    drift: int = 4,
    workers: int = 1,
    role: list[str] = typer.Option(None, help="restrict to specific roles"),
    cross_probe: bool = typer.Option(
        False, help="also probe percussion anchors under sibling trio probes"
    ),
    render_avg: int = typer.Option(3, help="renders averaged per measurement"),
) -> None:
    """Generate perturbation dataset shards (resumable)."""
    from timbre_graph_lab.gen import generate

    results = generate(
        LabConfig(), per_role=per_role, singles=singles, multis=multis,
        drift=drift, workers=workers, roles=list(role) if role else None,
        cross_probe=cross_probe, render_avg=render_avg,
    )
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "exists")
    console.print(f"done: {ok} new shards, {skipped} existing, {len(results)} total jobs")


@app.command()
def edges(
    k: int = typer.Option(4, help="nearest neighbours proposed per anchor"),
    points: int = typer.Option(5, help="interior interpolation points per edge"),
    role: list[str] = typer.Option(None, help="restrict to specific roles"),
) -> None:
    """Render + validate morph edges between same-role anchors (needs shards)."""
    from timbre_graph_lab.edges import build_graphs

    build_graphs(
        LabConfig(), roles=list(role) if role else None, k=k, n_points=points
    )


@app.command()
def tour(
    anchors: int = typer.Option(20, help="target anchors per role"),
    cap: int = typer.Option(400, help="max candidates screened per role"),
    keep: int = typer.Option(40, help="screened survivors kept for edges"),
    k: int = typer.Option(4, help="nearest neighbours proposed per anchor"),
    points: int = typer.Option(3, help="interior points rendered per hop"),
    role: list[str] = typer.Option(None, help="restrict to specific roles"),
    out: str = typer.Option("tour.json", help="artifact filename in workspace/"),
) -> None:
    """Build the SHIPPED dial: a validated tour through many presets per role."""
    from timbre_graph_lab.morph import save_graph
    from timbre_graph_lab.tour import build_tour_graph

    cfg = LabConfig()
    path = cfg.workspace / out
    graph = build_tour_graph(
        cfg, roles=list(role) if role else None, anchors=anchors, cap=cap,
        keep=keep, k=k, n_interior=points, save_to=path,
    )
    save_graph(graph, path)

    table = Table(title="tour")
    for col in ("role", "screened", "passed", "thru lens", "edges", "anchors",
                "min hop"):
        table.add_column(col, justify="left" if col == "role" else "right")
    for r, q in graph["quality"].items():
        if r == "_summary":
            continue
        shipped = len(q["anchors"])
        table.add_row(
            r,
            f"{q['n_pool']}{'*' if q['capped'] else ''}",
            str(q["n_passed"]),
            str(q.get("n_through_lens", "-")),
            f"{q.get('n_edges_valid', 0)}/{q.get('n_edges_proposed', 0)}",
            f"{shipped}/{anchors}" + (" SHORT" if q.get("shortfall") else ""),
            str(q.get("sweep", {}).get("worst_hop_travel", "-")),
        )
    console.print(table)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report = cfg.reports_dir / "tour-report.json"
    report.write_text(json.dumps(graph["quality"], indent=1))
    console.print(f"artifact -> {path}\nreport   -> {report}")


@app.command()
def patchmap(
    cap: int = typer.Option(200, help="max candidates screened per role"),
    points: int = typer.Option(40, help="anchors laid out on each role's map"),
    grid: int = typer.Option(10, help="NxN blend positions rendered to validate"),
    lenses: int = typer.Option(3, help="alternative worlds per role"),
    role: list[str] = typer.Option(None, help="restrict to specific roles"),
    out: str = typer.Option("patchmap.json", help="artifact filename in workspace/"),
) -> None:
    """Build the SHIPPED X/Y patch map: every point on it a checked sound."""
    from timbre_graph_lab.mapbuild import build_map_graph
    from timbre_graph_lab.morph import save_graph

    cfg = LabConfig()
    path = cfg.workspace / out
    graph = build_map_graph(
        cfg, roles=list(role) if role else None, cap=cap, n_points=points,
        grid_n=grid, n_lenses=lenses, save_to=path,
    )
    save_graph(graph, path)

    table = Table(title="patch map")
    for col in ("role", "screened", "passed", "lenses", "points per lens"):
        table.add_column(col, justify="left" if col == "role" else "right")
    for r, q in graph["quality"].items():
        if r == "_summary":
            continue
        ls = graph["roles"][r]["lenses"]
        table.add_row(
            r, str(q.get("n_pool", "-")), str(q.get("n_passed", "-")),
            str(len(ls)),
            ", ".join(str(len(l["points"])) for l in ls) or "-",
        )
    console.print(table)
    console.print(f"artifact -> {path}")


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
def parity(
    write_to: str = typer.Option(..., "--write", help="where to save this platform's measurement"),
    compare_to: str = typer.Option(None, "--compare", help="other platform's JSON to diff against"),
    per_role: int = 3,
) -> None:
    """Measure render parity for this platform; optionally gate against another.

    Run on macOS first (--write), then on the Linux pod with --compare.
    Exits non-zero if the gate fails, so it can guard a cloud run.
    """
    import json as _json

    from timbre_graph_lab.parity import compare as do_compare
    from timbre_graph_lab.parity import measure
    from timbre_graph_lab.parity import write as do_write

    cfg = LabConfig()
    data = measure(cfg, per_role=per_role)
    out = do_write(data, write_to)
    n_ok = sum(1 for s in data["samples"] if s["status"] == "ok")
    console.print(f"{data['platform']}: {n_ok}/{data['n_samples']} ok -> {out}")

    if compare_to:
        other = _json.loads(Path(compare_to).read_text())
        verdict = do_compare(data, other)
        console.print(
            f"content_identical={verdict['content_identical']} "
            f"matched={verdict['n_matched']} "
            f"delta_cos={verdict['delta_cosine_median']:.4f} "
            f"baseline_rel_err={verdict['baseline_rel_err_median']:.4f}"
        )
        do_write(verdict, Path(write_to).with_suffix(".verdict.json"))
        if verdict["PASS"]:
            console.print("[green]PARITY PASS[/green] — cloud rendering is trustworthy")
        else:
            console.print(
                "[red]PARITY FAIL[/red] — render corpus on macOS; use the pod "
                "for features/training only"
            )
            raise typer.Exit(1)


@app.command()
def probe(
    per_role_index: int = typer.Option(0, help="which corpus anchor to take per role"),
    out: str = typer.Option("responses.json", help="where to write the response set"),
    render_avg: int = typer.Option(3),
) -> None:
    """Measure Jacobians for one anchor per role (the on-demand prober).

    This is what replaces the learned model: rendering is ~100x real-time, so
    measuring the six patches the user actually picked beats predicting them
    (0.534 vs 0.135 delta-cosine — see docs/TRAINING.md C12).
    """
    import time

    from timbre_graph_lab.corpus import load_manifest
    from timbre_graph_lab.prober import probe_anchor, write_responses
    from timbre_graph_lab.worker import RenderWorker

    cfg = LabConfig()
    manifest = load_manifest(cfg)
    worker = RenderWorker(cfg)
    responses, paths = {}, {}
    for role in ROLES:
        pool = [e for e in manifest["entries"] if role in e["roles"]]
        if not pool:
            console.print(f"[yellow]{role}: no anchors[/yellow]")
            continue
        entry = pool[min(per_role_index, len(pool) - 1)]
        t0 = time.time()
        r = probe_anchor(worker, role, entry["path"], avg=render_avg,
                         name=entry["name"])
        if r is None:
            console.print(f"[yellow]{role}: {entry['name']} unusable[/yellow]")
            continue
        r.preset_id = entry["preset_id"]
        responses[role] = r
        paths[role] = entry["path"]
        console.print(
            f"{role:6s} {entry['name'][:26]:26s} "
            f"{int(r.usable.sum())}/{len(r.param_names)} live params  "
            f"{r.n_renders} renders  {time.time()-t0:.1f}s"
        )
    p = write_responses(responses, cfg.workspace / out)
    (cfg.workspace / "anchor_paths.json").write_text(json.dumps(paths, indent=1))
    console.print(f"responses -> {p}")


@app.command()
def verify(
    responses: str = typer.Option("responses.json"),
    n_gestures: int = typer.Option(4, help="random leader gestures to test"),
    leader: str = typer.Option("lead"),
    render_avg: int = typer.Option(3),
) -> None:
    """Render-verify coupling against knob-copy and no-move baselines."""
    import numpy as np

    from timbre_graph_lab.prober import read_responses
    from timbre_graph_lab.verify import summarize, verify_gesture
    from timbre_graph_lab.worker import RenderWorker

    cfg = LabConfig()
    resp = read_responses(cfg.workspace / responses)
    paths = json.loads((cfg.workspace / "anchor_paths.json").read_text())
    if leader not in resp:
        raise SystemExit(f"leader {leader!r} not in responses ({list(resp)})")

    worker = RenderWorker(cfg)
    lead = resp[leader]
    live = np.flatnonzero(lead.usable)
    rng = np.random.default_rng(SEED)
    results = []
    for g in range(n_gestures):
        dx = np.zeros(len(lead.param_names))
        # a plausible sound-design gesture: 1-3 live controls, moderate size
        for i in rng.choice(live, size=min(3, len(live)), replace=False):
            dx[i] = rng.choice([-1, 1]) * rng.uniform(0.05, 0.12)
        res = verify_gesture(worker, resp, paths, leader, dx, avg=render_avg)
        results.append(res)
        console.print(f"gesture {g+1}/{n_gestures}: " + "  ".join(
            f"{r['role']}={r['coupled']['cosine'] if isinstance(r.get('coupled'), dict) else 'x'}"
            for r in res["followers"]))

    summary = summarize(results)
    out = cfg.reports_dir / "verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "gestures": results}, indent=1))
    console.print("\n[bold]render-verified achieved-vs-intended cosine[/bold]")
    for arm in ("coupled", "copy", "none"):
        console.print(f"  {arm:8s} {summary[f'{arm}_cosine_median']}")
    console.print(f"  by role: {summary['by_role']}")
    console.print(
        f"\n{'[green]COUPLING BEATS KNOB-COPY[/green]' if summary.get('beats_knob_copy') else '[red]does NOT beat knob-copy[/red]'}"
        f"   (n={summary['n_observations']})  -> {out}"
    )


@app.command()
def morph(
    responses: str = typer.Option("responses.json"),
    axis: str = typer.Option("softer", help="semantic axis to sweep"),
    points: int = typer.Option(9, help="control positions on the dial"),
    budget: int = typer.Option(24, help="renders per control position"),
    out: str = typer.Option("", help="output path (default morph-<axis>.json)"),
) -> None:
    """Build the precomputed morph graph the plugin panel consumes."""
    from timbre_graph_lab.morph import build_morph_graph, save_graph
    from timbre_graph_lab.prober import read_responses
    from timbre_graph_lab.worker import RenderWorker

    cfg = LabConfig()
    resp = read_responses(cfg.workspace / responses)
    paths = json.loads((cfg.workspace / "anchor_paths.json").read_text())
    worker = RenderWorker(cfg)
    graph = build_morph_graph(worker, resp, paths, axis_name=axis,
                              n_points=points, budget=budget)
    dest = cfg.workspace / (out or f"morph-{axis}.json")
    save_graph(graph, dest)

    q = graph["quality"]
    console.print(f"\n[bold]morph graph — axis '{axis}'[/bold]")
    console.print(f"{'role':7s}{'neg end':>9s}{'pos end':>9s}{'monoton':>9s}{'max jump':>10s}")
    for role in ROLES:
        v = q.get(role)
        if not v:
            continue
        fmt = lambda d: (f"{d['endpoint_cosine']:+.2f}" if d and d.get("moves") else "hold")
        console.print(f"{role:7s}{fmt(v.get('negative')):>9s}{fmt(v.get('positive')):>9s}"
                      f"{v['monotonicity']:9.2f}{v['max_param_jump']:10.3f}")
    s = q["_summary"]
    console.print(f"\nroles moving {s['roles_moving']}  directions working {s['directions_working']}"
                  f"  median endpoint cos {s['median_endpoint_cosine']}")
    console.print(f"graph -> {dest}")


@app.command()
def axes(
    responses: str = typer.Option("responses.json"),
    budget: int = typer.Option(45, help="renders per (role, axis) probe"),
) -> None:
    """Measure which semantic axes each anchor can express (the coupling policy)."""
    from timbre_graph_lab.axes import AXES, achievability, best_axes, shared_axes
    from timbre_graph_lab.prober import read_responses
    from timbre_graph_lab.refine import make_surge_measure
    from timbre_graph_lab.worker import RenderWorker

    cfg = LabConfig()
    resp = read_responses(cfg.workspace / responses)
    paths = json.loads((cfg.workspace / "anchor_paths.json").read_text())
    worker = RenderWorker(cfg)

    tables: dict[str, dict[str, float]] = {}
    for role in ROLES:
        if role not in resp or not worker.load_preset(paths[role]):
            continue
        tables[role] = achievability(
            make_surge_measure(worker, resp[role], avg=1), resp[role], budget=budget
        )
    roles = list(tables)
    console.print("\n[bold]achievability[/bold] (0.00 = declined)")
    console.print(f"{'axis':10s}" + "".join(f"{r:>7s}" for r in roles))
    for name in AXES:
        console.print(f"{name:10s}" + "".join(f"{tables[r][name]:7.2f}" for r in roles))
    console.print("\nbest per role:")
    for r in roles:
        console.print(f"  {r:6s} " + ", ".join(f"{n} {c:.2f}" for n, c in best_axes(tables[r])))
    console.print("\nshared axes (candidates for one global dial):")
    for n, cnt, med in shared_axes(tables)[:5]:
        console.print(f"  {n:10s} {cnt}/{len(roles)} roles, median {med:.2f}")
    out = cfg.workspace / "achievability.json"
    out.write_text(json.dumps(tables, indent=1))
    console.print(f"\n-> {out}")


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
