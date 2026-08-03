"""Rebuild the shipped graph so every parameter it carries is measurably audible.

`sensitivity` used to come from the probe Jacobian's row norm. That is an
optimistic estimate: a parameter can register a Jacobian response yet leave the
render unchanged at the step sizes the panel actually uses, and the solver was
free to spend its whole gesture budget on exactly those parameters (the live
lead put its largest delta on `a_width`, audibility 0.46, while
`a_filter_1_cutoff` at 10.95 barely moved — so nothing was audible).

Here each anchor's `usable` mask is first narrowed to the parameters whose
render demonstrably changes, and `sensitivity` is set to that measured
magnitude. The solver then cannot pick an inert parameter, and the shipped
numbers are the same quantity the integration test verifies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.corpus import load_manifest
from timbre_graph_lab.descriptors import standardize
from timbre_graph_lab.morph import build_morph_graph, save_graph
from timbre_graph_lab.prober import probe_anchor, write_responses
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker

STEP = 0.08


def measured_sensitivity(worker: RenderWorker, role: str, names: list[str]) -> np.ndarray:
    """Render-change magnitude per parameter, in standardized descriptor units.

    Both directions are averaged so a parameter already near a rail is not
    written off for having nowhere to go on one side.
    """
    probe = get_probe(role, "short")
    sigma = float(np.linalg.norm(standardize(worker.noise_floor(probe, k=3, avg=2))))
    floor = 1.5 * max(3.0 * sigma, 0.02)
    z0 = standardize(worker.render_descriptors(probe, k=2))
    live = dict(worker.baseline_raw)

    out = np.zeros(len(names))
    for i, name in enumerate(names):
        base = live.get(name)
        if base is None:
            continue
        effects = []
        for step in (STEP, -STEP):
            if not (0.0 <= base + step <= 1.0):
                continue
            worker.apply_delta({name: step})
            z1 = standardize(worker.render_descriptors(probe, k=2))
            worker.restore_baseline()
            effects.append(float(np.linalg.norm(z1 - z0)))
        if effects and max(effects) > floor:
            out[i] = float(np.mean(effects)) / STEP   # per unit of parameter travel
    return out


def main() -> None:
    cfg = LabConfig()
    worker = RenderWorker(cfg)
    paths = json.loads((cfg.workspace / "anchor_paths.json").read_text())
    manifest = load_manifest(cfg)

    responses = {}
    for role, path in paths.items():
        resp = probe_anchor(worker, role, path, avg=3, name=Path(path).stem)
        if resp is None:
            raise SystemExit(f"{role}: chosen anchor failed to probe")
        entry = next((e for e in manifest["entries"] if e["path"] == path), None)
        if entry:
            resp.preset_id = entry["preset_id"]
        responses[role] = resp
        print(f"probed {role:6s} {resp.name[:26]:26s} "
              f"{int(resp.usable.sum())}/{len(resp.param_names)} live")

    for role, resp in responses.items():
        assert worker.load_preset(paths[role]), f"{role}: anchor will not load"
        sens = measured_sensitivity(worker, role, list(resp.param_names))
        live = sens > 0
        resp.usable = resp.usable & live          # solver may only move real params
        resp.meta["measured_sensitivity"] = [round(float(v), 6) for v in sens]
        top = int(np.argmax(sens)) if live.any() else -1
        print(f"{role:6s} audible={int(live.sum()):3d}/{len(sens):3d} "
              f"top={resp.param_names[top] if top >= 0 else '-':28s} "
              f"{sens[top] if top >= 0 else 0.0:7.3f}")

    write_responses(responses, cfg.workspace / "responses.json")
    graph = build_morph_graph(
        worker, responses, paths, axis_name="softer", n_points=9, budget=24
    )

    save_graph(graph, cfg.workspace / "morph-softer.json")
    print("\nquality:", graph["quality"]["_summary"])


if __name__ == "__main__":
    main()
