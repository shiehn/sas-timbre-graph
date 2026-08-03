"""Choose shipped anchors by MEASURED morphability, then build the artifact.

The first shipped graph took corpus index 0 per role, which handed the snare a
Keys patch and the hat a Leads patch — anchors with almost no reachable timbre
range, so those roles could not morph at all however the panel scaled the
gesture. Anchor choice, not the morph machinery, was the ceiling.

Screens several role-appropriate candidates per role, scores each by how many
of its core parameters measurably change the render (the same criterion the
integration test verifies), then probes the winners and writes the shipped
graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import LabConfig, ROLES
from timbre_graph_lab.corpus import load_manifest
from timbre_graph_lab.descriptors import standardize
from timbre_graph_lab.gen import core_params
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import RenderWorker, qc_audio

STEP = 0.08
CANDIDATES_PER_ROLE = 6
# Prefer anchors whose own category matches the role: a "Keys" patch used as a
# snare is what produced an unmorphable anchor in the first place.
PREFERRED_CATEGORIES = {
    "kick": ("Percussion",),
    "snare": ("Percussion",),
    "hat": ("Percussion",),
    "bass": ("Basses",),
    "pad": ("Pads", "Polysynths", "Chords"),
    "lead": ("Leads", "Plucks", "Keys", "Brass"),
}


# An anchor whose own renders wander this much cannot demonstrate a morph:
# whatever the dial does is buried under the patch's own drift, both in
# measurement and in the listener's ear.
MAX_SIGMA = 0.05


def score_anchor(worker: RenderWorker, role: str, path: str, params: list[str]) -> int:
    """How many core parameters measurably change this patch's render."""
    if not worker.load_preset(path):
        return -1
    probe = get_probe(role, "short")
    if not qc_audio(worker.render(probe)).ok:
        return -1
    sigma = float(np.linalg.norm(standardize(worker.noise_floor(probe, k=3, avg=2))))
    if sigma > MAX_SIGMA:
        print(f"  {role:6s} {Path(path).stem[:26]:26s} REJECT sigma={sigma:.3f}")
        return -1
    z0 = standardize(worker.render_descriptors(probe, k=2))
    claim = 1.5 * max(3.0 * sigma, 0.02)
    live = worker.baseline_raw
    n = 0
    for name in params:
        base = live.get(name)
        if base is None:
            continue
        step = STEP if base <= 0.5 else -STEP
        worker.apply_delta({name: step})
        z1 = standardize(worker.render_descriptors(probe, k=2))
        worker.restore_baseline()
        if float(np.linalg.norm(z1 - z0)) > claim:
            n += 1
    return n


def main() -> None:
    cfg = LabConfig()
    manifest = load_manifest(cfg)
    worker = RenderWorker(cfg)
    # v1 is the policy present on this machine; core_params narrows it to the
    # morphable core either way.
    policy_file = sorted(cfg.workspace.glob("policy/policy-v*.json"))[-1]
    policy_allowed = json.loads(policy_file.read_text())["allowed"]

    chosen: dict[str, str] = {}
    for role in ROLES:
        pool = [e for e in manifest["entries"] if role in e["roles"]]
        preferred = [e for e in pool if e["category"] in PREFERRED_CATEGORIES[role]]
        # role-appropriate category first, then anything valid for the role
        candidates = (preferred + [e for e in pool if e not in preferred])[
            :CANDIDATES_PER_ROLE
        ]
        best_name, best_path, best_score = "", "", -1
        for e in candidates:
            worker.load_preset(e["path"])
            params = core_params(policy_allowed, worker.baseline_raw)
            s = score_anchor(worker, role, e["path"], params)
            print(f"  {role:6s} {e['name'][:26]:26s} audible={s}")
            if s > best_score:
                best_name, best_path, best_score = e["name"], e["path"], s
        chosen[role] = best_path
        # written as we go: screening is the slow part, and losing it to a
        # failure in a later stage means paying for it twice
        (cfg.workspace / "anchor_paths.json").write_text(json.dumps(chosen, indent=1))
        print(f"{role:6s} -> {best_name}  ({best_score} audible params)\n")

    print("wrote anchor_paths.json — run rebuild_graph.py to probe and ship")


if __name__ == "__main__":
    main()
