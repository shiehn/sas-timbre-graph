# timbre-graph-lab

Training lab for the S&S **Timbre Graph** plugin: a self-supervised
forward/delta timbre model over six coupled Surge XT roles (Kick, Snare,
Hat, Bass, Chord Pad, Lead), built on the
[signals-to-surge](https://github.com/shiehn/signals-to-surge) hosting stack.

Full amended plan, benchmarks, and runbook: [`../docs/TRAINING.md`](../docs/TRAINING.md).
Original proposal: `../docs/proposal-v1.docx`.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]" --python .venv/bin/python
source .venv/bin/activate
```

Requires Surge XT installed at the default macOS locations (VST3 + factory
content). Override paths via `LabConfig` / `TGLAB_WORKSPACE`.

## Pipeline

```bash
tglab inventory   # corpus scan + role rules
tglab policy      # live-safe parameter allow-list
tglab pilot       # baseline render + QC
tglab gen         # perturbation dataset shards (resumable)
tglab train       # forward/delta proxy -> ONNX bundle
tglab bench       # render throughput
```

## Reshipping the graph the plugin bundles

`../assets/morph-softer.json` is the artifact the panel replays; it is produced
by two steps, both of which need Surge XT installed:

```bash
.venv/bin/python pick_anchors.py    # choose one anchor per role, by MEASURED morphability
.venv/bin/python rebuild_graph.py   # probe those anchors, narrow to audible params, solve
cp workspace/morph-softer.json ../assets/morph-softer.json
```

Anchor choice is the dominant factor in whether the dial is audible at all — a
patch with no reachable timbre range cannot be morphed however the panel scales
the gesture (see `docs/TRAINING.md` § C14). `pick_anchors.py` therefore ranks
candidates by how many parameters measurably change the render and rejects
patches whose own renders wander (`MAX_SIGMA`). Screening is the slow half and
writes `workspace/anchor_paths.json` as it goes, so a later failure does not
cost it twice.

After reshipping, re-run the integration suite below — it is the gate that the
new artifact is honest.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # pure-python, no Surge needed
.venv/bin/python -m pytest -m requires_surge tests/test_shipped_graph_integration.py -q
```

The second suite hosts real Surge and asserts, per role, that every parameter
the shipped graph carries exists, accepts a write, and — when marked audible —
measurably changes the render. It is what caught the silent-lead failure.

License: GPL-3.0-or-later (depends on pedalboard/GPL via synth2surge). The
plugin at the repo root is MIT; only exported model *artifacts* cross the
boundary, never GPL code.
