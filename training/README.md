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

## Reshipping the tour the plugin bundles

`../assets/tour.json` is the artifact the panel replays: per role, ~20 screened
Surge patches whose parameter values become the stops on the dial. One command
builds it (Surge XT must be installed):

```bash
.venv/bin/tglab tour --anchors 20      # ~25-35 min for all six roles
```

Read `workspace/reports/tour-report.json` BEFORE shipping — in particular the
per-role `shortfall` (fewer anchors than asked for) and `sweep.dropped`
(anchors the composed tour could not reach cleanly). Both are printed as they
happen; neither is ever silently absorbed. Then:

```bash
cp workspace/tour.json ../assets/tour.json
```

What the stages do, and why (full rationale in `src/timbre_graph_lab/tour.py`):

| stage | gate |
|---|---|
| screen | loads, passes render QC, own renders steady enough to hear a change through (`MAX_SIGMA`) |
| lens | the START anchor — role-appropriate category, steadiest renders. Its oscillators and routing colour the whole tour |
| effect | every candidate re-rendered THROUGH the lens; that is the only sound the dial can make |
| spread | keep the most perceptually spread survivors (farthest-point sampling) |
| edges | render the interpolation between neighbours: QC at every point, no wild detour, ends audibly apart |
| tour | deterministic maximum-travel path through the valid edges, starting at the lens |
| sweep | the whole tour replayed end to end exactly as the panel plays it; anchors that fail are dropped and the tour re-spliced |

The screening report per role is written incrementally to
`workspace/reports/anchor-screen-<role>.json`, so a later failure does not cost
that work twice.

After reshipping, re-run the integration suite below — it is the gate that the
new artifact is honest.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # pure-python, no Surge needed
.venv/bin/python -m pytest -m requires_surge tests/test_shipped_tour_integration.py -q
```

The second suite hosts real Surge and asserts, per role, that every parameter
the shipped graph carries exists, accepts a write, and — when marked audible —
measurably changes the render. It is what caught the silent-lead failure.

License: GPL-3.0-or-later (depends on pedalboard/GPL via synth2surge). The
plugin at the repo root is MIT; only exported model *artifacts* cross the
boundary, never GPL code.
