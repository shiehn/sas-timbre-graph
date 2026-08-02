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

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # pure-python, no Surge needed
```

License: GPL-3.0-or-later (depends on pedalboard/GPL via synth2surge). The
plugin at the repo root is MIT; only exported model *artifacts* cross the
boundary, never GPL code.
