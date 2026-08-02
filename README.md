# sas-timbre-graph

**Timbre Graph** — a Signals & Sorcery plugin hosting six fixed Surge XT
tracks (Kick, Snare, Hat, Bass, Chord Pad, Lead) that morph *together*
through one perceptual control surface. Drag one point; all six patches move
in a musically coherent direction. Unlink any track to freeze or hand-tweak
it while the other five keep morphing.

Two halves:

| Where | What | License |
|---|---|---|
| repo root | S&S panel plugin (currently a stub — standard `sas-plugin-template` shape) | MIT |
| `training/` | **The hard part**: self-supervised forward/delta timbre model over the six roles, trained from Surge's own renders (no taste labels) | GPL-3.0 |

Start here: [`docs/TRAINING.md`](docs/TRAINING.md) — amended plan, day-one
benchmarks, pipeline runbook, RunPod guidance. Original proposal:
`docs/proposal-v1.docx`.

## Plugin (stub)

```bash
npm install
npm test
npm run build
```

## Training lab

```bash
cd training
uv venv --python 3.11 .venv && uv pip install -e ".[dev]" --python .venv/bin/python
.venv/bin/tglab --help
```
