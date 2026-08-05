# Timbre Graph Plugin

A [Signals & Sorcery](https://signalsandsorcery.com) plugin for **sound exploration**: six Surge XT tracks behave as one instrument, so a single dial reshapes the whole ensemble at once and you go hunting for a sound instead of programming one.

<p align="center">
  <img src="assets/timbre-graph.png" alt="Timbre Graph" width="460" />
</p>

> Part of the **[Signals & Sorcery](https://signalsandsorcery.com)** ecosystem.

## What it does

Six fixed layers — Kick, Snare, Hat, Bass, Chord Pad, Lead — each on its own Surge XT track. One unlabelled dial moves all six together, and roughly every 5% of travel the ensemble settles into a genuinely different configuration, with the morph between them audible the whole way. Any track can be unlinked to freeze it while the other five keep moving. MIDI is never touched by the dial; it only ever changes sound.

## Architecture in one picture

The plugin is two halves joined by a small JSON file.

```
  OFFLINE (training/, Python + Surge XT + pedalboard)      SHIPPED        RUNTIME (TypeScript panel)
  ────────────────────────────────────────────────────    ──────────    ─────────────────────────────
  scan library → screen → measure → validate routes  →   tour.json  →   lerp between two snapshots
  ~10⁴ offline renders, ~100× real time                  ~180 KB        → setSynthParameters(relative)
```

Everything expensive happens ahead of time and is *measured on rendered audio*. At runtime there is no model, no solver and no inference — the panel interpolates between two snapshots and sends the difference, so the dial is instant and every point on it corresponds to a configuration that was actually rendered and checked.

## The runtime

The panel is built on the SDK's `panel-core` (`useGeneratorPanelCore` + `GeneratorPanelShell`), which owns track rows, the mixer strip, the sound drawer, shuffle and generation plumbing — the same machinery every other generator family uses. This repo contributes the parts that are specific to a coupled six-synth instrument:

- **Group semantics.** *Add Graph* creates one six-track group in a single click (one track per role, each with Surge XT loaded and the tour's start preset applied by `.fxp` path relative to the installed Surge content root). Group membership lives in scene data, and the dial only ever writes to members of a group — never to another panel's tracks.
- **The dial.** `paramsAt(controlPoints, snapshots, c)` is the whole engine: piecewise-linear interpolation between the two verified snapshots either side of `c`. The panel sends `paramsAt(c) − paramsAt(0)` as a **relative** parameter write (`host.setSynthParameters(trackId, deltas, 0, { relative: true })`, SDK 2.57.0), so the tour rides on whatever sound the track currently has instead of snapping to absolutes. Deltas are sent raw — both endpoints are validated configurations and the host clamps to each parameter's range.
- **Write discipline.** Dial writes are coalesced (newest position wins while a write is in flight) so dragging cannot queue stale writes, and each role is applied in isolation so one synth's failure cannot silence the other five.
- **Continuous parameters only.** Oscillator types, filter types, FM routing and unison voice counts are never written; there is no meaningful half-way between saw and FM, and stepping them produced clicks and garbage. This constraint is what shapes the entire offline design (below).

## The tour artifact

`assets/tour.json` (`tour-graph-v1`) is imported statically into the bundle, so the dial is live on first open — the training lab is for *re*-building it, not a user prerequisite. Per role it carries:

| field | meaning |
|---|---|
| `param_names` | the Surge parameters this role's tour writes. Per role, not global: which parameters exist depends on the start patch's oscillator types |
| `control_points` | ascending dial positions in 0..1, one per anchor |
| `anchors[]` | `preset_id`, name, and `fxp_path` relative to the Surge content root |
| `snapshots[][]` | absolute parameter values at each anchor, in `param_names` order |
| `declined` | this role has no validated tour and holds still rather than inventing motion |

A panel `import…` affordance loads a candidate artifact from disk (stored in project data) for testing before it is shipped.

## The offline lab

`training/` is a **measurement** lab, not a training run (see the next section). It hosts Surge XT headlessly through pedalboard/VST3 via the [signals-to-surge](https://github.com/shiehn/signals-to-surge) stack. Offline rendering is ~100× real time (~0.02 s for a 2.2 s clip), which is the fact the whole design leans on: it is cheaper to render a sound and look at it than to predict it.

Two conventions make a render trustworthy:

- **Determinism.** Surge runs oscillators at free-running phase with per-voice drift, so two identical renders differ. Every preset load forces `*retrigger*` on and `*drift*` to zero as a measurement convention, each measurement averages *k* renders, and every measured render is preceded by a throwaway settle render (parameter smoothing bleeds into the next buffer otherwise).
- **A perceptual basis.** Each render reduces to **20 DSP descriptors** — spectral centroid/bandwidth/rolloff/flatness/contrast, six band energies, ZCR, attack time, decay slope, envelope sparsity and flux, plus loudness and crest. Spectral and temporal dims are computed on RMS-normalised audio with loudness kept as its own separate dimensions, so "morphing" can never quietly become "turning it up". Descriptors are **standardised** by per-dimension corpus σ baked into the code — raw, two Hz-scale dimensions carry 85% of all delta energy and every distance silently becomes "spectral rolloff".

`tglab tour` then builds one tour per role in seven stages:

| stage | what happens | gate / algorithm |
|---|---|---|
| **screen** | every role-appropriate library patch is loaded and rendered through a short role-specific MIDI probe | must load, pass render QC (silence / flat-topped clipping / runaway level), and sit under a noise ceiling — a patch whose own renders wander (σ > 0.05) can never demonstrate a morph |
| **lens** | one patch becomes the start anchor | role-appropriate category first, then steadiest renders. Its oscillators and routing colour the entire tour |
| **effect** | every survivor's parameter values are applied *on top of the lens* and re-rendered | those values on different oscillators do not reproduce the patch they came from — they make something new, and that new sound is the only one the dial can make, so it is the only one worth measuring |
| **spread** | keep the most perceptually distinct survivors | farthest-point sampling in standardised descriptor space |
| **edges** | candidate hops between anchors | kNN proposal in descriptor space, then the interpolation *interior* is rendered: QC at every point, path length ≤ 3× the direct distance (no wild detour), and the two ends measurably far apart. Edge weight is the **measured travel** |
| **tour** | order ~20 anchors into a journey | greedy maximum-travel simple path: grow from every seed by always taking the heaviest hop, over the largest connected component containing the lens; keep the best by (anchor count, total travel). Deterministic; shortfalls are reported, never absorbed |
| **sweep** | the composed tour is replayed end to end exactly as the panel plays it | anchors that fail are dropped and the tour re-spliced |

Two design consequences worth knowing, both measured rather than assumed:

- **An anchor is not a preset reproduction.** Because only the lens preset is ever loaded, taking patch B's continuous values onto the lens's structure lands 0.79–5.3 normalised units away from B — usually *overshooting*. Arriving at B is a goal this runtime cannot have. But the *travel* per hop is 1.9–11.5 normalised units, against ~0.1 for the axis-morph design this replaced. So the gate is QC + measured travel, and an anchor is honestly a new configuration built from a real patch's values.
- **Validation uses the runtime's own lens.** Every candidate, every interpolation point and every gate is rendered through the loaded start preset, so validation and playback are the same signal and validity composes across hops.

## Is a model trained? No — and that was measured too

Nothing is trained at runtime, and no weights ship. The lab *did* build the obvious learned component — a role-conditioned FiLM MLP forward/delta proxy `F(params, role, probe) → z`, trained self-supervised on Surge's own renders over 2,111 anchors and 775,654 rows — and then benchmarked it out of the critical path:

| strategy for "how does this patch's sound move when this control moves?" | held-out delta-cosine |
|---|---|
| **probe the actual patch (render it)** | **+0.534** |
| population-mean Jacobian | +0.138 |
| single global ridge | +0.135 |
| nearest neighbour by parameters | +0.091 |
| trained MLP | +0.101 (a linear ridge scored 0.113) |

A preset's local parameter→timbre Jacobian is simply not a function of anything cheap that can be observed about it — averaging beating nearest-neighbour is the signature of that. Since a render costs ~30 ms, measuring the six patches actually chosen beats predicting them by ~4× with no model, no dataset versioning, no generalisation risk and no ONNX in an Electron app. `tglab gen` / `tglab train` survive as a research benchmark; the shipped path never touches them.

The lab also keeps the closed-loop tooling that came out of that work — measured per-anchor Jacobians (central finite differences), damped minimum-norm least-squares solving toward a named perceptual axis, and a line-search refiner that scores candidate moves against *real* renders (`tglab probe` / `axes` / `morph`). That produced a working single-axis dial whose flaw was structural: a minimum-norm step is by construction the *smallest* audible change, so from 0 to 100 the patch never stopped being itself. The tour replaced it as the shipped mechanism.

## MIDI

MIDI generation is completely separate from the dial and, unlike the sound, it *is* generative: each row calls `host.generateWithLLM` with the scene's key, chords and sibling tracks — the same machinery bass, pad and ensemble use. What is different is that **the role is the prompt**: there is no text box, because the creative control here is the dial. Deterministic guard rails then fold the model's output into the register each patch was measured in — percussion is pinned to its measured pitch (kick 36, snare 50, hat 66) and pitched roles are transposed by *whole octaves as a phrase*, so intervals survive. A failed generation surfaces as an error rather than as filler; a model failure that quietly played back a pattern would be indistinguishable from success.

## Install

From within Signals & Sorcery: **Settings > Manage Plugins > Add Plugin** and enter:

```
https://github.com/shiehn/sas-timbre-graph
```

Or clone manually into `~/.signals-and-sorcery/plugins/@signalsandsorcery/timbre-graph/`.

## Capabilities

| Capability | Required |
|------------|----------|
| `requiresLLM` | Yes — for the **notes**. The sound is never generated: the dial only replays measured configurations |
| `requiresSurgeXT` | Yes — six Surge XT instances at runtime, and headless rendering to build the tour |

## Development

```bash
npm install
npm test         # panel + interpolation + generation contract
npm run build    # tsup -> dist/ (the app consumes the built dist)
```

Rebuilding the shipped tour (macOS + Surge XT required):

```bash
cd training
uv venv --python 3.11 .venv && uv pip install -e ".[dev]" --python .venv/bin/python
.venv/bin/tglab tour --anchors 20        # ~25-35 min for all six roles
# read workspace/reports/tour-report.json — per-role `shortfall` and `sweep.dropped`
cp workspace/tour.json ../assets/tour.json

.venv/bin/python -m pytest tests/ -q     # pure-python, no Surge needed
.venv/bin/python -m pytest -m requires_surge tests/test_shipped_tour_integration.py -q
```

That last suite is the gate on an artifact being honest: it hosts real Surge and asserts, per role, that every parameter the shipped tour carries exists, accepts a write, and measurably changes the render. It is what caught a dial that moved every parameter correctly while the lead stayed silent.

Full measurement conventions, benchmarks and the decision log: [`docs/TRAINING.md`](docs/TRAINING.md) and [`training/README.md`](training/README.md).

Built with the [@signalsandsorcery/plugin-sdk](https://github.com/shiehn/sas-plugin-sdk); see the [Plugin SDK docs](https://signalsandsorcery.com/plugin-sdk/) for the API reference.

## The Signals & Sorcery Ecosystem

| Repo | Role |
|------|------|
| [signalsandsorcery.com](https://signalsandsorcery.com) | The platform |
| [sas-plugin-sdk](https://github.com/shiehn/sas-plugin-sdk) | Plugin SDK |
| [sas-drum-plugin](https://github.com/shiehn/sas-drum-plugin) | Drum generator |
| [sas-bass-plugin](https://github.com/shiehn/sas-bass-plugin) | Bass generator |
| [sas-pad-plugin](https://github.com/shiehn/sas-pad-plugin) | Pad generator |
| [sas-ensemble-plugin](https://github.com/shiehn/sas-ensemble-plugin) | Ensemble generator |

## License

MIT for the plugin. The training lab in `training/` is GPL-3.0-or-later (it depends on GPL audio tooling); only exported artifacts cross that boundary.
