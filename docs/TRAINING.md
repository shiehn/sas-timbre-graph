# Timbre Graph — Training Plan (amended)

The original proposal lives at `docs/proposal-v1.docx` ("Surge Ensemble Motion
Lab", 2026-08-02). Its core design — a self-supervised **forward/delta timbre
proxy** `F(params, role, probe) → z` trained on Surge's own renders, with all
gesture translation done by inference-time solving — **survives review and is
what this lab implements.** The amendments below come from benchmarking the
actual reuse stack (signals-to-surge) on the actual corpus machine (M4 Mac)
on day one.

## Measured benchmarks (M4, 2026-08-02)

| Measurement | Value |
|---|---|
| Surge XT VST3 host load (pedalboard) | 1.2 s |
| One-time param mapping build (per process) | 16.1 s |
| .fxp load via param map | 540/547 params matched |
| Offline render, 2.2 s clip | **0.02 s (~100× real-time)** |
| Offline render, 8.8 s clip | 0.07 s |
| Throughput | **~167k renders/hour/process** |
| Repeat-render determinism (settled) | maxdiff 6e-8 (feature-identical) |
| Repeat-render right after a param change | maxdiff 0.65 (**smoothing bleed**) |

Two consequences drive everything below: **rendering is nearly free on the
M4**, and **every measured render must be preceded by a throwaway settle
render** (encoded in `worker.py`).

## C0 — Renders were not reproducible (found + fixed, 2026-08-02)

The single most important finding of day one, and it invalidated the first
pilot. Surge runs oscillators at **free-running phase** (when retrigger is
off) with **per-voice drift**, so two identical renders differ. Measured
descriptor noise vs finite-difference signal on the original pipeline:

| Role | SNR before | SNR after fix |
|---|---|---|
| bass | **0.16** | deterministic (σ≈0) |
| pad | 0.68 | deterministic (σ≈0) |
| hat | 1.34 | ~9 (noise oscillator, irreducible) |
| snare | 36.9 | deterministic (σ≈0) |

The training targets were mostly measurement noise. The parity checker
caught it by failing a **Mac-vs-itself** comparison (delta-cosine 0.47 where
1.0 is required).

Fix (`worker.freeze_stochastic`, called on every preset load):
force all `*retrigger*` on and all `*drift*` to zero as a **measurement
convention**, average descriptors over `RENDER_AVG=3` renders, measure a
per-anchor **noise floor** and gate the sensitivity screen on SNR ≥ 3 rather
than a bare epsilon. Phase-locking barely affects timbre statistics, so the
convention costs nothing and the before/after pair is consistent by
construction.

Result — within-anchor ridge predictability (held-out edits, same anchor):

| | before | after |
|---|---|---|
| median across shards | 0.330 | **0.739** |
| pad (worst role) | 0.006 | **0.711** |

Mac-vs-itself parity now returns delta-cosine **1.0000**. Cost: ~4.5× more
renders per anchor (~90 s/anchor single core), which is what the RunPod
plan absorbs. **Any shard generated before this fix is garbage — regenerate,
never mix.**

## C0b — QC rejected loud renders as "clipped" (found + fixed 2026-08-02)

Caught mid-corpus-run: the gate was discarding **13% of all anchors** for a
defect that did not exist. It flagged any render with >1% of samples at or
above 0.999 as clipping — a meaningless test for float audio, since
pedalboard returns float32 and never truncates at unity. Measured on the
rejected set: peaks of **1.4–4.0 with a longest pinned-run of ONE sample**,
i.e. not a single flat-topped pair anywhere. They were simply hot patches.

Re-tested against the corrected gate: **29 of 29 clipping rejects recovered**
(Maj-Min Stab, Alias Pornography, Circus 1, the Kick Room family…), while all
21 `silent` rejects stayed silent — those are the genuine C5 preset-load gap.

The gate now judges waveform *shape*: reject only a flat-topped signal (≥8
consecutive samples pinned at the peak with peak ≤ unity), plus a new
runaway-level guard (~+30 dBFS) for self-oscillation.

Crucially this was a **pure accept/reject change, so shards already on disk
stayed valid** — spectral/temporal descriptors are computed on RMS-normalized
audio, which a test now pins directly (level-invariant to 2%, while the
explicit loudness dims still track level). Expected yield rose from ~65% to
~78% of jobs.

Lesson worth keeping: a QC threshold copied from fixed-point audio intuition
(0 dBFS = ceiling) is wrong in a float pipeline. Verify what a gate rejects
before trusting its rejection rate.

## C12 — PROBE, DON'T PREDICT (decisive result, 2026-08-03)

**The learned cross-preset forward proxy is unnecessary. Measure the six
anchors the user actually chose instead.** This supersedes the model-centric
parts of the proposal (§3, §5, §6) and is the single most important finding
of the build.

### What was measured

Full corpus: 2,145 anchors → 2,111 shards → 775,654 training rows (42× the
pilot). The role-conditioned FiLM MLP trained to a low training loss (0.485 →
0.070) and **still lost to a linear ridge baseline** on held-out presets:
delta-cosine **0.101 vs 0.113**. More data did not help: the pilot's 50
shards gave 0.027, and 42× more gave 0.101 — both ≈ 0.

Hypothesis at the time: anchors perturbed near-disjoint parameter subsets
(228/350 params never perturbed by any anchor; same-role Jaccard **0.205**),
so columns did not align across presets. Two bugs were fixed — switch
parameters were removed from the continuous allow-list (350 nominal → 97
genuinely continuous, via a behavioural probe) and a shared 34-parameter core
set was perturbed on every anchor — and a fresh 219-shard pilot was
generated. Column alignment improved exactly as intended, **Jaccard 0.205 →
0.732**. Cross-preset transfer did not move at all:

| | v1 | v2 (aligned columns) |
|---|---|---|
| Jaccard support overlap | 0.205 | **0.732** |
| Nearest-anchor transfer | 0.078 | **0.071** |
| Global ridge | 0.120 | 0.111 |

So the disjoint-support hypothesis was **wrong**. Every transfer strategy
tested on the full corpus (249 held-out presets) plateaus in the same place,
while probing the anchor itself is ~4× better:

| Strategy | delta-cosine |
|---|---|
| **probe the actual anchor (oracle)** | **+0.534** |
| nearest neighbour by descriptors | +0.075 |
| nearest neighbour by parameters | +0.091 |
| mean of 16 nearest | +0.144 |
| population-mean Jacobian | +0.138 |
| single global ridge | +0.135 |

Note that *averaging beats nearest-neighbour*. That is the signature of
idiosyncratic local behaviour: no individual similar patch predicts another,
so the population mean is the best any predictor can do — and it caps at
~0.14. A preset's local parameter→timbre Jacobian is simply not a function of
anything cheap we can observe about it.

### Why this is good news

The instrument never needed prediction. Rendering is ~100× real-time, so the
true local response of six *specific* chosen patches can be **measured** in
seconds:

- central finite differences over the 34-parameter core, averaged ×3 for
  noise ≈ 200–280 renders per anchor ≈ **4–8 s**
- six anchors ≈ **30–50 s serial, ~10 s across six processes**

That buys **0.534 instead of 0.135** — four times the accuracy — with no
model, no training pipeline, no dataset versioning, no generalization risk,
and no ONNX in the plugin. The morph graph is built from measured Jacobians
at anchor-selection time, exactly the C6 runtime that was already the plan.

### Consequences

- **Drop** the forward/delta model from the critical path (proposal Phases
  4–7). Keep the corpus + shards as a benchmark for future research.
- **Keep** everything that made measurement trustworthy: frozen-phase
  determinism (C0), the corrected QC gate (C0b), the probed parameter policy,
  the descriptor basis, and the per-anchor sensitivity screen.
- **Build next**: an on-demand anchor prober (measure 6 anchors → response
  matrices → morph graph), which is a small, deterministic component.
- A learned proxy only becomes interesting if instant graph-building matters
  more than accuracy, and the ceiling above says it would be a downgrade.

## C13 — Coupling: measured reality (2026-08-03)

The prober, solver and render-verifier are built (`prober.py`, `solver.py`,
`verify.py`, `tglab probe` / `tglab verify`). Probing six anchors takes **98 s
serial, ~25 s across six processes** and yields 8–26 live parameters per
anchor out of a 34-parameter core. What the render-verification then measured
changes the design in three ways.

### C13a — Descriptors MUST be standardized (another scale bug)

First verification looked triumphant: coupled 0.619 vs knob-copy −0.084. It
was an illusion. Raw descriptors span wildly different units, and measured
over the corpus **`rolloff85_mean` + `bandwidth_mean` alone carry 85% of all
delta energy** while 15 of 20 descriptors contribute ~0%. Every cosine — and
every solver target — was effectively "spectral rolloff" and nothing else.

`FEATURE_SCALE` (per-descriptor corpus std, baked into `descriptors.py` so the
runtime needs no corpus) now standardizes the space. Honest result after the
fix: coupled **0.057** vs copy 0.022. The same class of mistake as the QC gate
(C0b): *a unit assumption nobody checked*. Third time this project has been
bitten by one — check the scale before trusting any metric.

### C13b — Open-loop prediction is too weak; search instead

The Jacobian is linear and measured one parameter at a time, but Surge's
controls interact multiplicatively, so a combined solved move does not produce
the predicted sum. Since a render costs ~30 ms, the fix is to stop predicting:
seed from the Jacobian solve, then refine against **real renders**.

| | median achieved-vs-requested cosine |
|---|---|
| open-loop (Jacobian solve) | +0.048 |
| **closed-loop (+24 renders)** | **+0.186** |

Improved in **92%** of cases for 25 renders per target — under a second. This
is C12's lesson taken one step further: don't predict, *measure*, and keep
measuring inside the loop.

### C13c — Roles couple on DIFFERENT axes (the real coupling policy)

Percussion does not couple like melody, and that is physics rather than a bug:
a short low thump has a tiny reachable timbre space. Measured per semantic
axis (render-verified):

| | kick | snare | hat | bass | pad | lead |
|---|---|---|---|---|---|---|
| brighter | 0.02 | 0.01 | 0.26 | 0.11 | 0.41 | **0.46** |
| fuller | 0.05 | 0.09 | 0.07 | 0.29 | **0.50** | 0.37 |
| longer | −0.14 | **0.93** | **0.69** | 0.05 | −0.46 | 0.40 |
| snappier | −0.19 | 0.15 | 0.08 | 0.26 | 0.23 | −0.59 |

Asking a snare to get "brighter" achieves 0.01; asking it to get "longer"
achieves **0.93**. So the coupling policy is not a 6×6 scalar matrix (proposal
§3.1) but a **(role × axis) matrix**, and it can be *measured per anchor*
rather than hand-tuned: probe which axes an anchor can actually express, then
drive each role along its own best axis. One dial, six coherent moves, each in
the vocabulary its synth actually has.

### C13d — Closed-loop refiner: BUILT and validated

`refine.py` = Jacobian-seeded line search + adaptive coordinate refinement
against real renders, with an injectable `measure` so the search is testable
without audio. Objective is on-axis projection with a mild off-axis penalty.

Measured on real Surge, 60 renders per target, six anchors × four axes:

| | |
|---|---|
| pairs it chose to move | **10 of 24** |
| median cosine where it moved | **+0.523** |
| of those, above 0.4 | **10 of 10** |
| pairs it declined | 14 |
| speed | **5.8 s per target** (60 renders) |

**Declining is a feature, not a miss.** A kick asked to get "brighter" has
almost no reachable timbre space (11/34 live params), and the honest answer is
to leave it alone rather than invent a move — which is also the proposal's own
low-confidence fallback. So the instrument either moves convincingly (0.45 to
0.83) or stays put; it never produces incoherent motion. Reporting that
averages declines together with real moves is meaningless, hence
`RefineResult.moved`.

The refiner also rescues cases open-loop got *backwards*: snare/longer
−0.11 → **+0.83**, hat/longer −0.68 → **+0.45**, lead/snappier −0.56 → **+0.63**.

Caching the anchor's baseline descriptors (instead of re-rendering them on
every evaluation) cut ~15 s per target to 5.8 s — each measurement now costs
about one render rather than four.

### The measured coupling policy

Not hand-tuned, not learned — probed:

| role | best axis | cosine |
|---|---|---|
| snare | longer | **+0.83** |
| lead | longer | +0.73 |
| bass | snappier | +0.64 |
| pad | snappier | +0.59 |
| hat | longer | +0.45 |
| kick | *(declined all four)* | — |

This is C13c made operational: at probe time, test each anchor against the
axis library and record what it can express. The dial then drives every role
along its own best axis, which is what makes one control produce six coherent
moves instead of five wrong ones.

### C13e — Axis library + the complete measured coupling policy

The kick declining every spectral axis was diagnosed, not worked around. Its
Jacobian shows **0.000 reach in `band_high` and `band_air`** — "brighter" is
physically impossible for it, not merely hard. Its expressive dimensions are
envelope and dynamics. A second cause was sign coupling: the original `longer`
axis paired `decay_slope+` with `attack_time+`, while the kick's own reachable
direction pairs `decay_slope+` with `attack_time-`, so the request was
unreachable even though both components move individually.

`axes.py` now holds a named axis library (spectral / envelope / dynamics /
texture) plus `achievability()`, which render-verifies each axis per anchor.
The library is deliberately redundant — `longer` and `boomier` differ only in
that attack sign — because measurement, not taste, picks the winner.

**Full matrix, render-verified (0.00 = declined), 6 roles × 9 axes, 301 s:**

| axis | kick | snare | hat | bass | pad | lead |
|---|---|---|---|---|---|---|
| brighter | 0.00 | 0.00 | 0.37 | 0.00 | 0.46 | 0.45 |
| fuller | 0.00 | 0.00 | 0.00 | 0.00 | 0.44 | 0.45 |
| wider | 0.00 | 0.69 | 0.60 | 0.39 | 0.76 | **0.79** |
| longer | 0.71 | **0.84** | 0.54 | 0.00 | 0.00 | 0.73 |
| boomier | 0.66 | 0.00 | **0.81** | 0.00 | 0.00 | 0.00 |
| tighter | **0.81** | **0.86** | 0.69 | 0.61 | **0.83** | 0.00 |
| punchier | **0.81** | 0.00 | 0.00 | **0.72** | 0.51 | **0.84** |
| softer | 0.75 | 0.82 | 0.63 | 0.56 | 0.76 | 0.71 |
| rougher | 0.00 | 0.00 | 0.00 | 0.60 | 0.68 | 0.00 |

The kick went from declining all four original axes to **0.66-0.81 on four**.
Every role now has at least three axes above 0.40.

**A single dial is viable.** Axes ranked by how many roles can express them:

| axis | roles | median cosine |
|---|---|---|
| **softer** | **6 / 6** | 0.73 |
| tighter | 5 / 6 | 0.81 |
| punchier | 4 / 6 | 0.76 |
| wider | 4 / 6 | 0.73 |
| longer | 4 / 6 | 0.72 |

`softer` moves all six coherently; `tighter` moves five at 0.81. So the
unlabeled dial has real, measured backing — and where a role cannot follow it
holds still rather than misbehaving.

Spectral axes are the weak ones (percussion cannot reach them at all), which
inverts the intuition the proposal started from: **the ensemble couples through
envelope and dynamics, not brightness.**

### C13f — The morph graph is BUILT (the artifact the panel consumes)

`morph.py` precomputes the dial: for a chosen anchor set and one semantic axis
it stores an **absolute parameter snapshot per synth at every control
position**, each render-verified on the way. Runtime is then pure
interpolation — `params_at()` is the whole engine, a handful of lines that port
trivially to TypeScript. No solver, no model, no ONNX, zero latency, and
per-track unlink is just "stop applying this row".

Built by **walking** rather than solving each position independently: every
step targets a small increment and warm-starts from the previous solution, so
steps stay near the radius where the Jacobian is valid (the C13b failure) and
consecutive snapshots stay close.

**First real graph — axis `softer`, 9 control points, built in 159 s (44 KB):**

| role | negative end | positive end | monotonicity | max param jump |
|---|---|---|---|---|
| kick | +0.55 | +0.62 | 0.75 | 0.053 |
| snare | *holds* | **+0.80** | 1.00 | 0.019 |
| hat | *holds* | +0.58 | 1.00 | 0.044 |
| bass | +0.48 | +0.56 | 1.00 | 0.023 |
| pad | +0.49 | **+0.71** | 0.88 | 0.020 |
| lead | **+0.70** | +0.64 | 0.88 | 0.033 |

- **roles moving 6/6**, directions working **10/12**
- median endpoint cosine **0.599**, median monotonicity **0.938**
- worst parameter jump **0.053**, and at runtime the largest parameter change
  per 1% of dial travel is **0.004** — the dial is smooth to the touch.

Endpoint cosines closely track the achievability table (snare 0.80 vs 0.82,
pad 0.71 vs 0.76, bass 0.56 vs 0.56), so the walk loses almost nothing versus
refining a single target.

**Expressiveness is asymmetric.** Snare and hat get *softer* but cannot get
*harder* — already-tight patches have nowhere to go — so they hold still on
that side rather than inventing motion. Two of twelve directions are dead and
that is the honest answer, not a defect.

**A reporting lesson, for the third time.** The first quality read looked bad
(hat "median cosine 0.02") because it averaged every control point: near the
centre the move is deliberately tiny so its measured direction is mostly render
noise, and it also averaged the dead direction into the live one. `quality()`
now reports **per direction, at the endpoints**. Same mistake as the QC gate
and the unstandardized descriptors — *check what a metric is actually
averaging before believing it.*

### C13g — Panel + registration DONE; one SDK gap blocks live audio

Shipped:
- **`TimbreGraphPanel.tsx`** — six always-visible tracks, the morph dial, and a
  per-track link toggle. `paramsAt()` is the whole runtime: linear
  interpolation between two verified snapshots. Dial writes are **coalesced**
  (newest position wins while a write is in flight) so dragging cannot queue
  stale parameter writes. A track that cannot follow the current direction
  reads *holding*; unlinked reads *frozen*.
- **CLI**: `tglab morph` (build the graph) and `tglab axes` (measure the
  coupling policy).
- **Registered in sas-app**: `file:` dep in `package.json`, import + builtin
  entry in `src/plugins/index.ts` (`sortOrder: 10`, chat moved to 11), and
  `DEFAULT_BUILTIN_PANEL_ORDER` in `LoopWorkstation.tsx`.
  **`defaultEnabled: false`** — the panel needs a morph artifact built offline,
  so opting in avoids a dead panel on first launch. sas-app typechecks clean.

**SDK gap — CLOSED.** `host.setSynthParameters(trackId, params, pluginIndex?)`
ships in **SDK 2.57.0**. The engine addresses parameters by index while callers
hold names, and indices are not stable across plugin versions, so the host
resolves names once against `listPluginParameters` and then writes each value.
Unknown names **reject the whole call** rather than half-applying — a partially
applied snapshot corresponds to no verified control position, which is worse
than refusing. Implemented in `plugin-host-mixins/instrument.ts` behind
`assertCapability('requiresSurgeXT')` + `assertOwned(trackId)`, mirrored in
sas-app's in-tree types, `PLUGIN_SDK_VERSION` bumped to 2.57.0, and entered in
`plugin-host-coverage-ledger.ts` (reason `plugin-runtime`) so the guard test
passes. The panel still treats the method as optional so an older host degrades
to a message instead of crashing.

An alternative needing zero SDK change: have `tglab morph` capture Surge's
**base64 state** at each control point and drive the panel through
`setPluginState`. Rejected for now — a Surge state is ~50-200 KB, so nine
points × six roles is ~10 MB per graph versus 44 KB today.

### C13h — X/Y pad and cached achievability

**Two-axis pad** (`build_xy_graph` / `params_at_xy`): the two axes are refined
independently and their parameter moves summed, which costs 2 x n_points
searches instead of n_points squared. Summing is a linearity assumption of
exactly the kind that failed in C13b, so **every corner is re-rendered** and
its achieved direction stored as `corner_cosine`. Measured on the six anchors
(softer x tighter, 95 s): median corner cosine **+0.285**, best corners 0.66 to
0.86 (kick ++ 0.86, pad -+ 0.71), but snare and lead go negative in places. So
the pad is real and usable where `corner_cosine` is healthy, and the artifact
says exactly where that is — prefer the single-axis dial elsewhere.

**Cached achievability** (`achievability_cached`): a patch's reachable axes
depend only on the patch, the probe and the policy — never on the other five
anchors. Keyed on (preset_id, role, policy version, axis set) on disk, so
re-picking a previously seen anchor is free instead of ~50 s.

### What to build next

1. Panel UI for the X/Y pad (single-axis dial ships now).
2. Surface `corner_cosine` in the panel so weak corners visibly soften.
3. Kick already has good axes via `tighter`/`punchier`/`boomier`; a `tighter`
   graph measured median endpoint cosine **0.792**, higher than `softer` though
   with fewer live directions — worth offering axis choice in the UI.

Semantic axes are defined in descriptor space (brighter / fuller / snappier /
longer / rougher) — interpretable, role-agnostic to state, role-specific to
realize. That is the honest version of "an unlabeled dial that morphs through
convincing configurations".

## Challenges / amendments to the proposal

**C1 — Rendering is not the bottleneck; drop the render-farm plan.**
The proposal's stated "largest uncertainty" (deterministic high-throughput
Surge rendering) is a non-issue at 100× real-time: even a 1M-render corpus is
an overnight single-machine job with 8 worker processes. The RunPod §9 plan
shrinks to *one* GPU pod for CLAP/model work (see RunPod section).

**C2 — Descriptors primary, CLAP secondary.**
The proposal makes a PCA of (CLAP ⊕ descriptors) the perceptual basis. CLAP
deltas under small ε-perturbations are noise-dominated (the proposal itself
lists this as a top risk), and naive standardization of a 512-dim CLAP block
against ~20 descriptor dims lets CLAP swamp the basis. v0/v1 train on the
**20 explicit descriptors only** (`descriptors.py`: loudness kept as separate
dims, spectral/temporal computed loudness-normalized). Shards store raw
descriptor vectors so a CLAP block can be added later as a *separately
weighted block* without re-rendering. Gate: add CLAP only if the v1
listening tests show gestures the descriptor basis can't represent.

**C3 — Short probes for the dataset; 4-bar clips for ears only.**
Perturbation data uses ~4 s role probes (`probes.py`), not the canonical
four-bar clips — 4× cheaper renders *and* 4× cheaper feature extraction with
no loss for *local* timbre deltas. Canonical probes exist for human A/B
listening and validation renders.

**C4 — Role labels gate anchors, not training coverage.**
Factory reality: `Percussion` holds **9 patches total** (a few kicks, one
snare, zero hats). If role classification gated training data, kick/snare/hat
would starve. Reframe: the model learns `F(x, role-probe) → z`; *any*
QC-passing patch can be probed under any register-compatible role probe
(cross-probing, `tglab gen --role ...`). Role classification only decides
which presets appear as **runtime anchors** in menus. Also: the installed
**2,371 third-party patches** are in-scope for corpus scanning from day one
(keyword rules over author folders) — factory-only was never going to feed
six roles.

**C5 — Known fidelity gap: param-map preset loading.**
Pedalboard `set_state` doesn't work for Surge, so presets load via the
auto-discovered parameter map (synth2surge `preset_loader.py`). Anything not
on the automation surface — **wavetable selection**, mod-routing topology,
FX types/params, filter subtypes, tuning files — silently keeps its default.
Measured day-one: ~46/591 XML params are unmappable, and **osc-param ranges
that depend on osc type do not re-range through automation** (e.g. `Snare
Tight` wants `a_osc1_param5 = 19.5` under osc type 5; the automation surface
clamps it at 1.0 → the patch loads silent). `worker.py` recalibrates osc
params under the loaded types, which fixes the in-range cases; the
out-of-range family (FM-style ratios) is unfixable through this surface and
gets dropped by the QC gate (pilot: snare 4/10, hat 6/10 pass — the pools
are 60/61 deep, so this thins percussion but doesn't starve it).
The training state is still
*self-consistent* (we train on exactly the state we can reproduce through the
same surface at runtime), but a "loaded" patch may not sound like the same
patch loaded natively in Surge. Two mitigations, deferred deliberately:
(a) a `surgepy` spike — the official Surge Python bindings load .fxp natively
and render headlessly on macOS *and* Linux, but they're not on PyPI, so it's
a from-source build (~30-min spike, do before scaling past pilot);
(b) design the runtime plugin to apply **relative deltas on top of the
host's native preset load**, not absolute snapshots, so the fidelity gap
cancels out for small morphs.

**C6 — Runtime = precomputed graph for the dial, tiny local solver for
leader-follower (revised 2026-08-02 after product clarification).** The
product has TWO interaction modes and both ship in the plugin:
(a) the **unlabeled discovery dial** — a path through the timbre graph; all
six patches morph along precomputed, validated trajectories; runtime is pure
interpolation, and (b) **live leader-follower** — touch any synth's param
(in Surge's own editor or programmatically) and the other linked tracks
respond. (b) does *not* need ONNX or Python at runtime: per-anchor response
matrices (~24 params × 20 features per synth) are baked into the graph
artifact, and the follower solve is a damped least-squares over those —
microseconds in plain TypeScript. Unlink = stop applying updates to that
track. Integration prerequisite for (b): the engine must surface plugin
parameter-change events for leader detection (tracktion/JUCE parameter
listeners) — wiring task, noted for Phase 8.

The graph has two data layers, both rendered by this lab:
- **local**: perturbation shards around each anchor (the model's food, and
  the per-anchor response matrices)
- **edges** (`tglab edges`): validated morph routes between same-role
  anchors — kNN proposals in descriptor space, each edge rendered along the
  continuous-param interpolation and gated on QC + endpoint accuracy +
  path detour. Audio decides structural compatibility (an edge between
  patches with incompatible osc/filter structure misses its endpoint and is
  rejected); no fingerprint bookkeeping. This is what lets the dial *travel*
  between configurations instead of only wiggling around one.

**C7 — No autograd-through-ONNX at runtime.**
Where the solver does need Jacobians (lab test bench, graph precompute),
compute them by **batched finite differences on the proxy** — a ~50-wide
batch of MLP forwards is microseconds — or read them off the delta head.
If the panel ever needs live inference, hand-roll the MLP forward pass in TS
from the exported weights (it's 3 residual blocks) rather than shipping
onnxruntime native deps into Electron.

**C8 — Compressed schedule: pilot-train on day one.**
The proposal's phases 0–3 (weeks) assumed rendering was expensive and corpus
curation had to precede everything. With C1 true and keyword-rules corpus
bootstrapping, the right day-one move is: pilot corpus → FD screen → shards →
**train v0 tonight**. The proposal's *actual* warning — don't build a huge
corpus before the no-ML gesture proof — stands: the FD screens generated for
every shard ARE the no-ML proof data (local Jacobians per anchor), so Phase 3
falls out of the same run. Scale-up waits for the ear test.

**C9 — Loudness is excluded structurally, not just statistically.**
All `*volume*`/`*level*`-adjacent global params are deny-listed in
`policy.py`; descriptor loudness dims are separate; QC rejects silent/clipped
renders. "Followers just turn up the volume" is made impossible three ways.

**C10 — Register identity is protected.**
`*octave*`/`*pitch*` (coarse) params are deny-listed — a morph must never
transpose a follower out of its role register (MIDI is sacred; so is the
register the MIDI implies). Fine detune/dispersion stays allowed.

**C11 — "Learn the param matrix alone" (considered, rejected 2026-08-02).**
Proposal from product review: train a network on the joint parameter vectors
of role-correct preset combinations — no rendering — so weights encode which
follower params should move when leader params move. Rejected on two
grounds, recorded here because the reasoning shapes the design:
(1) *No label variance.* Every combination of individually-valid presets is
labeled GOOD, and independently-drawn positives contain zero cross-layer
statistical dependency — a discriminator trained on such data can only learn
per-layer validity, which the curated preset list already provides for free.
There is no signal in the combinations for any loss to extract.
(2) *No cross-synth alignment without audio.* Six patch formats are six
private languages; the param matrix gives each language's grammar but no
dictionary between them. The shared descriptor space (rendered audio) is the
interlingua that makes "move in the same direction" well-defined across
different synth architectures — and within one synth it is what
distinguishes audibly smooth directions from dead params and cliff edges.
What survives from the idea: per-role parameter-manifold structure IS real
and learnable without audio; we use a nonparametric version (anchors + trust
radius), and a per-role preset-VAE prior remains a candidate V2 regularizer.

## What stays exactly as proposed

- Forward/delta self-supervised objective, role conditioning, FiLM MLP (§3, §6.4)
- Locality: trust radius, damped minimum-norm solving, locks (§3.2, §6.5)
- MIDI is sacred; presets are trusted anchors; no taste labels (§1.1)
- Split by preset, never by perturbation (§Phase 1)
- Kill/pivot criteria (§8.3) — all adopted verbatim
- Deliberate out-of-scope list (§1.2)

## Pipeline (implemented in `training/`)

```
tglab inventory   # scan factory + 3rd-party, keyword role rules -> corpus manifest
tglab policy      # introspect live host -> continuous live-safe allow-list
tglab pilot       # baseline render + QC gate per role
tglab gen         # FD screen -> seeded gesture plans -> .npz shards (resumable)
                  #   --cross-probe (percussion trio) --render-avg N --per-role 0=all
tglab edges       # kNN morph routes between same-role anchors, render-validated
tglab train       # forward/delta proxy -> ONNX + manifest bundle
tglab parity      # macOS<->Linux render parity gate (guards cloud runs)
tglab bench       # render throughput on this machine
```

Dataset unit, versioning (probe/policy/corpus versions in every shard),
QC gates, and the model bundle contract follow the proposal's §6.1 and
Appendix C.

## Tonight's runbook (EOD 2026-08-02)

```bash
cd sas-timbre-graph/training && source .venv/bin/activate
tglab inventory                      # expect: bass/pad/lead rich; kick/snare/hat thin but nonzero
tglab policy
tglab pilot --per-role 10            # QC pass-rate report; eyeball reports/pilot.json
tglab gen --per-role 10 --workers 1  # ~60 anchors x ~300 renders — fine in one process
tglab train --epochs 60              # minutes on M4 (MPS); exports workspace/models/tg-v0
```

Success gate for tonight (from proposal §8.1, adjusted): val delta-cosine
median **> ridge baseline** and > 0.6 absolute on the descriptor basis.
0.80 is the v1 target after corpus scale-up, not the v0 bar. If v0 can't
beat ridge, the perceptual basis needs work before any scale-up (kill
criterion #1).

Scale-up (tomorrow+): `--per-role 40 --workers 8`, add cross-probing for
kick/snare/hat coverage, then the two-synth no-ML gesture proof from the FD
caches, then the six-synth bench.

## RunPod — click-by-click

Full runbook: **[RUNPOD.md](RUNPOD.md)**. Summary below.

## RunPod — what to actually set up

**Training is not the reason to rent anything** — the model trains in ~11 s
on MPS. **Rendering the corpus is**, and after the C0 fix it costs ~90 s of
one core per anchor. So the correct rental is **cores, not VRAM**:

- **CPU pod, Compute Optimized (`cpu5c`), 32 vCPU / 64 GB, on-demand**
- Plain Ubuntu template, 40 GB container disk, no network volume needed
- Full-corpus run ≈ 48 min ≈ **under $1**
- **Skip**: GPU pods (the GPU idles), A100/H100, Serverless, Spot
  (preemption wastes the batch), network volumes for single sessions

A GPU pod only becomes correct if C2's gate opens and CLAP embeddings enter
the feature basis — then an RTX 4090 for the embedding pass is worthwhile.

The proposal's Linux-parity gate (§9.5) stands: never render corpus data on
Linux until the same preset + probe produces matching descriptors vs macOS;
until then RunPod does features/training only, on Mac-rendered features.

## C14 — The anchor was the ceiling, not the morph (2026-08-03)

Live play exposed a failure the whole pipeline had been reporting as success:
the dial moved, every parameter write landed, the artifact validated — and the
lead never changed audibly. Integration tests (`tests/test_shipped_graph_integration.py`)
were written to assert the thing that actually matters: for every role, every
parameter the artifact marks audible must measurably change a real render.
They failed immediately, and the two causes were both upstream of the morph.

### C14a — `‖J‖` overstates audibility, and the solver chased it

Shipped `sensitivity` came from the probe Jacobian's row norm. A parameter can
register a finite-difference response at the probe's `EPS = 0.04` yet leave the
render unchanged at the step sizes the panel uses. Worse, the damped
minimum-norm solve *prefers* those parameters: reaching a target through a
weakly-coupled control needs a large delta, so the least audible parameters
attract the biggest moves. The live lead spent its whole gesture on `a_width`
(measured audibility 0.46) while `a_filter_1_cutoff` (10.95) barely moved.

Fix, in two places so it cannot drift back:
- each anchor's `usable` mask is narrowed to parameters whose render
  demonstrably changes, so the solver *cannot* select an inert control;
- `sensitivity` ships that measured magnitude (`meta["measured_sensitivity"]`,
  preferred over `‖J‖` by `_sensitivity()` in `morph.py`).

### C14b — Anchor choice, measured

The first graph took corpus index 0 per role. That handed the hat
`Leads/Chatter.fxp`, which has **zero** measurably audible parameters — no
amount of gesture scaling can morph a patch with no reachable timbre range.
Anchors are now screened (`pick_anchors.py`) by measured audible-parameter
count, which raised the kick from 3 to 15 and the hat from 0 to 13:

| role | anchor | audible / probed |
|---|---|---|
| kick | `Percussion/Kick Tech 2.fxp` | 15 / 34 |
| snare | `Keys/Experiment.fxp` | 14 / 28 |
| hat | `Plucks/That Comb Magic.fxp` | 13 / 28 |
| bass | `Basses/Bass 1.fxp` | 17 / 34 |
| pad | `Chords/Tek Stab.fxp` | 16 / 30 |
| lead | `Brass/OB-8 Jump.fxp` | 17 / 34 |

Graph quality rose to **6/6 roles moving, 10/12 directions working**
(median endpoint cosine 0.625, monotonicity 0.875).

### C14c — A patch that wanders cannot demonstrate a morph

`Chords/Inharmonic Stab.fxp` screened well but collapsed to 1 audible
parameter on re-measurement: its own noise floor is σ ≈ 0.43 against bass's
0.001, so the audibility threshold rises to ~1.95 and only a monster parameter
clears it. `freeze_stochastic()` cannot help — the patch uses a genuine noise
oscillator. Two consequences:

- screening now rejects anchors above `MAX_SIGMA = 0.05`, because a patch whose
  own renders differ run-to-run will also mask the dial in the listener's ear;
- σ estimated from a handful of renders is *itself* unstable (the same patch
  measured 0.43 and then under the gate minutes later), so measurement and
  verification must average identically or they compare different quantities.
  The integration test previously took `z1` from a single render while
  sensitivity came from an averaged one; both now use `render_descriptors(k=2)`.

### Two roles are still musically mismatched

`snare` is a Keys patch and `hat` is a Plucks patch. The role-appropriate
Percussion candidates (`Snare Tight`, `Snare 2`, `Closed Hat`) all score −1 —
they fail to load or render silent, the same ~10% preset-load gap tracked
against a possible `surgepy` spike. Morphability was chosen over category
fidelity so every role can actually move; closing the load gap is what would
let a real hi-hat back in.

## C15 — The anchor was never the point: ship a TOUR (2026-08-03)

C14 made the dial audible. Live play still judged it boring, and the complaint
named the cause exactly: "from 0-100 it sounds like someone is still using the
same patch the whole time but they maybe turned 1 or 2 parameters."

That is precisely what the artifact contained. Measured on the shipped
`morph-softer.json`: the kick moved 8 of 34 parameters across the ENTIRE dial,
the largest by 0.0185 of its range. The morph solved a damped minimum-norm
least-squares step — by construction *the smallest* parameter change that
travels along one perceptual axis, chosen so "the patch keeps its identity".
The objective function was the opposite of what the product wanted.

### C15a — Continuous-only interpolation cannot arrive at another preset

`edges.py` (written after the 2026-08-02 "travel between configurations"
requirement, never run) proposed the fix: validate morph routes BETWEEN
same-role anchors and gate on the endpoint landing near B. The first real run
rejected **11 of 12 bass edges**, median endpoint error 0.94 against a 0.35
gate.

Direct measurement of why:

| pair | ‖z_B − z_A‖ | achieved travel | endpoint err |
|---|---|---|---|
| 0-1 | 1.60 | 7.34 | 5.33 |
| 0-3 | 4.06 | 2.29 | 0.93 |
| 3-4 | 4.19 | 2.76 | 0.79 |

The runtime writes only continuous allow-list parameters; oscillator types,
filter types and FM routing stay at the loaded patch's values. So B's cutoff
and envelope settings land on A's oscillators and mean something else
entirely — usually *overshooting* B rather than approaching it.

The conclusion is not that the morph is broken. **Travel per hop is 1.9–11.5
normalized units** where the whole old dial moved 0.08–0.14. Arriving at B was
a means the runtime cannot have; travelling somewhere new, cleanly, is the
product. The gate is now QC + measured travel, not endpoint arrival, and an
"anchor" is honestly a *new configuration built from a real patch's values*,
not a reproduction of that patch.

### C15b — Measure through the lens the runtime actually has

Because only anchor 0's preset is ever loaded, every later position is heard
through ITS structure. Pairwise validation under each edge's own A-anchor
therefore validated sounds the user never hears, and validity does not compose
across hops. `tour.py` loads the lens once and renders every candidate, every
interpolation point and every gate through it, so validation and playback are
the same signal. The composed tour is then swept end to end as a final gate.

### C15c — The writable parameter set is not fixed

`shared_basis` exists because Surge names oscillator parameters by oscillator
type. Measured: presets of one role exposed 90, 93 and 97 of the 97-parameter
allow-list, and **the same preset offered 97 on one load and 87 on the next**,
depending on load history. A basis taken from any single patch would ship
parameters the runtime silently drops. The tour ships the intersection over
that role's screened candidates — every anchor then has a genuinely measured
value for every parameter it carries — and all writes skip names the live host
does not expose, matching `setSynthParameters`' lenient relative mode in the
app.

### What the panel does now

Delete, not adjust: depth control, sensitivity weighting, dial-fraction
scaling and the ±0.6 per-parameter clamp are all gone. The dial spans 0..1 and
sends `paramsAt(c) − paramsAt(0)` **raw**, relative to the live sound. Any
scaling would stop the dial reaching the configurations the artifact promises;
the values are already in [0,1] and the host clamps to each parameter's range.

Artifact: `tour-graph-v1` (`assets/tour.json`), per-role `param_names`,
`control_points` (0..1, one per anchor), `anchors[]` and absolute `snapshots`.
Build with `tglab tour`; see `training/README.md` for the runbook.

## Licensing note

Corpus manifests retain source + category; 3rd-party patches ship with Surge
under mixed licenses. Fine for local training. Before *distributing* trained
weights, audit the license metadata (proposal §10 risk table) — the corpus
manifest already records what went in.
