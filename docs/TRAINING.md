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

## Licensing note

Corpus manifests retain source + category; 3rd-party patches ship with Surge
under mixed licenses. Fine for local training. Before *distributing* trained
weights, audit the license metadata (proposal §10 risk table) — the corpus
manifest already records what went in.
