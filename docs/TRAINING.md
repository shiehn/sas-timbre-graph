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

**C6 — Runtime = precomputed morph graph, not live solving (the plugin's
namesake).** The end-user surface is an X/Y pad morphing all six synths, with
per-track unlink. That does **not** need real-time Jacobian solving in
Electron: the moment six anchors are chosen, solve the whole morph field
*offline* (seconds — grid of latent points → per-synth param snapshots along
smooth trajectories), ship it to the panel as a **timbre graph artifact**, and
runtime is pure interpolation: zero-latency, guaranteed smooth, trivially
unlinkable (stop applying updates to that track), no ONNX in the plugin.
Live leader-follower coupling (the proposal's §3.2) stays in the *lab test
bench* where a Python process can afford it. Revisit real-time solving only
if graph precompute feels too laggy after a manual patch tweak (it re-solves
in background seconds).

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
tglab train       # forward/delta proxy -> ONNX + manifest bundle
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

## RunPod — what to actually set up

**Answer to "can the M4 train it in 30 min": yes.** v0 (pilot corpus,
descriptor basis, FiLM-MLP) trains in **minutes** on MPS; even a
full-corpus v1 is an under-an-hour M4 job. Rendering (the feared cost) is
~free per C1. So: **no RunPod needed for the pilot, and none tonight.**

Where RunPod earns its keep later:
1. **CLAP block extraction** over a full corpus (if C2's gate opens)
2. **Model/hyper-parameter sweeps** (many v1 configs × seeds)
3. **Linux container parity** for eventual reproducible cloud runs

Recommended setup (one-time, ~20 min):
- **Account + $25 credit**, SSH key, API key (store as `RUNPOD_API_KEY`)
- **1× RTX 4090 24 GB, Secure Cloud, on-demand** (~$0.69/hr as of 2026-07-27
  pricing) — the largest job in this plan fits comfortably in 24 GB
- **Network Volume 100 GB** (~$7/mo) in the same region as 4090 stock —
  shards/checkpoints/manifests live here; audio never leaves the Mac
  (features only), so 100 GB is generous
- Template: current `runpod/pytorch` 2.x/py3.11/cu12.x image, then
  `pip install "git+https://github.com/shiehn/sas-timbre-graph#subdirectory=training"`
- **Skip**: A100/L40S/H100 (nothing here needs >24 GB), Serverless (stateful
  batch jobs), CPU render fleet (C1), Spot for training (use on-demand; Spot
  is fine for shard-sharded feature extraction which is resumable)

The proposal's Linux-parity gate (§9.5) stands: never render corpus data on
Linux until the same preset + probe produces matching descriptors vs macOS;
until then RunPod does features/training only, on Mac-rendered features.

## Licensing note

Corpus manifests retain source + category; 3rd-party patches ship with Surge
under mixed licenses. Fine for local training. Before *distributing* trained
weights, audit the license metadata (proposal §10 risk table) — the corpus
manifest already records what went in.
