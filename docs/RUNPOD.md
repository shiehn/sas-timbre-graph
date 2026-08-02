# RunPod runbook — Timbre Graph corpus run

**This job is CPU-bound, not GPU-bound.** Corpus generation is Surge XT
plugin hosting (one core per worker); training the model takes ~11 seconds.
Rent cores, not VRAM. A GPU pod is wasted money here.

## 0. Before you rent — measured facts

| Fact | Value | Why it matters |
|---|---|---|
| Render cost | ~20 ms per 4 s probe | rendering is ~100× real-time |
| Renders per anchor | ~700 (noise floor + FD screen + 164 gestures × avg 3) | the averaging is what makes the data usable |
| Wall time per anchor | **~90 s single core** | measured on M4 after the reproducibility fix |
| Surge XT version | **1.3.4** — must match the Mac | preset sha1s must match or shards aren't comparable |
| Content shipped by the Linux .deb | factory + 3rd-party patches | no need to upload any patch content |

Anchor budget → core-hours:

| Run | Anchors | Core-hours | 32 vCPU wall time | Cost @ ~$1/hr |
|---|---|---|---|---|
| Scale-up (per-role 40) | ~240 | ~6 | **~13 min** | ~$0.25 |
| Full corpus (per-role 150) | ~900 | ~23 | **~48 min** | ~$0.80 |
| Everything usable | ~2,000 | ~50 | ~1 h 45 m | ~$1.75 |

Even the biggest run is pocket change; the pod's *idle* time while you read
logs will cost more than the compute. Budget $5 and you cannot overspend.

## 1. Create the pod

1. runpod.io → sign in → **Billing** → load $10 (minimum useful balance).
2. Left nav → **Pods** → **Deploy**.
3. At the top of the deploy page switch the toggle from **GPU** to **CPU**.
   (If you cannot find CPU pods, fall back to the cheapest GPU pod with a
   high vCPU count — see §6.)
4. Instance type: **Compute Optimized** (`cpu5c`, or `cpu3c` if 5 is out of
   stock). Pick **32 vCPU / 64 GB**. 16 vCPU is fine too, just 2× slower.
5. Template: **RunPod Ubuntu 22.04** (any plain Ubuntu image; we install
   everything ourselves). If only PyTorch templates are offered, take one —
   it is Ubuntu underneath and works fine.
6. Disk: **Container disk 40 GB**. You do **not** need a network volume for
   a single session — results are ~200 MB and we copy them back at the end.
   (Add a 50 GB network volume only if you plan several sessions.)
7. Deploy On-Demand. **Do not use Spot** — a preemption mid-run wastes the
   whole batch. On-demand for under an hour costs cents.
8. Wait for the pod to show **Running**, then **Connect → Start Web Terminal**
   (or SSH if you added your key).

## 2. Bootstrap (one line, ~4 minutes)

Paste into the pod terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/shiehn/sas-timbre-graph/main/training/cloud/bootstrap.sh | bash
```

It installs Surge XT 1.3.4 + system libs, clones this repo, builds the venv,
and finishes with a smoke test that loads Surge and renders one probe. If the
smoke test prints a render time and `qc=True`, the pod is good.

Then:

```bash
cd /workspace/sas-timbre-graph/training && source .venv/bin/activate
nproc          # confirm your core count
```

## 3. Upload the macOS parity baseline

The gate that decides whether cloud-rendered data is trustworthy. From
**your Mac**:

```bash
cd ~/sas-platform/sas-timbre-graph/training
runpodctl send workspace/reports/parity-macos.json
# prints a one-time code; run the matching `runpodctl receive <code>` on the pod
```

No `runpodctl`? Simplest alternative — the file is small:

```bash
# on the Mac
gh gist create workspace/reports/parity-macos.json          # prints a URL
# on the pod
curl -fsSL <raw-gist-url> -o /workspace/parity-macos.json
```

## 4. Run the gate — **do not skip this**

```bash
tglab inventory
tglab policy
tglab parity --write /tmp/parity-linux.json --compare /workspace/parity-macos.json
```

Expected: `content_identical=True`, `delta_cos≈1.0`, **PARITY PASS**.

- **PASS** → the pod renders the same perceptual answers as your Mac. Proceed.
- **FAIL** → stop. Linux Surge is behaving differently; the correct fallback
  is to render on the Mac and use the pod only for training. Save
  `/tmp/parity-linux.verdict.json` and send it to me — the per-row breakdown
  says which presets and which descriptors diverged.

The command exits non-zero on failure, so you can chain it:
`tglab parity ... && tglab gen ...`

## 5. Generate + train

**The quality-max run (recommended — quality was named the goal):** every
usable anchor, percussion cross-probing, deeper local sampling, extra
averaging where noise oscillators live, then the morph-edge graph:

```bash
tglab gen --per-role 0 --cross-probe \
  --singles 120 --multis 200 --drift 8 \
  --workers $(( $(nproc) - 2 )) 2>&1 | tee /workspace/gen.log

tglab edges --k 4 2>&1 | tee /workspace/edges.log
tglab train --epochs 80 2>&1 | tee /workspace/train.log
```

~2,500 anchor-jobs (2,145 anchors + ~350 percussion cross-probes) at
~4 min each ≈ 165 core-hours ≈ **~5.5 h on 32 vCPU ≈ $6**, plus minutes for
edges and seconds for training. Start it, check the first dozen log lines
look healthy, walk away.

Smaller variants if you want a fast read first:

```bash
# scale-up (~40 min on 32 vCPU with the deeper plans)
tglab gen --per-role 40 --cross-probe --workers $(( $(nproc) - 2 ))
```

Leave 2 cores free — each worker is a full Surge host and the box gets
unresponsive at 100% saturation.

`gen` is **resumable**: existing shards are skipped, so if anything dies just
re-run the same command.

Watch for in the log: `status: ok` with a `median_snr` well above 3.
A rash of `too-noisy` / `insensitive` means something is wrong with the
Linux render path — stop and check parity again.

## 6. Get the results back

```bash
cd /workspace/sas-timbre-graph/training
tar czf /workspace/results.tar.gz workspace/shards workspace/models workspace/reports
du -h /workspace/results.tar.gz

# then either:
runpodctl send /workspace/results.tar.gz     # and `runpodctl receive` on the Mac
# or use the RunPod web UI file browser to download it
```

## 7. **Terminate the pod**

RunPod bills while the pod exists, including when idle and when merely
stopped (storage). Pods → your pod → **Terminate**. Verify it disappears
from the list. This is the only step that can actually cost you money if
forgotten.

---

## Fallback: no CPU pods available

If CPU pods are out of stock in every region, take the cheapest GPU pod with
a high vCPU count (an **RTX A4000/A5000** pod typically bundles 9–16 vCPU at
~$0.30–0.45/hr). The GPU sits idle — you are buying its cores. Everything
above works unchanged; just set `--workers` from `nproc`.

## Why not just run it on the Mac?

You can — with `--workers 8` the scale-up run is roughly 25–30 minutes, but
it pins every performance core and the machine becomes unpleasant to use.
The full-corpus run is ~3 hours of that. The pod exists to buy your laptop
back for under a dollar.
