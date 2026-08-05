"""The tour: a dial that TRAVELS between configurations instead of orbiting one.

The shipped morph graph kept each role inside one anchor's neighbourhood — a
minimum-norm walk along a perceptual axis, deliberately the smallest audible
change. Live play judged it boring for exactly that reason: from 0 to 100 the
patch never stops being itself. What the dial is FOR (user requirement,
2026-08-02, restated 2026-08-03) is a journey through many good-sounding
configurations: arrive at one, keep turning, morph into the next.

## What "arriving at a preset" can and cannot mean (measured 2026-08-03)

The runtime loads ONE preset per role and thereafter writes only continuous
allow-list parameters, as relative deltas. Discrete/structural parameters —
oscillator and filter types, FM routing, unison voice counts — are never
written, because there is no meaningful half-way between saw and FM and
because they were the source of clicks and garbage.

That has a consequence the first edge run made unmissable: taking preset B's
continuous values onto preset A's structure does NOT reproduce B. Measured
over bass pairs, the endpoint landed 0.79–5.3 normalized distances away from
B — usually *overshooting*, because B's cutoff and envelope settings mean
something entirely different on A's oscillators. The original edge gate
(`endpoint_err <= 0.35`, "the morph must arrive at B") rejected 11 of 12
edges for failing a goal this runtime cannot have.

What the same measurement showed is that the *travel* is enormous: 1.9–11.5
normalized units per hop, against a shipped axis morph whose entire sweep
moved a handful of parameters by ~2% of their range. So the tour's anchors
are not "preset B's sound"; they are **new configurations built from a real
patch's values**, heard through the start anchor's structure — which is
precisely "patch randomization except the patches morph into configurations
known to be good".

Everything is therefore measured through the LENS the runtime actually has:
the start anchor is loaded once, and every candidate, every interpolation
point and every gate is rendered through it. Validation and playback see the
same sound.

## The pipeline, per role

1. screen: every corpus candidate (capped, loudly) must load, pass render QC
   and sit under the stochastic noise ceiling (C14c). ~9 renders each.
2. lens:   choose the start anchor — role-appropriate category, steadiest
   renders. It defines the structure for the whole tour.
3. effect: render every survivor's parameter vector THROUGH the lens. Drop
   the ones that turn to garbage there. This is the sound the dial will make.
4. spread: keep the M most perceptually spread survivors (farthest-point
   sampling in effective descriptor space).
5. edges:  kNN proposals, each validated by rendering the interior of the
   interpolation — QC everywhere, no wild detour, and the two ends must be
   audibly apart. Weight is the MEASURED travel.
6. tour:   a deterministic maximum-travel simple path of ~N anchors over the
   valid edges. Shortfalls are recorded, never papered over.
7. sweep:  the composed tour re-rendered end to end, gating that every
   position is clean and every hop actually moves. Anchors that fail are
   dropped and the tour re-spliced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from timbre_graph_lab.config import (
    LabConfig,
    POLICY_VERSION,
    PROBE_VERSION,
    ROLES,
    SEED,
)
from timbre_graph_lab.corpus import load_manifest
from timbre_graph_lab.descriptors import standardize
from timbre_graph_lab.edges import MAX_DETOUR, edge_metrics, knn_edges
from timbre_graph_lab.morph import portable_fxp_path
from timbre_graph_lab.probes import get_probe
from timbre_graph_lab.worker import (
    BUILD_MARGIN,
    RenderWorker,
    qc_audio,
    qc_loudness,
)

TOUR_VERSION = "tour-graph-v1"

DEFAULT_ANCHORS = 20
DEFAULT_CAP = 400  # candidates screened per role; the whole pool if smaller
DEFAULT_KEEP = 40  # screened survivors kept for edge validation
N_INTERIOR = 3  # interpolation points rendered between two anchors

# An anchor whose own renders wander this much cannot demonstrate a morph
# (C14c): the dial's work is buried under the patch's own drift.
#
# PER ROLE, because percussion is noise. A hi-hat, snare or clap IS a noise
# burst, `freeze_stochastic()` cannot make a genuine noise oscillator
# repeatable, and run-to-run spectral variation in a noise burst is inaudible
# as instability — it is the sound. A single melodic ceiling threw away the
# real drums and left the roles wearing melodic patches: measured 2026-08-03 it
# rejected `Hi-Hat` (0.18), `Hat Closed` (0.098), `Hat Open` (0.127),
# `Clap` (0.48) and `Kick 909ish` (0.46), so the hat's lens ended up a bass
# patch and the snare's a marimba. Percussion is screened for LOUDNESS and
# LOADABILITY; its measurement noise is handled by averaging more renders
# (PERC_RENDER_AVG) rather than by discarding the patch.
MAX_SIGMA = 0.05
PERCUSSIVE_ROLES = ("kick", "snare", "hat")
MAX_SIGMA_PERCUSSIVE = 0.75
# sigma/sqrt(k): more averaging buys a steady measurement of an unsteady patch
PERC_RENDER_AVG = 4
# below this many self-declared drums, a restricted pool is worse than a mixed one
MIN_PERCUSSIVE_POOL = 6


def sigma_ceiling(role: str) -> float:
    return MAX_SIGMA_PERCUSSIVE if role in PERCUSSIVE_ROLES else MAX_SIGMA


def render_avg(role: str) -> int:
    return PERC_RENDER_AVG if role in PERCUSSIVE_ROLES else 2


# Two positions this close in standardized descriptor space are the same
# sound to a listener. Deliberately stricter than the shipped integration
# test's floor (max(3*sigma, 0.1)) so the artifact passes its own gate with
# headroom rather than sitting on it.
MIN_HOP_TRAVEL = 0.3

# How many start anchors to try before settling for the best of them, and what
# counts as "this lens works": a lens through which most of the pool collapses
# cannot yield a twenty-stop tour however good the path search is.
MAX_LENS_TRIES = 5
LENS_SURVIVAL = 0.5
LENS_PROBE = 40  # candidates rendered to judge a lens before committing

# Prefer anchors whose own category matches the role: a "Keys" patch used as
# a snare is what produced an unmorphable anchor in the first place (C14b).
PREFERRED_CATEGORIES = {
    "kick": ("Percussion",),
    "snare": ("Percussion",),
    "hat": ("Percussion",),
    "bass": ("Basses",),
    "pad": ("Pads", "Polysynths", "Chords"),
    "lead": ("Leads", "Plucks", "Keys", "Brass"),
}

# ...but the NAME is the stronger signal, and for drums it is the only one.
# Surge's `Percussion` category holds four patches in the whole 2062-patch
# library and no hi-hats at all; the actual drums live in third-party packs
# under the pack's own name — `Closed Hat` is category `Altenberg`, `Acoustic
# Snare` is `Cybersoda`. Ranking on category alone therefore handed the hat a
# bass patch and the snare a marimba. When a patch says what it is, believe it.
ROLE_NAME_PATTERNS = {
    "kick": re.compile(r"\b(kick|bass ?drum|bd|808|909|707|606)\b", re.I),
    "snare": re.compile(r"\b(snare|clap|rim ?shot|rim)\b", re.I),
    "hat": re.compile(r"\b(hi-?hat|hat|cymbal|ride|crash|shaker|tambourine)\b", re.I),
    "bass": re.compile(r"\b(bass|sub)\b", re.I),
    "pad": re.compile(r"\b(pad|string|choir|atmos)\b", re.I),
    "lead": re.compile(r"\b(lead|pluck|key|brass|bell|arp)\b", re.I),
}


def names_the_role(name: str, role: str) -> bool:
    """Does this patch's own name say it is this role?"""
    pat = ROLE_NAME_PATTERNS.get(role)
    return bool(pat and pat.search(name))


# Folders that mean "this is percussion" whatever the filename says.
DRUM_FOLDERS = ("Drums", "Percussion")


def is_percussive(c: "Candidate", role: str) -> bool:
    """Percussion by name OR by the folder it ships in.

    Name alone was too narrow: it kept `Closed Hat` but dropped `Perc 7` and
    `Tom L`, which are perfectly good neighbours on a drum tour. Folder alone
    is too broad to pick a LENS, which is why rank_lenses still puts the name
    first.
    """
    return names_the_role(c.name, role) or c.subcategory in DRUM_FOLDERS


def delta_to(
    x_target: np.ndarray, base: dict[str, float], param_names: list[str]
) -> dict[str, float]:
    """The raw-space move from the loaded preset's baseline to `x_target`.

    Names absent from the live host are SKIPPED, not defaulted. Surge exposes
    oscillator parameters by oscillator type, and the exposed set turns out to
    depend on load history — the same preset offered 97 allow-list parameters
    on one load and 87 on the next (measured 2026-08-03). The app's host does
    exactly this in relative mode (`skipUnknown`), so skipping keeps the lab
    and the runtime writing the same subset instead of diverging.
    """
    return {
        n: float(x_target[j] - base[n])
        for j, n in enumerate(param_names)
        if n in base
    }


@dataclass
class Candidate:
    """One screened, loadable, QC-passing preset."""

    preset_id: str
    name: str
    category: str
    subcategory: str  # what the patch IS: Drums, Basses, Leads...
    path: str  # absolute, authoring machine
    sigma: float
    raw: dict[str, float]  # its own allow-list values at baseline
    z0: np.ndarray  # raw descriptors of the preset ON ITS OWN
    # this candidate's values expressed on the LENS's parameter basis —
    # filled in by project_to_lens_basis, because which parameters exist at
    # all depends on the loaded oscillator types (see that function).
    x: np.ndarray | None = None
    # descriptors of that vector heard through the lens anchor — the sound
    # the dial will actually make. Filled in by measure_effective.
    z_eff: np.ndarray | None = None
    # measured RMS of that same render, and the gain trim (in raw parameter
    # units on GAIN_PARAM) applied to bring it to the lens's level.
    rms_eff: float = 0.0
    trim: float = 0.0


def shared_basis(cands: list[Candidate], allowed: list[str]) -> list[str]:
    """The parameters EVERY candidate of this role actually has, in allow-list
    order; each candidate's values are projected onto it.

    Surge's parameter list is not preset-independent — oscillator parameters
    are named by oscillator type, so `a_osc_1_shape` exists on one patch and
    not the next, and the exposed set even varies with load history. Shipping
    a basis one patch happens to expose would ship parameters the runtime
    silently drops. The intersection is the set that travels: every anchor
    has a genuine measured value for it, so no position on the dial is built
    from a substituted number.
    """
    if not cands:
        return []
    common = set(cands[0].raw)
    for c in cands[1:]:
        common &= set(c.raw)
    param_names = [n for n in allowed if n in common]
    for c in cands:
        c.x = np.array([c.raw[n] for n in param_names], dtype=np.float64)
    return param_names


# ---------------------------------------------------------------------------
# 1. screen
# ---------------------------------------------------------------------------


def screen_role_pool(
    cfg: LabConfig,
    worker: RenderWorker,
    role: str,
    allowed: list[str],
    cap: int = DEFAULT_CAP,
) -> tuple[list[Candidate], list[dict]]:
    """Screen every (capped) corpus candidate for a role.

    Returns (survivors, report_rows). The report is also written
    incrementally to reports/anchor-screen-<role>.json — screening is the slow
    part, and losing it to a later failure means paying for it twice.
    """
    manifest = load_manifest(cfg)
    pool = [e for e in manifest["entries"] if role in e["roles"]]
    preferred = [e for e in pool if e["category"] in PREFERRED_CATEGORIES[role]]
    ordered = preferred + [e for e in pool if e not in preferred]
    if len(ordered) > cap:
        print(f"{role}: screening {cap}/{len(ordered)} candidates (capped)")
        ordered = ordered[:cap]

    probe = get_probe(role, "short")
    avg = render_avg(role)
    report_path = cfg.reports_dir / f"anchor-screen-{role}.json"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    survivors: list[Candidate] = []
    rows: list[dict] = []

    def note(entry: dict, status: str, sigma: float | None = None) -> None:
        rows.append(
            {
                "preset_id": entry["preset_id"],
                "name": entry["name"],
                "category": entry["category"],
                "status": status,
                **({} if sigma is None else {"sigma": round(sigma, 4)}),
            }
        )
        report_path.write_text(json.dumps(rows, indent=1))

    for entry in ordered:
        if not worker.load_preset(entry["path"]):
            note(entry, "load-failed")
            continue
        qc = qc_audio(worker.render(probe))
        if not qc.ok:
            note(entry, f"qc-{qc.reason}")
            continue
        # avg=2 matches how every descriptor here is measured (k=2), so sigma
        # and the distances it gates are the same quantity (C14c).
        sigma = float(
            np.linalg.norm(standardize(worker.noise_floor(probe, k=3, avg=avg)))
        )
        if sigma > sigma_ceiling(role):
            note(entry, "sigma-reject", sigma)
            continue
        base = worker.baseline_raw
        z0 = worker.render_descriptors(probe, k=avg)
        survivors.append(
            Candidate(
                preset_id=entry["preset_id"],
                name=entry["name"],
                category=entry["category"],
                subcategory=entry.get("subcategory", ""),
                path=entry["path"],
                sigma=sigma,
                raw={n: base[n] for n in allowed if n in base},
                z0=np.asarray(z0, dtype=np.float64),
            )
        )
        note(entry, "ok", sigma)

    return survivors, rows


# ---------------------------------------------------------------------------
# 2. lens (pure) + 3. effective sound through it
# ---------------------------------------------------------------------------


def rank_lenses(cands: list[Candidate], role: str) -> list[int]:
    """Candidate start anchors, best first.

    A RANKING rather than a single pick, because how good a lens is cannot be
    known until other patches are heard through it. Measured 2026-08-03: the
    lead's best-ranked lens (`Simple Waveguide`) passed every screen on its
    own and then let **1 of 358** candidates render — a waveguide's parameters
    mean something so different that other patches' values collapse to
    silence on it. The caller tries these in order and keeps the first that
    actually admits a tour.

    Role-appropriate category first (the tour inherits the lens's oscillators
    and routing, so a Keys patch as the snare lens colours all twenty
    anchors), then steadiest renders. Deterministic; ties break on preset_id.
    """
    if not cands:
        raise ValueError(f"{role}: no candidates to choose a lens from")
    preferred = PREFERRED_CATEGORIES[role]
    return sorted(
        range(len(cands)),
        key=lambda i: (
            # what the patch CALLS itself outranks the folder it sits in
            not names_the_role(cands[i].name, role),
            cands[i].subcategory not in DRUM_FOLDERS
            if role in PERCUSSIVE_ROLES else False,
            cands[i].category not in preferred,
            cands[i].sigma,
            cands[i].preset_id,
        ),
    )


def choose_lens(cands: list[Candidate], role: str) -> int:
    """The single best-ranked lens (see rank_lenses)."""
    return rank_lenses(cands, role)[0]


def _mean_rms(worker: RenderWorker, probe, k: int) -> tuple[float, bool]:
    """RMS averaged over k renders, and whether every one passed the gate.

    A single render is not a level for percussion: a hi-hat or snare IS noise,
    which is why their sigma ceiling is 0.75. Normalising against one sample
    chased that noise — the snare's anchors were reported 2.04 dB apart at
    build time and measured 25.3 dB apart afterwards (2026-08-03).
    """
    vals, ok = [], True
    for _ in range(max(1, k)):
        qc = qc_loudness(worker.render(probe), BUILD_MARGIN)
        if not qc.ok:
            ok = False
            break
        vals.append(qc.rms)
    return (float(np.mean(vals)) if vals else 0.0), ok

def probe_subset(n: int, k: int) -> list[int]:
    """`k` indices spread evenly over `n`. Deterministic. Pure.

    Used to judge a candidate lens cheaply: whether a structure admits other
    patches shows up in a few dozen renders, and paying a full pool's worth
    per rejected lens is what makes retrying one unaffordable.
    """
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def measure_effective(
    worker: RenderWorker,
    role: str,
    cands: list[Candidate],
    lens: int,
    param_names: list[str],
    only: list[int] | None = None,
) -> list[Candidate]:
    """Render every candidate's parameter vector through the lens anchor.

    This is the only sound the dial can make, so it is the only sound worth
    measuring. Candidates that turn to garbage through this structure are
    dropped: their values are unusable here whatever they sound like alone.
    Returns the survivors with `z_eff` filled in, lens first.

    `only` restricts the work to those indices (the lens is always included),
    for cheaply judging a candidate lens before committing to it.
    """
    probe = get_probe(role, "short")
    if not worker.load_preset(cands[lens].path):
        return []
    base = worker.baseline_raw
    wanted = None if only is None else set(only) | {lens}
    out: list[Candidate] = []
    for i, c in enumerate(cands):
        if wanted is not None and i not in wanted:
            continue
        if i == lens:
            c.z_eff = np.asarray(c.z0, dtype=np.float64)
            # The lens needs its own level measured, not just its descriptors:
            # it is the TARGET every other anchor is trimmed to. Taking z0's
            # loudness would be wrong anyway (measured on a different load),
            # and leaving it at 0 silently disables normalization entirely.
            c.rms_eff = _mean_rms(worker, probe, render_avg(role))[0]
            out.append(c)
            continue
        worker.apply_delta(delta_to(c.x, base, param_names))
        qc = qc_loudness(worker.render(probe), BUILD_MARGIN)
        if not qc.ok:
            worker.restore_baseline()
            continue
        c.rms_eff = _mean_rms(worker, probe, render_avg(role))[0]
        c.z_eff = np.asarray(
            worker.render_descriptors(probe, k=render_avg(role)), dtype=np.float64
        )
        worker.restore_baseline()
        out.append(c)
    # the lens leads: it is anchor 0's structure and the artifact's start
    out.sort(key=lambda c: c.preset_id != cands[lens].preset_id)
    return out


# ---------------------------------------------------------------------------
# 3b. loudness normalization — the ear-safety gate
# ---------------------------------------------------------------------------

# The one output-level control present in every role's basis.
GAIN_PARAM = "a_vca_gain"
# How far an anchor may sit from the lens's level once trimmed. Comfortably
# inside "no lurch", and any anchor that cannot be brought here is dropped.
LOUDNESS_TOL_DB = 3.0
# How much louder than the lens an in-between position may get. Looser than the
# anchor tolerance because a transient midpoint is less fatiguing than a
# sustained one, but far tighter than the 24 dB the ungated build produced.
MIDPOINT_TOL_DB = 6.0
# Bisection bounds on the trim, in raw parameter units.
TRIM_STEPS = 7




def normalize_loudness(
    worker: RenderWorker,
    role: str,
    cands: list[Candidate],
    lens: int,
    param_names: list[str],
    tol_db: float = LOUDNESS_TOL_DB,
) -> tuple[list[Candidate], dict]:
    """Trim every anchor to the LENS's loudness, verified by re-rendering.

    The tour swings output gain as part of the timbre: measured on the first
    shipped artifact, snare's `a_vca_gain` moved 1.0 -> 0.564 in ONE dial step
    and kick/bass swept `a_waveshaper_drive` across its whole range, with
    anchors named "Play Louder" and "Serious Distortion". Those go out as raw
    relative deltas, clamped only at each parameter's own range and never at a
    loudness target — which is how a dial sweep came to hurt someone.

    So each anchor's gain is bisected until its rendered RMS matches the lens,
    and the result is RE-RENDERED to confirm rather than predicted. Anchors
    that cannot be tamed are dropped: an untameable anchor is a patch whose
    loudness IS its character, and there is no safe way to put it on a dial.
    """
    if not cands:
        return [], {"target_rms": 0.0, "dropped": 0, "spread_db": 0.0}

    probe = get_probe(role, "short")
    target = cands[lens].rms_eff
    if target <= 0 or GAIN_PARAM not in param_names:
        return cands, {
            "target_rms": round(float(target), 6),
            "dropped": 0,
            "spread_db": None,
            "reason": "no gain control in this role's basis" if target > 0 else "silent lens",
        }
    gi = param_names.index(GAIN_PARAM)

    if not worker.load_preset(cands[lens].path):
        return cands, {"target_rms": 0.0, "dropped": 0, "spread_db": None,
                       "reason": "lens would not reload"}
    base = worker.baseline_raw

    kept: list[Candidate] = []
    dropped: list[dict] = []
    for i, c in enumerate(cands):
        if i == lens:
            kept.append(c)
            continue
        # bisect the gain until the render matches the lens's level
        lo, hi = 0.0, 1.0
        best_err, best_gain, best_rms = None, float(c.x[gi]), c.rms_eff
        for _ in range(TRIM_STEPS):
            mid = 0.5 * (lo + hi)
            probe_x = c.x.copy()
            probe_x[gi] = mid
            worker.apply_delta(delta_to(probe_x, base, param_names))
            rms, ok = _mean_rms(worker, probe, render_avg(role))
            worker.restore_baseline()
            if not ok:
                hi = mid          # too hot (or broken) — back the gain off
                continue
            err = abs(20.0 * np.log10(max(rms, 1e-9) / target))
            if best_err is None or err < best_err:
                best_err, best_gain, best_rms = err, mid, rms
            if rms > target:
                hi = mid
            else:
                lo = mid
        if best_err is None or best_err > tol_db:
            dropped.append({"preset_id": c.preset_id, "name": c.name,
                            "err_db": None if best_err is None else round(best_err, 2)})
            continue
        c.trim = float(best_gain - c.x[gi])
        c.x[gi] = float(best_gain)
        c.rms_eff = best_rms
        kept.append(c)

    # The space BETWEEN anchors is gated later, in validate_tour_sweep, where
    # the consecutive pairs are the ones the dial actually travels. Gating
    # preset_id-ordered pairs here (as a first attempt did) checks approaches
    # that never happen and lets the real ones through.

    levels = [
        20.0 * np.log10(max(c.rms_eff, 1e-9) / target) for c in kept if c.rms_eff > 0
    ]
    spread = round(float(max(levels) - min(levels)), 2) if levels else 0.0
    print(
        f"{role}: loudness-normalized {len(kept)}/{len(cands)} anchors "
        f"(spread {spread} dB, dropped {len(dropped)})"
    )
    return kept, {
        "target_rms": round(float(target), 6),
        "dropped": len(dropped),
        "dropped_detail": dropped[:10],
        "spread_db": spread,
        "tol_db": tol_db,
    }


# ---------------------------------------------------------------------------
# 4. spread (pure)
# ---------------------------------------------------------------------------


def farthest_point_sample(
    zs: np.ndarray, seed: int, ids: list[str], m: int
) -> list[int]:
    """Indices of the m most spread rows of zs, always including `seed`. Pure.

    Greedily adds whichever row maximizes its minimum distance to the kept
    set. Ties break on preset_id so re-runs agree exactly.
    """
    n = len(zs)
    if n == 0:
        return []
    if n <= m:
        return list(range(n))
    kept = [seed]
    d_min = np.linalg.norm(zs - zs[seed], axis=1)
    while len(kept) < m:
        nxt = max(
            (i for i in range(n) if i not in kept),
            key=lambda i: (d_min[i], ids[i]),
        )
        kept.append(nxt)
        d_min = np.minimum(d_min, np.linalg.norm(zs - zs[nxt], axis=1))
    return sorted(kept)


# ---------------------------------------------------------------------------
# 5. edges — validate the interpolation the runtime will actually play
# ---------------------------------------------------------------------------


def validate_role_edges(
    worker: RenderWorker,
    role: str,
    cands: list[Candidate],
    lens_path: str,
    param_names: list[str],
    k: int = 4,
    n_interior: int = N_INTERIOR,
    min_travel: float = MIN_HOP_TRAVEL,
) -> list[dict]:
    """kNN proposals in effective space, each validated by rendering.

    Both endpoints are already known clean through the lens, so what this
    tests is the ROUTE between them: every interior point must pass QC, the
    path must not detour wildly, and the two ends must be audibly apart —
    an edge between two positions that sound the same is not a journey.
    """
    probe = get_probe(role, "short")
    zs = np.stack([standardize(c.z_eff) for c in cands])
    if not worker.load_preset(lens_path):
        return []
    base = worker.baseline_raw

    edges: list[dict] = []
    for ia, ib in knn_edges(zs, k=k):
        a, b = cands[ia], cands[ib]
        travel = float(np.linalg.norm(zs[ia] - zs[ib]))
        row = {"ia": ia, "ib": ib, "travel": round(travel, 4)}
        if travel < min_travel:
            edges.append({**row, "valid": False, "reason": "no-travel"})
            continue

        z_rows, ok = [], True
        for t in np.linspace(0.0, 1.0, n_interior + 2)[1:-1]:
            x_t = a.x * (1 - t) + b.x * t
            worker.apply_delta(delta_to(x_t, base, param_names))
            # loudness too: a route that passes THROUGH a scream is not a
            # usable route, however well its endpoints measure
            if not qc_loudness(worker.render(probe), BUILD_MARGIN).ok:
                ok = False
                break
            z_rows.append(
                standardize(worker.render_descriptors(probe, k=render_avg(role)))
            )
        worker.restore_baseline()
        if not ok:
            edges.append({**row, "valid": False, "reason": "qc-fail-on-path"})
            continue

        z_path = np.vstack([np.stack(z_rows), zs[ib][None, :]])
        _, detour = edge_metrics(z_path, zs[ia], zs[ib])
        valid = detour <= MAX_DETOUR
        edges.append(
            {
                **row,
                "valid": valid,
                "detour": round(detour, 4),
                **({} if valid else {"reason": "detour"}),
            }
        )
    return edges


# ---------------------------------------------------------------------------
# 6. tour (pure)
# ---------------------------------------------------------------------------


def select_tour(
    n_nodes: int,
    valid_edges: list[tuple[int, int, float]],
    ids: list[str],
    target: int,
    start: int | None = None,
) -> tuple[list[int], dict]:
    """A deterministic maximum-travel simple path over the valid edges. Pure.

    Largest connected component first (or the component holding `start`, when
    given — the lens anchor must be on the tour because it supplies the
    structure). Inside it, grow a simple path from every seed by repeatedly
    extending whichever end has the heaviest edge to an unvisited neighbour;
    keep the best path (most nodes, then total travel). Over-length paths keep
    their max-travel contiguous window; shortfalls are reported, never
    silently absorbed.
    """
    adj: dict[int, dict[int, float]] = {i: {} for i in range(n_nodes)}
    for a, b, w in valid_edges:
        adj[a][b] = max(w, adj[a].get(b, 0.0))
        adj[b][a] = max(w, adj[b].get(a, 0.0))

    seen: set[int] = set()
    components: list[list[int]] = []
    for node in range(n_nodes):
        if node in seen or not adj[node]:
            continue
        comp, stack = [], [node]
        seen.add(node)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(sorted(comp))
    isolated = n_nodes - sum(len(c) for c in components)

    if start is not None:
        components = [c for c in components if start in c]
    if not components:
        return [], {
            "wanted": target, "shipped": 0, "component_size": 0,
            "isolated_nodes": isolated,
        }

    # largest component; ties on the smallest preset_id it contains
    comp = min(components, key=lambda c: (-len(c), min(ids[i] for i in c)))

    def grow(seed: int, one_way: bool, avoid_dead_ends: bool) -> list[int]:
        """Greedy simple path from `seed`.

        `one_way` extends only the tail, so the seed stays at position 0 —
        what a fixed lens requires, since the panel loads that preset and
        starts the dial there.

        `avoid_dead_ends` adapts Warnsdorff's rule to a LONGEST path. Plain
        Warnsdorff ("step to the fewest-onward neighbour") is built for
        Hamiltonian tours, where a degree-1 node must be taken while it is
        still reachable; here stepping into one ENDS the walk, so a node with
        no onward option is taken only when nothing else remains. Otherwise
        prefer the fewest onward options, which keeps well-connected nodes
        available for later.

        Pure heaviest-hop greed strands itself — measured 2026-08-03, it
        walked 3 anchors out of a 31-node connected pad component and 5 out of
        25 for the kick, because the biggest hops hang off the sparse edge of
        the graph and there is no way back. Travel stays the tie-break, so
        among equally safe steps the dial still takes the bigger one.
        """
        path = [seed]
        on_path = {seed}
        while True:
            best: tuple[bool, int, float, str, int, bool] | None = None
            ends = ((False, path[-1]),) if one_way else ((True, path[0]), (False, path[-1]))
            for at_head, end in ends:
                for v, w in adj[end].items():
                    if v in on_path:
                        continue
                    onward = sum(1 for u in adj[v] if u not in on_path and u != v)
                    cand = (
                        # a dead end is a last resort, never a preference
                        avoid_dead_ends and onward == 0,
                        onward if avoid_dead_ends else 0,
                        -w,
                        ids[v],
                        v,
                        at_head,
                    )
                    if best is None or cand[:4] < best[:4]:
                        best = cand
            if best is None:
                return path
            _, _, _, _, node, at_head = best
            path.insert(0, node) if at_head else path.append(node)
            on_path.add(node)

    def travel_of(path: list[int]) -> float:
        return sum(adj[a][b] for a, b in zip(path, path[1:]))

    def better(p: list[int], cur: list[int]) -> bool:
        """More anchors first, then more travel. Reaching twenty stops matters
        more than any single hop being large — a short tour is the failure the
        whole redesign exists to avoid."""
        if len(p) != len(cur):
            return len(p) > len(cur)
        return travel_of(p) > travel_of(cur)

    if start is not None:
        best_path = []
        # both heuristics, because neither dominates: dead-end avoidance wins
        # on sparse graphs, raw greed on dense ones where nothing strands
        for avoid in (True, False):
            p = grow(start, one_way=True, avoid_dead_ends=avoid)
            if better(p, best_path):
                best_path = p
    else:
        best_path = []
        for seed in sorted(comp, key=lambda i: ids[i]):
            for avoid in (True, False):
                p = grow(seed, one_way=False, avoid_dead_ends=avoid)
                if ids[p[0]] > ids[p[-1]]:
                    p = p[::-1]  # canonical direction so re-runs agree
                if better(p, best_path):
                    best_path = p

    if len(best_path) > target:
        if start is not None:
            best_path = best_path[:target]  # the lens holds position 0
        else:
            hop = [adj[a][b] for a, b in zip(best_path, best_path[1:])]
            best_start, best_w = 0, -1.0
            for s in range(len(best_path) - target + 1):
                w = sum(hop[s : s + target - 1])
                if w > best_w + 1e-12:
                    best_start, best_w = s, w
            best_path = best_path[best_start : best_start + target]

    return best_path, {
        "wanted": target,
        "shipped": len(best_path),
        "component_size": len(comp),
        "isolated_nodes": isolated,
    }


# ---------------------------------------------------------------------------
# 7. sweep — gate the composed tour exactly as the panel will play it
# ---------------------------------------------------------------------------


def sweep_tour(
    worker: RenderWorker,
    role: str,
    cands: list[Candidate],
    path: list[int],
    param_names: list[str],
    n_interior: int = N_INTERIOR,
) -> tuple[list[float], list[int], list[float]]:
    """Render the whole tour under path[0]'s structure, as the panel plays it.

    Returns (per-hop travel, positions that render unusably, per-hop LOUDEST
    interior level in dB relative to the lens). A hop whose interior fails QC
    gets zero travel and its endpoint is reported broken, so the caller can
    drop it and re-splice; a hop whose interior is merely loud is reported so
    the caller can decide.
    """
    probe = get_probe(role, "short")
    if not worker.load_preset(cands[path[0]].path):
        return [], list(range(len(path))), []
    base = worker.baseline_raw
    snaps = [cands[i].x for i in path]
    ref, _ = _mean_rms(worker, probe, render_avg(role))
    ref = max(ref, 1e-9)
    loud: dict[int, float] = {}

    def render_at(x_t: np.ndarray, hop: int = -1) -> np.ndarray | None:
        # THE PANEL'S DELTA, not an absolute target.
        #
        # `delta_to(x_t, base, ...)` lands exactly on x_t. The panel cannot do
        # that: it sends `snapshot[c] - snapshot[0]` as a RELATIVE write on top
        # of whatever the track currently holds, so what plays is
        # `live + x_t - snapshot[0]`. Those agree only if the lens loads
        # identically every time, and it does not — `_fix_conditional_params`
        # recalibrates oscillator ranges against the previously loaded patch,
        # so the same preset can expose 97 parameters on one load and 87 on the
        # next. Validating the absolute form meant the build checked a
        # configuration the runtime never plays: a kick anchor passed every
        # build gate and rendered a dead-consistent 5.07 peak in the panel,
        # over a 4.0 ceiling (2026-08-03).
        worker.apply_delta(
            {
                n: float(x_t[j] - snaps[0][j])
                for j, n in enumerate(param_names)
                if n in base and abs(x_t[j] - snaps[0][j]) > 1e-5
            }
        )
        if not qc_loudness(worker.render(probe), BUILD_MARGIN).ok:
            return None
        rms, ok = _mean_rms(worker, probe, render_avg(role))
        if hop >= 0 and ok and rms > 0:
            db = 20.0 * np.log10(rms / ref)
            loud[hop] = max(loud.get(hop, -999.0), db)
        return standardize(worker.render_descriptors(probe, k=render_avg(role)))

    zs: list[np.ndarray | None] = [render_at(s) for s in snaps]
    broken = [i for i, z in enumerate(zs) if z is None]

    travels: list[float] = []
    for h in range(1, len(path)):
        a, b = snaps[h - 1], snaps[h]
        za, zb = zs[h - 1], zs[h]
        interior_ok = True
        for t in np.linspace(0.0, 1.0, n_interior + 2)[1:-1]:
            if render_at(a * (1 - t) + b * t, hop=h) is None:
                interior_ok = False
                break
        if za is None or zb is None or not interior_ok:
            if interior_ok is False and h not in broken:
                broken.append(h)
            travels.append(0.0)
            continue
        travels.append(float(np.linalg.norm(zb - za)))
    worker.restore_baseline()
    hop_db = [loud.get(h, -999.0) for h in range(1, len(path))]
    return travels, sorted(set(broken)), hop_db


def validate_tour_sweep(
    worker: RenderWorker,
    role: str,
    cands: list[Candidate],
    path: list[int],
    param_names: list[str],
    n_interior: int = N_INTERIOR,
    min_travel: float = MIN_HOP_TRAVEL,
) -> tuple[list[int], dict]:
    """Drop anchors the composed tour cannot reach cleanly, then re-splice.

    Pairwise edges were validated two at a time; a tour is played end to end.
    This gates the shipped experience itself: every position renders, and
    every hop moves the sound. Position 0 is the lens and is never dropped —
    without it there is no structure to hear the rest through.
    """
    kept = list(path)
    dropped: list[dict] = []
    while len(kept) >= 2:
        travels, broken, hop_db = sweep_tour(
            worker, role, cands, kept, param_names, n_interior
        )
        # never drop the lens; a broken lens means the role cannot ship
        droppable = [i for i in broken if i != 0]
        if droppable:
            idx = droppable[0]
            dropped.append({"preset_id": cands[kept[idx]].preset_id, "reason": "qc"})
            kept.pop(idx)
            continue
        if broken:
            return [], {
                "dropped": dropped, "worst_hop_travel": None,
                "reason": "lens renders unusably",
            }
        # A hop whose MIDDLE is much louder than the lens is the shape that
        # hurt: both endpoints level-matched, the approach between them not.
        # Measured before this gate existed, kick's midpoints ran +12.8 dB
        # over its median while its anchors sat 8 dB apart.
        hot = [h for h, db in enumerate(hop_db, start=1) if db > MIDPOINT_TOL_DB]
        if hot:
            idx = hot[0]
            dropped.append({
                "preset_id": cands[kept[idx]].preset_id,
                "reason": "hot-approach",
                "db_over_lens": round(hop_db[idx - 1], 1),
            })
            kept.pop(idx)
            continue

        still = [h for h, t in enumerate(travels, start=1) if t < min_travel]
        if not still:
            return kept, {
                "dropped": dropped,
                "worst_hop_travel": round(min(travels), 4) if travels else None,
            }
        dropped.append(
            {
                "preset_id": cands[kept[still[0]]].preset_id,
                "reason": "no-travel",
                "travel": round(travels[still[0] - 1], 4),
            }
        )
        kept.pop(still[0])
    return kept, {"dropped": dropped, "worst_hop_travel": None}


# ---------------------------------------------------------------------------
# assemble + orchestrate
# ---------------------------------------------------------------------------


def role_entry(
    cands: list[Candidate],
    path: list[int],
    param_names: list[str],
    cfg: LabConfig | None = None,
) -> dict:
    """The per-role artifact entry. Pure given resolved candidates.

    `param_names` is per role, not global: which parameters exist depends on
    the lens preset's oscillator types (see project_to_lens_basis).
    """
    n = len(path)
    if n > 1:
        controls = [round(i / (n - 1), 6) for i in range(n)]
    else:
        controls = [0.0] * n
    return {
        "param_names": param_names,
        "control_points": controls,
        "anchors": [
            {
                "preset_id": cands[i].preset_id,
                "name": cands[i].name,
                "fxp_path": portable_fxp_path(cands[i].path, cfg),
            }
            for i in path
        ],
        "snapshots": [np.round(cands[i].x, 6).tolist() for i in path],
        "declined": n < 2,
    }


def build_role_tour(
    cfg: LabConfig,
    worker: RenderWorker,
    role: str,
    allowed: list[str],
    anchors: int,
    cap: int,
    keep: int,
    k: int,
    n_interior: int,
) -> tuple[dict, dict]:
    """Screen → lens → effect → spread → edges → tour → sweep, for one role."""
    survivors, rows = screen_role_pool(cfg, worker, role, allowed, cap=cap)
    n_screened = len(rows)
    print(f"{role}: {len(survivors)}/{n_screened} candidates pass screening")

    quality: dict = {
        "n_pool": n_screened,
        "capped": n_screened >= cap,
        "n_passed": len(survivors),
    }
    if len(survivors) < 2:
        quality.update(
            {"n_kept": len(survivors), "n_edges_valid": 0, "anchors": [],
             "shortfall": {"wanted": anchors, "shipped": 0,
                           "reason": "too few candidates survived screening"}}
        )
        return role_entry(survivors, [], [], cfg), quality

    # A drum's identity is its envelope, and every stop on the tour is another
    # patch's continuous values. Tour a kick through pad settings and the long
    # release stops it being a kick — which is exactly how the first shipped
    # drums read ("completely unconvincing"). So percussion tours only among
    # patches that call themselves percussion; the melodic roles keep the full
    # pool, where crossing categories is the point.
    if role in PERCUSSIVE_ROLES:
        named = [c for c in survivors if is_percussive(c, role)]
        if len(named) >= MIN_PERCUSSIVE_POOL:
            print(f"{role}: restricting to {len(named)} percussion patches")
            survivors = named
            quality["restricted_to_named"] = len(named)
        else:
            print(
                f"{role}: only {len(named)} self-declared {role} patches survived "
                f"— keeping the full pool of {len(survivors)}"
            )
            quality["restricted_to_named"] = None

    param_names = shared_basis(survivors, allowed)

    # A lens is only as good as what survives being heard through it, and that
    # cannot be known in advance — so try ranked candidates until one admits a
    # usable pool. Without this the lead shipped nothing at all: its top-ranked
    # lens let 1 of 358 candidates render.
    sample = probe_subset(len(survivors), LENS_PROBE)
    tried: list[dict] = []
    lens, best_rate = -1, -1.0
    for candidate_lens in rank_lenses(survivors, role)[:MAX_LENS_TRIES]:
        got = measure_effective(
            worker, role, survivors, candidate_lens, param_names, only=sample
        )
        rate = len(got) / max(1, len(sample))
        name = survivors[candidate_lens].name
        print(f"{role}: lens {name!r} admits {len(got)}/{len(sample)} probed")
        tried.append({"name": name, "probe_rate": round(rate, 3)})
        if rate > best_rate:
            lens, best_rate = candidate_lens, rate
        if rate >= LENS_SURVIVAL:
            break

    quality["lens_attempts"] = tried
    # commit: measure the WHOLE pool through the lens that won
    live = measure_effective(worker, role, survivors, lens, param_names)
    print(f"{role}: {len(live)}/{len(survivors)} render cleanly through the lens")
    quality["n_through_lens"] = len(live)
    if len(live) < 2:
        quality.update(
            {"n_kept": len(live), "n_edges_valid": 0, "anchors": [],
             "shortfall": {"wanted": anchors, "shipped": 0,
                           "reason": "nothing renders through any candidate lens"}}
        )
        return role_entry(live, [], param_names, cfg), quality
    lens_path = survivors[lens].path
    print(
        f"{role}: lens = {survivors[lens].name} ({survivors[lens].category}), "
        f"{len(param_names)} shared params"
    )

    # EAR SAFETY: level every anchor to the lens before anything selects on
    # them, so spread/edges/tour all reason about tamed sounds.
    live, loud_q = normalize_loudness(worker, role, live, 0, param_names)
    quality["loudness"] = loud_q
    if len(live) < 2:
        quality.update(
            {"n_kept": len(live), "n_edges_valid": 0, "anchors": [],
             "shortfall": {"wanted": anchors, "shipped": 0,
                           "reason": "nothing survived loudness normalization"}}
        )
        return role_entry(live, [], param_names, cfg), quality

    zs = np.stack([standardize(c.z_eff) for c in live])
    kept_idx = farthest_point_sample(zs, 0, [c.preset_id for c in live], keep)
    kept = [live[i] for i in kept_idx]
    lens_idx = kept_idx.index(0)

    edges = validate_role_edges(
        worker, role, kept, lens_path, param_names, k=k, n_interior=n_interior
    )
    valid = [(e["ia"], e["ib"], e["travel"]) for e in edges if e["valid"]]
    travels = sorted(e["travel"] for e in edges)
    if travels:
        print(
            f"{role}: {len(valid)}/{len(edges)} edges valid "
            f"(travel min {travels[0]:.2f} / med {travels[len(travels) // 2]:.2f} "
            f"/ max {travels[-1]:.2f})"
        )

    path, tour_info = select_tour(
        len(kept), valid, [c.preset_id for c in kept], anchors, start=lens_idx
    )
    sweep_q: dict = {"dropped": [], "worst_hop_travel": None}
    if len(path) >= 2:
        path, sweep_q = validate_tour_sweep(
            worker, role, kept, path, param_names, n_interior=n_interior
        )

    shipped = len(path)
    if shipped < anchors:
        print(
            f"{role}: SHORTFALL — shipped {shipped}/{anchors} anchors "
            f"(component {tour_info['component_size']}, "
            f"isolated {tour_info['isolated_nodes']}, "
            f"dropped in sweep {len(sweep_q['dropped'])})"
        )
    quality.update(
        {
            "n_kept": len(kept),
            "lens": {"preset_id": kept[lens_idx].preset_id,
                     "name": kept[lens_idx].name,
                     "category": kept[lens_idx].category},
            "n_edges_proposed": len(edges),
            "n_edges_valid": len(valid),
            "edge_travel": travels,
            "anchors": [
                {"preset_id": kept[i].preset_id, "name": kept[i].name,
                 "sigma": round(kept[i].sigma, 4)}
                for i in path
            ],
            "sweep": sweep_q,
            "shortfall": None if shipped >= anchors else {**tour_info, "shipped": shipped},
        }
    )
    entry = role_entry(kept, path, param_names, cfg)
    entry["role"] = role
    return entry, quality


def assemble(role_entries: dict[str, dict], quality: dict[str, dict]) -> dict:
    """The shipped artifact from whatever roles are finished. Pure."""
    return {
        "version": TOUR_VERSION,
        "policy_version": POLICY_VERSION,
        "probe_version": PROBE_VERSION,
        "seed": SEED,
        "roles": role_entries,
        "quality": {
            **quality,
            "_summary": {
                "anchors_by_role": {
                    r: len(e["anchors"]) for r, e in role_entries.items()
                },
                "roles_declined": [
                    r for r, e in role_entries.items() if e["declined"]
                ],
            },
        },
    }


def build_tour_graph(
    cfg: LabConfig | None = None,
    roles: list[str] | None = None,
    anchors: int = DEFAULT_ANCHORS,
    cap: int = DEFAULT_CAP,
    keep: int = DEFAULT_KEEP,
    k: int = 4,
    n_interior: int = N_INTERIOR,
    save_to: str | Path | None = None,
) -> dict:
    """Build every role's tour and assemble the shipped artifact.

    When `save_to` is given the artifact is rewritten after EVERY role. Six
    roles is the better part of an hour of rendering, and the roles are
    independent — losing kick through pad because lead raised is a bill
    nobody should pay twice (the same reason screening reports stream).
    """
    cfg = cfg or LabConfig()
    worker = RenderWorker(cfg)
    allowed = json.loads(cfg.policy_path.read_text())["allowed"]

    role_entries: dict[str, dict] = {}
    quality: dict[str, dict] = {}

    for role in roles or ROLES:
        entry, q = build_role_tour(
            cfg, worker, role, allowed, anchors, cap, keep, k, n_interior
        )
        entry["role"] = role
        role_entries[role] = entry
        quality[role] = q
        if save_to is not None:
            from timbre_graph_lab.morph import save_graph

            save_graph(assemble(role_entries, quality), save_to)
            print(f"{role}: saved -> {save_to}")

    return assemble(role_entries, quality)
