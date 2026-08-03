"""Live-safe continuous parameter allow-list ("parameter policy").

Built by introspecting the live pedalboard parameter surface: discrete /
stepped / string-valued parameters are excluded structurally, then a name
deny-list removes globals that must never participate in coupling (master
volume, scene/FX routing, tuning, tempo-sync switches). The result is
persisted as a versioned JSON artifact that dataset shards and the runtime
solver both reference.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

from timbre_graph_lab.config import POLICY_VERSION, LabConfig
from timbre_graph_lab.worker import RenderWorker

# Structural or global controls that must never move during coupling.
DENY_PATTERNS = [
    "*volume*",            # loudness must not masquerade as timbre
    "*scene*",
    "*type*",              # osc/filter type selectors
    "*category*",
    "*polymode*", "*poly_limit*",
    "*splitpoint*", "*split_key*",
    "*fx*bypass*", "*bypass*", "*disable*",
    "*tempo*", "*sync*",
    "*tuning*", "*scl*", "*kbm*",
    "*mpe*", "*pitch_bend*", "*pbrange*",
    "*portamento_curve*",
    "*macro*",             # macros are user meta-controls, not timbre params
    "*send*",              # FX sends: keep FX static in v1
    "*return*",
    "*character*",
    "*octave*",            # register identity is sacred (MIDI is sacred)
    "*pitch*",             # coarse pitch: same reason (fine detune is allowed)
    "*keytrack*",
    "*unison_voices*",     # voice-count is steppy/clicky live
    # Structural switches. These leaked into the v1 allow-list and then
    # DOMINATED the sensitivity screen, because flipping a mute or a routing
    # switch produces a far bigger descriptor delta than any continuous
    # timbre move — crowding real parameters out of the per-anchor budget.
    "*mute*",
    "*solo*",
    "*route*",
    "*retrigger*",         # also fights the frozen-phase determinism convention
]

ALLOW_EXCEPTIONS = [
    # fine detune is a legitimate timbre move even though "*pitch*" is denied
    "*_osc?_dispersion*",
]

# Only scene A participates in v1: most patches are single-scene, and moving
# scene B params on a scene-A patch is dead weight.
SCENE_PREFIX_DENY = ("b_",)


def _denied(name: str) -> bool:
    lname = name.lower()
    for pat in ALLOW_EXCEPTIONS:
        if fnmatch.fnmatch(lname, pat):
            return False
    return any(fnmatch.fnmatch(lname, pat) for pat in DENY_PATTERNS)


_NUMERIC_DISPLAY = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)")
_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)


def _is_discrete(param) -> bool:
    """Decide continuity by PROBING the parameter, not by trusting metadata.

    Every metadata signal on this surface lies: pedalboard reports
    `num_steps = 2147483647` and `is_boolean = False` for every Surge
    parameter (so the v1 test never fired and 21 switches were treated as
    continuous), while `range` is unreported — `(None, None, None)` — for
    plenty of genuinely continuous controls including every envelope stage
    (rejecting on that basis threw away amp/filter EG attack, decay, release).

    Sweeping raw 0->1 and reading the display string is unambiguous:

        cutoff   -> '13.75 Hz', '89.87 Hz', ...   5 distinct, numeric
        eg_attack-> '0.0 ms', '37.2 ms', ...      5 distinct, numeric
        mute     -> 'Off', 'On'                   2 distinct, text
        route    -> 'Filter 1', 'Both'            3 distinct, text
        filter_configuration -> 'Serial 1', 'Dual 2'  5 distinct, TEXT

    So: continuous iff the sweep yields many distinct values AND every one of
    them reads as a number. The last example is why distinctness alone is not
    enough.
    """
    rng = getattr(param, "range", None)
    if rng is not None:
        t = tuple(rng)
        if len(t) >= 2 and isinstance(t[0], bool) and isinstance(t[1], bool):
            return True  # declared boolean switch
    try:
        saved = param.raw_value
        seen = []
        for r in _SWEEP:
            param.raw_value = r
            seen.append(str(getattr(param, "string_value", "")))
        param.raw_value = saved
    except Exception:
        return True  # unprobeable -> keep it out of the live-safe set
    if len(set(seen)) < 4:
        return True
    return not all(_NUMERIC_DISPLAY.match(v) for v in seen)


def build_policy(worker: RenderWorker) -> dict:
    """Introspect the live host and produce the policy dict."""
    plugin = worker.host._plugin  # noqa: SLF001 — introspection by design
    allowed: list[str] = []
    excluded: dict[str, str] = {}

    for name, param in plugin.parameters.items():
        lname = name.lower()
        if lname.startswith(SCENE_PREFIX_DENY):
            excluded[name] = "scene-b"
            continue
        if _is_discrete(param):
            excluded[name] = "discrete"
            continue
        if _denied(name):
            excluded[name] = "deny-list"
            continue
        allowed.append(name)

    return {
        "policy_version": POLICY_VERSION,
        "n_allowed": len(allowed),
        "allowed": sorted(allowed),
        "excluded": excluded,
    }


def write_policy(policy: dict, cfg: LabConfig | None = None) -> Path:
    cfg = cfg or LabConfig()
    out = cfg.policy_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=1))
    return out


def load_policy(cfg: LabConfig | None = None) -> dict:
    cfg = cfg or LabConfig()
    return json.loads(cfg.policy_path.read_text())
