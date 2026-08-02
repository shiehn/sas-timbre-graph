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
        # String-valued / stepped params are discrete: skip structurally.
        try:
            n_steps = getattr(param, "num_steps", None)
            valid_str = getattr(param, "string_value", None)
        except Exception:
            n_steps, valid_str = None, None
        is_discrete = bool(getattr(param, "is_boolean", False))
        if n_steps is not None and 0 < n_steps <= 32:
            is_discrete = True
        if is_discrete:
            excluded[name] = "discrete"
            continue
        if _denied(name):
            excluded[name] = "deny-list"
            continue
        allowed.append(name)
        _ = valid_str  # introspection only

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
