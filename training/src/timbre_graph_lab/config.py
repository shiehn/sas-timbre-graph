"""Central configuration: paths, render settings, versions, seeds.

Everything that affects dataset comparability is versioned here. Bump
PROBE_VERSION / POLICY_VERSION when probes or the parameter policy change;
datasets record these so stale shards are never silently mixed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROBE_VERSION = "probes-v1"
POLICY_VERSION = "policy-v1"
CORPUS_VERSION = "corpus-v1"

ROLES = ["kick", "snare", "hat", "bass", "pad", "lead"]

SEED = 20260802


def _default_workspace() -> Path:
    env = os.environ.get("TGLAB_WORKSPACE")
    if env:
        return Path(env)
    # training/workspace/ (gitignored)
    return Path(__file__).resolve().parents[2] / "workspace"


# macOS first, then the Linux (.deb / tarball) layouts used on RunPod.
# Env overrides win so a pod can point anywhere.
_VST3_CANDIDATES = [
    "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3",
    "/usr/lib/vst3/Surge XT.vst3",
    "/usr/local/lib/vst3/Surge XT.vst3",
    "~/.vst3/Surge XT.vst3",
]
_CONTENT_CANDIDATES = [
    "/Library/Application Support/Surge XT",
    "/usr/share/surge-xt",
    "/usr/local/share/surge-xt",
    "~/.local/share/surge-xt",
]


def _first_existing(env_var: str, candidates: list[str], fallback: str) -> Path:
    env = os.environ.get(env_var)
    if env:
        return Path(env).expanduser()
    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p
    return Path(fallback).expanduser()


def _default_vst3() -> Path:
    return _first_existing("TGLAB_SURGE_VST3", _VST3_CANDIDATES, _VST3_CANDIDATES[0])


def _content_root() -> Path:
    return _first_existing(
        "TGLAB_SURGE_CONTENT", _CONTENT_CANDIDATES, _CONTENT_CANDIDATES[0]
    )


def _default_factory() -> Path:
    return _content_root() / "patches_factory"


def _default_third_party() -> Path:
    return _content_root() / "patches_3rdparty"


@dataclass
class LabConfig:
    surge_vst3: Path = field(default_factory=_default_vst3)
    factory_patches_dir: Path = field(default_factory=_default_factory)
    third_party_patches_dir: Path = field(default_factory=_default_third_party)
    workspace: Path = field(default_factory=_default_workspace)
    sample_rate: int = 44100

    @property
    def corpus_path(self) -> Path:
        return self.workspace / "corpus" / f"{CORPUS_VERSION}.json"

    @property
    def policy_path(self) -> Path:
        return self.workspace / "policy" / f"{POLICY_VERSION}.json"

    @property
    def shards_dir(self) -> Path:
        return self.workspace / "shards"

    @property
    def models_dir(self) -> Path:
        return self.workspace / "models"

    @property
    def reports_dir(self) -> Path:
        return self.workspace / "reports"


def configs_dir() -> Path:
    """training/configs/ (committed, versioned)."""
    return Path(__file__).resolve().parents[2] / "configs"
