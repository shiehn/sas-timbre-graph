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


@dataclass
class LabConfig:
    surge_vst3: Path = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")
    factory_patches_dir: Path = Path(
        "/Library/Application Support/Surge XT/patches_factory"
    )
    third_party_patches_dir: Path = Path(
        "/Library/Application Support/Surge XT/patches_3rdparty"
    )
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
