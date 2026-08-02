"""Corpus scan + role assignment over installed Surge patch content.

Scans factory and 3rd-party .fxp trees, applies the keyword/category rules
from configs/roles.yaml, and writes a corpus manifest JSON. Rules only
bootstrap the corpus; render-time QC is the real gate. Manual overrides
(configs/corpus_overrides.yaml, optional) persist curation decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from timbre_graph_lab.config import CORPUS_VERSION, LabConfig, ROLES, configs_dir


@dataclass
class CorpusEntry:
    preset_id: str  # sha1 of file bytes (stable across machines)
    path: str
    name: str
    source: str  # "factory" | "3rdparty"
    category: str  # factory folder / 3rd-party author folder
    roles: list[str] = field(default_factory=list)


def _load_rules() -> dict:
    with open(configs_dir() / "roles.yaml") as f:
        return yaml.safe_load(f)


def _load_overrides() -> dict[str, list[str]]:
    p = configs_dir() / "corpus_overrides.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    # {preset name or id: [roles] or []}
    return {str(k): list(v or []) for k, v in data.items()}


def _matches_any(text: str, keywords: list[str]) -> bool:
    return any(k.lower() in text for k in keywords)


def assign_roles(name: str, path_parts: list[str], rules: dict) -> list[str]:
    """Pure rule evaluation — unit-testable without any Surge install."""
    lname = name.lower()
    reject = rules.get("reject_global", {})
    if any(part in reject.get("categories", []) for part in path_parts):
        return []
    if _matches_any(lname, reject.get("keywords", [])):
        return []

    assigned: list[str] = []
    for role in ROLES:
        r = rules["roles"][role]
        if _matches_any(lname, r.get("reject_keywords", [])):
            continue
        in_category = any(part in r.get("categories", []) for part in path_parts)
        by_keyword = _matches_any(lname, r.get("keywords", []))
        # Category alone is enough for melodic roles; percussion roles need a
        # keyword hit because Percussion mixes kicks/snares/toms/other.
        if role in ("kick", "snare", "hat"):
            if by_keyword and (in_category or "Percussion" not in path_parts):
                assigned.append(role)
        elif in_category or by_keyword:
            assigned.append(role)
    return assigned


def scan(cfg: LabConfig | None = None) -> list[CorpusEntry]:
    cfg = cfg or LabConfig()
    rules = _load_rules()
    overrides = _load_overrides()
    entries: list[CorpusEntry] = []

    for source, root in [
        ("factory", cfg.factory_patches_dir),
        ("3rdparty", cfg.third_party_patches_dir),
    ]:
        if not root.exists():
            continue
        for fxp in sorted(root.rglob("*.fxp")):
            rel = fxp.relative_to(root)
            parts = list(rel.parts[:-1])
            name = fxp.stem
            preset_id = hashlib.sha1(fxp.read_bytes()).hexdigest()[:16]

            if name in overrides:
                roles = overrides[name]
            elif preset_id in overrides:
                roles = overrides[preset_id]
            else:
                roles = assign_roles(name, parts, rules)
            if not roles:
                continue
            entries.append(
                CorpusEntry(
                    preset_id=preset_id,
                    path=str(fxp),
                    name=name,
                    source=source,
                    category=parts[0] if parts else "",
                    roles=roles,
                )
            )
    return entries


def write_manifest(entries: list[CorpusEntry], cfg: LabConfig | None = None) -> Path:
    cfg = cfg or LabConfig()
    out = cfg.corpus_path
    out.parent.mkdir(parents=True, exist_ok=True)
    by_role = {r: sum(1 for e in entries if r in e.roles) for r in ROLES}
    payload = {
        "corpus_version": CORPUS_VERSION,
        "counts_by_role": by_role,
        "n_entries": len(entries),
        "entries": [asdict(e) for e in entries],
    }
    out.write_text(json.dumps(payload, indent=1))
    return out


def load_manifest(cfg: LabConfig | None = None) -> dict:
    cfg = cfg or LabConfig()
    return json.loads(cfg.corpus_path.read_text())
