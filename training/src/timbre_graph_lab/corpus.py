"""Corpus scan + role assignment over installed Surge patch content.

Scans factory and 3rd-party .fxp trees, applies the keyword/category rules
from configs/roles.yaml, and writes a corpus manifest JSON. Rules only
bootstrap the corpus; render-time QC is the real gate. Manual overrides
(configs/corpus_overrides.yaml, optional) persist curation decisions.
"""

from __future__ import annotations

import hashlib
import re
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
    # What the patch IS, as opposed to who made it.
    #
    # Third-party is uniformly `author/subcategory/patch.fxp` (2371 files, zero
    # exceptions), so `category` — which is parts[0] — holds the AUTHOR for
    # every one of the 1557 third-party entries, and the thing that says
    # "Drums" or "Basses" was computed for role assignment and then thrown
    # away. Downstream that made category-based preference useless for
    # third-party patches and produced the conclusion that "Surge has 4
    # percussion presets" when there are 145 `Drums` + 94 `Percussion` files
    # on disk.
    subcategory: str = ""
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
    """WORD-boundary match, not substring.

    Substring matching cut both ways and was wrong in both directions:
    `Harpsichord`, `Sharp Lead` and `Guitarp` were globally rejected for
    containing "arp" (34 patches lost), while `Chatter`, `That Comb Magic` and
    `Hate` were assigned the `hat` role and went on to become hi-hat anchors.
    A keyword is a word.
    """
    return any(
        re.search(rf"\b{re.escape(k.lower())}\b", text) for k in keywords
    )


def assign_roles(name: str, path_parts: list[str], rules: dict) -> list[str]:
    """Pure rule evaluation — unit-testable without any Surge install."""
    lname = name.lower()
    reject = rules.get("reject_global", {})
    if any(part in reject.get("categories", []) for part in path_parts):
        return []
    if _matches_any(lname, reject.get("keywords", [])):
        return []

    assigned: list[str] = []
    perc_roles = ("kick", "snare", "hat")

    def allowed(role: str) -> bool:
        return not _matches_any(
            lname, rules["roles"][role].get("reject_keywords", [])
        )

    # Which drum does the NAME claim to be? A specific claim settles it.
    named_perc = [
        role
        for role in perc_roles
        if allowed(role)
        and _matches_any(lname, rules["roles"][role].get("keywords", []))
    ]

    for role in ROLES:
        r = rules["roles"][role]
        if not allowed(role):
            continue
        in_category = any(part in r.get("categories", []) for part in path_parts)
        by_keyword = _matches_any(lname, r.get("keywords", []))

        if role in perc_roles:
            # The filename says WHICH drum; the folder only says THAT it is a
            # drum. So a specific name wins outright — `Kick 909ish` must not
            # also become a hi-hat candidate just for sitting in Percussion/.
            # Only when the name claims nothing ("Perc 7", "Tom L") does folder
            # membership admit it to all three, because an unlabelled drum is a
            # useful neighbour on any drum tour and we cannot tell which.
            #
            # Previously ONLY the keyword counted (the folder guard was a
            # tautology — verified over 8980 evaluations, it never changed an
            # outcome), so the pools were pure filename matches: 51/60/61
            # candidates against the 239 files in drum folders.
            if named_perc:
                if role in named_perc:
                    assigned.append(role)
            elif in_category:
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
                    # third-party: author/SUBCATEGORY/patch — factory: the
                    # category folder is already parts[0]
                    subcategory=parts[-1] if parts else "",
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
