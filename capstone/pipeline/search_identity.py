"""Stable content identity for a search configuration, independent of runtime caps.

`config_digest` is what a search records in its manifest and what the single holdout
look checks: the same config gives the same digest in any checkout, and a config that
differs in anything that can change a result gives another.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

from capstone.funnel import thresholds_dict
from capstone.pipeline.config import E2EConfig

# E2EConfig fields added after configs were first hashed: at their default they stay
# out of the digest, so existing configs keep theirs.
_OPTIONAL_FIELDS = ("deepen_on_budget", "paper_mode", "placebo", "placebo_seed")
_OPTIONAL_FIELDS += ("regime_operators",)  # the market-regime operators
_OPTIONAL_FIELDS += ("deepen_max_children_per_family",)  # the per-family deepen cap
_RUNTIME_FIELDS = ("concurrency", "resources")  # never change a result: never in the digest
_DEFAULTS = {f.name: f.default for f in fields(E2EConfig) if f.name in _OPTIONAL_FIELDS}

__all__ = ["config_digest"]


def config_digest(config: E2EConfig) -> str:
    """SHA-256 of the config's content (every field, nested ones included;
    thresholds as a gate config registers them). The paper file enters by its
    content, not its absolute path, so a moved or re-checked-out repo hashes the
    same and an edited paper list does not."""
    papers = hashlib.sha256(Path(config.papers).read_bytes()).hexdigest()
    content = asdict(config) | {
        "thresholds": thresholds_dict(config.thresholds),
        "papers": f"sha256:{papers}",
    }
    for name in _OPTIONAL_FIELDS:  # newer fields enter only when set, as thresholds' do
        if content[name] == _DEFAULTS[name]:
            del content[name]
    content = {k: v for k, v in content.items() if k not in _RUNTIME_FIELDS}
    text = json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
