"""The frozen comparison rule: which saved portfolio a bundle and the portfolio gate use.

A pure function of the frozen `[portfolios]` section and the pooled study's report; it
never ranks comparisons by results (the study registers `winner_selection: false`).
Rules: `named` (an id, a `profile:mode` pair or a mode) and `first_full_pool` (the first
evaluated full-pool comparison in registered order, optionally of one profile).
"""

from __future__ import annotations

from pathlib import Path

RULES = ("named", "first_full_pool")
FULL_POOL = "full_pool"
SPEC = "portfolio-spec.toml"


class RuleError(ValueError):
    """The frozen section does not name a usable rule (a configuration refusal)."""


def _norm(text):
    return str(text).strip().lower().replace("-", "_")


def _named(rows, comparison):
    if not comparison:
        raise RuleError("rule `named` needs `[portfolios] comparison`")
    by_id = [row for row in rows if row["id"] == comparison]
    if by_id:
        return by_id
    wanted = _norm(comparison)
    full = [row for row in rows if row.get("kind") == FULL_POOL]
    if ":" in wanted:
        profile, _, mode = wanted.partition(":")
        return [r for r in full if r.get("profile") == profile and r.get("mode") == mode]
    return [r for r in full if r.get("mode") == wanted]


def _first_full_pool(rows, comparison):
    profile = _norm(comparison) if comparison else None
    full = [
        row
        for row in rows
        if row.get("kind") == FULL_POOL
        and row.get("status") == "evaluated"
        and (profile is None or row.get("profile") == profile)
    ]
    return full[:1]


def choose(rule, comparison, report, root):
    """(chosen row or None, reason). `reason` says why nothing matched (a recorded outcome)."""
    if rule not in RULES:
        raise RuleError(f"unknown comparison_rule {rule!r}; one of {', '.join(RULES)}")
    rows = list(report.get("comparisons", []))
    found = _named(rows, comparison) if rule == "named" else _first_full_pool(rows, comparison)
    if not found:
        return None, f"rule {rule!r} ({comparison!r}) matches no registered comparison"
    if len(found) > 1:
        ids = ", ".join(sorted(row["id"] for row in found))
        return None, f"rule {rule!r} ({comparison!r}) is ambiguous: {ids}"
    row = found[0]
    if row.get("status") != "evaluated":
        return None, f"comparison {row['id']} is {row.get('status')!r}, not evaluated"
    if not (Path(root) / "comparisons" / row["id"] / SPEC).is_file():
        return None, f"comparison {row['id']} has no saved {SPEC}"
    return row, None
