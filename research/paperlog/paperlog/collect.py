"""Volume-targeted collection.

A run aims for a target number of *new* papers to screen rather than a fixed
date window. It starts with the normal window and escalates only if it falls
short, cheapest step first. Everything fetched is deduplicated against the log
before it counts, so widening never re-screens what you already have.

Levels
  0  normal window
  1+ window doubled, up to max_lookback_days
  then depth: more pages per query
  then exploration: extra broad queries and a looser prefilter
  (the run loop then falls back to backlog and rescreening, which fetch nothing)
"""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class Plan:
    level: int
    since: str
    depth: int = 1
    exploration: bool = False
    loosen_prefilter: bool = False

    def describe(self) -> str:
        bits = [f"since {self.since}"]
        if self.depth > 1:
            bits.append(f"depth x{self.depth}")
        if self.exploration:
            bits.append("exploration queries")
        if self.loosen_prefilter:
            bits.append("loose prefilter")
        return ", ".join(bits)


def base_since(con, cfg, since_arg=None, days_arg=None) -> str:
    if since_arg:
        return since_arg
    if days_arg:
        return (date.today() - timedelta(days=days_arg)).isoformat()
    last = con.execute(
        "SELECT started FROM runs WHERE finished IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if last:
        d = date.fromisoformat(last["started"][:10]) - timedelta(days=cfg.get("overlap_days", 7))
        return d.isoformat()
    return (date.today() - timedelta(days=cfg.get("lookback_days_first_run", 60))).isoformat()


def ladder(cfg: dict, start_since: str) -> list[Plan]:
    """The escalation sequence. Each step is tried only if the previous fell short."""
    max_days = cfg.get("max_lookback_days", 365)
    floor = (date.today() - timedelta(days=max_days)).isoformat()
    plans = [Plan(0, start_since)]

    since = start_since
    level = 1
    while since > floor and level <= cfg.get("max_window_doublings", 4):
        span = max((date.today() - date.fromisoformat(since)).days, 7)
        since = max((date.today() - timedelta(days=span * 2)).isoformat(), floor)
        plans.append(Plan(level, since))
        level += 1

    plans.append(Plan(level, since, depth=2))
    plans.append(Plan(level + 1, since, depth=3, exploration=True))
    plans.append(Plan(level + 2, since, depth=3, exploration=True, loosen_prefilter=True))
    return plans
