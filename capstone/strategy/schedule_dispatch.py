"""Explicit formation dispatch without modifying hash-bound legacy constructors.

These are formation decisions, plus the exit decisions of a version-2 calendar
window (its schedule's own `decision_due`; the exit target is cash). Membership
exits require the shared exit-only fixed-share path, not a fabricated daily
rebalance schedule.
"""

from __future__ import annotations

import pandas as pd

from capstone.strategy import input_memo


def decision_due(spec, panel, as_of):
    """Memoized per (spec, panel inputs, session) inside an evaluation scope (B3)."""
    return input_memo.panel_cached(
        "decision_due", spec, panel, (as_of,), lambda: _due(spec, panel, as_of)
    )


def _due(spec, panel, as_of):
    from capstone.strategy.compiled import CompiledStrategySpec
    from capstone.strategy.compiled import decision_due as compiled_due
    from capstone.strategy.contract import StrategySpec
    from capstone.strategy.diversified import DiversifiedStrategySpec
    from capstone.strategy.diversified import decision_due as diversified_due
    from capstone.strategy.program_schedule import session_date
    from capstone.strategy.recorded import require_current_implementation
    from capstone.strategy.reference import BenchmarkSpec, ReferenceStrategySpec
    from capstone.strategy.reference import decision_due as reference_due
    from capstone.strategy.research_spec import ResearchStrategySpec

    require_current_implementation(spec)
    at = session_date(as_of, "decision")
    if at not in panel.universe.index:
        raise ValueError("formation decision must be an observed session")
    if panel.calendar is not None and at not in pd.DatetimeIndex(panel.calendar):
        raise ValueError("formation decision is absent from the authoritative calendar")
    leaf_types = (
        CompiledStrategySpec,
        DiversifiedStrategySpec,
        StrategySpec,
        ReferenceStrategySpec,
        ResearchStrategySpec,
        BenchmarkSpec,
    )
    if isinstance(spec, leaf_types):
        if at < session_date(spec.anchor, "anchor"):
            raise ValueError("formation decision must be on or after anchor")
    if isinstance(spec, CompiledStrategySpec):
        return compiled_due(spec, panel, at)
    if isinstance(spec, DiversifiedStrategySpec):
        return diversified_due(spec, panel, at)
    if isinstance(spec, StrategySpec):
        dates = pd.DatetimeIndex(panel.universe.index)
        anchor = session_date(spec.anchor, "anchor")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(
                "legacy anchored calendar requires unique increasing observation sessions"
            )
        if anchor not in dates or at not in dates or at < anchor:
            raise ValueError("legacy calendar must contain anchor and decision on or after anchor")
        return (dates.get_loc(at) - dates.get_loc(anchor)) % spec.trading.hold == 0
    if isinstance(spec, (ReferenceStrategySpec, ResearchStrategySpec, BenchmarkSpec)):
        return reference_due(spec, panel, at)
    raise ValueError(f"unsupported strategy schedule type: {type(spec).__name__}")
