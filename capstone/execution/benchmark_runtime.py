"""Strict historical benchmark inputs; sizing remains the shared share planner."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

import numpy as np
import pandas as pd

from capstone.execution.planning import share_plan
from capstone.factors.evaluation import factor_signal
from capstone.factors.tree import parse
from capstone.strategy import input_memo
from capstone.strategy.historical_benchmark import HistoricalBenchmarkSpec
from capstone.strategy.program import FormationEligibilityError, eligible_members
from capstone.strategy.program_schedule import ProgramSchedule, session_calendar, session_date
from capstone.strategy.reference import benchmark_target, decision_due
from capstone.strategy.research import digest


def validate_benchmark(spec, panel):
    input_memo.panel_cached(
        "validate_benchmark", spec, panel, (), lambda: _validate_benchmark(spec, panel)
    )


def _validate_benchmark(spec, panel):
    if type(spec) is not HistoricalBenchmarkSpec or spec.data_version != panel.data_version:
        raise ValueError("historical benchmark requires its exact type and frozen data version")
    columns = panel.universe.columns
    if columns.has_duplicates or any(
        isinstance(name, bool) or not isinstance(name, Integral) or name <= 0 for name in columns
    ):
        raise ValueError("historical benchmark instruments require positive integer identities")


def benchmark_due(spec, panel, as_of):
    """Memoized per (spec, panel inputs, session) inside an evaluation scope."""
    return input_memo.panel_cached(
        "benchmark_due", spec, panel, (as_of,), lambda: _due(spec, panel, as_of)
    )


def _due(spec, panel, as_of):
    validate_benchmark(spec, panel)
    at = session_date(as_of)
    dates = session_calendar(panel.calendar, spec.anchor, at)
    observed = panel.universe.index
    required = dates[(dates >= pd.Timestamp(spec.anchor)) & (dates <= at)]
    actual = observed.sort_values()
    actual = actual[(actual >= pd.Timestamp(spec.anchor)) & (actual <= at)]
    if observed.has_duplicates or not actual.equals(required):
        raise ValueError("benchmark observation sessions differ from the authoritative calendar")
    ordered = panel.universe if observed.is_monotonic_increasing else panel.universe.sort_index()
    return decision_due(spec.legacy, replace(panel, universe=ordered, calendar=dates), at)


def _audit(spec, at, members, excluded):
    evidence = {
        "benchmark_id": spec.id,
        "formation": str(at.date()),
        "eligible": [int(name) for name in members],
        "excluded": excluded,
        "policy": spec.eligibility_rule,
        "formal_admission": False,
    }
    evidence["id"] = digest(evidence)
    return evidence


def _availability(panel, name, dates, decision):
    supplied = (panel.availability or {}).get(name)
    if not isinstance(supplied, pd.Series) or supplied.index.has_duplicates:
        raise ValueError("benchmark feature availability is missing or ambiguous: " + name)
    available = pd.to_datetime(supplied.reindex(dates))
    if available.isna().any() or any(
        day.tz is not None or day != day.normalize() or day > decision for day in available
    ):
        raise ValueError("benchmark feature availability is unqualified: " + name)


def benchmark_weights(spec, panel, formation):
    """Existing arithmetic with audited required-history eligibility at formation.

    The known baseline sums231 returns ending21 sessions before formation. Its
    252-session span is not an exact paper replication. Unused recent21 returns
    do not authorize replacing a missing required observation.
    """
    if not benchmark_due(spec, panel, formation):
        raise ValueError("benchmark targets require an actual scheduled formation")
    at = session_date(formation)
    membership = panel.universe.loc[at].sort_index()
    if membership.isna().any() or not membership.isin([True, False]).all():
        raise ValueError("benchmark formation membership is missing")
    members = membership.index[membership.astype(bool)]
    fields, excluded = {}, {}
    universe = membership.to_frame().T
    if spec.version == 2 and spec.kind != "cash":
        # Data-property eligibility shared with programs: identical in every cost scenario.
        members, rows = eligible_members(
            panel, at, need_cap=spec.kind == "cap_weight_universe_proxy"
        )
        for row in rows:
            excluded.setdefault(str(row["name"]), row["reason"])
        universe = membership.index.isin(members)
        universe = pd.DataFrame([universe], index=[at], columns=membership.index)
    if spec.kind != "cash" and not len(members):
        message = "historical benchmark has no eligible formation members"
        if spec.version == 1:
            raise ValueError(message)
        raise FormationEligibilityError(message, _audit(spec, at, members, excluded))
    availability = {}
    if spec.kind == "cap_weight_universe_proxy":
        frame = panel.fields.get("cap")
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.index.has_duplicates
            or frame.columns.has_duplicates
        ):
            raise ValueError("benchmark capitalization field is missing or ambiguous")
        _availability(panel, "cap", pd.DatetimeIndex([at]), at)
        row = frame.reindex(index=[at], columns=members)
        if (
            not np.isfinite(row.to_numpy()).all()
            or row.le(0).any().any()
            or not np.isfinite(row.sum(axis=1).iloc[0])
        ):
            raise ValueError("benchmark capitalization must be finite positive point-in-time USD")
        fields["cap"] = frame.loc[[at]].reindex(columns=universe.columns).copy()
        availability["cap"] = panel.availability["cap"].loc[[at]]
    elif spec.kind == "known_momentum_12_1":
        calendar = pd.DatetimeIndex(panel.calendar).sort_values()
        offset = calendar.get_loc(at)
        if offset < 251:
            raise ValueError("known momentum needs its complete252-session formation span")
        span = calendar[offset - 251 : offset + 1]
        required = span[:231]
        universe = panel.universe.reindex(index=span, columns=membership.index).copy()
        frame = panel.fields.get("returns")
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.index.has_duplicates
            or frame.columns.has_duplicates
            or not span.isin(frame.index).all()
        ):
            raise ValueError("known momentum is missing required global return sessions")
        _availability(panel, "returns", required, at)
        values = frame.reindex(index=required, columns=members)
        observed = values.to_numpy(dtype=float)
        if np.isinf(observed).any() or np.nanmin(observed, initial=0) < -1:
            raise ValueError("known momentum return observations have invalid domains")
        complete = values.notna().all(axis=0)
        excluded.update(
            {
                str(name): "required_formation_return_history_incomplete"
                for name in members[~complete]
            }
        )
        members = members[complete]
        if len(members) < 2:
            raise FormationEligibilityError(
                "known momentum requires at least two complete-history members",
                _audit(spec, at, members, excluded),
            )
        universe.loc[at] = False
        universe.loc[at, members] = True
        fields["returns"] = frame.reindex(index=span, columns=universe.columns).copy()
        availability["returns"] = panel.availability["returns"].reindex(span)
    elif spec.kind != "cash" and not len(members):
        raise ValueError("universe proxy has no eligible formation members")
    qualified = replace(
        panel,
        fields=fields,
        universe=universe,
        availability=availability,
        calendar=pd.DatetimeIndex(panel.calendar).sort_values(),
        target_cache=None,
    )
    if spec.kind == "known_momentum_12_1":
        score = factor_signal(parse("shift(ts_sum(returns, 231), 21)"), qualified).loc[at, members]
        with np.errstate(over="ignore", invalid="ignore"):
            centered = score - score.mean()
            finite = np.isfinite(score.to_numpy()).all() and np.isfinite(centered.to_numpy()).all()
            finite = finite and np.isfinite(centered.abs().sum())
        if not finite:
            raise FormationEligibilityError(
                "known momentum has an undefined sum or weight-conversion domain",
                _audit(spec, at, members, excluded),
            )
    book = benchmark_target(spec.legacy, qualified, at)
    book = book[book != 0].sort_index().rename("weight")
    if not np.isfinite(book.to_numpy()).all() or book.abs().sum() > 1 + 1e-10:
        raise FormationEligibilityError(
            "historical benchmark produced invalid weights", _audit(spec, at, members, excluded)
        )
    audit = _audit(spec, at, members, excluded)
    book.attrs["formation_eligibility"] = {spec.id: audit}
    return book


def execution_target(spec, panel, as_of, version):
    from capstone.execution.contracts import BENCHMARK_VERSIONS

    if version not in BENCHMARK_VERSIONS:
        raise ValueError("historical benchmark targets require explicit execution v5")
    return benchmark_weights(spec, panel, as_of)


def benchmark_plan(
    spec, panel, as_of, current, prices, capital, *, weights, min_notional, fee_bps, carry=None
):
    """`carry` (v6 only) is share_plan's held carry / fallback-exit state; v5 passes None."""
    if weights is not None:
        plan = share_plan(
            weights,
            current,
            prices,
            capital,
            min_notional=min_notional,
            side=spec.side,
            fee_bps=fee_bps,
            carry=carry,
        )
        plan["intent"] = "formation_rebalance"
        return plan
    dates = pd.DatetimeIndex(panel.calendar).sort_values()
    # Membership exits are sticky; re-entry cannot resurrect a pre-exit position.
    clock = (
        ProgramSchedule(interval=21)
        if spec.kind == "known_momentum_12_1"
        else ProgramSchedule(kind="month_end", interval=1)
    )
    candidates = clock.formations(dates, spec.anchor, as_of)
    formation = candidates[-1] if len(candidates) else None
    allowed = set()
    if formation is not None:
        book = benchmark_weights(spec, panel, formation)
        history = panel.universe.reindex(
            index=dates[(dates >= formation) & (dates <= as_of)], columns=book.index
        )
        if history.isna().any().any() or not history.isin([True, False]).all().all():
            raise ValueError("benchmark membership history is missing")
        allowed = set(book.index[history.astype(bool).all(axis=0)])
    exiting = {name: quantity for name, quantity in current.items() if name not in allowed}
    plan = share_plan(
        pd.Series(dtype=float),
        exiting,
        prices,
        capital,
        min_notional=min_notional,
        side=spec.side,
        fee_bps=fee_bps,
        carry=carry,
    )
    survivors = {name: quantity for name, quantity in current.items() if name in allowed}
    for key in ("targets", "projected"):
        plan[key] = dict(sorted((survivors | plan[key]).items()))
    plan["intent"] = "membership_exit_only"
    return plan
