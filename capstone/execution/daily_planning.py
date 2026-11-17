"""Shared daily planning choice; every trade still uses the existing share planner."""

from __future__ import annotations

import math
from copy import deepcopy

import pandas as pd

from capstone.execution.benchmark_runtime import benchmark_due, benchmark_plan, execution_target
from capstone.execution.contracts import (
    BENCHMARK_VERSIONS,
    HELD_ACTION_VERSIONS,
    WINDOW_START_VERSIONS,
    ExecutionContractBlocked,
    execution_leaves,
    execution_policy,
    program_positions,
    validate_program_instruments,
)
from capstone.execution.exit_planning import program_exit_plan
from capstone.execution.planning import positive_price, share_plan
from capstone.execution.replay_state import EVENT_VERSIONS, freeze_intent
from capstone.strategy.contract import target_weights
from capstone.strategy.historical_benchmark import HistoricalBenchmarkSpec
from capstone.strategy.portfolio import PortfolioSpec
from capstone.strategy.program import ProgramStrategySpec, program_weights
from capstone.strategy.program_schedule import CalendarWindowSchedule
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec
from capstone.strategy.schedule_dispatch import decision_due
from capstone.strategy.selected_portfolio import SelectedPortfolioSpec


def active_programs(specs, as_of):
    roots = tuple(specs) if isinstance(specs, (tuple, list)) else (specs,)
    programs = []
    for spec in roots:
        if isinstance(spec, ProgramStrategySpec):
            programs.append(spec)
        elif isinstance(spec, SelectedPortfolioSpec):
            for member in spec.members(as_of):
                programs.extend(active_programs(member, as_of))
        elif isinstance(spec, PortfolioSpec):
            for member in spec.specs:
                programs.extend(active_programs(member, as_of))
        else:
            raise ExecutionContractBlocked("unsupported_program_exit_member")
    return tuple(programs)


def calendar_window_exit(specs, panel, as_of):
    """True on a version-2 calendar window's exit decision; its target book is cash."""
    at = pd.Timestamp(as_of)
    return any(
        spec.exit_rule == "calendar_window_exit"
        and at in spec.schedule.exits(panel.calendar, spec.anchor, at)
        for spec in active_programs(specs, at)
    )


def plan_completed_session(
    specs,
    panel,
    as_of,
    current,
    prices,
    capital,
    *,
    weights=None,
    min_notional=100.0,
    fee_bps=0.0,
    allocation_changed=False,
    execution_version=None,
    carry=None,
):
    """Never infer an allocation change from price/capital drift or fresh optimization."""
    if execution_version in BENCHMARK_VERSIONS:
        return benchmark_plan(
            specs,
            panel,
            as_of,
            current,
            prices,
            capital,
            weights=weights,
            min_notional=min_notional,
            fee_bps=fee_bps,
            carry=carry,
        )
    contract = execution_policy(specs)
    validate_program_instruments(specs, panel)
    if contract["route"] == "program-v1":
        current = program_positions(current)
    roots = tuple(specs) if isinstance(specs, (tuple, list)) else (specs,)
    scheduled = [decision_due(spec, panel, as_of) for spec in roots]
    if type(allocation_changed) is not bool:
        raise ExecutionContractBlocked("allocation_transition_contract_invalid")
    if any(scheduled) or allocation_changed:
        if weights is None:
            raise ExecutionContractBlocked("formation_target_book_missing")
        plan = share_plan(
            weights,
            current,
            prices,
            capital,
            min_notional=min_notional,
            side=contract["profile"],
            fee_bps=fee_bps,
            carry=carry,
        )
        plan["intent"] = "allocation_rebalance" if allocation_changed else "formation_rebalance"
        if (
            not allocation_changed
            and contract["route"] == "program-v1"
            and calendar_window_exit(specs, panel, as_of)
        ):
            plan["intent"] = "calendar_window_exit"
        return plan
    if contract["route"] == "program-v1":
        members = active_programs(specs, as_of)
        if not members:
            if len(roots) == 1 and isinstance(roots[0], SavedProgramPortfolioSpec):
                plan = share_plan(
                    pd.Series(dtype=float),
                    current,
                    prices,
                    capital,
                    min_notional=min_notional,
                    side=contract["profile"],
                    fee_bps=fee_bps,
                    carry=carry,
                )
                plan["intent"] = "allocated_cash_exit_only"
                return plan
            if any(current.values()):
                raise ExecutionContractBlocked("cash_selection_account_state_unresolved")
            return {
                "targets": {},
                "projected": {},
                "orders": [],
                "omitted": [],
                "sizing_capital": capital,
                "estimated_fees": 0.0,
                "intent": "hold_cash",
            }
        plan = program_exit_plan(
            members,
            panel,
            as_of,
            current,
            prices,
            capital,
            min_notional=min_notional,
            fee_bps=fee_bps,
            carry=carry,
        )
        plan["intent"] = "membership_exit_only"
        return plan
    return {
        "targets": dict(current),
        "projected": dict(current),
        "orders": [],
        "omitted": [],
        "sizing_capital": capital,
        "estimated_fees": 0.0,
        "intent": "hold_fixed_shares",
    }


def lifecycle_plan(
    specs,
    panel,
    as_of,
    current,
    prices,
    capital,
    *,
    weights,
    attribution,
    states,
    min_notional=100.0,
    fee_bps=0.0,
    carry=None,
):
    """(target book, plan) with registry lifecycle states applied.

    `states` maps a program id, or a saved portfolio's id standing for all its members, to
    "retired" or "suspended"; `attribution` is the plan's per-strategy target book. Every
    trade still goes through the shared planners:

    - Held: current names that a suspended member's standing book (its book at the session
      before `as_of`) holds and no active member's book does keep their shares; they are
      never sold until the member is resumed (ordinary planning again) or retired.
    - On a decision the target book is the active members' attribution alone: a suspended
      or retired coefficient goes to cash and is never handed to the others; the ordinary
      planner rebalances everything not held, so a retired member's names close.
    - Between decisions `program_exit_plan` over the allocated active members closes the
      names none of them holds (a retired member's) and keeps every surviving share.
    """
    roots = tuple(specs) if isinstance(specs, (tuple, list)) else (specs,)
    if execution_policy(roots)["route"] != "program-v1":
        raise ExecutionContractBlocked("lifecycle_states_require_program_route")
    state = {}
    for root in roots:
        for member in root.specs if isinstance(root, SavedProgramPortfolioSpec) else (root,):
            if not isinstance(member, ProgramStrategySpec):
                raise ExecutionContractBlocked("lifecycle_states_require_program_route")
            state[member.id] = (member, states.get(member.id, states.get(root.id, "active")))
    at = pd.Timestamp(as_of)
    calendar = pd.DatetimeIndex(panel.calendar)
    prior = calendar[calendar < at]
    standing = set()
    for member, status in state.values():
        if status == "suspended" and len(prior) and prior[-1] >= pd.Timestamp(member.anchor):
            book = program_weights(member, panel, prior[-1])
            standing |= {int(n) for n, w in book.items() if w}
    current = program_positions(current)
    decision = any(decision_due(root, panel, at) for root in roots)
    if decision:
        parts = [
            pd.Series(attribution[m.id], dtype=float)
            for m, status in state.values()
            if status == "active" and m.id in attribution
        ]
        weights = pd.concat(parts).groupby(level=0).sum() if parts else pd.Series(dtype=float)
        weights = weights[weights != 0].sort_index().rename("weight")
        weights.index = weights.index.astype("int64").rename("permno")
        allowed = set(weights.index)
    else:
        survivors = [m for m in active_programs(roots, at) if state[m.id][1] == "active"]
        allowed = {int(n) for m in survivors for n, w in program_weights(m, panel, at).items() if w}
    held = {n: q for n, q in current.items() if n in standing and n not in allowed}
    rest = {n: q for n, q in current.items() if n not in held}
    options = {"min_notional": min_notional, "fee_bps": fee_bps, "carry": carry}
    if decision:
        plan = plan_completed_session(
            roots, panel, at, rest, prices, capital, weights=weights, **options
        )
    elif survivors:
        plan = program_exit_plan(survivors, panel, at, rest, prices, capital, **options)
    else:
        side = execution_policy(roots)["profile"]
        plan = share_plan(pd.Series(dtype=float), rest, prices, capital, side=side, **options)
    if not decision:
        plan["intent"] = "lifecycle_exit_only"
    for key in ("targets", "projected"):
        plan[key] = dict(sorted((plan[key] | held).items()))
    return weights, plan


def program_initialization_due(spec, panel, prior):
    if type(spec) is HistoricalBenchmarkSpec:
        return prior >= pd.Timestamp(spec.anchor) and benchmark_due(spec, panel, prior)
    return all(
        prior >= pd.Timestamp(leaf.anchor) for leaf in execution_leaves(spec)
    ) and decision_due(spec, panel, prior)


def window_start_due(spec, panel, prior, version):
    """(contracts.WINDOW_START) A fresh calendar-window Program account whose prior
    session is not a window decision still qualifies; it holds cash until the schedule's
    next decision. Only the known calendar decides this, never data after `prior`."""
    return (
        version in WINDOW_START_VERSIONS
        and type(spec) is ProgramStrategySpec
        and isinstance(spec.schedule, CalendarWindowSchedule)
        and prior >= pd.Timestamp(spec.anchor)
    )


def initial_replay_plan(inputs, prior, rates):
    spec, panel = inputs["spec"], inputs["panel"]
    if prior is None or (
        inputs["execution"]["route"] == "program-v1"
        and not program_initialization_due(spec, panel, prior)
    ):
        return None
    if prior not in inputs["closes"].index:
        raise ValueError("initial investment requires the previous completed-session prices")
    plan = share_plan(
        target_weights(spec, panel, prior),
        {},
        inputs["closes"].loc[prior].to_dict(),
        inputs["capital"],
        min_notional=inputs["min_notional"],
        side=getattr(spec, "side", "long_short"),
        fee_bps=rates(prior),
    )
    plan["decision"] = str(prior.date())
    return plan


def program_replay_plan(
    spec,
    panel,
    as_of,
    positions,
    prices,
    capital,
    *,
    min_notional,
    fee_bps,
    terminated,
    recovery_requested=False,
    execution_version=None,
    source_intent=None,
    cash=None,
    receivables=(),
    carry=None,
):
    event_protocol = execution_version in EVENT_VERSIONS
    if recovery_requested and not event_protocol:
        raise ExecutionContractBlocked("program_pending_intent_recovery_unqualified")
    if execution_version in HELD_ACTION_VERSIONS and calendar_unconfirmed(spec, panel, as_of):
        # v5: the known calendar (one session past 2020) cannot confirm the next month
        # end, so no window decision is due; hold for a session the replay never reaches.
        plan = {
            "targets": dict(positions),
            "projected": dict(positions),
            "orders": [],
            "omitted": [],
            "sizing_capital": capital,
            "estimated_fees": 0.0,
            "intent": "calendar_unconfirmed_hold",
            "decision": str(pd.Timestamp(as_of).date()),
        }
        source = {"kind": "exit_only", "strategy_id": spec.id, "decision": plan["decision"]}
        return freeze_intent(plan, positions, cash, receivables, decision=as_of, source=source)
    due = (
        benchmark_due(spec, panel, as_of)
        if execution_version in BENCHMARK_VERSIONS
        else decision_due(spec, panel, as_of)
    )
    book = None
    if due:
        book = (
            execution_target(spec, panel, as_of, execution_version)
            if execution_version in BENCHMARK_VERSIONS
            else target_weights(spec, panel, as_of)
        )
    deferred = (
        not due
        and event_protocol
        and source_intent is not None
        and source_intent["kind"] == "formation"
    )
    excluded = []
    if deferred:
        book = sticky_source_book(source_intent, spec, panel, as_of)
        if execution_version in HELD_ACTION_VERSIONS:
            book, excluded = requalified_entries(book, positions, panel, as_of, prices, terminated)
    if book is not None and any(book.get(name, 0) != 0 for name in terminated):
        raise ExecutionContractBlocked("retired_instrument_in_program_targets")
    if deferred:
        plan = share_plan(
            book,
            positions,
            prices,
            capital,
            min_notional=min_notional,
            side=spec.side,
            fee_bps=fee_bps,
            carry=carry,
        )
        plan["intent"] = "deferred_formation_rebalance"
        plan["omitted"] += excluded
    else:
        plan = plan_completed_session(
            spec,
            panel,
            as_of,
            positions,
            prices,
            capital,
            weights=book,
            min_notional=min_notional,
            fee_bps=fee_bps,
            execution_version=execution_version,
            carry=carry,
        )
    plan["decision"] = str(pd.Timestamp(as_of).date())
    if event_protocol:
        if due:
            if (
                source_intent is not None
                and source_intent["kind"] == "formation"
                and source_intent["formation_session"] != plan["decision"]
            ):
                plan["superseded_source_intent_id"] = source_intent["source_intent_id"]
            source_intent = {
                "kind": "formation",
                "strategy_id": spec.id,
                "formation_session": plan["decision"],
                "book": {str(n): float(w) for n, w in book.items()},
                "construction_evidence": deepcopy(book.attrs),
            }
        elif source_intent is None:
            source_intent = {
                "kind": "exit_only",
                "strategy_id": spec.id,
                "decision": plan["decision"],
            }
        if source_intent["kind"] == "formation":
            plan["construction_evidence"] = deepcopy(source_intent.get("construction_evidence", {}))
        return freeze_intent(
            plan, positions, cash, receivables, decision=as_of, source=source_intent
        )
    return plan


def sticky_source_book(source, spec, panel, as_of):
    """A deferred formation retains its frozen selected book; only actual exits apply."""
    if source["kind"] != "formation" or source["strategy_id"] != spec.id:
        raise ExecutionContractBlocked("deferred_formation_source_incompatible")
    formed, at = pd.Timestamp(source["formation_session"]), pd.Timestamp(as_of)
    due = (
        benchmark_due(spec, panel, formed)
        if type(spec) is HistoricalBenchmarkSpec
        else decision_due(spec, panel, formed)
    )
    if formed > at or not due:
        raise ExecutionContractBlocked("deferred_formation_source_incompatible")
    book = pd.Series({int(n): float(w) for n, w in source["book"].items()})
    if not book.index.is_unique or not all(pd.notna(book)) or book.abs().sum() > 1 + 1e-10:
        raise ExecutionContractBlocked("deferred_formation_book_invalid")
    dates = pd.DatetimeIndex(panel.calendar)
    held = dates[(dates >= formed) & (dates <= at)]
    if not held.isin(panel.universe.index).all():
        raise ExecutionContractBlocked("deferred_formation_membership_missing")
    membership = panel.universe.reindex(index=held, columns=book.index)
    if membership.isna().any().any() or not membership.isin([True, False]).all().all():
        raise ExecutionContractBlocked("deferred_formation_membership_missing")
    return book.where(membership.astype(bool).all(axis=0), 0).rename("weight")


def calendar_unconfirmed(spec, panel, as_of):
    """True when a calendar window's schedule cannot be evaluated at `as_of` because the
    known calendar ends before the month end an entry would precede (mirrors
    CalendarWindowSchedule._windows, which raises there)."""
    schedule = getattr(spec, "schedule", None)
    if not isinstance(schedule, CalendarWindowSchedule):
        return False
    after = int((pd.DatetimeIndex(panel.calendar) > pd.Timestamp(as_of)).sum())
    return after <= max(0, -schedule.enter_offset)


def requalified_entries(book, positions, panel, as_of, prices, terminated=()):
    """v5/v7: a deferred formation re-planned on a later session keeps its frozen book,
    but an unheld name that has since been retired by a corporate action, or lost its
    qualified quote or sizing price, is not entered (data properties, so every cost
    scenario agrees); held names keep v4/v6 carry and fallback exits."""
    at = pd.Timestamp(as_of)
    spreads = panel.half_spread.loc[at] if at in panel.half_spread.index else pd.Series(dtype=float)
    book, excluded = book.copy(), []
    for name, weight in book.items():
        if not weight or positions.get(int(name)):
            continue
        spread = float(spreads.get(name, math.nan))
        if int(name) in terminated:
            reason = "retired_instrument"
        elif not math.isfinite(spread) or spread < 0:
            reason = "missing_qualified_quote"
        elif not positive_price(prices.get(name)):
            reason = "nonpositive_sizing_price"
        else:
            continue
        book[name] = 0.0
        excluded.append(
            {"permno": int(name), "shares": 0, "reason": "deferred_entry_unqualified:" + reason}
        )
    return book, excluded


def replay_actions(actions, day, positions, cash, program_route, opens=None):
    try:
        return actions.apply(day, positions, cash, opens)
    except ValueError as exc:
        if program_route:
            raise ExecutionContractBlocked(
                "program_corporate_action_transition_unqualified", str(exc)
            ) from exc
        raise
