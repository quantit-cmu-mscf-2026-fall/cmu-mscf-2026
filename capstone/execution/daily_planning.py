"""Shared daily planning choice; every trade still uses the existing share planner."""

from __future__ import annotations

import pandas as pd

from capstone.execution.benchmark_runtime import benchmark_due, benchmark_plan
from capstone.execution.contracts import (
    BENCHMARK_VERSIONS,
    WINDOW_START_VERSIONS,
    ExecutionContractBlocked,
    execution_leaves,
    execution_policy,
    program_positions,
    validate_program_instruments,
)
from capstone.execution.exit_planning import program_exit_plan
from capstone.execution.planning import share_plan
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


def replay_actions(actions, day, positions, cash, program_route, opens=None):
    try:
        return actions.apply(day, positions, cash, opens)
    except ValueError as exc:
        if program_route:
            raise ExecutionContractBlocked(
                "program_corporate_action_transition_unqualified", str(exc)
            ) from exc
        raise
