"""Daily fixed-share replay with next-session fills and explicit cash accounting."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import pandas as pd

from capstone import runlog as runlog
from capstone.execution.contracts import (
    ExecutionContractBlocked,
    execution_contract,
    validate_program_instruments,
)
from capstone.execution.daily_planning import program_replay_plan, replay_actions
from capstone.execution.planning import execution_risk, share_plan
from capstone.execution.replay_accounting import (
    ReplayCosts,
    carried_close,
    carried_open,
    fallback_exits,
    fill_orders,
    held_carry,
    legacy_actions,
    mark_session,
    open_gaps,
    order_rates,
)
from capstone.execution.replay_identity import completed_replay_metadata, register_replay
from capstone.execution.replay_state import (
    EVENT_VERSIONS,
    account_state_id,
    checked_checkpoint,
    final_event_metadata,
    initial_account,
    replay_start_context,
    retain_intent_evidence,
)
from capstone.factors.reference import sessions
from capstone.pipeline.actions import ActionBook, action_rules
from capstone.strategy import input_memo
from capstone.strategy.contract import target_weights
from capstone.strategy.research import digest
from capstone.strategy.schedule_dispatch import decision_due


def replay(
    spec,
    panel,
    opens,
    closes,
    *,
    start,
    end,
    capital,
    costs: ReplayCosts,
    splits=None,
    dividends=None,
    cash_settlements=None,
    min_notional=100.0,
    events=None,
    cost_schedule=None,
    on_completed_session=None,
    execution_version="fixed-share-next-open-v2",
    checkpoint=None,
):
    """Fixed-share replay; opt-in v3 uses observed events and verified account checkpoints."""
    if pd.Timestamp(end) > pd.Timestamp("2020-12-31"):
        raise ValueError("historical research replay cannot open the sealed holdout")
    execution = execution_contract(spec, version=execution_version)
    event_protocol = execution_version in EVENT_VERSIONS
    checkpoint = checked_checkpoint(
        execution_version, checkpoint, events, (splits, dividends, cash_settlements)
    )
    validate_program_instruments(spec, panel)
    if on_completed_session is not None and not input_memo.active():
        # Observers may retain caller references; replay owns a stable input snapshot.
        # An evaluation's memo scope reads its own frames; its observer only appends.
        inputs = (spec, panel, opens, closes, splits, dividends, cash_settlements, events)
        *inputs, cost_schedule = deepcopy((*inputs, cost_schedule))
        spec, panel, opens, closes, splits, dividends, cash_settlements, events = inputs
    # Eligibility sees share_plan's closes; fresh cache (no cross-scenario reuse).
    cache = None if panel.target_cache is None else {}
    panel = replace(panel, sizing_prices=closes, target_cache=cache)
    program_route = execution["route"] in {"program-v1", "benchmark-history-v1"}
    calendar = sessions(panel.calendar, end)
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if dates.empty or capital <= 0:
        raise ValueError("replay needs sessions and positive capital")
    if events is not None and any(v is not None for v in (splits, dividends, cash_settlements)):
        raise ValueError("observed events and legacy action arrays cannot be mixed")
    actions = ActionBook(events, **action_rules(execution))
    scalar_cost = costs.commission_bps + costs.half_spread_bps + costs.slippage_bps

    def rates(decision):
        return scalar_cost if cost_schedule is None else cost_schedule.rates(decision)

    for prices in (opens, closes):
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError("duplicate execution observations")
        if not dates.isin(prices.index).all():
            raise ValueError("execution prices must cover every replay session")

    identity_inputs = dict(
        spec=spec,
        panel=panel,
        opens=opens,
        closes=closes,
        start=start,
        end=end,
        capital=capital,
        costs=costs,
        min_notional=min_notional,
        execution=execution,
        splits=splits,
        dividends=dividends,
        cash_settlements=cash_settlements,
        events=events,
        cost_schedule=cost_schedule,
    )
    context = replay_start_context(identity_inputs, checkpoint, calendar, dates, actions)
    context["carry"] = carry = held_carry(execution, cost_schedule)
    if event_protocol:
        identity_inputs.update(
            initialization=context["initialization"],
            checkpoint_id=None if checkpoint is None else checkpoint["checkpoint_id"],
        )
    identity = register_replay(identity_inputs)
    state = initial_account(identity_inputs, context, checkpoint, calendar, dates, actions, rates)
    cash, positions, pending, previous, terminated, formations = (
        state[name]
        for name in ("cash", "positions", "pending", "previous", "terminated", "formations")
    )
    history, order_history, action_history = [], [], []
    for day in dates:
        row_open, row_close = opens.loc[day], closes.loc[day]
        components = {
            key: 0.0 for key in ("commission", "spread", "slippage", "borrow", "financing")
        }
        held_before_actions = set(positions)
        event_result = replay_actions(actions, day, positions, cash, program_route, row_open)
        session = (costs, cost_schedule, rates, components)
        cash, cost, turnover, exits = fallback_exits(
            event_result, opens, closes, day, positions, carry, session
        )
        order_history.extend(exits)
        cash, distribution_pnl, split_names, settled, legacy_action_changed = legacy_actions(
            day,
            positions,
            cash,
            row_open,
            splits,
            dividends,
            cash_settlements,
            event_result,
            event_result["distribution_pnl"],
            action_history,
            terminated,
        )
        replan_after_action, recovery_source = False, None
        row_open = carried_open(row_open, positions, carry, pending)
        if (
            program_route
            and not event_protocol
            and (
                legacy_action_changed
                or (split_names | settled) & held_before_actions
                or event_result["payments_applied"]
            )
        ):
            raise ExecutionContractBlocked("program_corporate_action_intent_unqualified")
        if event_protocol:
            positions = {n: q for n, q in sorted(positions.items()) if q}
        if pending is not None and (not event_protocol or pending["orders"]):
            relevant = (
                held_before_actions | set(positions) | {o["permno"] for o in pending["orders"]}
            )
            affected = (split_names | settled) & relevant
            for order in pending["orders"]:
                name = order["permno"]
                if splits is not None:
                    ratio = float(splits.loc[day, name])
                    if not math.isfinite(ratio) or ratio <= 0:
                        raise ValueError("invalid pending-instrument split")
                    if ratio != 1:
                        affected.add(name)
            account_changed = event_protocol and pending["account_state_id"] != account_state_id(
                positions, cash, actions.receivables
            )
            if affected or event_result["payments_applied"] or account_changed:
                if program_route and not event_protocol:
                    raise ExecutionContractBlocked(
                        "program_pending_corporate_action_intent_unqualified"
                    )
                order_history.append(
                    {"date": str(day.date()), "status": "expired_corporate_action"}
                )
                replan_after_action = True
                if event_protocol:
                    recovery_source = pending["source_intent"]
                    order_history[-1].update(
                        intent_id=pending["intent_id"], source_intent=recovery_source
                    )
            else:
                required = relevant
                missing = [
                    n
                    for n in sorted(required)
                    if n not in row_open
                    or not math.isfinite(float(row_open[n]))
                    or row_open[n] <= 0
                ]
                if missing and carry and carry.get("open_gap_sessions"):
                    pending, missing, row_open, gap = open_gaps(
                        pending, missing, positions, opens, day, carry
                    )
                    order_history.extend(gap)
                if missing:
                    order_history.append(
                        {
                            "date": str(day.date()),
                            "status": "blocked_missing_execution_price",
                            "permnos": missing,
                        }
                    )
                    if program_route and not event_protocol:
                        raise ExecutionContractBlocked(
                            "program_pending_execution_price_unqualified"
                        )
                    if event_protocol:
                        recovery_source = pending["source_intent"]
                        order_history[-1].update(
                            intent_id=pending["intent_id"], source_intent=recovery_source
                        )
                    pending = {"orders": [], "decision": pending["decision"]}
                    replan_after_action = True
                before = (
                    cash
                    + actions.value
                    + sum(
                        q * float(row_open[n])
                        for n, q in positions.items()
                        if q and n not in missing
                    )
                )
                projected = dict(positions)
                for order in pending["orders"]:
                    n = order["permno"]
                    projected[n] = projected.get(n, 0) + order["shares"]
                risk = (
                    {"blockers": []}
                    if missing
                    else execution_risk(
                        {
                            "projected_shares": projected,
                            "capital": before,
                            "profile": getattr(spec, "side", "long_short"),
                            "execution_cost_bps": order_rates(pending, rates),
                            "limits": {
                                "max_gross": 1,
                                "max_weight": 1,
                                "max_order_notional": float("inf"),
                            },
                        },
                        {"positions": positions, "cash": cash},
                        row_open.to_dict(),
                    )
                )
                if risk["blockers"]:
                    order_history.append(
                        {
                            "date": str(day.date()),
                            "status": "blocked_open_risk",
                            "blockers": risk["blockers"],
                        }
                    )
                    if program_route and not event_protocol:
                        raise ExecutionContractBlocked(
                            "program_pending_open_risk_recovery_unqualified",
                            ",".join(risk["blockers"]),
                        )
                    if event_protocol:
                        recovery_source = pending["source_intent"]
                        order_history[-1].update(
                            intent_id=pending["intent_id"], source_intent=recovery_source
                        )
                        replan_after_action = True
                    pending = {"orders": []}
                cash, filled_cost, filled_turnover, filled = fill_orders(
                    pending,
                    positions,
                    cash,
                    row_open,
                    costs,
                    cost_schedule,
                    rates,
                    day,
                    components,
                    event_protocol=event_protocol,
                )
                cost += filled_cost
                turnover += filled_turnover
                order_history.extend(filled)
        positions = {n: q for n, q in positions.items() if q}
        row_close = carried_close(day, positions, opens.loc[day], row_close, carry, order_history)
        cash, equity, row = mark_session(
            day,
            positions,
            cash,
            row_close,
            previous,
            cost,
            turnover,
            components,
            costs,
            cost_schedule,
            actions,
            distribution_pnl,
            getattr(spec, "side", "long_short"),
            event_protocol=event_protocol,
        )
        retain_intent_evidence(row, pending)
        history.append(row)
        if on_completed_session is not None:
            on_completed_session(deepcopy(history[-1]))
        previous, pending = equity, None
        if program_route:
            pending = program_replay_plan(
                spec,
                panel,
                day,
                positions,
                closes.loc[day].to_dict(),
                equity - max(actions.value, 0),
                min_notional=min_notional,
                fee_bps=rates(day),
                terminated=terminated,
                recovery_requested=replan_after_action,
                execution_version=execution_version,
                source_intent=recovery_source,
                cash=cash,
                receivables=actions.receivables,
                carry=carry,
            )
            if event_protocol and pending["source_intent"].get("formation_session") == str(
                day.date()
            ):
                formations.append(str(day.date()))
        elif decision_due(spec, panel, day) or replan_after_action:
            book = target_weights(spec, panel, day)
            if any(book.get(n, 0) != 0 for n in terminated):
                raise ValueError("a cash-settled instrument remains in the target universe")
            pending = share_plan(
                book,
                positions,
                closes.loc[day].to_dict(),
                equity - max(actions.value, 0),
                min_notional=min_notional,
                side=getattr(spec, "side", "long_short"),
                fee_bps=rates(day),
                carry=carry,
            )
            pending["decision"] = str(day.date())
    metadata = completed_replay_metadata(
        identity_inputs, identity, initialization_qualifying=context["initialization_qualifying"]
    )
    result = {
        "history": pd.DataFrame(history).set_index("date"),
        "orders": order_history,
        "corporate_actions": action_history + actions.history,
        "pending": pending,
        "positions": positions,
        "cash": cash,
        "receivables": actions.receivables,
        **metadata,
        "account_hash": digest({"positions": positions, "cash": cash}),
    }
    if event_protocol:
        result.update(
            final_event_metadata(
                identity_inputs,
                context,
                checkpoint,
                calendar,
                dates[-1],
                positions,
                cash,
                actions,
                pending,
                terminated,
                previous,
                formations,
            )
        )
    return result
