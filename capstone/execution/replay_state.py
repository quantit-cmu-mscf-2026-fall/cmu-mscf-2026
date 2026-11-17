"""Frozen Program recovery intents and replay account initialization."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from capstone.execution.contracts import (
    BENCHMARK_VERSIONS,
    PROGRAM_EVENTS_VERSIONS,
    WINDOW_START_MODE,
    ExecutionContractBlocked,
    program_positions,
)
from capstone.execution.planning import carry_marks
from capstone.execution.prefix_protocol import PROGRAM_VERSIONS as PREFIX_PROGRAM_VERSIONS
from capstone.execution.replay_checkpoint import (
    checkpoint_config,
    make_checkpoint,
    portable,
    restore_checkpoint,
)
from capstone.execution.replay_checkpoint import (
    checkpoint_document as checkpoint_document,
)
from capstone.execution.replay_checkpoint import (
    read_checkpoint as read_checkpoint,
)
from capstone.execution.replay_checkpoint import (
    write_checkpoint as write_checkpoint,
)
from capstone.strategy.research import digest
from capstone.strategy.saved_portfolio import EXECUTION_VERSIONS as SAVED_VERSIONS

EVENT_VERSIONS = (
    *PROGRAM_EVENTS_VERSIONS,
    *SAVED_VERSIONS,
    *BENCHMARK_VERSIONS,
    *PREFIX_PROGRAM_VERSIONS,
)


def retain_intent_evidence(row, pending):
    if pending:
        for key in ("formation_eligibility", "construction_evidence", "source_intent"):
            if key in pending:
                row[key] = deepcopy(pending[key])


def checked_checkpoint(version, record, events, legacy_arrays):
    event = version in EVENT_VERSIONS
    if record is not None and not event:
        raise ExecutionContractBlocked("checkpoint_requires_event_protocol")
    if event and (events is None or any(value is not None for value in legacy_arrays)):
        raise ExecutionContractBlocked("event_protocol_requires_observed_event_coverage")
    parsed = None if record is None else checkpoint_document(record)
    if parsed is not None and parsed["protocol"] != version:
        raise ExecutionContractBlocked("checkpoint_protocol_incompatible")
    return parsed


def account_state_id(positions, cash, receivables):
    return digest(
        {
            "positions": {str(n): int(q) for n, q in sorted(positions.items()) if q},
            "cash": float(cash),
            "receivables": receivables,
        }
    )


def freeze_intent(plan, positions, cash, receivables, *, decision, source):
    source = portable(source)
    if "source_intent_id" not in source:
        source["source_intent_id"] = digest(source)
    content = deepcopy(plan)
    content.update(
        decision=str(pd.Timestamp(decision).date()),
        source_intent=source,
        account_state_id=account_state_id(positions, cash, receivables),
    )
    content.pop("intent_id", None)
    content = portable(content)
    content["intent_id"] = digest(content)
    return content


def verify_intent(plan):
    if not isinstance(plan, dict) or plan.get("intent_id") != digest(
        {k: v for k, v in plan.items() if k != "intent_id"}
    ):
        raise ExecutionContractBlocked("checkpoint_pending_intent_integrity_failure")
    for field in ("targets", "projected"):
        program_positions(plan[field])
    for order in plan["orders"]:
        program_positions({order["permno"]: order["shares"]})
    source = plan["source_intent"]
    if source.get("source_intent_id") != digest(
        {k: v for k, v in source.items() if k != "source_intent_id"}
    ):
        raise ExecutionContractBlocked("checkpoint_pending_source_integrity_failure")
    if source.get("kind") not in {"formation", "exit_only"}:
        raise ExecutionContractBlocked("checkpoint_pending_source_incompatible")
    return deepcopy(plan)


def prefix_inputs_hash(inputs, origin, through):
    from capstone.execution.replay_identity import replay_identity

    return replay_identity(
        **(
            inputs
            | {
                "start": origin,
                "end": through,
                "checkpoint_id": None,
            }
        )
    )["inputs_hash"]


def replay_start_context(inputs, checkpoint, calendar, dates, actions):
    from capstone.execution.daily_planning import program_initialization_due, window_start_due

    prior = calendar[calendar.get_loc(dates[0]) - 1] if calendar.get_loc(dates[0]) else None
    program = inputs["execution"]["route"] in {"program-v1", "benchmark-history-v1"}
    initialize = prior is not None and (
        not program or program_initialization_due(inputs["spec"], inputs["panel"], prior)
    )
    event = inputs["execution"]["version"] in EVENT_VERSIONS
    origin = str(dates[0].date()) if checkpoint is None else checkpoint["origin_session"]
    context = {"origin_session": origin, "prior": prior, "initialize": initialize}
    if event:
        context["initialization"] = (
            "restored_replay_checkpoint"
            if checkpoint is not None
            else "new_account_at_scheduled_formation"
            if initialize
            else WINDOW_START_MODE
            if prior is not None
            and window_start_due(
                inputs["spec"], inputs["panel"], prior, inputs["execution"]["version"]
            )
            else "diagnostic_flat_until_scheduled_formation"
        )
        if checkpoint is not None and pd.Timestamp(checkpoint["session"]) >= dates[0]:
            raise ExecutionContractBlocked("checkpoint_requires_next_authoritative_session")
        relevant = actions.events[
            actions.events.event_date.between(pd.Timestamp(origin), dates[-1])
        ]
        if not relevant.event_date.isin(calendar).all():
            raise ExecutionContractBlocked("event_effective_session_not_in_calendar")
        for column in ("event_date", "pay_date", "observed_pay_date"):
            if column in relevant:
                values = pd.to_datetime(relevant[column].dropna())
                if any(at.tz is not None or at != at.normalize() for at in values):
                    raise ExecutionContractBlocked("event_timestamp_precision_unqualified")
        if "pay_date" in relevant:
            delayed = relevant[
                relevant.kind.isin(["settlement", "conversion"]) & relevant.pay_date.notna()
            ]
            if not pd.to_datetime(delayed.pay_date).eq(delayed.event_date).all():
                raise ExecutionContractBlocked("nondividend_cash_payment_timing_unqualified")
        if "observed_pay_date" in relevant:
            observed = relevant[
                relevant.kind.isin(["settlement", "conversion"])
                & relevant.observed_pay_date.notna()
            ]
            if not pd.to_datetime(observed.observed_pay_date).le(observed.event_date).all():
                raise ExecutionContractBlocked("nondividend_cash_payment_timing_unqualified")
    return context


def initial_account(inputs, context, checkpoint, calendar, dates, actions, rates):
    from capstone.execution.daily_planning import initial_replay_plan, program_replay_plan

    capital = inputs["capital"]
    state = {
        "cash": float(capital),
        "positions": {},
        "pending": None,
        "previous": float(capital),
        "terminated": set(),
        "formations": [],
    }
    event = inputs["execution"]["version"] in EVENT_VERSIONS
    context["config"] = (
        checkpoint_config(
            inputs["spec"],
            inputs["panel"],
            inputs["execution"],
            capital,
            inputs["costs"],
            inputs["min_notional"],
        )
        if event
        else None
    )
    if checkpoint is not None:
        state.update(
            restore_checkpoint(
                checkpoint,
                config=context["config"],
                start=dates[0],
                calendar=calendar,
                closes=inputs["closes"],
                actions=actions,
                spec=inputs["spec"],
                panel=inputs["panel"],
                rates=rates,
                carry=context.get("carry"),
                prefix_inputs_hash=prefix_inputs_hash(
                    inputs, context["origin_session"], checkpoint["session"]
                ),
            )
        )
    elif context["initialize"]:
        prior = context["prior"]
        if event:
            if prior not in inputs["closes"].index:
                raise ValueError(
                    "initial investment requires the previous completed-session prices"
                )
            state["pending"] = program_replay_plan(
                inputs["spec"],
                inputs["panel"],
                prior,
                {},
                inputs["closes"].loc[prior].to_dict(),
                capital,
                min_notional=inputs["min_notional"],
                fee_bps=rates(prior),
                terminated=set(),
                execution_version=inputs["execution"]["version"],
                cash=capital,
                receivables=[],
                carry=context.get("carry"),
            )
            state["formations"].append(str(prior.date()))
        else:
            state["pending"] = initial_replay_plan(inputs, prior, rates)
    context["initialization_qualifying"] = inputs["execution"]["route"] != "program-v1"
    if event:
        context["initialization_qualifying"] = (
            checkpoint["origin_initialization"]["qualifying"]
            if checkpoint is not None
            else bool(context["initialize"]) or context["initialization"] == WINDOW_START_MODE
        )
        context["origin_initialization"] = (
            deepcopy(checkpoint["origin_initialization"])
            if checkpoint is not None
            else {
                "mode": context["initialization"],
                "qualifying": context["initialization_qualifying"],
            }
        )
    return state


def final_checkpoint(
    inputs,
    context,
    calendar,
    day,
    positions,
    cash,
    actions,
    pending,
    terminated,
    previous,
    formations,
):
    # A carried name's checkpoint mark is its last valid price (v3/v4 held carry).
    carry = context.get("carry")
    closes = carry_marks(inputs["closes"].loc[day], positions, carry and carry["marks"])[0]
    return make_checkpoint(
        config=context["config"],
        protocol=inputs["execution"]["version"],
        origin_initialization=context["origin_initialization"],
        origin_session=context["origin_session"],
        session=day,
        next_session=calendar[calendar.get_loc(day) + 1],
        positions=positions,
        cash=cash,
        actions=actions,
        pending=pending,
        terminated=terminated,
        previous=previous,
        closes=closes,
        formations=formations,
        prefix_inputs_hash=prefix_inputs_hash(inputs, context["origin_session"], day),
    )


def final_event_metadata(
    inputs,
    context,
    checkpoint,
    calendar,
    day,
    positions,
    cash,
    actions,
    pending,
    terminated,
    previous,
    formations,
):
    return {
        "checkpoint": final_checkpoint(
            inputs,
            context,
            calendar,
            day,
            positions,
            cash,
            actions,
            pending,
            terminated,
            previous,
            formations,
        ),
        "seed_checkpoint_id": None if checkpoint is None else checkpoint["checkpoint_id"],
    }
