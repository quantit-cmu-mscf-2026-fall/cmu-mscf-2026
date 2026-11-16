"""Canonical completed-session checkpoint integrity, valuation and storage."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from capstone.execution.contracts import (
    BENCHMARK_VERSIONS,
    PROGRAM_EVENTS_VERSIONS,
    WINDOW_START_MODE,
    ExecutionContractBlocked,
    program_positions,
)
from capstone.execution.planning import positive_price
from capstone.execution.prefix_protocol import PROGRAM_VERSIONS as PREFIX_PROGRAM_VERSIONS
from capstone.locking import exclusive_file
from capstone.strategy.research import canonical, digest
from capstone.strategy.saved_portfolio import EXECUTION_VERSIONS as SAVED_VERSIONS

# Default protocol of make_checkpoint; program-events-v4 checkpoints pass it explicitly.
PROGRAM_EVENTS_VERSION = PROGRAM_EVENTS_VERSIONS[0]


def portable(value):
    return json.loads(canonical(value))


def checkpoint_document(record):
    try:
        return _checkpoint_document(record)
    except ExecutionContractBlocked:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ExecutionContractBlocked("checkpoint_schema_invalid") from exc


def _checkpoint_document(record):
    required = {
        "schema_version",
        "protocol",
        "config",
        "origin_session",
        "session",
        "next_session",
        "positions",
        "cash",
        "receivables",
        "applied_event_ids",
        "pending",
        "terminated",
        "previous_equity",
        "previous_close",
        "prefix_inputs_hash",
        "formation_sessions",
        "provenance",
        "checkpoint_id",
        "origin_initialization",
    }
    if (
        not isinstance(record, dict)
        or set(record) != required
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != 1
        or record.get("protocol")
        not in {
            *PROGRAM_EVENTS_VERSIONS,
            *SAVED_VERSIONS,
            *BENCHMARK_VERSIONS,
            *PREFIX_PROGRAM_VERSIONS,
        }
        or record.get("checkpoint_id")
        != digest({k: v for k, v in record.items() if k != "checkpoint_id"})
    ):
        raise ExecutionContractBlocked("checkpoint_integrity_failure")
    for key in ("origin_session", "session", "next_session"):
        at = pd.Timestamp(record[key])
        if (
            pd.isna(at)
            or at.tz is not None
            or at != at.normalize()
            or (key != "next_session" and at > pd.Timestamp("2020-12-31"))
        ):
            raise ExecutionContractBlocked("checkpoint_session_incompatible")
    if not record["origin_session"] <= record["session"] < record["next_session"]:
        raise ExecutionContractBlocked("checkpoint_session_incompatible")
    for key in ("applied_event_ids", "formation_sessions", "terminated"):
        values = record[key]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ExecutionContractBlocked("checkpoint_provenance_invalid")
    if any(type(n) is not int or n <= 0 for n in record["terminated"]):
        raise ExecutionContractBlocked("checkpoint_terminated_identity_invalid")
    initialization = record["origin_initialization"]
    if (
        not isinstance(initialization, dict)
        or set(initialization) != {"mode", "qualifying"}
        or type(initialization["qualifying"]) is not bool
        or type(initialization["mode"]) is not str
    ):
        raise ExecutionContractBlocked("checkpoint_origin_initialization_incompatible")
    return deepcopy(record)


def checkpoint_config(spec, panel, execution, capital, costs, min_notional):
    return {
        "strategy_id": spec.id,
        "data_version": panel.data_version,
        "profile": spec.side,
        "execution_contract": execution["id"],
        "origin_capital": float(capital),
        "costs": asdict(costs),
        "min_notional": float(min_notional),
    }


def make_checkpoint(
    *,
    config,
    protocol=PROGRAM_EVENTS_VERSION,
    origin_initialization,
    origin_session,
    session,
    next_session,
    positions,
    cash,
    actions,
    pending,
    terminated,
    previous,
    closes,
    prefix_inputs_hash,
    formations,
):
    held = {n: q for n, q in sorted(positions.items()) if q}
    content = {
        "schema_version": 1,
        "protocol": protocol,
        "config": config,
        "origin_initialization": origin_initialization,
        "origin_session": origin_session,
        "session": str(pd.Timestamp(session).date()),
        "next_session": str(pd.Timestamp(next_session).date()),
        "positions": {str(n): int(q) for n, q in held.items()},
        "cash": float(cash),
        "receivables": deepcopy(actions.receivables),
        "applied_event_ids": sorted(actions.applied),
        "pending": deepcopy(pending),
        "terminated": sorted(terminated),
        "previous_equity": float(previous),
        "previous_close": {str(n): float(closes[n]) for n in held},
        "prefix_inputs_hash": prefix_inputs_hash,
        "formation_sessions": list(formations),
        "provenance": "produced_by_shared_fixed_share_replay",
    }
    content = portable(content)
    return content | {"checkpoint_id": digest(content)}


def restore_checkpoint(
    record,
    *,
    config,
    start,
    calendar,
    closes,
    actions,
    prefix_inputs_hash,
    spec,
    panel,
    rates,
    carry=None,
):
    try:
        return _restore_checkpoint(
            record,
            config=config,
            start=start,
            calendar=calendar,
            closes=closes,
            actions=actions,
            prefix_inputs_hash=prefix_inputs_hash,
            spec=spec,
            panel=panel,
            rates=rates,
            carry=carry,
        )
    except ExecutionContractBlocked:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ExecutionContractBlocked("checkpoint_schema_invalid") from exc


def _restore_checkpoint(
    record,
    *,
    config,
    start,
    calendar,
    closes,
    actions,
    prefix_inputs_hash,
    spec,
    panel,
    rates,
    carry=None,
):
    """Integrity and valuation are verified; source qualification is a separate gate."""
    from capstone.execution.replay_state import account_state_id, verify_intent

    record = checkpoint_document(record)
    if record["config"] != portable(config) or record["prefix_inputs_hash"] != prefix_inputs_hash:
        raise ExecutionContractBlocked("checkpoint_inputs_or_implementation_incompatible")
    prior = pd.Timestamp(record["session"])
    at = pd.Timestamp(start)
    if (
        prior not in calendar
        or at not in calendar
        or calendar.get_loc(at) != calendar.get_loc(prior) + 1
        or record["next_session"] != str(at.date())
    ):
        raise ExecutionContractBlocked("checkpoint_requires_next_authoritative_session")
    positions = program_positions(record["positions"])
    cash, previous = record["cash"], record["previous_equity"]
    if any(isinstance(v, bool) or not math.isfinite(v) for v in (cash, previous)) or previous <= 0:
        raise ExecutionContractBlocked("checkpoint_invalid_cash_or_equity")
    if config["profile"] == "long_only" and (
        cash < -1e-8 or any(q < 0 for q in positions.values())
    ):
        raise ExecutionContractBlocked("checkpoint_long_only_account_invalid")
    marks = record["previous_close"]
    if set(marks) != {str(n) for n, q in positions.items() if q} or any(
        not math.isfinite(marks[str(n)])
        or marks[str(n)] <= 0
        or (
            float(closes.loc[prior, n]) != marks[str(n)]
            # v3/v4 held carry: no valid close, so the recorded last valid mark stands.
            and (carry is None or positive_price(closes.loc[prior, n]))
        )
        for n, q in positions.items()
        if q
    ):
        raise ExecutionContractBlocked("checkpoint_previous_valuation_incompatible")
    if carry is not None:
        carry["marks"].update({n: marks[str(n)] for n, q in positions.items() if q})
    claims = record["receivables"]
    ids = [claim["event_id"] for claim in claims]
    if len(ids) != len(set(ids)) or any(
        not math.isfinite(claim["amount"]) or pd.Timestamp(claim["pay_date"]) <= prior
        for claim in claims
    ):
        raise ExecutionContractBlocked("checkpoint_receivables_invalid")
    value = (
        cash
        + sum(claim["amount"] for claim in claims)
        + sum(q * marks[str(n)] for n, q in positions.items() if q)
    )
    if not math.isclose(value, previous, rel_tol=1e-12, abs_tol=1e-8):
        raise ExecutionContractBlocked("checkpoint_account_does_not_reconcile")
    applied = set(record["applied_event_ids"])
    known = set(actions.events.loc[actions.events.event_date <= prior, "event_id"])
    if not applied <= known or not set(ids) <= applied:
        raise ExecutionContractBlocked("checkpoint_event_provenance_incompatible")
    pending = record["pending"]
    from capstone.execution.benchmark_runtime import benchmark_due, execution_target
    from capstone.execution.daily_planning import program_initialization_due, window_start_due
    from capstone.strategy.contract import target_weights
    from capstone.strategy.schedule_dispatch import decision_due

    def due(day):
        return (
            benchmark_due(spec, panel, day)
            if record["protocol"] in BENCHMARK_VERSIONS
            else decision_due(spec, panel, day)
        )

    formations = record["formation_sessions"]
    if formations != sorted(formations) or any(
        pd.Timestamp(day) > prior or not due(day) for day in formations
    ):
        raise ExecutionContractBlocked("checkpoint_formation_provenance_invalid")
    origin = pd.Timestamp(record["origin_session"])
    origin_prior = calendar[calendar.get_loc(origin) - 1] if calendar.get_loc(origin) else None
    qualified_origin = origin_prior is not None and program_initialization_due(
        spec, panel, origin_prior
    )
    window = (
        not qualified_origin
        and origin_prior is not None
        and window_start_due(spec, panel, origin_prior, record["protocol"])
    )
    expected_origin = {
        "mode": "new_account_at_scheduled_formation"
        if qualified_origin
        else WINDOW_START_MODE
        if window
        else "diagnostic_flat_until_scheduled_formation",
        "qualifying": bool(qualified_origin or window),
    }
    if record["origin_initialization"] != expected_origin or (
        qualified_origin and str(origin_prior.date()) not in formations
    ):
        raise ExecutionContractBlocked("checkpoint_origin_initialization_incompatible")
    if pending is None:
        raise ExecutionContractBlocked("checkpoint_pending_plan_missing")
    pending = verify_intent(pending)
    if pending["decision"] != record["session"] or pending["account_state_id"] != account_state_id(
        positions, cash, claims
    ):
        raise ExecutionContractBlocked("checkpoint_pending_account_incompatible")
    source = pending["source_intent"]
    if record["protocol"] in PREFIX_PROGRAM_VERSIONS:
        from capstone.execution.prefix_protocol import formation_panel

        panel = formation_panel(panel, prior)
    if source["strategy_id"] != spec.id:
        raise ExecutionContractBlocked("checkpoint_pending_source_incompatible")
    if source["kind"] == "formation":
        original = (
            execution_target(spec, panel, source["formation_session"], record["protocol"])
            if record["protocol"] in BENCHMARK_VERSIONS
            else target_weights(spec, panel, source["formation_session"])
        )
        if (
            source["formation_session"] not in formations
            or canonical(source["book"])
            != canonical({str(n): float(w) for n, w in original.items()})
            or canonical(source.get("construction_evidence", {})) != canonical(original.attrs)
        ):
            raise ExecutionContractBlocked("checkpoint_pending_source_incompatible")
    _verify_pending_economics(record, spec, panel, positions, cash, claims, closes, rates, carry)
    actions.receivables, actions.applied = deepcopy(claims), applied
    return {
        "positions": positions,
        "cash": cash,
        "previous": previous,
        "pending": pending,
        "terminated": set(record["terminated"]),
        "origin_session": record["origin_session"],
        "formations": list(record["formation_sessions"]),
    }


def _verify_pending_economics(record, spec, panel, positions, cash, claims, closes, rates, carry):
    """Rebuild instructions with the same planner; hashes alone do not prove quantities."""
    from capstone.execution.daily_planning import program_replay_plan

    pending = record["pending"]
    expected = program_replay_plan(
        spec,
        panel,
        record["session"],
        positions,
        closes.loc[pd.Timestamp(record["session"])].to_dict(),
        record["previous_equity"] - max(sum(claim["amount"] for claim in claims), 0),
        min_notional=record["config"]["min_notional"],
        fee_bps=rates(record["session"]),
        terminated=set(record["terminated"]),
        recovery_requested=True,
        execution_version=record["protocol"],
        source_intent=pending["source_intent"],
        cash=cash,
        receivables=claims,
        carry=carry,
    )
    # Historical supersession is retained as provenance; the prior abandoned source
    # is not needed to rederive the current source's exact economic instructions.
    excluded = {"intent_id", "superseded_source_intent_id"}
    if canonical({k: v for k, v in pending.items() if k not in excluded}) != canonical(
        {k: v for k, v in expected.items() if k not in excluded}
    ):
        raise ExecutionContractBlocked("checkpoint_pending_instructions_incompatible")


def write_checkpoint(record, path):
    record = checkpoint_document(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical(record) + "\n"
    with exclusive_file(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError("refusing to replace a different replay checkpoint")
            return path
        temporary = path.with_suffix(path.suffix + ".partial")
        if temporary.exists():
            if temporary.read_text(encoding="utf-8") != text:
                raise FileExistsError(
                    "quarantined incomplete checkpoint requires explicit resolution"
                )
        else:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    return path


def read_checkpoint(path):
    try:
        return checkpoint_document(json.loads(Path(path).read_text(encoding="utf-8")))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ExecutionContractBlocked):
            raise
        raise ExecutionContractBlocked("checkpoint_integrity_failure") from exc
