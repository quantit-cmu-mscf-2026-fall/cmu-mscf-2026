"""Canonical replay input identity; an execution version alone is not an input hash."""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import pandas as pd

from capstone import runlog
from capstone.execution.contracts import (
    BENCHMARK_VERSIONS,
    PREFIX_VERSIONS,
    PROGRAM_EVENTS_VERSIONS,
    ExecutionContractBlocked,
)
from capstone.execution.prefix_protocol import PROGRAM_VERSIONS as PREFIX_PROGRAM_VERSIONS
from capstone.pipeline.market_costs import HistoricalCostSchedule
from capstone.release import runtime_versions
from capstone.strategy import input_memo
from capstone.strategy.allocation_state import frame_identity  # single definition; re-exported
from capstone.strategy.portfolio import PortfolioSpec
from capstone.strategy.research import canonical, digest
from capstone.strategy.saved_portfolio import EXECUTION_VERSIONS as SAVED_VERSIONS
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec

REPLAY_ROW = "reference-replay"
TAGS = ["reference-v2", "replay"]
# Replays a killed attempt of the same registered output already logged: a
# resumed attempt consumes these instead of logging the same trial a second time.
_RESUMED = contextvars.ContextVar("replays_logged_before_resume", default=None)
# Inside a parallel evaluation's worker the rows are captured, not logged; the
# parent logs them in task order (`log_registered`) before it reads the results.
_CAPTURED = contextvars.ContextVar("replay_rows_captured", default=None)
NOTES = "Registered before replay results exist."


def replay_key(params):
    """The trial identity of one `reference-replay` ledger row: its params as the ledger
    holds them (a JSON round trip), so a row read back matches the one about to be logged."""
    return canonical(json.loads(json.dumps(params)))


@contextmanager
def resume_scope():
    """A scope in which a resumed attempt may name replays already on the ledger
    (`already_logged`); outside one, every replay is logged."""
    token = _RESUMED.set(Counter())
    try:
        yield
    finally:
        _RESUMED.reset(token)


def already_logged(keys):
    """Inside `resume_scope`: these replay rows (by `replay_key`) were logged by the
    killed attempt being resumed, before it looked at their results."""
    pending = _RESUMED.get()
    if pending is not None:
        pending.update(keys)


@contextmanager
def captured():
    """A scope whose replays hand their ledger params to the yielded list instead of
    logging them (a parallel evaluation's worker)."""
    rows = []
    token = _CAPTURED.set(rows)
    try:
        yield rows
    finally:
        _CAPTURED.reset(token)


def log_registered(params):
    """Log one `reference-replay` row, unless the killed attempt being resumed already
    logged the same trial (`already_logged`)."""
    pending, key = _RESUMED.get(), replay_key(params)
    if pending is not None and pending[key] > 0:
        pending[key] -= 1  # the killed attempt logged this trial already
        return
    runlog.log_run(REPLAY_ROW, params=params, tags=list(TAGS), notes=NOTES)


def register_replay(inputs):
    """Bind and register the opportunity before the engine computes any results.

    Exactly once across a kill: a replay the killed attempt being resumed already logged
    (`already_logged`, same params) is not logged again; its row was written before that
    attempt saw any result, so the rule "log before looking" still holds."""
    if isinstance(inputs["spec"], SavedProgramPortfolioSpec):
        inputs["spec"].verify_history(inputs["panel"], inputs["end"])
    identity = replay_identity(**inputs)
    spec, schedule, events = inputs["spec"], inputs["cost_schedule"], inputs["events"]
    params = {
        "strategy_id": spec.id,
        "data_version": inputs["panel"].data_version,
        "start": str(inputs["start"]),
        "end": str(inputs["end"]),
        "costs": asdict(inputs["costs"]),
        "capital": inputs["capital"],
        "execution": "next-session-open",
        "execution_contract": inputs["execution"]["id"],
        "execution_version": inputs["execution"]["version"],
        "initialization": identity["initialization"],
        "execution_inputs_hash": identity["inputs_hash"],
        "execution_report_id": identity["report_id"],
        "min_notional": inputs["min_notional"],
        "event_hash": None if events is None else digest(events.to_json(date_format="iso")),
        "cost_schedule": None if schedule is None else schedule.id,
        "return_basis": "funded_total_return",
    }
    capture = _CAPTURED.get()
    if capture is not None:
        capture.append(params)
    else:
        log_registered(params)
    return identity


def completed_replay_metadata(inputs, identity, *, initialization_qualifying=None):
    from capstone.execution.contracts import execution_contract

    if (
        execution_contract(inputs["spec"], version=inputs["execution"]["version"])
        != inputs["execution"]
    ):
        raise ExecutionContractBlocked("execution_source_changed_during_replay")
    with input_memo.bypass():  # an honest re-hash: memoized hashes are checked here
        unchanged = replay_identity(**inputs) == identity
    if not unchanged:
        raise ExecutionContractBlocked("execution_input_changed_during_replay")
    return {
        "return_basis": "funded_total_return",
        "execution_contract": inputs["execution"],
        "initialization": identity["initialization"],
        "initialization_qualifying": (
            inputs["execution"]["route"] != "program-v1"
            if initialization_qualifying is None
            else initialization_qualifying
        ),
        "formal_admission": False,
        "execution_report_id": identity["report_id"],
        "execution_inputs_hash": identity["inputs_hash"],
        "strategy_id": inputs["spec"].id,
    }


def _hash(frame, end):
    """frame_identity, memoized per frame object and end inside an evaluation scope."""
    end = str(pd.Timestamp(end))
    return input_memo.cached("frame", (frame,), (end,), lambda: frame_identity(frame, end))


def table_identity(frame):
    if frame is None:
        return None
    frame = frame.sort_index(axis=1)
    metadata = canonical(
        {"columns": frame.columns.tolist(), "dtypes": [str(dtype) for dtype in frame.dtypes]}
    )
    values = np.sort(
        pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    ).tobytes()
    return hashlib.sha256(metadata.encode() + values).hexdigest()


def allocation_identity(spec, end):
    if not isinstance(spec, PortfolioSpec):
        return None
    record = {
        "mode": spec.mode,
        "returns": frame_identity(spec.returns, end),
        "members": {member.id: allocation_identity(member, end) for member in spec.specs},
    }
    if isinstance(spec, SavedProgramPortfolioSpec):
        record.update(decisions=spec.decision_prefix(end), decision_prefix_id=spec.prefix_id(end))
    return record


def cost_identity(schedule, end, *, prefix_only=False):
    if schedule is None:
        return None
    if type(schedule) is not HistoricalCostSchedule:
        raise ExecutionContractBlocked("cost_schedule_identity_unqualified")
    record = {
        "spreads": _hash(schedule.spreads, end),
        "costs": asdict(schedule.costs),
        "multiplier": schedule.multiplier,
    }
    # V2 preserves its recorded whole-source identity. V3 checkpoints may acquire
    # later quotes without changing the actual cost inputs of their prior prefix.
    if not prefix_only:
        record["id"] = schedule.id
    return record


def replay_identity(
    spec,
    panel,
    opens,
    closes,
    *,
    start,
    end,
    capital,
    costs,
    min_notional,
    execution,
    splits,
    dividends,
    cash_settlements,
    events,
    cost_schedule,
    checkpoint_id=None,
    initialization=None,
):
    if execution["version"] in PREFIX_VERSIONS:
        from capstone.execution.prefix_protocol import canonical_inputs

        panel, opens, closes, events, cost_schedule = canonical_inputs(
            panel, opens, closes, events, cost_schedule, end
        )
    features = {name: _hash(frame, end) for name, frame in sorted(panel.fields.items())}
    availability = {
        name: _hash(frame, end) for name, frame in sorted((panel.availability or {}).items())
    }
    observed_events = None
    if events is not None:
        cut = pd.Timestamp(end)
        observed_events = input_memo.cached(
            "events",
            (events,),
            (str(cut),),
            lambda: table_identity(events[pd.to_datetime(events.event_date) <= cut]),
        )
    fundamentals = panel.fundamentals
    if fundamentals is not None:
        fundamentals = fundamentals[pd.to_datetime(fundamentals.published_at) <= pd.Timestamp(end)]
    calendar = pd.DatetimeIndex(panel.calendar).sort_values()
    # The first following session binds the boundary without admitting unrelated future rows.
    calendar = calendar[calendar <= pd.Timestamp(end)].append(
        calendar[calendar > pd.Timestamp(end)][:1]
    )
    inputs = {
        "features": features,
        "availability": availability,
        "universe": _hash(panel.universe, end),
        "half_spread": _hash(panel.half_spread, end),
        "sic": _hash(panel.sic, end),
        "company": _hash(panel.company, end),
        "opens": _hash(opens, end),
        "closes": _hash(closes, end),
        "splits": _hash(splits, end),
        "dividends": _hash(dividends, end),
        "cash_settlements": _hash(cash_settlements, end),
        "events": observed_events,
        "calendar": [str(day.date()) for day in calendar],
        "cost_schedule": cost_identity(
            cost_schedule,
            end,
            prefix_only=execution["version"]
            in {
                *PROGRAM_EVENTS_VERSIONS,
                *SAVED_VERSIONS,
                *BENCHMARK_VERSIONS,
                *PREFIX_PROGRAM_VERSIONS,
            },
        ),
        "allocation_inputs": allocation_identity(spec, end),
        "fundamentals": table_identity(fundamentals),
    }
    initialization = initialization or (
        "diagnostic_flat_until_scheduled_formation"
        if execution["route"] == "program-v1"
        else "legacy_prior_completed_session"
    )
    descriptor = {
        "strategy_id": spec.id,
        "data_version": panel.data_version,
        "start": str(start),
        "end": str(end),
        "capital": capital,
        "costs": asdict(costs),
        "min_notional": min_notional,
        "inputs_hash": digest(inputs),
        "execution_contract": execution["id"],
        "initialization": initialization,
        "runtime": runtime_versions(),
    }
    if execution["version"] in {
        *PROGRAM_EVENTS_VERSIONS,
        *SAVED_VERSIONS,
        *BENCHMARK_VERSIONS,
        *PREFIX_PROGRAM_VERSIONS,
    }:
        descriptor["seed_checkpoint_id"] = checkpoint_id
    return {**descriptor, "report_id": digest(descriptor)}
