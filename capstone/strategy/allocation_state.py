"""Saved chronological decisions over the existing V2 coefficient optimizer."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict

import pandas as pd

from capstone.factors.research_contracts import canonical, digest
from capstone.strategy.program_schedule import session_date
from capstone.strategy.robust_allocation import MODES, AllocationPolicyV2, optimize_allocation


def frame_identity(frame, end):
    """Digest of a table's columns, dtypes and rows through ``end`` (later rows never count)."""
    if frame is None:
        return None
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.loc[frame.index <= pd.Timestamp(end)].sort_index().sort_index(axis=1)
    metadata = canonical(
        {"columns": frame.columns.tolist(), "dtypes": [str(dtype) for dtype in frame.dtypes]}
    )
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes()
    return hashlib.sha256(metadata.encode() + values).hexdigest()


# v1: retrospective fixed pool. v2: the pool at each decision is the set of members
# with an authenticated prefix receipt available strictly before it; departures are explicit
# capital transitions to cash. v1 records keep verifying and the fixed route still makes them.
FIXED_POOL_VERSION = "funded-program-allocation-state-v1"
VERSION = "funded-program-allocation-state-v2"
VERSIONS = (FIXED_POOL_VERSION, VERSION)
CLOCK = "prior_qualified_prefix"
BASIS = "funded_incremental_over_zero_cash_v1"
RECEIPT_KEYS = ("receipt_id", "available_at", "history_start")
DISCOVERY_END = "2020-12-31"


def current_input(returns, at, names):
    """(content hash, start) a post-2020 decision's current provenance binds.

    Only the pool's prior columns: a candidate joining later does not change it.
    """
    from capstone.execution.bundle import frame_hash

    prefix = returns.loc[returns.index < at].reindex(columns=sorted(names))
    return frame_hash(prefix), prefix.index.min() if len(prefix) else at


def _day(value):
    return str(session_date(value, "membership availability").date())


def membership_events(events):
    """Canonical receipt/departure events; availability is the only membership clock."""
    rows = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("strategy_id"), str):
            raise ValueError("membership events require a strategy identifier")
        if event.get("kind") == "receipt" and set(event) == {"strategy_id", "kind", *RECEIPT_KEYS}:
            key = event["receipt_id"]
            if not isinstance(key, str) or len(key) != 64:
                raise ValueError("membership receipt requires its authenticated identifier")
            row = dict(event, history_start=_day(event["history_start"]))
        elif event.get("kind") == "departure" and set(event) == {
            "strategy_id",
            "kind",
            "available_at",
            "reason",
        }:
            if not isinstance(event["reason"], str) or not event["reason"]:
                raise ValueError("a departure requires its explicit reason")
            row = dict(event)
        else:
            raise ValueError("membership events are receipts or departures")
        row["available_at"] = _day(event["available_at"])
        rows.append(row)
    return tuple(sorted(rows, key=canonical))


def pool_at(events, at):
    """Latest receipt per member available before ``at``; departures are permanent."""
    known = sorted(
        (row for row in membership_events(events) if session_date(row["available_at"]) < at),
        key=lambda row: row["available_at"],
    )
    departed = {}
    for row in known:
        if row["kind"] == "departure":
            departed.setdefault(row["strategy_id"], row["reason"])
    latest = {
        row["strategy_id"]: row
        for row in known
        if row["kind"] == "receipt" and row["strategy_id"] not in departed
    }
    return latest, departed


def _verify_membership(record):
    pool = record["pool_ids"]
    membership = record.get("membership")
    if (
        record.get("membership_clock") != CLOCK
        or not isinstance(membership, dict)
        or set(membership) != set(pool)
        or any(
            not isinstance(row, dict)
            or set(row) != set(RECEIPT_KEYS)
            or row["available_at"] >= record["decision_session"]
            for row in membership.values()
        )
        or not isinstance(record.get("insufficient_history_ids"), list)
        or not set(record["insufficient_history_ids"]) <= set(pool)
        or any(record["weights"][name] != 0 for name in record["insufficient_history_ids"])
        or not isinstance(record.get("entered"), list)
        or not set(record["entered"]) <= set(pool)
        or "capital_transition" not in record
    ):
        raise ValueError("chronological membership is not bound to prior receipts")
    transition = record["capital_transition"]
    if transition is None:
        return
    retired = transition.get("retired") if isinstance(transition, dict) else None
    if (
        not isinstance(retired, dict)
        or set(transition) != {"retired", "to_cash", "reason"}
        or not retired
        or set(retired) & set(pool)
        or not isinstance(transition["reason"], dict)
        or set(transition["reason"]) != set(retired)
        or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
            for w in retired.values()
        )
        or not isinstance(transition["to_cash"], (int, float))
        or abs(sum(retired.values()) - transition["to_cash"]) > 1e-12
    ):
        raise ValueError("invalid capital transition to cash")


def verify_decision(record):
    if not isinstance(record, dict) or record.get("id") != digest(
        {k: v for k, v in record.items() if k != "id"}
    ):
        raise ValueError("allocation decision integrity failure")
    if record.get("version") not in VERSIONS or record.get("return_basis") != BASIS:
        raise ValueError("incompatible allocation decision")
    if record.get("profile") not in {"long_only", "long_short"}:
        raise ValueError("explicit portfolio profile required")
    weights = record.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(record.get("pool_ids", ())):
        raise ValueError("allocation must identify the complete frozen pool")
    if any(
        isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
        for w in weights.values()
    ):
        raise ValueError("invalid allocation coefficient")
    cash = record.get("cash_weight")
    if not isinstance(cash, (int, float)) or isinstance(cash, bool) or not math.isfinite(cash):
        raise ValueError("invalid cash coefficient")
    if cash < 0 or abs(sum(weights.values()) + cash - 1) > 1e-8:
        raise ValueError("allocation capital does not reconcile")
    decision, execution = (
        session_date(record["decision_session"]),
        session_date(record["execution_session"]),
    )
    if decision >= execution:
        raise ValueError("allocation must precede execution")
    if record["history_end"] is not None and session_date(record["history_end"]) >= decision:
        raise ValueError("allocation history must precede the decision")
    if record["version"] == VERSION:
        _verify_membership(record)
    if record["decision_session"] > DISCOVERY_END and not isinstance(
        record.get("current_provenance"), dict
    ):
        raise ValueError("a post-2020 decision records its current provenance")
    return record


def make_decision(
    returns,
    *,
    as_of,
    execution_session,
    profile,
    input_evidence_id,
    pool_ids=None,
    membership=None,
    previous=None,
    mode="median_window_sharpe",
    policy=None,
    current=None,
):
    """Coefficients are fractions of funded capital; positive cash earns zero.

    Standalone returns already include stock execution/carry costs. Subtract the
    explicitly declared zero-return cash reference once; do not charge them
    again here. The final combined stock book must receive its own single replay.

    ``pool_ids`` is the retrospective fixed pool (v1 records). ``membership`` is the
    chronological clock (v2 records): receipt and departure events, see ``_chronological``.
    """
    if (pool_ids is None) == (membership is None):
        raise ValueError("supply exactly one of pool_ids or membership")
    policy = policy or AllocationPolicyV2()
    if not isinstance(policy, AllocationPolicyV2):
        raise ValueError("new funded decisions require the corrected V2 optimizer")
    if profile not in {"long_only", "long_short"}:
        raise ValueError("separate explicit profile required")
    if mode not in MODES or not isinstance(returns, pd.DataFrame):
        raise ValueError("known allocation mode and funded-return dataframe required")
    if (
        not isinstance(input_evidence_id, str)
        or len(input_evidence_id) != 64
        or any(c not in "0123456789abcdef" for c in input_evidence_id)
    ):
        raise ValueError("complete historical return evidence identity required")
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or returns.index.has_duplicates
        or returns.columns.has_duplicates
    ):
        raise ValueError("unique dated return observations required")
    at, execution = session_date(as_of), session_date(execution_session)
    if at >= execution:
        raise ValueError("discovery decision and following execution session required")
    if at > pd.Timestamp(DISCOVERY_END):  # the same gate as target_weights
        # The authorized current-feature route is not part of this module yet: fail closed.
        raise ValueError(
            "decisions after the discovery period need the authorized current-feature route"
        )
    if previous is not None:
        verify_decision(previous)
        if previous["profile"] != profile or previous["decision_session"] >= str(at.date()):
            raise ValueError("previous state must be earlier and have the same profile")
        if (previous["version"] == VERSION) != (membership is not None):
            raise ValueError("previous state must use the same membership clock")
    if membership is not None:
        record = _chronological(
            returns, at, execution, membership, profile, input_evidence_id, previous, mode, policy
        )
    else:
        record = _fixed(
            returns, at, execution, pool_ids, profile, input_evidence_id, previous, mode, policy
        )
    if at > pd.Timestamp(DISCOVERY_END):  # bound into the id; re-checked on every recompute
        record["current_provenance"] = current.to_dict()
    record["id"] = digest(record)
    return verify_decision(record)


def _fixed(returns, at, execution, pool_ids, profile, input_evidence_id, previous, mode, policy):
    names = sorted(pool_ids)
    if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("distinct frozen pool identifiers required")
    if set(returns.columns) != set(names):
        raise ValueError("funded return columns must identify the frozen pool exactly")
    past = returns.loc[returns.index < at].sort_index().reindex(columns=names)
    prior = {}
    if previous is not None:
        if set(previous["weights"]) - set(names):
            raise ValueError("pool retirement needs an explicit capital transition")
        prior = previous["weights"]
    result, original_id = _optimize(past, past, at, mode, policy, prior)
    record = _record(
        result, original_id, past, at, execution, profile, names, input_evidence_id, previous
    )
    record["policy"] = asdict(policy)
    record["optimizer_record_id"] = original_id
    record["formal_admission"] = False
    return record


def _optimize(past, window, at, mode, policy, prior):
    if len(window.columns):
        incremental = window.subtract(pd.Series(0.0, index=window.index), axis=0)
        result = optimize_allocation(incremental, at, mode=mode, policy=policy, previous=prior)
        return result, result.pop("id")
    result = {
        "weights": {},
        "cash_weight": 1.0,
        "status": "empty_pool_cash",
        "history_end": str(past.index[-1].date()) if len(past) else None,
    }
    return result, None


def _record(result, original_id, past, at, execution, profile, names, evidence, previous):
    return {
        **result,
        "version": FIXED_POOL_VERSION,
        "profile": profile,
        "pool_ids": names,
        "decision_session": str(at.date()),
        "execution_session": str(execution.date()),
        "return_basis": BASIS,
        "source_return_basis": "funded_total_return",
        "coefficient_units": "fraction_of_funded_capital",
        "cash_reference": "zero_return_cash",
        "risk_free_subtractions": 0,
        "cash_reference_subtractions": 1,
        "standalone_costs_deducted_again": False,
        "input_evidence_id": evidence,
        "return_prefix_hash": frame_identity(past, at),
        "cash_prefix_hash": frame_identity(pd.Series(0.0, index=past.index), at),
        "previous_state_id": None if previous is None else previous["id"],
        "previous_coefficient_convention": "last_saved_targets; not reconstructed_sleeve_wealth",
    }


def _chronological(returns, at, execution, events, profile, evidence, previous, mode, policy):
    """Pool = members whose latest receipt is available strictly before ``at``.

    Each member contributes only its own prefix (``history_start`` through its
    receipt's ``available_at``). Every pool receipt must share one availability, so a
    stale member cannot hide behind older evidence. A member below the policy minimum
    stays in the pool at weight zero; the optimizer sees the common window of the
    others. A previous member absent from the pool needs a departure available before
    ``at``; its saved coefficient moves to cash in an explicit ``capital_transition``.
    """
    latest, departed = pool_at(events, at)
    names = sorted(latest)
    ends = {row["available_at"] for row in latest.values()}
    previous_pool = set(previous["pool_ids"]) if previous is not None else set()
    missing = previous_pool - set(names)
    if len(ends) > 1 or missing - set(departed):
        raise ValueError(
            "pool retirement or a stale prefix receipt needs an explicit capital transition"
        )
    if set(names) - set(returns.columns):
        raise ValueError("funded return columns must cover every receipt-qualified member")
    past = returns.loc[returns.index < at].sort_index().reindex(columns=names).astype(float)
    if names:
        past = past.loc[past.index <= session_date(ends.pop())]
    for name in names:
        past.loc[past.index < session_date(latest[name]["history_start"]), name] = float("nan")
    allocatable = [name for name in names if past[name].notna().sum() >= policy.minimum]
    starts = [session_date(latest[name]["history_start"]) for name in allocatable]
    window = past.loc[past.index >= max(starts), allocatable] if starts else past[allocatable]
    prior = {
        name: weight
        for name, weight in (previous["weights"] if previous is not None else {}).items()
        if name in allocatable
    }
    result, original_id = _optimize(past, window, at, mode, policy, prior)
    if names and not allocatable:
        result["status"] = "insufficient_history_cash"
    result["weights"] = {name: result["weights"].get(name, 0.0) for name in names}
    retired = {name: previous["weights"][name] for name in sorted(missing)}
    record = _record(result, original_id, past, at, execution, profile, names, evidence, previous)
    record.update(
        version=VERSION,
        policy=asdict(policy),
        optimizer_record_id=original_id,
        formal_admission=False,
        membership_clock=CLOCK,
        membership={name: {key: latest[name][key] for key in RECEIPT_KEYS} for name in names},
        entered=sorted(set(names) - previous_pool),
        insufficient_history_ids=sorted(set(names) - set(allocatable)),
        capital_transition={
            "retired": retired,
            "to_cash": sum(retired.values()),
            "reason": {name: departed[name] for name in retired},
        }
        if retired
        else None,
    )
    return record
