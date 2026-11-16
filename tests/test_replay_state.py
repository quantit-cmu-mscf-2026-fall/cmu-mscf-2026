"""Synthetic tests for frozen recovery intents and checkpoint-protocol checks."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.replay_checkpoint import checkpoint_document, make_checkpoint
from capstone.execution.replay_state import (
    EVENT_VERSIONS,
    account_state_id,
    checked_checkpoint,
    freeze_intent,
    retain_intent_evidence,
    verify_intent,
)
from capstone.strategy.research import digest

PLAN = {
    "targets": {101: 10, 102: 0},
    "projected": {101: 10},
    "orders": [{"permno": 101, "shares": 10}],
    "omitted": [],
    "sizing_capital": 1000.0,
    "estimated_fees": 0.0,
    "intent": "formation_rebalance",
}
SOURCE = {"kind": "formation", "strategy_id": "s1", "formation_session": "2000-01-03"}
ACCOUNT = ({101: 10}, 900.0, [])


def frozen(plan=PLAN, source=SOURCE, **over):
    return freeze_intent(plan, *ACCOUNT, decision="2000-01-04", source=source | over)


def test_account_state_id_ignores_flat_names_and_key_order():
    base = account_state_id({102: 3, 101: 1, 103: 0}, 5.0, [])
    assert base == account_state_id({101: 1, 102: 3}, 5.0, [])
    assert base != account_state_id({101: 1, 102: 3}, 6.0, [])
    assert base != account_state_id({101: 1, 102: 4}, 5.0, [])
    assert base != account_state_id({101: 1, 102: 3}, 5.0, [{"event_id": "e"}])


def test_freeze_intent_is_self_identifying_and_does_not_mutate_the_plan():
    before = deepcopy(PLAN)
    intent = frozen()
    assert PLAN == before
    assert intent["decision"] == "2000-01-04"
    assert intent["account_state_id"] == account_state_id(*ACCOUNT)
    assert intent["source_intent"]["source_intent_id"] == digest(SOURCE)
    body = {k: v for k, v in intent.items() if k != "intent_id"}
    assert intent["intent_id"] == digest(body)
    assert frozen() == intent
    assert verify_intent(intent) == intent
    assert verify_intent(intent) is not intent


def test_freeze_intent_replaces_a_stale_intent_id():
    assert frozen(PLAN | {"intent_id": "stale"}) == frozen()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda i: i.update(sizing_capital=1.0),
        lambda i: i.update(intent_id="0" * 64),
        lambda i: i.pop("intent_id"),
    ],
)
def test_verify_intent_refuses_tampering(mutate):
    intent = frozen()
    mutate(intent)
    with pytest.raises(ExecutionContractBlocked, match="pending_intent_integrity_failure"):
        verify_intent(intent)


def test_verify_intent_refuses_non_mappings():
    with pytest.raises(ExecutionContractBlocked, match="pending_intent_integrity_failure"):
        verify_intent([])


def reseal(intent, **over):
    body = {k: v for k, v in intent.items() if k != "intent_id"} | over
    return body | {"intent_id": digest(body)}


def test_verify_intent_checks_the_source_after_a_valid_outer_seal():
    intent = frozen()
    tampered = deepcopy(intent)
    tampered["source_intent"]["strategy_id"] = "other"
    with pytest.raises(ExecutionContractBlocked, match="pending_source_integrity_failure"):
        verify_intent(reseal(tampered))
    kind = frozen(kind="rebalance", source_intent_id=None)
    source = {k: v for k, v in kind["source_intent"].items() if k != "source_intent_id"}
    source["source_intent_id"] = digest(source)
    with pytest.raises(ExecutionContractBlocked, match="pending_source_incompatible"):
        verify_intent(reseal(intent, source_intent=source))


@pytest.mark.parametrize(
    "over",
    [
        {"targets": {101: 1.5}},
        {"projected": {"abc": 1}},
        {"orders": [{"permno": True, "shares": 1}]},
    ],
)
def test_verify_intent_refuses_malformed_positions(over):
    intent = frozen()
    with pytest.raises((ExecutionContractBlocked, ValueError)):
        verify_intent(reseal(intent, **over))


def test_exit_only_sources_are_accepted():
    intent = frozen(kind="exit_only")
    assert verify_intent(intent)["source_intent"]["kind"] == "exit_only"


def test_retain_intent_evidence_copies_only_the_evidence_keys():
    row, pending = (
        {},
        {
            "formation_eligibility": {"a": [1]},
            "construction_evidence": {"b": 2},
            "source_intent": {"kind": "formation"},
            "cash": 1,
        },
    )
    retain_intent_evidence(row, pending)
    assert set(row) == {"formation_eligibility", "construction_evidence", "source_intent"}
    pending["construction_evidence"]["b"] = 3
    assert row["construction_evidence"] == {"b": 2}
    retain_intent_evidence(row, None)
    assert len(row) == 3


def test_checked_checkpoint_enforces_the_event_protocol():
    version = EVENT_VERSIONS[0]
    assert checked_checkpoint(version, None, pd.DataFrame(), (None, None)) is None
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_requires_event_protocol"):
        checked_checkpoint("fixed-share-next-open-v2", {"x": 1}, None, ())
    with pytest.raises(ExecutionContractBlocked, match="requires_observed_event_coverage"):
        checked_checkpoint(version, None, None, ())
    with pytest.raises(ExecutionContractBlocked, match="requires_observed_event_coverage"):
        checked_checkpoint(version, None, pd.DataFrame(), (None, 1))
    assert checked_checkpoint("fixed-share-next-open-v2", None, None, (None,)) is None


def test_checked_checkpoint_refuses_a_checkpoint_of_another_protocol():
    record = make_checkpoint(
        config={"profile": "long_only", "min_notional": 1.0},
        origin_initialization={"mode": "m", "qualifying": False},
        origin_session="2000-01-03",
        session="2000-01-05",
        next_session="2000-01-06",
        positions={101: 10},
        cash=900.0,
        actions=SimpleNamespace(receivables=[], applied=set()),
        pending=None,
        terminated=set(),
        previous=1000.0,
        closes={101: 10.0},
        prefix_inputs_hash="h",
        formations=[],
    )
    assert checked_checkpoint(record["protocol"], record, [], ()) == checkpoint_document(record)
    other = next(v for v in EVENT_VERSIONS if v != record["protocol"])
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_protocol_incompatible"):
        checked_checkpoint(other, record, [], ())
