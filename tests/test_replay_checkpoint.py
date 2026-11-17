"""Synthetic checkpoint tests: integrity, canonical storage and restore refusals."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.replay_checkpoint import (
    PROGRAM_EVENTS_VERSION,
    checkpoint_document,
    make_checkpoint,
    portable,
    read_checkpoint,
    restore_checkpoint,
    write_checkpoint,
)
from capstone.strategy.research import digest

CALENDAR = pd.bdate_range("2000-01-03", "2000-01-14")
PRIOR, NEXT = CALENDAR[2], CALENDAR[3]
CONFIG = {"profile": "long_only", "min_notional": 1.0}
INIT = {"mode": "diagnostic_flat_until_scheduled_formation", "qualifying": False}


def build(**over):
    actions = SimpleNamespace(receivables=[], applied={"e1"})
    args = {
        "config": CONFIG,
        "origin_initialization": INIT,
        "origin_session": str(CALENDAR[0].date()),
        "session": PRIOR,
        "next_session": NEXT,
        "positions": {101: 10, 102: 0},
        "cash": 900.0,
        "actions": actions,
        "pending": {"intent_id": "x"},
        "terminated": {7, 3},
        "previous": 1000.0,
        "closes": {101: 10.0},
        "prefix_inputs_hash": "h",
        "formations": ["2000-01-03"],
    }
    return make_checkpoint(**(args | over))


def resealed(**over):
    record = build() | over
    record.pop("checkpoint_id")
    return record | {"checkpoint_id": digest(record)}


def test_make_checkpoint_is_canonical_and_self_identifying():
    record = build()
    assert record["positions"] == {"101": 10}
    assert record["terminated"] == [3, 7]
    assert record["protocol"] == PROGRAM_EVENTS_VERSION
    body = {k: v for k, v in record.items() if k != "checkpoint_id"}
    assert record["checkpoint_id"] == digest(body)
    assert build() == record
    assert checkpoint_document(record) == record
    assert checkpoint_document(record) is not record


def test_portable_normalises_to_json_types():
    assert portable({"b": (1, 2), "a": {"z": 1.5, "y": None}}) == {
        "a": {"y": None, "z": 1.5},
        "b": [1, 2],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(cash=1.0),
        lambda r: r.update(extra=1),
        lambda r: r.pop("pending"),
        lambda r: r.update(schema_version=True),
    ],
)
def test_tampering_is_refused(mutate):
    record = build()
    mutate(record)
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_integrity_failure"):
        checkpoint_document(record)


@pytest.mark.parametrize(
    "over, code",
    [
        ({"next_session": "2000-01-03"}, "checkpoint_session_incompatible"),
        (
            {"session": "2021-01-04", "next_session": "2021-01-05"},
            "checkpoint_session_incompatible",
        ),
        ({"terminated": [1, 1]}, "checkpoint_provenance_invalid"),
        ({"terminated": [0]}, "checkpoint_terminated_identity_invalid"),
        (
            {"origin_initialization": {"mode": "m", "qualifying": 1}},
            "checkpoint_origin_initialization_incompatible",
        ),
    ],
)
def test_resealed_but_invalid_documents_are_refused(over, code):
    with pytest.raises(ExecutionContractBlocked, match=code):
        checkpoint_document(resealed(**over))


def test_write_read_round_trip_and_idempotence(tmp_path):
    record = build()
    path = write_checkpoint(record, tmp_path / "sub" / "cp.json")
    assert read_checkpoint(path) == record
    assert write_checkpoint(record, path) == path
    assert not path.with_suffix(".json.partial").exists()


def test_write_refuses_to_replace_a_different_checkpoint(tmp_path):
    path = write_checkpoint(build(), tmp_path / "cp.json")
    with pytest.raises(FileExistsError):
        write_checkpoint(build(cash=901.0, previous=1001.0), path)


def test_write_quarantines_a_foreign_partial_file(tmp_path):
    path = tmp_path / "cp.json"
    path.with_suffix(".json.partial").write_text("junk", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_checkpoint(build(), path)
    assert not path.exists()


def test_read_refuses_corrupt_files(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_integrity_failure"):
        read_checkpoint(path)


@pytest.fixture
def stubbed_modules(monkeypatch):
    # replay_state and daily_planning are imported lazily; the refusals below fire first.
    stub = SimpleNamespace(
        account_state_id=None,
        verify_intent=None,
        program_initialization_due=None,
        window_start_due=None,
        program_replay_plan=None,
    )
    for name in ("replay_state", "daily_planning"):
        monkeypatch.setitem(sys.modules, f"capstone.execution.{name}", stub)


def restore(record, **over):
    args = {
        "config": CONFIG,
        "start": NEXT,
        "calendar": CALENDAR,
        "closes": pd.DataFrame({101: 10.0}, index=CALENDAR),
        "actions": SimpleNamespace(
            receivables=[],
            applied=set(),
            events=pd.DataFrame({"event_date": [], "event_id": []}),
        ),
        "prefix_inputs_hash": "h",
        "spec": None,
        "panel": None,
        "rates": lambda day: 0.0,
    }
    return restore_checkpoint(record, **(args | over))


@pytest.mark.parametrize(
    "over, code",
    [
        ({"config": {"profile": "long_only", "min_notional": 2.0}}, "inputs_or_implementation"),
        ({"prefix_inputs_hash": "other"}, "inputs_or_implementation"),
        ({"start": CALENDAR[5]}, "requires_next_authoritative_session"),
    ],
)
def test_restore_refuses_incompatible_inputs(stubbed_modules, over, code):
    with pytest.raises(ExecutionContractBlocked, match=code):
        restore(build(), **over)


def test_restore_refuses_stale_marks_and_unreconciled_accounts(stubbed_modules):
    with pytest.raises(ExecutionContractBlocked, match="previous_valuation"):
        restore(build(), closes=pd.DataFrame({101: 11.0}, index=CALENDAR))
    with pytest.raises(ExecutionContractBlocked, match="account_does_not_reconcile"):
        restore(resealed(cash=901.0))


def test_restore_wraps_a_malformed_record(stubbed_modules):
    record = deepcopy(build())
    record["positions"] = {"not-a-permno": 1}
    body = {k: v for k, v in record.items() if k != "checkpoint_id"}
    record["checkpoint_id"] = digest(body)
    with pytest.raises(ExecutionContractBlocked):
        restore(record)
