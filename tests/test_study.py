import json

import pytest

from capstone import runlog
from capstone.pipeline import study


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


def test_definition_is_a_bounded_deterministic_candidate_budget():
    first, second = study.definition(), study.definition()
    assert first == second
    candidates = first["candidates"]
    ids = [c["candidate_id"] for c in candidates]
    assert len(ids) == first["candidate_limit"] == 36 and len(set(ids)) == 36
    assert first["evaluation"][1] == "2020-12-31"
    assert first["holdout_opened"] is False and first["broker_connection"] is False
    assert {c["side"] for c in candidates} == set(study.SIDES)


def test_freeze_logs_before_writing_and_never_overwrites(tmp_path, ledger):
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic")
    target = tmp_path / "registration.json"
    value = study.freeze(target, sources={"source": source})
    entries = runlog.read_entries()
    assert [e["name"] for e in entries] == ["qualification-study-registration"]
    assert entries[0]["params"]["study_hash"] == value["study_hash"]
    assert value["sources"] == {"source": study.file_digest(source)}
    assert study.load(target) == value
    with pytest.raises(FileExistsError):
        study.freeze(target)
    assert len(runlog.read_entries()) == 1


def test_load_refuses_a_changed_registration(tmp_path):
    target = tmp_path / "registration.json"
    study.freeze(target)
    value = json.loads(target.read_text(encoding="utf-8"))
    value["study"]["seed"] = 405
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        study.load(target)


def test_write_once_and_file_digest(tmp_path):
    path = study.write_once(tmp_path / "a" / "b.json", {"b": 1, "a": 2})
    assert path.read_text(encoding="utf-8") == '{"a":2,"b":1}\n'
    with pytest.raises(FileExistsError):
        study.write_once(path, {})
    assert study.file_digest(path) == study.file_digest(path) and len(study.file_digest(path)) == 64


def test_reconcile_explains_every_source_row_and_variant(tmp_path):
    registration = {"study_hash": "h"}
    entries = [{"name": "a", "ts_utc": "t1"}, {"name": "b", "ts_utc": "t2"}]
    ok = {
        "0": {"classification": "strategy_trial", "variant_ids": ["v1"], "reason": "run"},
        "1": {"classification": "diagnostic", "variant_ids": [], "reason": "check"},
    }
    result = study.reconcile(registration, entries, ["v1"], ok)
    assert result["reconciled"] and result["conservative_trial_entries"] == 1
    bad = study.reconcile(registration, entries, ["v1", "v2"], {"0": ok["0"], "1": {}})
    assert not bad["reconciled"]
    assert "unclassified_source_row:1" in bad["blockers"]
    assert "unmapped_return_column:v2" in bad["blockers"]
    unknown = study.reconcile(
        registration, entries, ["v1"], {**ok, "0": {**ok["0"], "variant_ids": ["zz"]}}
    )
    assert "unknown_return_column:0" in unknown["blockers"]
    with pytest.raises(ValueError, match="register"):
        study.reconcile({}, entries, [], {})
    target = tmp_path / "accounting.json"
    study.record_reconciliation(target, result)
    assert runlog.read_entries()[-1]["name"] == "qualification-ledger-reconciliation"
    with pytest.raises(FileExistsError):
        study.record_reconciliation(target, result)
