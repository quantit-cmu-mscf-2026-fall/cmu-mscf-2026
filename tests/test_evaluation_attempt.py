"""Checkpointed evaluation attempts: resume, refusal and exactly-once ledger, synthetic only."""

import json

import pytest

from capstone import runlog
from capstone.execution import replay_identity
from capstone.pipeline import evaluation_attempt as module
from capstone.pipeline.evaluation_attempt import (
    ATTEMPT_SCHEMA,
    checkpoint_opportunity,
    open_attempt,
)

UNITS = [("opp-a", ["out/a"]), ("opp-b", ["out/b"])]
OUTPUTS = ["out/a", "out/b", "out/controls"]


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def start(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    return root, open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")


def test_new_attempt_records_schema_and_ledger_position(tmp_path):
    root, attempt = start(tmp_path)
    assert attempt["schema"] == ATTEMPT_SCHEMA and attempt["status"] == "started"
    assert attempt["completed_opportunities"] == [] and attempt["ledger_lines"] == 0
    assert json.loads((root / "attempt.json").read_text(encoding="utf-8")) == attempt


def test_resume_keeps_committed_evidence_and_discards_the_rest(tmp_path):
    root, attempt = start(tmp_path)
    write(root, "out/a/result.txt", "first")
    checkpoint_opportunity(root, attempt, "opp-a", ["out/a"], {"status": "evaluated"})
    write(root, "out/b/partial.txt", "interrupted")  # never committed
    write(root, "out/controls/c.txt", "controls")
    resumed = open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")
    assert [e["id"] for e in resumed["completed_opportunities"]] == ["opp-a"]
    assert (root / "out/a/result.txt").read_text(encoding="utf-8") == "first"
    assert not (root / "out/b/partial.txt").exists()
    assert not (root / "out/controls/c.txt").exists()
    (recovery,) = resumed["recoveries"]
    assert recovery["resumed_after"] == 1
    assert sorted(recovery["discarded"]) == ["out/b/partial.txt", "out/controls/c.txt"]


def test_resume_refuses_changed_missing_or_added_evidence(tmp_path):
    root, attempt = start(tmp_path)
    write(root, "out/a/result.txt", "first")
    checkpoint_opportunity(root, attempt, "opp-a", ["out/a"], {"status": "evaluated"})
    write(root, "out/a/result.txt", "edited")
    with pytest.raises(ValueError, match="changed or incomplete"):
        open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")
    write(root, "out/a/result.txt", "first")
    write(root, "out/a/extra.txt", "added")
    with pytest.raises(ValueError, match="changed or incomplete"):
        open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")
    (root / "out/a/extra.txt").unlink()
    (root / "out/a/result.txt").unlink()
    with pytest.raises(ValueError, match="changed or incomplete"):
        open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")


def test_resume_refuses_another_manifest_or_order_and_legacy_attempts(tmp_path):
    root, attempt = start(tmp_path)
    write(root, "out/a/result.txt", "first")
    checkpoint_opportunity(root, attempt, "opp-a", ["out/a"], {})
    with pytest.raises(ValueError, match="do not belong"):
        open_attempt(root, "m2", UNITS, OUTPUTS, legacy="old attempt")
    with pytest.raises(ValueError, match="do not belong"):
        open_attempt(root, "m1", UNITS[::-1], OUTPUTS, legacy="old attempt")
    (root / "attempt.json").write_text(
        json.dumps({"manifest_id": "m1", "status": "started"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="old attempt"):
        open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")


def test_unfinished_atomic_write_is_removed(tmp_path):
    root, _ = start(tmp_path)
    (root / "attempt.json.tmp").write_text("{", encoding="utf-8")
    open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")
    assert not list(root.glob("*.json.tmp"))


def replay_params(name):
    return {"strategy_id": name, "n": 1}


def test_resume_logs_each_replay_trial_exactly_once(tmp_path):
    root, _ = start(tmp_path)  # the attempt began with an empty ledger
    # the killed attempt registered one replay (before its result existed), then was cut
    # off in the middle of a second write
    runlog.log_run(replay_identity.REPLAY_ROW, params=replay_params("s1"))
    runlog.log_run("unrelated", params={"x": 1})
    with runlog.ledger_path().open("a", encoding="utf-8") as stream:
        stream.write('{"name": "reference-replay", "par\n')
    assert module._logged_since({"ledger_lines": 0}) == [
        replay_identity.replay_key(replay_params("s1"))
    ]
    assert module._logged_since({}) == []  # an attempt that predates the count
    before = len(runlog.ledger_path().read_text(encoding="utf-8").splitlines())
    with replay_identity.resume_scope():
        open_attempt(root, "m1", UNITS, OUTPUTS, legacy="old attempt")
        replay_identity.log_registered(replay_params("s1"))  # already on the ledger: skipped
        replay_identity.log_registered(replay_params("s2"))  # a new trial: logged once
    lines = runlog.ledger_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == before + 1
    assert json.loads(lines[-1])["params"] == replay_params("s2")
