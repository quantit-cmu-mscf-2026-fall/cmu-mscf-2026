import sqlite3

import pytest

from capstone.pipeline import offline_recovery as recovery


def journal_with(path, plan_id, states):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE plans (id TEXT)")
    db.execute("CREATE TABLE instructions (id TEXT, plan TEXT, state TEXT)")
    db.execute("INSERT INTO plans VALUES (?)", (plan_id,))
    db.executemany(
        "INSERT INTO instructions VALUES (?, ?, ?)",
        [(name, plan_id, state) for name, state in states.items()],
    )
    db.commit()
    db.close()


def test_nothing_to_recover_without_a_journal(tmp_path):
    assert recovery.interrupted_simulation(tmp_path, "plan") is None


def test_a_saved_broker_means_the_simulation_finished(tmp_path):
    journal_with(tmp_path / "execution.sqlite", "plan", {"a": "filled"})
    (tmp_path / "simulated-broker.json").write_text("{}")
    assert recovery.interrupted_simulation(tmp_path, "plan") is None


def test_a_plan_never_journaled_is_not_an_interruption(tmp_path):
    journal_with(tmp_path / "execution.sqlite", "other", {"a": "filled"})
    assert recovery.interrupted_simulation(tmp_path, "plan") is None


def test_interrupted_simulation_reports_instruction_states(tmp_path):
    journal_with(
        tmp_path / "execution.sqlite", "plan", {"b": "unknown", "a": "filled", "c": "submitting"}
    )
    assert recovery.interrupted_simulation(tmp_path, "plan") == {
        "a": "filled",
        "b": "unknown",
        "c": "submitting",
    }


def test_recover_needs_the_sessions_market(tmp_path, monkeypatch):
    journal_with(tmp_path / "execution.sqlite", "plan", {"a": "unknown"})
    monkeypatch.setattr(
        recovery, "read_plan", lambda path: {"plan_id": "plan", "execution_session": "2019-03-01"}
    )
    with pytest.raises(ValueError, match="synthetic market"):
        recovery.recover_simulation(None, "plan.json", tmp_path, market=None)
    with pytest.raises(ValueError, match="synthetic market"):
        recovery.recover_simulation(None, "plan.json", tmp_path, market={"session": "2019-03-04"})


def test_recover_returns_none_when_nothing_was_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "read_plan", lambda path: {"plan_id": "plan"})
    assert recovery.recover_simulation(None, "plan.json", tmp_path, market={}) is None
