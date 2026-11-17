"""Autopilot core: stage ordering, typed reasons and the chained journal."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from capstone.pipeline import blocker_glossary
from capstone.pipeline.autopilot import reasons as reasons_module
from capstone.pipeline.autopilot.journal import (
    EVENTS,
    GENESIS,
    STATE,
    Journal,
    JournalError,
    sha256_file,
)
from capstone.pipeline.autopilot.reasons import (
    CLASSES,
    CORE,
    Blocked,
    NotImplementedBlocker,
    ReasonSpec,
    Stop,
    record,
)
from capstone.pipeline.autopilot.stage import Context, Stage, StageResult, order

REASONS = {
    "inbox_empty": ReasonSpec("Source", "drop a file"),
    "spend_stop": ReasonSpec("Spend", "a person decides"),
    "missing_input": ReasonSpec("Operator"),  # cleared_by comes from the blocker glossary
}


def toy(name, deps=(), **more):
    def run(ctx):
        return StageResult()

    return Stage(name=name, unit="T", run=run, depends_on=tuple(deps), **more)


# --- DAG ----------------------------------------------------------------------------


def test_order_is_topological_and_stable():
    toys = [toy("c", deps=["a", "b"]), toy("b", deps=["a"]), toy("x"), toy("a")]
    assert [s.name for s in order(toys)] == ["x", "a", "b", "c"]
    with pytest.raises(ValueError, match="unknown stage 'zz'"):
        order([toy("a", deps=["zz"])])
    with pytest.raises(ValueError, match="cycle among"):
        order([toy("a", deps=["b"]), toy("b", deps=["a"])])
    with pytest.raises(ValueError, match="declared twice"):
        order([toy("a"), toy("a")])


def test_a_stage_section_defaults_to_its_name():
    assert toy("a").section == "a"
    assert toy("a", section="shared").section == "shared"


def test_a_stage_result_carries_a_reason_code_exactly_when_it_is_not_done():
    assert StageResult().status == "done"
    assert StageResult(status="blocked", code="inbox_empty").code == "inbox_empty"
    with pytest.raises(ValueError, match="exactly when"):
        StageResult(status="blocked")
    with pytest.raises(ValueError, match="exactly when"):
        StageResult(status="done", code="inbox_empty")
    with pytest.raises(ValueError, match="unknown stage status"):
        StageResult(status="maybe", code="x")


# --- typed reasons --------------------------------------------------------------------


def test_a_declared_reason_is_recorded_with_its_class_and_what_clears_it():
    row = record("inbox_empty", "nothing in the inbox", REASONS)
    assert row == {
        "code": "inbox_empty",
        "class": "Source",
        "detail": "nothing in the inbox",
        "cleared_by": "drop a file",
    }


def test_a_reason_without_its_own_text_takes_the_blocker_glossary_row():
    row = record("missing_input", "x", REASONS)
    assert row["class"] == "Operator"
    assert row["cleared_by"] == blocker_glossary.CLEARED_BY["missing_input"]


def test_an_undeclared_reason_is_refused():
    with pytest.raises(KeyError):
        record("made_up", "y", REASONS)
    with pytest.raises(ValueError, match="unknown blocker class"):
        ReasonSpec("Vibes")


def test_every_core_reason_has_a_known_class_and_a_way_to_clear_it():
    assert set(CLASSES) >= {spec.klass for spec in CORE.values()}
    for code, spec in CORE.items():
        row = record(code, "", CORE)
        assert row["cleared_by"], code
        assert spec.cleared_by or code in blocker_glossary.CLEARED_BY, code
    assert reasons_module.CORE is CORE


def test_blocked_and_stop_carry_a_code_and_a_stop_is_a_blocked():
    blocked, stop = Blocked("inbox_empty", "d"), Stop("spend_stop")
    assert (blocked.status, blocked.code, blocked.detail) == ("blocked", "inbox_empty", "d")
    assert (stop.status, stop.code, str(stop)) == ("stopped", "spend_stop", "spend_stop")
    assert isinstance(stop, Blocked)
    stub = NotImplementedBlocker("daily", "owner-x")
    assert stub.code == "stage_not_implemented" and "owner-x" in stub.detail


# --- context ---------------------------------------------------------------------------


def make_context(tmp_path, *, real_model_calls=False, services=None):
    config = SimpleNamespace(
        run_root=tmp_path / "run",
        core={"policy_id": "p1", "real_model_calls": real_model_calls},
        sections={"a": {"k": 1}},
        section_of={"a": "a"},
    )
    return Context(config, Journal(config.run_root), "a", services or {})


def test_context_reads_its_section_reviewer_and_other_stages_data(tmp_path):
    ctx = make_context(tmp_path)
    assert ctx.section == {"k": 1}
    assert ctx.reviewer == "autopilot:p1"
    assert ctx.path("x", "y.txt") == tmp_path / "run" / "x" / "y.txt"
    ctx.journal.update("b", state="done", data={"made": "b"})
    assert ctx.data("b") == {"made": "b"}
    ctx.data("b")["made"] = "changed"  # a copy: the journal is not edited through it
    assert ctx.data("b") == {"made": "b"}
    assert ctx.record("never-run")["state"] == "pending"


def test_a_model_client_is_injected_or_a_persons_opt_in_and_never_real_under_pytest(
    tmp_path, monkeypatch
):
    fake = object()
    assert make_context(tmp_path, services={"model_client": fake}).model_client() is fake
    with pytest.raises(Stop, match="inject a fake") as caught:  # even with the opt-in
        make_context(tmp_path, real_model_calls=True).model_client()
    assert caught.value.code == "model_client_not_authorized"
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    with pytest.raises(Stop, match="real_model_calls is false"):
        make_context(tmp_path).model_client()
    assert make_context(tmp_path, real_model_calls=True).model_client() is None


def test_context_events_are_stamped_with_the_stage(tmp_path):
    ctx = make_context(tmp_path)
    row = ctx.event("started", n=1)
    assert (row["kind"], row["stage"], row["n"]) == ("started", "a", 1)


# --- journal ---------------------------------------------------------------------------


def test_journal_state_write_is_atomic(tmp_path, monkeypatch):
    journal = Journal(tmp_path)
    journal.update("a", state="done")

    def killed(source, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        journal.update("a", state="blocked")
    monkeypatch.undo()
    assert Journal(tmp_path).stage("a")["state"] == "done"  # the previous state, intact
    assert Journal(tmp_path).stage("never-run")["state"] == "pending"
    (tmp_path / STATE).write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(JournalError, match="schema_version"):
        Journal(tmp_path)


def test_event_log_is_an_append_only_chain_that_survives_a_torn_tail(tmp_path):
    ticks = iter(f"t{n}" for n in range(9))
    journal = Journal(tmp_path, clock=lambda: next(ticks))
    journal.event("one", n=1)
    journal.event("two")
    path = tmp_path / EVENTS
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write('{"seq": 2, "kind": "tor')  # a process killed mid-append
    journal.event("three")
    assert [(e["seq"], e["kind"], e["at"]) for e in Journal(tmp_path).events()] == [
        (0, "one", "t0"),
        (1, "two", "t1"),
        (2, "three", "t2"),
    ]
    assert journal.verify_events() == 3
    lines = path.read_text(encoding="utf-8").split("\n")
    edited = "\n".join([lines[0].replace('"n":1', '"n":2'), *lines[1:]])
    path.write_text(edited, encoding="utf-8", newline="\n")
    with pytest.raises(JournalError, match="event 1 breaks the chain"):
        journal.verify_events()
    path.write_text("\n".join(lines[1:]), encoding="utf-8", newline="\n")  # a dropped line
    with pytest.raises(JournalError, match="event 0 breaks the chain"):
        journal.verify_events()


def test_the_first_event_chains_from_genesis_and_an_empty_log_verifies(tmp_path):
    journal = Journal(tmp_path)
    assert journal.verify_events() == 0
    assert journal.event("one")["prev"] == GENESIS


def test_a_state_update_merges_fields_and_a_reload_sees_it(tmp_path):
    journal = Journal(tmp_path)
    journal.update("a", state="running", attempts=1)
    journal.update("a", state="done")
    again = Journal(tmp_path).stage("a")
    assert (again["state"], again["attempts"], again["crashes"]) == ("done", 1, 0)
    assert not (tmp_path / "AUTOPILOT.json.tmp").exists()


def test_sha256_file_hashes_the_bytes(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
