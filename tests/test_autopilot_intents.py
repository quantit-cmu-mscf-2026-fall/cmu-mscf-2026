"""Write-ahead intents and exactly-once events: resume reuses the first attempt's values."""

from __future__ import annotations

import pytest

from capstone import faults
from capstone.faults import InjectedFault
from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import intents
from capstone.pipeline.autopilot.intents import IntentFile, event_once, intent_id
from capstone.pipeline.autopilot.journal import Journal
from capstone.pipeline.autopilot.stage import Context, Stage, StageResult


def make_context(tmp_path, stage="effect"):
    toys = [Stage(name=stage, unit="T", run=lambda ctx: StageResult())]
    core = {"run_root": str(tmp_path / "run"), "policy_id": "p1"}
    config = cfg.validate({"autopilot": core}, toys, path=tmp_path / "autopilot.toml")
    return Context(config, Journal(config.run_root), stage)


def test_intent_ids_are_deterministic_and_depend_on_every_part():
    assert intent_id("a", 1) == intent_id("a", 1)
    assert intent_id("a", 1) != intent_id("a", 2)
    assert intent_id("a", 1) != intent_id(1, "a")
    assert len(intent_id("a")) == 16


def test_the_first_open_wins_and_a_resumed_attempt_reuses_its_values(tmp_path):
    ctx = make_context(tmp_path)
    file = IntentFile.of(ctx)
    assert file.path == ctx.path("autopilot", "intents", "effect.json")
    assert file.read() == {}
    first = file.open("k", at="2026-01-01T00:00:00Z", calls=3)
    again = IntentFile.of(ctx).open("k", at="2099-01-01T00:00:00Z", calls=9)
    assert first == again == {"status": "open", "at": "2026-01-01T00:00:00Z", "calls": 3}
    assert file.get("missing") is None


def test_put_replaces_and_close_keeps_the_payload_and_adds_the_result(tmp_path):
    file = IntentFile(tmp_path / "x" / "intents.json")
    file.open("k", calls=1)
    file.put("k", calls=2)
    assert file.get("k") == {"status": "open", "calls": 2}
    closed = file.close("k", result="ok")
    assert closed == {"status": "closed", "calls": 2, "result": "ok"}
    assert file.close("other", result="r") == {"status": "closed", "result": "r"}
    assert not list(file.path.parent.glob("*.tmp"))  # the replace is atomic


def test_an_event_is_written_once_however_often_the_effect_is_found_done(tmp_path):
    ctx = make_context(tmp_path)
    first = event_once(ctx, "k1", "effect_done", n=1)
    second = event_once(ctx, "k1", "effect_done", n=2)
    other = event_once(ctx, "k2", "effect_done", n=3)
    assert second == first and second["n"] == 1 and other["n"] == 3
    rows = [e for e in ctx.journal.events() if e["kind"] == "effect_done"]
    assert [(e["once"], e["n"]) for e in rows] == [("k1", 1), ("k2", 3)]


def test_a_kill_between_the_effect_and_its_event_loses_nothing(tmp_path, monkeypatch):
    """The event is recorded on every pass that finds the effect done, so a crash after
    the effect and before the event is repaired by the resumed attempt."""
    monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
    monkeypatch.setenv("CAPSTONE_FAULT_AT", "autopilot:effect:after_effect")
    faults._fault_plan.cache_clear()
    faults._fault_hits.clear()
    ctx = make_context(tmp_path)
    effects = []

    def attempt():
        file = IntentFile.of(ctx)
        record = file.open("k", at=ctx.now())
        if not effects:
            effects.append(record["at"])  # the effect, once
        intents.fault("autopilot:effect:after_effect")
        event_once(ctx, "k", "effect_done", at=record["at"])
        file.close("k", done=True)

    try:
        with pytest.raises(InjectedFault):
            attempt()
        assert not [e for e in ctx.journal.events() if e["kind"] == "effect_done"]
        attempt()  # the resumed attempt: the effect is not repeated, the event is written
        attempt()  # a further pass repeats nothing
    finally:
        faults._fault_plan.cache_clear()
        faults._fault_hits.clear()
    rows = [e for e in ctx.journal.events() if e["kind"] == "effect_done"]
    assert len(rows) == 1 and len(effects) == 1 and rows[0]["at"] == effects[0]
    assert IntentFile.of(ctx).get("k")["status"] == "closed"


def test_the_fault_point_is_inert_without_the_test_gate(monkeypatch):
    monkeypatch.delenv("CAPSTONE_FAULT_INJECTION", raising=False)
    faults._fault_plan.cache_clear()
    intents.fault("autopilot:effect:after_effect")
