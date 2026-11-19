"""Autopilot tick: stage ordering, STOP/clear, crash recovery, hooks and the spend-guard seam.

Synthetic stages only; no data, model or broker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from capstone.locking import exclusive_file
from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import runner
from capstone.pipeline.autopilot.config import Key
from capstone.pipeline.autopilot.journal import LOCK, STATE, Journal
from capstone.pipeline.autopilot.reasons import Blocked, ReasonSpec, Stop
from capstone.pipeline.autopilot.stage import Stage, StageResult

REPO = Path(__file__).resolve().parents[1]

REASONS = {
    "inbox_empty": ReasonSpec("Source", "drop a file"),
    "spend_stop": ReasonSpec("Spend", "a person decides"),
    "missing_input": ReasonSpec("Operator"),  # cleared_by comes from the blocker glossary
}


def toy(name, run=None, deps=(), **more):
    def write(ctx):
        ctx.path(name).mkdir(parents=True, exist_ok=True)
        ctx.path(name, "out.txt").write_text(name, encoding="utf-8")
        return StageResult(artifacts=[ctx.path(name, "out.txt")], data={"made": name})

    more.setdefault("reasons", REASONS)
    return Stage(name=name, unit="T", run=run or write, depends_on=tuple(deps), **more)


def make_config(tmp_path, toys, **sections):
    core = {"run_root": str(tmp_path / "run"), "policy_id": "p1"}
    return cfg.validate({"autopilot": core, **sections}, toys, path=tmp_path / "autopilot.toml")


def raising(error):
    def run(ctx):
        raise error

    return run


KILL_SCRIPT = """
import os, sys
from pathlib import Path
from capstone.pipeline.autopilot import config as cfg, runner
from capstone.pipeline.autopilot.stage import Stage, StageResult


root = Path(sys.argv[1])

def a(ctx):
    with (root / "a-runs.txt").open("a") as stream:
        stream.write("x")
    return StageResult(data={"pid": os.getpid()})

def b(ctx):
    half = ctx.path("b-half.txt")
    if not half.exists():
        half.parent.mkdir(parents=True, exist_ok=True)
        half.write_text("half")
        os._exit(9)  # killed mid-stage: no cleanup, no journal write
    ctx.path("b.txt").write_text(half.read_text() + "+rest")
    return StageResult(artifacts=[ctx.path("b.txt")], data={"pid": os.getpid()})

toys = [Stage("a", "T", a), Stage("b", "T", b, depends_on=("a",))]
raw = {"autopilot": {"run_root": str(root / "run"), "policy_id": "p1"}}
report = runner.tick(cfg.validate(raw, toys), toys)
print(report.line())
sys.exit(report.exit_code)
"""


def test_a_killed_tick_resumes_in_a_fresh_process(tmp_path):
    script = tmp_path / "kill.py"
    script.write_text(KILL_SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    command = [sys.executable, str(script), str(tmp_path)]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 9, first.stderr
    killed = Journal(tmp_path / "run")
    assert killed.stage("a")["state"] == "done" and killed.stage("b")["state"] == "running"
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "autopilot: ok (2/2 stages done, ran 1)"
    journal = Journal(tmp_path / "run")
    a, b = journal.stage("a"), journal.stage("b")
    assert (tmp_path / "a-runs.txt").read_text() == "x"  # the done stage was not run again
    assert b["state"] == "done" and b["attempts"] == 2 and a["data"]["pid"] != b["data"]["pid"]
    assert (tmp_path / "run/b.txt").read_text() == "half+rest"
    starts = [e for e in journal.events() if e["kind"] == "stage_started" and e["stage"] == "b"]
    assert [e["resumed"] for e in starts] == [False, True]
    assert journal.verify_events() == len(journal.events())
    # the lock a killed process held is released by the operating system
    third = subprocess.run(command, env=env, capture_output=True, text=True)
    assert third.returncode == 0 and "ran 0" in third.stdout


# --- lock ---------------------------------------------------------------------------


def test_a_second_tick_is_refused_while_the_lock_is_held(tmp_path):
    toys = [toy("a")]
    config = make_config(tmp_path, toys)
    with exclusive_file(config.run_root / LOCK):
        report = runner.tick(config, toys)
    assert report.exit_code == 2 and report.ran == []
    assert report.problems[0]["reason"]["code"] == "autopilot_locked"
    assert not (config.run_root / STATE).exists()  # the refused tick wrote nothing
    assert runner.tick(config, toys).exit_code == 0


# --- tick ---------------------------------------------------------------------------


def test_tick_runs_ready_stages_in_order_and_is_idempotent(tmp_path):
    toys = [toy("c", deps=["b"]), toy("b", deps=["a"]), toy("a")]
    config = make_config(tmp_path, toys)
    report = runner.tick(config, toys)
    assert report.ran == ["a", "b", "c"] and report.exit_code == 0
    assert report.line() == "autopilot: ok (3/3 stages done, ran 3)"
    record = Journal(config.run_root).stage("b")
    assert record["data"] == {"made": "b"} and list(record["artifacts"]) == ["b/out.txt"]
    assert len(record["artifacts"]["b/out.txt"]) == 64
    again = runner.tick(config, toys)
    assert again.ran == [] and again.exit_code == 0
    kinds = [e["kind"] for e in Journal(config.run_root).events()]
    assert kinds == ["run_started"] + ["stage_started", "stage_finished"] * 3


def test_a_blocked_stage_skips_its_dependents_and_is_retried(tmp_path):
    inbox = tmp_path / "inbox"

    def a(ctx):
        if not inbox.exists():
            raise Blocked("inbox_empty", "nothing\n in the inbox")
        return StageResult()

    toys = [toy("a", a), toy("b", deps=["a"]), toy("free")]
    config = make_config(tmp_path, toys)
    report = runner.tick(config, toys)
    assert report.ran == ["a", "free"] and report.exit_code == 2
    assert report.states == {"a": "blocked", "free": "done", "b": "pending"}
    assert report.line() == (
        "autopilot: blocked a: inbox_empty: nothing in the inbox (cleared by: drop a file)"
    )
    reason = Journal(config.run_root).stage("a")["reason"]
    assert reason["class"] == "Source" and reason["code"] == "inbox_empty"
    inbox.mkdir()
    after = runner.tick(config, toys)
    assert after.ran == ["a", "b"] and after.exit_code == 0
    assert Journal(config.run_root).stage("a")["reason"] is None


def test_a_stop_halts_the_run_until_a_person_clears_it(tmp_path):
    calls = []

    def a(ctx):
        calls.append(1)
        if len(calls) == 1:
            return StageResult(status="stopped", code="spend_stop", detail="301 > 300")
        return StageResult()

    toys = [toy("a", a), toy("later")]
    config = make_config(tmp_path, toys)
    first = runner.tick(config, toys)
    assert first.ran == ["a"] and first.states["later"] == "pending"  # the tick ended at the STOP
    again = runner.tick(config, toys)
    assert again.ran == [] and again.exit_code == 2 and len(calls) == 1
    assert again.line().startswith("autopilot: stopped a: spend_stop: 301 > 300 (cleared by:")
    with pytest.raises(ValueError, match="not stopped"):
        runner.clear(config, toys, "later", reviewer="reviewer-a", reason="x")
    with pytest.raises(ValueError, match="reviewer and a reason"):
        runner.clear(config, toys, "a", reviewer=" ", reason="x")
    runner.clear(config, toys, "a", reviewer="reviewer-a", reason="new authorization")
    cleared = Journal(config.run_root).events()[-1]
    assert (cleared["kind"], cleared["reviewer"]) == ("stop_cleared", "reviewer-a")
    assert cleared["was"]["reason"]["code"] == "spend_stop"
    assert runner.tick(config, toys).exit_code == 0 and len(calls) == 2


def test_crashes_are_typed_then_stop_at_the_limit(tmp_path):
    toys = [toy("a", raising(RuntimeError("boom"))), toy("exits", raising(SystemExit(2)))]
    config = make_config(tmp_path, toys)
    first = runner.tick(config, toys)
    assert [p["reason"]["code"] for p in first.problems] == ["stage_crashed", "stage_crashed"]
    assert first.problems[0]["reason"]["detail"] == "RuntimeError: boom"
    runner.tick(config, toys)
    third = runner.tick(config, toys)
    assert third.problems[0]["status"] == "stopped"
    assert third.problems[0]["reason"]["code"] == "stage_crash_limit"
    assert Journal(config.run_root).stage("a")["crashes"] == 3


def test_reasons_are_typed_and_undeclared_codes_are_flagged(tmp_path):
    toys = [
        toy("glossary", raising(Blocked("missing_input", "x"))),
        toy("rogue", raising(Stop("made_up", "y"))),
    ]
    config = make_config(tmp_path, toys)
    report = runner.tick(config, toys)
    known, rogue = (p["reason"] for p in report.problems)
    assert known["class"] == "Operator" and "produce it with" in known["cleared_by"]
    assert rogue["code"] == "undeclared_reason" and rogue["detail"] == "made_up: y"
    assert report.problems[1]["status"] == "stopped"  # a STOP stays a STOP
    with pytest.raises(ValueError, match="unknown blocker class"):
        ReasonSpec("Vibes")
    with pytest.raises(ValueError, match="exactly when"):
        StageResult(status="blocked")
    not_a_result = [toy("a", lambda ctx: None)]
    wrong = runner.tick(make_config(tmp_path / "w", not_a_result), not_a_result)
    assert "not StageResult" in wrong.problems[0]["reason"]["detail"]


def test_a_changed_or_missing_artifact_is_a_stop(tmp_path):
    missing = [toy("a", lambda ctx: StageResult(artifacts=["nope.txt"]))]
    report = runner.tick(make_config(tmp_path / "m", missing), missing)
    assert report.problems[0]["reason"]["code"] == "no_stage_artifact"
    toys = [toy("a"), toy("b", deps=["a"])]
    config = make_config(tmp_path, toys)
    assert runner.tick(config, toys).exit_code == 0
    (config.run_root / "a/out.txt").write_text("tampered", encoding="utf-8")
    after = runner.tick(config, toys)
    assert after.problems == [
        {
            "stage": "a",
            "status": "stopped",
            "reason": Journal(config.run_root).stage("a")["reason"],
        }
    ]
    assert after.problems[0]["reason"]["code"] == "artifact_changed"
    assert after.problems[0]["reason"]["detail"] == "a/out.txt"


def test_a_started_stage_freezes_its_section_and_the_policy(tmp_path):
    toys = [toy("a", schema={"n": Key("int", 1)}), toy("b", deps=["a"])]
    assert runner.tick(make_config(tmp_path, toys), toys).exit_code == 0
    changed = runner.tick(make_config(tmp_path, toys, a={"n": 2}), toys)
    assert changed.ran == [] and changed.problems[0]["stage"] == "a"
    assert changed.problems[0]["reason"]["code"] == "config_section_changed"
    raw = {"autopilot": {"run_root": str(tmp_path / "run"), "policy_id": "p2"}}
    other = runner.tick(cfg.validate(raw, toys), toys)
    assert other.problems[0]["reason"]["code"] == "policy_changed" and other.ran == []


def test_recurring_stages_run_every_tick_and_hooks_run_last(tmp_path):
    runs, seen = [], []

    def daily(ctx):
        runs.append(ctx.reviewer)
        ctx.event("session", n=len(runs))
        return StageResult(data={"sessions": len(runs), "upstream": ctx.data("a")})

    def hook(ctx, report):
        seen.append((ctx.stage, list(report.ran), ctx.record("daily")["data"]["sessions"]))

    def broken(ctx, report):
        raise OSError("smtp down")

    toys = [toy("a"), toy("daily", daily, deps=["a"], recurring=True)]
    config = make_config(tmp_path, toys)
    clock = iter(f"t{n}" for n in range(99))
    services = {"clock": lambda: next(clock)}
    assert runner.tick(config, toys, hooks=[hook], services=services).exit_code == 0
    report = runner.tick(config, toys, hooks=[hook, broken])
    assert runs == ["autopilot:p1"] * 2
    assert seen == [(None, ["a", "daily"], 1), (None, ["daily"], 2)]
    assert report.exit_code == 2 and report.problems[0]["reason"]["code"] == "ops_hook_failed"
    journal = Journal(config.run_root)
    assert journal.stage("daily")["data"] == {"sessions": 2, "upstream": {"made": "a"}}
    sessions = [e for e in journal.events() if e["kind"] == "session"]
    assert [e["stage"] for e in sessions] == ["daily", "daily"]
    assert journal.events()[0]["at"] == "t0"  # the injected clock stamps the journal


def test_stages_log_to_the_project_ledger_and_a_second_run_root_sees_the_first(
    tmp_path, monkeypatch
):
    from capstone import runlog

    project = tmp_path / "project"
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(project))
    counts = []

    def trial(ctx):
        runlog.log_run("autopilot-search")
        counts.append(runlog.trial_count("autopilot-search"))
        return StageResult(data={})

    toys = [toy("a", trial)]
    for name in ("run1", "run2"):
        core = {"run_root": str(tmp_path / name), "policy_id": "p1"}
        config = cfg.validate({"autopilot": core}, toys, path=tmp_path / "autopilot.toml")
        runner.tick(config, toys)
    assert counts == [1, 2]
    assert os.environ["CAPSTONE_LEDGER_DIR"] == str(project)
    assert not (tmp_path / "run1" / "ledger").exists()
    lines = (project / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    tags = json.loads(lines[1])["tags"]
    assert tags == [f"autopilot_run_root:{tmp_path / 'run2'}", "autopilot_policy:p1"]
    assert "AUTOPILOT_RUN_ROOT" not in os.environ


def test_only_runs_a_units_own_stages_over_a_seeded_journal(tmp_path):
    other = raising(RuntimeError("another unit's stage"))
    toys = [toy("up", other), toy("mine", deps=["up"]), toy("down", other, deps=["mine"])]
    config = make_config(tmp_path, toys)
    waiting = runner.tick(config, toys, only=("mine",))
    assert waiting.ran == [] and waiting.exit_code == 0 and waiting.states["mine"] == "pending"
    Journal(config.run_root).update("up", state="done", data={"k": 1})
    report = runner.tick(config, toys, only=("mine",))
    assert report.ran == ["mine"] and report.exit_code == 0
    assert report.states == {"up": "done", "mine": "done", "down": "pending"}
    with pytest.raises(ValueError, match="unknown stages in only"):
        runner.tick(config, toys, only=("nope",))


# --- spend-guard seam ------------------------------------------------------------------


def spending(ceiling, ran):
    def run(ctx):
        ran.append(1)
        return StageResult()

    return toy("buy", run, spend=lambda ctx: ceiling, spend_scope="calls")


def test_a_spending_stage_passes_the_callers_guard_before_it_runs(tmp_path):
    ran, seen = [], []

    def guard(ctx, scope, ceiling):
        seen.append((ctx.stage, scope, ceiling))

    toys = [spending(7, ran)]
    config = make_config(tmp_path, toys)
    assert runner.tick(config, toys, spend_guard=guard).exit_code == 0
    assert seen == [("buy", "calls", 7)] and ran == [1]


def test_a_guard_that_stops_keeps_the_stage_from_running(tmp_path):
    ran = []

    def guard(ctx, scope, ceiling):
        raise Stop("spend_stop", f"{ceiling} calls not authorized")

    toys = [spending(7, ran)]
    config = make_config(tmp_path, toys)
    report = runner.tick(config, toys, spend_guard=guard)
    assert ran == [] and report.problems[0]["status"] == "stopped"
    assert report.problems[0]["reason"]["code"] == "spend_stop"


def test_a_spending_stage_with_no_guard_supplied_stops_and_a_none_ceiling_needs_none(tmp_path):
    ran = []
    toys = [spending(7, ran)]
    report = runner.tick(make_config(tmp_path / "a", toys), toys)
    assert ran == [] and report.problems[0]["reason"]["code"] == "spend_guard_missing"
    assert report.problems[0]["status"] == "stopped"
    free = [spending(None, ran)]
    assert runner.tick(make_config(tmp_path / "b", free), free).exit_code == 0 and ran == [1]


def test_status_lists_every_stage_without_running_or_locking(tmp_path):
    toys = [toy("a"), toy("b", deps=["a"])]
    config = make_config(tmp_path, toys)
    table = runner.status(config, toys)
    assert [r["stage"] for r in table["stages"]] == ["a", "b"]
    assert {r["state"] for r in table["stages"]} == {"pending"}
    assert table["policy_id"] is None and table["events"] == 0
    assert not (config.run_root / STATE).exists()
    runner.tick(config, toys)
    after = runner.status(config, toys)
    assert {r["state"] for r in after["stages"]} == {"done"} and after["policy_id"] == "p1"
