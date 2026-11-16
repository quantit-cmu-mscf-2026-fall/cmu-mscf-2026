"""Spend guard: caller-supplied ceilings, cross-run accounting and the cross-run lock.

Synthetic run roots only; no data, model or broker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import runner, spend
from capstone.pipeline.autopilot.journal import Journal
from capstone.pipeline.autopilot.reasons import CORE, Blocked, Stop
from capstone.pipeline.autopilot.stage import Context, Stage, StageResult

REPO = Path(__file__).resolve().parents[1]
POLICY = {"run_ceiling": 200, "global_ceiling": 300}  # two runs of 200 cannot both fit
WAIT = 60.0  # seconds a test waits for a child's signal file


def toy(name, run=None, deps=(), **more):
    def write(ctx):
        return StageResult()

    return Stage(name=name, unit="T", run=run or write, depends_on=tuple(deps), **more)


def make_config(tmp_path, toys, **sections):
    core = {"run_root": str(tmp_path / "run"), "policy_id": "p1"}
    return cfg.validate({"autopilot": core, **sections}, toys, path=tmp_path / "autopilot.toml")


CHILD = r"""
import json, sys, time
from contextlib import nullcontext
from pathlib import Path

from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import spend
from capstone.pipeline.autopilot.journal import Journal
from capstone.pipeline.autopilot.reasons import Blocked
from capstone.pipeline.autopilot.stage import Context

runs, name, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
policy = {**json.loads(sys.argv[4]), "runs_dir": str(runs)}
raw = {"autopilot": {"run_root": str(runs / name), "policy_id": "p1"}, "spend": policy}
config = cfg.validate(raw, [], path=runs.parent / "autopilot.toml")
if mode == "hold":  # take the lock, say so, and wait to be killed
    with spend.cross_run_lock(config.sections["spend"]):
        (runs.parent / f"held-{name}").write_text("held")
        time.sleep(600)
if mode == "unlocked":  # the negative control: the guard without the cross-run lock
    spend.cross_run_lock = lambda policy: nullcontext()
counted = spend.committed


def slow(root):  # read first, then hold the window open so the two checks overlap
    found = counted(root)
    time.sleep(2.0)
    return found


spend.committed = slow
(runs.parent / f"ready-{name}").write_text("ready")
go = runs.parent / "go"
while not go.exists():
    time.sleep(0.01)
try:
    spend.guard(Context(config, Journal(config.run_root), "freeze"), "research", 200)
    print(json.dumps({"name": name, "outcome": "passed"}))
except Blocked as refusal:
    print(json.dumps({"name": name, "outcome": refusal.code, "status": refusal.status}))
"""


def wait_for(path):
    deadline = time.monotonic() + WAIT
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.01)


def autopilot_run(runs, name):
    """An autopilot run root that has not passed its first guard (counts 0)."""
    (runs / name).mkdir(parents=True)
    Journal(runs / name).save()


def child(tmp_path, name, mode):
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "tests")])}
    command = [sys.executable, str(script), str(tmp_path / "runs"), name, mode]
    return subprocess.Popen(
        command + [json.dumps(POLICY)], env=env, stdout=subprocess.PIPE, text=True
    )


def race(tmp_path, mode):
    runs = tmp_path / "runs"
    for name in ("ra", "rb"):
        autopilot_run(runs, name)
    procs = [child(tmp_path, name, mode) for name in ("ra", "rb")]
    for name in ("ra", "rb"):
        wait_for(tmp_path / f"ready-{name}")
    (tmp_path / "go").write_text("go")
    outcomes = {}
    for proc in procs:
        out, _ = proc.communicate(timeout=WAIT)
        assert proc.returncode == 0, out
        result = json.loads(out.strip().splitlines()[-1])
        outcomes[result.pop("name")] = result
    return runs, outcomes


def context(tmp_path, name, **policy):
    runs = tmp_path / "runs"
    raw = {
        "autopilot": {"run_root": str(runs / name), "policy_id": "p1"},
        "spend": {k: v for k, v in {**POLICY, "runs_dir": str(runs), **policy}.items() if v},
    }
    config = cfg.validate(raw, [], path=tmp_path / "autopilot.toml")
    return Context(config, Journal(config.run_root), "freeze")


def test_two_processes_freezing_at_once_cannot_both_fit_the_global_ceiling(tmp_path):
    runs, outcomes = race(tmp_path, "locked")
    passed = [name for name, result in outcomes.items() if result["outcome"] == "passed"]
    refused = [result for result in outcomes.values() if result["outcome"] != "passed"]
    assert len(passed) == 1, outcomes
    assert refused == [{"outcome": "spend_global_ceiling_exceeded", "status": "stopped"}]
    (winner,) = passed
    loser = ({"ra", "rb"} - {winner}).pop()
    guard_file = Path("autopilot", "spend-guard.json")
    assert json.loads((runs / winner / guard_file).read_text())["scopes"] == {"research": 200}
    assert not (runs / loser / guard_file).exists()  # a refusal reserves nothing


def test_without_the_lock_both_concurrent_freezes_pass_the_negative_control(tmp_path):
    _, outcomes = race(tmp_path, "unlocked")
    assert [result["outcome"] for result in outcomes.values()] == ["passed", "passed"]


def test_a_killed_lock_holder_releases_the_lock(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    autopilot_run(runs, "holder")
    holder = child(tmp_path, "holder", "hold")
    try:
        wait_for(tmp_path / "held-holder")
        monkeypatch.setattr(spend, "LOCK_WAIT", 0.2)
        ctx = context(tmp_path, "mine")
        with pytest.raises(Blocked) as busy:
            spend.guard(ctx, "research", 20)
        assert busy.value.code == "spend_lock_busy" and busy.value.status == "blocked"
        assert not ctx.path("autopilot", "spend-guard.json").exists()
    finally:
        holder.kill()
        holder.communicate(timeout=WAIT)
    assert spend.guard(ctx, "research", 20)["global_total"] == 20
    assert "spend_lock_busy" in CORE


def test_a_single_run_is_unchanged_and_the_lock_file_is_not_a_run(tmp_path):
    autopilot_run(tmp_path / "runs", "only")
    ctx = context(tmp_path, "only")
    check = spend.guard(ctx, "research", 200)
    assert check["other_runs"] == {} and check["global_total"] == 200
    assert (tmp_path / "runs" / spend.LOCK).is_file()
    assert spend.other_runs(ctx) == []  # the lock file is never counted as a run
    assert spend.guard(ctx, "research", 200) == check  # a resumed stage passes again
    events = [e for e in ctx.journal.events() if e["kind"] == "spend_guard"]
    assert len(events) == 2 and events[0]["global_total"] == 200
    assert spend.guard(context(tmp_path, "solo", runs_dir=None), "research", 1)["run_total"] == 1


def accounting(root, calls, folder="source-run"):
    (root / folder).mkdir(parents=True, exist_ok=True)
    document = {"outbound_reservations": calls, "requests": []}
    (root / folder / "request-accounting.json").write_text(json.dumps(document), encoding="utf-8")


def guard_file(root):
    return json.loads((root / "autopilot/spend-guard.json").read_text(encoding="utf-8"))


def test_the_runner_passes_the_callers_guard_before_a_spending_stage_runs(tmp_path):
    ran = []

    def run(ctx):
        ran.append(ctx.stage)
        return StageResult()

    toys = [
        toy("freeze", run, spend=lambda ctx: 24, spend_scope="research"),
        toy("off", run, deps=["freeze"], spend=lambda ctx: None),
        toy("refine", run, deps=["off"], spend=lambda ctx: 4),
    ]
    unset = runner.tick(
        make_config(tmp_path / "unset", toys), toys, spend_guard=spend.guard
    )  # run_ceiling defaults to 0
    assert ran == [] and unset.problems[0]["status"] == "stopped"
    assert unset.problems[0]["reason"]["code"] == "spend_run_ceiling_exceeded"
    assert unset.problems[0]["reason"]["class"] == "Spend"
    config = make_config(tmp_path, toys, spend={"run_ceiling": 27, "global_ceiling": 300})
    report = runner.tick(config, toys, spend_guard=spend.guard)
    assert ran == ["freeze", "off"]  # refine: 24 + 4 > 27, stopped before its run
    assert report.problems[0]["stage"] == "refine"
    assert report.problems[0]["reason"]["code"] == "spend_run_ceiling_exceeded"
    assert guard_file(config.run_root)["scopes"] == {"research": 24}
    runner.clear(config, toys, "refine", reviewer="cal", reason="raised the ceiling")
    raised = make_config(tmp_path, toys, spend={"run_ceiling": 28, "global_ceiling": 300})
    changed = runner.tick(
        raised, toys, spend_guard=spend.guard
    )  # the ceilings are frozen at the first check
    assert changed.problems[0]["reason"]["code"] == "spend_policy_changed" and len(ran) == 2
    fresh = make_config(tmp_path / "fresh", toys, spend={"run_ceiling": 28, "global_ceiling": 300})
    assert runner.tick(fresh, toys, spend_guard=spend.guard).exit_code == 0 and ran[2:] == [
        "freeze",
        "off",
        "refine",
    ]
    assert guard_file(fresh.run_root)["scopes"] == {"research": 24, "refine": 4}
    checks = [e for e in Journal(fresh.run_root).events() if e["kind"] == "spend_guard"]
    assert [(e["stage"], e["scope"], e["run_total"]) for e in checks] == [
        ("freeze", "research", 24),
        ("refine", "refine", 28),
    ]


def test_the_guard_counts_every_accounted_run_toward_the_global_ceiling(tmp_path):
    toys = [toy("a")]
    accounting(tmp_path / "r1", 200)
    accounting(tmp_path / "r1", 60, "source-run/refinements")
    (tmp_path / "r2/autopilot").mkdir(parents=True)  # frozen by another autopilot, not started
    planned = {"schema_version": 1, "policy": {}, "scopes": {"research": 20}}
    (tmp_path / "r2/autopilot/spend-guard.json").write_text(json.dumps(planned), encoding="utf-8")
    policy = {"run_ceiling": 24, "global_ceiling": 300, "accounting_roots": ["r1", "r2", "run"]}
    config = make_config(tmp_path, toys, spend=policy)
    ctx = Context(config, Journal(config.run_root), "a")
    check = spend.guard(ctx, "research", 20)
    assert check["global_total"] == 300 and sorted(check["other_runs"].values()) == [20, 260]
    assert spend.guard(ctx, "research", 20) == check  # a resumed stage passes again
    with pytest.raises(Stop, match="spend_global_ceiling_exceeded: 21 \\+ other runs 280 > 300"):
        spend.guard(ctx, "research", 21)
    with pytest.raises(Stop, match="spend_run_ceiling_exceeded"):
        spend.guard(ctx, "refine", 5)
    assert guard_file(config.run_root)["scopes"] == {"research": 20}  # a refusal saves nothing
    accounting(config.run_root, 21)  # this run reserved more than its ceiling says
    with pytest.raises(Stop, match="spend_global_ceiling_exceeded"):
        spend.guard(ctx, "research", 20)

    def refused(name, **policy):
        other = make_config(tmp_path / name, toys, spend=policy)
        with pytest.raises(Stop) as stop:
            spend.guard(Context(other, Journal(other.run_root), "a"), "research", 1)
        return stop.value.code

    (tmp_path / "empty/there").mkdir(parents=True)
    unreadable = "spend_accounting_unreadable"  # never counted as zero
    assert (
        refused("empty", run_ceiling=9, global_ceiling=9, accounting_roots=["there"]) == unreadable
    )
    assert (
        refused("gone", run_ceiling=9, global_ceiling=9, accounting_roots=["nowhere"]) == unreadable
    )
    assert refused("zero", run_ceiling=9, global_ceiling=0) == "spend_global_ceiling_exceeded"
    with pytest.raises(Stop, match="spend_accounting_unreadable"):
        spend.guard(ctx, "research", True)


def test_the_ceilings_are_the_callers_and_a_larger_number_is_honoured(tmp_path):
    toys = [toy("a")]
    config = make_config(tmp_path, toys, spend={"run_ceiling": 900, "global_ceiling": 800})
    ctx = Context(config, Journal(config.run_root), "a")
    assert spend.guard(ctx, "research", 800)["global_ceiling"] == 800
    with pytest.raises(Stop, match="spend_global_ceiling_exceeded"):
        spend.guard(ctx, "refine", 1)  # 801 > the caller's 800, not a built-in number


def test_the_budget_is_returned_for_the_run_ledger(tmp_path):
    config = make_config(tmp_path, [toy("a")], spend={"run_ceiling": 7, "global_ceiling": 9})
    assert spend.budget_params(config) == {"spend_budget": {"run_ceiling": 7, "global_ceiling": 9}}
    unset = make_config(tmp_path / "unset", [toy("a")])
    assert spend.budget_params(unset)["spend_budget"] == {"run_ceiling": 0, "global_ceiling": 0}
