"""Autopilot operations hook: alerts (typed, de-duplicated, resumable) and status.

Toy stages only; no network, no credentials, no scheduler. The notifier is injected or a
tiny module written to a temp folder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import ops, runner
from capstone.pipeline.autopilot.journal import Journal
from capstone.pipeline.autopilot.ops import alert as ap_alert
from capstone.pipeline.autopilot.ops import common, hook, status
from capstone.pipeline.autopilot.reasons import Blocked, ReasonSpec, Stop
from capstone.pipeline.autopilot.stage import Stage, StageResult

REPO = Path(__file__).resolve().parents[1]

REASONS = {
    "spend_stop": ReasonSpec("Spend", "a person decides"),
    "inbox_empty": ReasonSpec("Source", "drop a file"),
}
SECTIONS, HOOKS = ops.discover()
CLEAR = {"reviewer": "reviewer-a", "reason": "test"}


def toy(name, run=None, deps=(), **more):
    def write(ctx):
        ctx.path(name).mkdir(parents=True, exist_ok=True)
        ctx.path(name, "out.txt").write_text(name, encoding="utf-8")
        return StageResult(artifacts=[ctx.path(name, "out.txt")])

    more.setdefault("reasons", REASONS)
    return Stage(name=name, unit="T", run=run or write, depends_on=tuple(deps), **more)


def gated(flag, error):
    """Raises `error` while `flag` exists, else writes its output."""
    plain = toy("gated").run

    def run(ctx):
        if Path(flag).exists():
            raise error
        return plain(ctx)

    return toy("gated", run)


def make_config(tmp_path, toys, **sections):
    core = {"run_root": str(tmp_path / "run"), "policy_id": "p1"}
    raw = {"autopilot": core, **sections}
    return cfg.validate(raw, toys, extra=SECTIONS, path=tmp_path / "autopilot.toml")


def tick(config, toys, **services):
    return runner.tick(config, toys, hooks=HOOKS, services=services)


def codes(report):
    return [p["reason"]["code"] for p in report.problems]


def alert_of(root):
    return json.loads((root / "ALERT.json").read_text(encoding="utf-8"))


def history(root):
    return common.read_lines(common.ops_dir(root) / ap_alert.HISTORY)


# --- alerts ----------------------------------------------------------------------------


def test_a_stop_writes_alert_json_and_md_and_a_clean_tick_keeps_them(tmp_path):
    flag = tmp_path / "flag"
    flag.touch()
    toys = [gated(flag, Stop("spend_stop", "ceiling reached"))]
    config = make_config(tmp_path, toys)
    sent = []
    report = tick(config, toys, notifier=sent.append)
    root = config.run_root
    assert report.exit_code == 2 and codes(report) == ["spend_stop"]
    alert = alert_of(root)
    assert alert["severity"] == "stop" and alert["policy_id"] == "p1"
    assert alert["problems"][0]["reason"] == {
        "code": "spend_stop",
        "class": "Spend",
        "detail": "ceiling reached",
        "cleared_by": "a person decides",
    }
    assert alert["notifier"]["status"] == "sent" and alert["states"] == {"gated": "stopped"}
    assert len(sent) == 1 and sent[0]["problems"] == alert["problems"]
    assert sent[0]["notifier"]["status"] == "pending"  # what the notifier saw, before outcome
    text = (root / "ALERT.md").read_text(encoding="utf-8")
    assert "spend_stop" in text and "a person decides" in text and "Autopilot alert: stop" in text
    assert len(history(root)) == 1
    # still stopped: the same STOP is not a new problem , so no second alert
    before = (root / "ALERT.json").read_bytes()
    assert tick(config, toys, notifier=sent.append).exit_code == 2 and len(sent) == 1
    runner.clear(config, toys, "gated", **CLEAR)
    flag.unlink()
    assert tick(config, toys, notifier=sent.append).exit_code == 0
    assert (root / "ALERT.json").read_bytes() == before  # alerts are history
    assert (root / "ALERT.md").is_file() and len(sent) == 1 and len(history(root)) == 1


def test_blocked_alerts_unless_on_blocked_is_false(tmp_path):
    flag = tmp_path / "flag"
    flag.touch()
    toys = [gated(flag, Blocked("inbox_empty", "nothing yet"))]
    quiet = make_config(tmp_path, toys, alert={"on_blocked": False})
    assert tick(quiet, toys).exit_code == 2
    assert not (quiet.run_root / "ALERT.json").exists()
    loud = make_config(tmp_path / "b", toys)
    assert tick(loud, toys).exit_code == 2
    alert = alert_of(loud.run_root)
    assert alert["severity"] == "blocked" and alert["notifier"]["status"] == "none"
    assert alert["problems"][0]["reason"]["code"] == "inbox_empty"


NOTIFIER = """import json, pathlib
def send(alert):
    out = pathlib.Path(alert["run_root"]).parent / "sent.jsonl"
    with out.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(alert["problems"][0]["reason"]["code"]) + "\\n")
def boom(alert):
    raise ConnectionError("smtp down")
"""


def test_the_configured_notifier_is_called_and_its_failures_are_typed(tmp_path, monkeypatch):
    (tmp_path / "ops_toy_notifier.py").write_text(NOTIFIER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    toys = [gated(tmp_path, Stop("spend_stop", "x"))]  # tmp_path exists: always stops
    config = make_config(tmp_path, toys, alert={"notifier": "ops_toy_notifier:send"})
    assert tick(config, toys).exit_code == 2
    assert (tmp_path / "sent.jsonl").read_text(encoding="utf-8") == '"spend_stop"\n'
    assert alert_of(config.run_root)["notifier"]["target"] == "ops_toy_notifier:send"
    for target, needle in [
        ("ops_toy_notifier:boom", "ConnectionError: smtp down"),
        ("ops_toy_notifier:missing", "is not a callable"),
        ("no_such_ops_module:send", "notifier module 'no_such_ops_module'"),
        ("not-a-target", "is not module:function"),
    ]:
        base = tmp_path / target.replace(":", "_")
        failing = make_config(base, toys, alert={"notifier": target})
        report = tick(failing, toys)
        assert codes(report) == ["spend_stop", "alert_notifier_failed"], target
        assert needle in report.problems[1]["reason"]["detail"]
        alert = alert_of(failing.run_root)
        assert alert["notifier"]["status"] == "failed" and needle in alert["notifier"]["detail"]
        assert [p["reason"]["code"] for p in alert["problems"]] == codes(report)


# --- kill and resume in fresh processes ------------------------------------------------

KILL_SCRIPT = """import os, sys
from pathlib import Path
from capstone.pipeline.autopilot import config as cfg, ops, runner
from capstone.pipeline.autopilot.reasons import ReasonSpec, Stop
from capstone.pipeline.autopilot.stage import Stage, StageResult

base, mode = Path(sys.argv[1]), sys.argv[2]

def daily(ctx):
    return StageResult(data={"last_session": "2024-01-02", "sessions": 1})

def stopper(ctx):
    if (base / "stop-flag").exists():
        raise Stop("spend_stop", "ceiling reached")
    return StageResult()

toys = [
    Stage("daily", "T", daily, recurring=True),
    Stage("guard", "T", stopper, reasons={"spend_stop": ReasonSpec("Spend", "a person")}),
]
sections, hooks = ops.discover()
raw = {"autopilot": {"run_root": str(base / "run"), "policy_id": "p1"}}
config = cfg.validate(raw, toys, extra=sections, path=base / "autopilot.toml")

def killed_notifier(alert):
    os._exit(9)

def recording_notifier(alert):
    with (base / "sent.txt").open("a", encoding="utf-8") as stream:
        stream.write(alert["severity"] + "\\n")

services = {"notifier": recording_notifier}
if mode == "kill-notify":
    services["notifier"] = killed_notifier
report = runner.tick(config, toys, hooks=hooks, services=services)
print(report.line())
sys.exit(report.exit_code)
"""


def run_script(tmp_path, mode):
    script = tmp_path / "kill.py"
    script.write_text(KILL_SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    command = [sys.executable, str(script), str(tmp_path), mode]
    return subprocess.run(command, env=env, capture_output=True, text=True)


def test_a_tick_killed_inside_the_notifier_alerts_again_in_a_fresh_process(tmp_path):
    (tmp_path / "stop-flag").touch()
    first = run_script(tmp_path, "kill-notify")
    assert first.returncode == 9, first.stderr
    root = tmp_path / "run"
    pending = alert_of(root)
    assert pending["severity"] == "stop" and pending["notifier"]["status"] == "pending"
    assert history(root) == []  # appended only once the notifier has answered
    second = run_script(tmp_path, "normal")
    assert second.returncode == 2 and "spend_stop" in second.stdout
    assert alert_of(root)["notifier"]["status"] == "sent"
    assert (tmp_path / "sent.txt").read_text(encoding="utf-8") == "stop\n"
    assert len(history(root)) == 1


# --- alert hygiene -----------------------------------------------------------------

A4_REASONS = {
    **REASONS,
    "data_stale": ReasonSpec("Data", "restart the export"),
    "spend_lock_busy": ReasonSpec("Operator", "nothing: the next tick resumes"),
}


def scripted(script, recurring=True):
    """A toy stage raising what `script` (a JSON file) says: {"raise": "stop" | "block" |
    null, "code", "detail"}; null writes its output."""
    plain = toy("gated").run

    def run(ctx):
        step = json.loads(Path(script).read_text(encoding="utf-8"))
        if step["raise"] == "stop":
            raise Stop(step["code"], step["detail"])
        if step["raise"] == "block":
            raise Blocked(step["code"], step["detail"])
        return plain(ctx)

    return toy("gated", run, recurring=recurring, reasons=A4_REASONS)


def setup(tmp_path, recurring=True):
    script = tmp_path / "script.json"
    toys = [scripted(script, recurring)]
    return make_config(tmp_path, toys), toys, script


def say(script, kind, code="", detail=""):
    script.write_text(json.dumps({"raise": kind, "code": code, "detail": detail}), "utf-8")


def test_a_lasting_stop_alerts_once_and_again_only_after_a_clear(tmp_path):
    config, toys, script = setup(tmp_path, recurring=False)
    say(script, "stop", "spend_stop", "ceiling reached")
    sent = []
    for _ in range(4):  # stopped for four ticks: one alert, one history row
        assert tick(config, toys, notifier=sent.append).exit_code == 2
    assert len(sent) == 1 and len(history(config.run_root)) == 1
    assert sent[0]["severity"] == "stop" and len(sent[0]["new_problems"]) == 1
    # a person clears it without fixing the cause: the stage runs again and stops again,
    # with the same code and detail; that is a new occurrence, so it alerts again
    runner.clear(config, toys, "gated", **CLEAR)
    assert tick(config, toys, notifier=sent.append).exit_code == 2
    assert len(sent) == 2 and sent[1]["new_problems"] != sent[0]["new_problems"]
    assert tick(config, toys, notifier=sent.append).exit_code == 2 and len(sent) == 2


def test_the_same_code_for_another_session_is_a_new_problem(tmp_path):
    config, toys, script = setup(tmp_path)
    sent = []
    say(script, "block", "data_stale", "newest session 2027-03-01 is 9 days old")
    for _ in range(3):
        tick(config, toys, notifier=sent.append)
    assert len(sent) == 1 and sent[0]["new_blockers"] == ["gated|data_stale"]
    say(script, "block", "data_stale", "newest session 2027-03-02 is 9 days old")
    tick(config, toys, notifier=sent.append)
    tick(config, toys, notifier=sent.append)
    assert len(sent) == 2  # another detail: alerted once more, then quiet again
    assert sent[1]["problems"][0]["reason"]["detail"].startswith("newest session 2027-03-02")


def test_a_progress_detail_does_not_make_a_new_problem(tmp_path):
    config, toys, script = setup(tmp_path)
    sent = []
    for waited in (10, 20, 30):  # a lock held across ticks: the wait moves on
        detail = f"another run held the lock for {waited}s; the next tick retries"
        say(script, "block", "spend_lock_busy", detail)
        tick(config, toys, notifier=sent.append)
    assert len(sent) == 1


def test_the_stop_that_follows_a_blocker_is_alerted(tmp_path):
    config, toys, script = setup(tmp_path, recurring=False)
    sent = []
    say(script, "block", "inbox_empty", "nothing yet")
    tick(config, toys, notifier=sent.append)
    say(script, "stop", "spend_stop", "ceiling reached")
    tick(config, toys, notifier=sent.append)
    assert [a["severity"] for a in sent] == ["blocked", "stop"]


def test_a_seen_file_from_older_reads_as_empty(tmp_path):
    config, toys, script = setup(tmp_path)
    say(script, "block", "inbox_empty", "nothing yet")
    seen = common.ops_dir(config.run_root) / ap_alert.SEEN
    common.write_json(seen, {"blocked": ["gated|inbox_empty"]})  # the older format
    sent = []
    tick(config, toys, notifier=sent.append)
    tick(config, toys, notifier=sent.append)
    assert len(sent) == 1  # alerted once under the new key, then deduplicated
    assert json.loads(seen.read_text("utf-8"))["schema"] == ap_alert.SEEN_SCHEMA


def test_problem_key_is_status_stage_code_detail_and_a_stops_attempt(tmp_path):
    def row(status, stage, code, detail):
        reason = {"code": code, "detail": detail, "class": "Data", "cleared_by": "x"}
        return {"stage": stage, "status": status, "reason": reason}

    key = ap_alert.problem_key
    assert key(row("blocked", "daily", "data_stale", "a")) != key(
        row("blocked", "daily", "data_stale", "b")
    )
    assert key(row("blocked", "daily", "data_stale", "a")) != key(
        row("stopped", "daily", "data_stale", "a")
    )
    assert key(row("blocked", None, "spend_lock_busy", "held 1s")) == key(
        row("blocked", None, "spend_lock_busy", "held 2s")
    )
    journal = Journal(tmp_path / "run")
    journal.update("daily", attempts=3)
    stopped = key(row("stopped", "daily", "data_corrupt", "x"), journal)
    assert stopped.endswith("|a3")
    assert not key(row("blocked", "daily", "data_stale", "x"), journal).endswith("|a3")


# --- discovery, config and status ------------------------------------------------------


def test_the_hook_declares_its_section_with_safe_defaults(tmp_path):
    assert hook.after_tick in HOOKS and len(HOOKS) == 1
    config = make_config(tmp_path, [toy("a")])
    assert config.sections["alert"] == {"notifier": None, "on_blocked": True}
    for code, spec in common.REASONS.items():  # every ops reason is a typed, known class
        assert spec.klass == "Operator", code


def test_status_is_written_each_tick_and_reflects_the_journal(tmp_path):
    flag = tmp_path / "flag"
    flag.touch()
    toys = [toy("a"), gated(flag, Blocked("inbox_empty", "nothing yet"))]
    config = make_config(tmp_path, toys)
    assert not config.run_root.exists()
    assert status.write(config, ["a", "gated"]).get("ok") is True  # nothing run, nothing written
    assert not config.run_root.exists()
    assert tick(config, toys).exit_code == 2
    root = config.run_root
    written = json.loads((common.ops_dir(root) / "status.json").read_text("utf-8"))
    assert written["attention"] == ["gated"] and written["ok"] is False
    assert written["alert"]["severity"] == "blocked" and written["alerts_total"] == 1
    assert written["counts"] == {"blocked": 1, "done": 1}
    md = (common.ops_dir(root) / "status.md").read_text("utf-8")
    assert "inbox_empty" in md and "drop a file" in md
    flag.unlink()
    assert tick(config, toys).exit_code == 0
    again = json.loads((common.ops_dir(root) / "status.json").read_text("utf-8"))
    assert again["ok"] is True and again["alerts_total"] == 1  # the alert stays as history


def test_a_broken_journal_is_a_typed_status_problem_not_a_traceback(tmp_path):
    toys = [toy("a")]
    config = make_config(tmp_path, toys)
    assert tick(config, toys).exit_code == 0
    events = config.run_root / "AUTOPILOT.events.jsonl"
    events.write_text(events.read_text(encoding="utf-8").replace('"seq":0', '"seq":7'), "utf-8")
    report = tick(config, toys)
    assert codes(report) == ["ops_status_failed"] and "breaks the chain" in report.line()
    alert_doc = json.loads((config.run_root / "ALERT.json").read_text("utf-8"))
    assert [p["reason"]["code"] for p in alert_doc["problems"]] == ["ops_status_failed"]


def test_atomic_writes_replace_and_a_torn_line_is_closed_off_not_extended(tmp_path):
    path = tmp_path / "x" / "doc.json"
    common.write_json(path, {"b": 1, "a": 2})
    assert path.read_text("utf-8") == "{" + chr(10) + '  "a": 2,' + chr(10) + '  "b": 1' + chr(
        10
    ) + "}" + chr(10)
    assert common.read_json(tmp_path / "missing.json", "d") == "d"
    log = tmp_path / "log.jsonl"
    common.append_line(log, {"n": 1})
    with log.open("a", encoding="utf-8") as stream:
        stream.write('{"n": 2')  # a kill tore this line
    common.append_line(log, {"n": 3})
    assert common.read_lines(log) == [{"n": 1}, {"n": 3}]
