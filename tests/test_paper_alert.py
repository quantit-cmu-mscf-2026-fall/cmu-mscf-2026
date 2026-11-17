"""A paper tick's refusals and failures reach an ALERT.

Synthetic errors and notifiers only: no broker, no connection, no network. The
tick itself is not involved; `alerting` wraps whatever raises inside it.
"""

from __future__ import annotations

import json

import pytest

from capstone.execution import paper_alert
from capstone.execution.broker import PaperGuardError
from capstone.execution.deploy import Refused
from capstone.faults import InjectedFault
from capstone.pipeline.autopilot.ops import alert as autopilot_alert

CLOCK = "2020-06-24T21:00:00Z"
CONTEXT = {"tag": "deploy-2020.06.23", "session": "2020-06-23"}


class Coded(RuntimeError):
    """A refusal that carries its own reason code, as the tick's refusals do."""

    def __init__(self, code, detail="detail"):
        super().__init__(detail)
        self.code = code


def read_alert(folder):
    return json.loads((folder / autopilot_alert.JSON_FILE).read_text(encoding="utf-8"))


def tick(folder, error, calls, *, notifier=None):
    """One tick that raises `error` inside the alerting wrapper."""
    with (
        pytest.raises(type(error)),
        paper_alert.alerting(
            folder, notifier=notifier or calls.append, context=CONTEXT, clock=lambda: CLOCK
        ),
    ):
        raise error


def test_errors_are_classified_into_a_code_and_a_status():
    assert paper_alert.classify(Coded("kill_switch")) == ("kill_switch", "stopped")
    assert paper_alert.classify(Coded("no_session")) == ("no_session", "blocked")
    assert paper_alert.classify(PaperGuardError("x")) == ("paper_guard", "stopped")
    assert paper_alert.classify(Refused("x")) == ("deploy_refused", "stopped")
    assert paper_alert.classify(ValueError("x")) == ("paper_tick_failed", "stopped")
    row = paper_alert.problem_row(ValueError("boom"))
    assert row["stage"] == paper_alert.STAGE == "paper_tick"
    assert row["reason"]["detail"] == "ValueError: boom"  # an unexpected error names its type
    assert row["reason"]["cleared_by"]


def test_a_refusal_alerts_once_per_new_problem_and_a_clean_tick_forgets(tmp_path):
    out, calls = tmp_path / "paper", []
    for _ in range(2):
        tick(out, Coded("kill_switch", "kill file set"), calls)
    assert len(calls) == 1  # the same problem twice: one notification
    alert = read_alert(out)
    assert alert["schema"] == autopilot_alert.SCHEMA and alert["severity"] == "stop"
    (row,) = alert["problems"]
    assert row["stage"] == paper_alert.STAGE and row["reason"]["code"] == "kill_switch"
    assert row["reason"]["cleared_by"]
    assert alert["notifier"]["status"] == "repeat"
    assert alert["paper"] == CONTEXT and alert["policy_id"] == "deploy-2020.06.23"
    text = (out / autopilot_alert.MD_FILE).read_text(encoding="utf-8")
    assert text.startswith("# Autopilot alert: stop")
    history = (out / autopilot_alert.HISTORY).read_text(encoding="utf-8").splitlines()
    assert len(history) == 1 and calls[0]["notifier"]["status"] == "pending"
    paper_alert.cleared(out)  # a tick that succeeded
    tick(out, Coded("kill_switch", "kill file set"), calls)
    assert len(calls) == 2  # the problem came back after a clean tick: alerted again


def test_another_detail_is_a_new_problem(tmp_path):
    out, calls = tmp_path / "paper", []
    tick(out, Coded("no_session", "2020-06-23"), calls)
    tick(out, Coded("no_session", "2020-06-24"), calls)
    assert len(calls) == 2
    assert calls[0]["severity"] == "blocked" and calls[0]["new_blockers"] == [
        "paper_tick|no_session"
    ]


def test_a_failing_notifier_never_hides_the_ticks_error(tmp_path):
    out = tmp_path / "paper"

    def failing(alert):
        raise RuntimeError("pager down")

    tick(out, Coded("kill_switch"), [], notifier=failing)  # the Coded error is what is raised
    alert = read_alert(out)
    assert alert["notifier"]["status"] == "failed" and "pager down" in alert["notifier"]["detail"]
    assert not (out / paper_alert.SEEN).exists()  # it will be sent again


def test_a_notifier_given_as_module_function_is_resolved(tmp_path):
    out = tmp_path / "paper"
    tick(out, Coded("kill_switch"), [], notifier="test_paper_alert:record")
    assert read_alert(out)["notifier"] == {
        "target": "test_paper_alert:record",
        "status": "sent",
        "detail": "",
    }
    assert RECEIVED and RECEIVED[-1]["problems"][0]["reason"]["code"] == "kill_switch"
    tick(tmp_path / "other", Coded("kill_switch"), [], notifier="no_such_module:nope")
    assert read_alert(tmp_path / "other")["notifier"]["status"] == "failed"


RECEIVED = []


def record(alert):
    RECEIVED.append(alert)


def test_a_paper_guard_refusal_is_notified_and_writes_nothing(tmp_path):
    out, calls = tmp_path / "paper", []
    tick(out, PaperGuardError("not a paper account"), calls)
    tick(out, PaperGuardError("not a paper account"), calls)
    assert [c["problems"][0]["reason"]["code"] for c in calls] == ["paper_guard"] * 2
    assert calls[0]["severity"] == "stop"
    assert not out.exists()  # nothing under the paper folder, and sent on every tick


def test_a_simulated_kill_is_not_a_refusal_and_an_error_alerts_only_once(tmp_path):
    out, calls = tmp_path / "paper", []
    tick(out, InjectedFault("killed"), calls)
    assert calls == [] and not out.exists()
    error = Coded("kill_switch")
    with pytest.raises(Coded), paper_alert.alerting(out, notifier=calls.append):
        with paper_alert.alerting(out, notifier=calls.append):  # the inner wrapper alerts
            raise error
    assert len(calls) == 1 and getattr(error, paper_alert.MARK) is True


def test_the_notifier_comes_from_the_autopilot_config(tmp_path):
    config = tmp_path / "autopilot.toml"
    config.write_text('[alert]\nnotifier = "test_paper_alert:record"\n', encoding="utf-8")
    assert paper_alert.notifier_from_config(config) == "test_paper_alert:record"
    config.write_text("[autopilot]\n", encoding="utf-8")
    assert paper_alert.notifier_from_config(config) is None
