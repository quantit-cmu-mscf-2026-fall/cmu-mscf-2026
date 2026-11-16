"""Typed early warnings, before the run commits to a long cash wait or blocks.

Toy stages under the real names the warnings read (`portfolios`, `admission`, `daily`,
`final_protocol`) and the real operations hook. Synthetic only; the notifier is a list.
"""

import json

import pytest
from test_autopilot_ops import SECTIONS, gated, toy

from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import runner
from capstone.pipeline.autopilot.config import Key
from capstone.pipeline.autopilot.journal import Journal
from capstone.pipeline.autopilot.ops import common, early_warnings, hook
from capstone.pipeline.autopilot.reasons import Blocked
from capstone.pipeline.autopilot.stage import StageResult

MINIMUM = 1260


def make_config(tmp_path, toys, **sections):
    raw = {"autopilot": {"run_root": str(tmp_path / "run"), "policy_id": "p1"}, **sections}
    return cfg.validate(raw, toys, extra=SECTIONS, path=tmp_path / "autopilot.toml")


def tick(config, toys, **services):
    return runner.tick(config, toys, hooks=[hook.after_tick], services=services)


def warnings_file(config):
    return common.ops_dir(config.run_root) / early_warnings.FILE


def codes(alerts):
    return [p["reason"]["code"] for a in alerts for p in a["problems"]]


def write_spec(path, chronological=True, minimum=MINIMUM):
    definition = {"policy": {"minimum": minimum}, "specifications": {}}
    if chronological:
        definition["membership_clock"] = {"rule": "prefix"}
    payload = json.dumps({"definition": definition})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"payload = {json.dumps(payload)}\n", encoding="utf-8")
    return path


# --- unset runs_dir ----------------------------------------------------------------------


def test_an_unset_runs_dir_warns_once_when_the_run_may_call(tmp_path):
    toys = [toy("a")]
    config = make_config(tmp_path, toys, spend={"run_ceiling": 4})
    sent = []
    for _ in range(3):
        report = tick(config, toys, notifier=sent.append)
        assert report.exit_code == 0 and report.problems == []  # a warning is not a problem
    assert codes(sent) == ["spend_runs_dir_unset"] and sent[0]["severity"] == "warning"
    assert sent[0]["problems"][0]["reason"]["class"] == "Spend"
    assert "[spend] runs_dir" in sent[0]["problems"][0]["reason"]["cleared_by"]
    assert not (config.run_root / "ALERT.json").exists()  # warnings are not alerts
    recorded = json.loads(warnings_file(config).read_text("utf-8"))
    assert [w["reason"]["code"] for w in recorded["warnings"]] == ["spend_runs_dir_unset"]
    status = json.loads((common.ops_dir(config.run_root) / "status.json").read_text("utf-8"))
    assert status["ok"] is True and status["warnings"] == recorded["warnings"]
    assert "spend_runs_dir_unset" in (common.ops_dir(config.run_root) / "status.md").read_text(
        "utf-8"
    )


@pytest.mark.parametrize("spend", [{"run_ceiling": 0}, {"run_ceiling": 4, "runs_dir": "runs"}])
def test_no_runs_dir_warning_without_calls_or_with_a_runs_dir(tmp_path, spend):
    (tmp_path / "runs").mkdir()
    toys = [toy("a")]
    config = make_config(tmp_path, toys, spend=spend)
    sent = []
    assert tick(config, toys, notifier=sent.append).exit_code == 0
    assert sent == [] and not warnings_file(config).exists()


def test_a_failed_notifier_sends_the_warning_again_next_tick(tmp_path):
    toys = [toy("a")]
    config = make_config(tmp_path, toys, spend={"run_ceiling": 4})
    calls = []

    def down(notice):
        calls.append(notice)
        raise ConnectionError("smtp down")

    report = tick(config, toys, notifier=down)
    assert [p["reason"]["code"] for p in report.problems] == ["alert_notifier_failed"]
    assert tick(config, toys, notifier=calls.append).exit_code == 0
    assert tick(config, toys, notifier=calls.append).exit_code == 0
    assert codes(calls) == ["spend_runs_dir_unset", "spend_runs_dir_unset"]


# --- missing final-protocol manifest -----------------------------------------------------


def protocol_toys(flag):
    gate = gated(flag, Blocked("inbox_empty", "the gate waits"))
    final = toy(
        "final_protocol", deps=("gated",), schema={"manifest": Key("path", None, "SHA256SUMS")}
    )
    return [gate, final]


def test_a_missing_manifest_warns_before_final_protocol_runs(tmp_path):
    flag = tmp_path / "flag"
    flag.touch()
    toys = protocol_toys(flag)
    for section, needle in [({}, "is not set"), ({"manifest": "nope/SHA256SUMS"}, "not a file")]:
        base = tmp_path / (needle.replace(" ", "-"))
        base.mkdir()
        config = make_config(base, toys, final_protocol=section)
        sent = []
        tick(config, toys, notifier=sent.append)
        assert Journal(config.run_root).stage("final_protocol")["state"] == "pending"
        warned = [a for a in sent if a["severity"] == "warning"]
        assert codes(warned) == ["final_protocol_manifest_missing"]
        assert warned[0]["problems"][0]["stage"] == "final_protocol"
        assert needle in warned[0]["problems"][0]["reason"]["detail"]


def test_a_manifest_that_exists_or_a_done_stage_does_not_warn(tmp_path):
    flag = tmp_path / "flag"
    toys = protocol_toys(flag)
    (tmp_path / "SHA256SUMS").write_text("", encoding="utf-8")
    config = make_config(tmp_path, toys, final_protocol={"manifest": "SHA256SUMS"})
    sent = []
    assert tick(config, toys, notifier=sent.append).exit_code == 0
    assert sent == []
    unset = make_config(tmp_path / "b", toys)
    assert tick(unset, toys, notifier=sent.append).exit_code == 0  # all done in one tick
    # the stage is done by the time the hook looks: nothing left to block
    assert sent == [] and not warnings_file(unset).exists()


# --- chronological cash period -----------------------------------------------------------


def chrono_toys(tmp_path, chronological=True, admit=None):
    """`portfolios` chooses a comparison (its spec written chronological or not);
    `admission` waits on a flag, then admits `admit`."""
    flag = tmp_path / "admission-waits"
    flag.touch()

    def portfolios(ctx):
        write_spec(ctx.path("portfolios", "cmp", "portfolio-spec.toml"), chronological)
        return StageResult(data={"comparison_dir": "portfolios/cmp", "comparison_id": "cmp"})

    def admission(ctx):
        if flag.exists():
            raise Blocked("inbox_empty", "the gate waits")
        return StageResult(data={"portfolios": admit or [], "candidates": []})

    toys = [
        toy("portfolios", portfolios),
        toy("admission", admission, deps=("portfolios",)),
        toy("daily", deps=("admission",), schema={"saved_portfolio": Key("path", None)}),
    ]
    return toys, flag


def test_a_chosen_chronological_comparison_warns_before_admission_and_daily(tmp_path):
    toys, flag = chrono_toys(tmp_path, admit=["cmp"])
    config = make_config(tmp_path, toys)
    sent = []
    tick(config, toys, notifier=sent.append)
    assert Journal(config.run_root).stage("daily")["state"] == "pending"  # nothing planned
    (warned,) = [a for a in sent if a["severity"] == "warning"]
    (row,) = warned["problems"]
    assert (row["stage"], row["reason"]["code"]) == ("daily", "chronological_cash_period")
    assert f"{MINIMUM} prefix receipts" in row["reason"]["detail"]
    assert "portfolios/cmp/portfolio-spec.toml" in row["reason"]["detail"]
    assert "about 5 years" in row["reason"]["detail"]
    flag.unlink()
    for _ in range(2):  # admitted and traded: the same warning is not sent again
        tick(config, toys, notifier=sent.append)
    assert len([a for a in sent if a["severity"] == "warning"]) == 1


def test_no_cash_warning_for_a_plain_comparison_or_when_no_portfolio_is_admitted(tmp_path):
    (tmp_path / "plain").mkdir()
    toys, _ = chrono_toys(tmp_path / "plain", chronological=False)
    config = make_config(tmp_path / "plain", toys)
    sent = []
    tick(config, toys, notifier=sent.append)
    assert [a for a in sent if a["severity"] == "warning"] == []
    toys, flag = chrono_toys(tmp_path, admit=[])  # chronological, but admission admits none
    flag.unlink()
    config = make_config(tmp_path, toys)
    assert tick(config, toys, notifier=sent.append).exit_code == 0
    assert [a for a in sent if a["severity"] == "warning"] == []


def test_a_chronological_saved_portfolio_warns(tmp_path):
    toys, _ = chrono_toys(tmp_path, chronological=False)
    spec = write_spec(tmp_path / "vK.toml", minimum=60)
    config = make_config(tmp_path, toys, daily={"saved_portfolio": str(spec)})
    sent = []
    tick(config, toys, notifier=sent.append)
    (warned,) = [a for a in sent if a["severity"] == "warning"]
    assert "60 prefix receipts" in warned["problems"][0]["reason"]["detail"]
