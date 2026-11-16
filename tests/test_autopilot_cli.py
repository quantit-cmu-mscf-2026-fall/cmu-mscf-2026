"""`python -m capstone.pipeline.autopilot`: stage discovery, tick / status / clear.

Toy stages only (the DAG is monkeypatched), so the test holds whichever stage modules
exist; no data, no model, no network.
"""

from __future__ import annotations

import json
import types

import pytest

from capstone.pipeline.autopilot import __main__ as cli
from capstone.pipeline.autopilot import ops, stages
from capstone.pipeline.autopilot.journal import STATE
from capstone.pipeline.autopilot.reasons import Blocked, ReasonSpec
from capstone.pipeline.autopilot.stage import Stage, StageResult

CLEAR = ["--reviewer", "reviewer-a", "--reason", "test"]
REASONS = {"inbox_empty": ReasonSpec("Source", "drop a file")}


def toy(name, run=None, deps=(), **more):
    more.setdefault("reasons", REASONS)
    return Stage(
        name=name, unit="T", run=run or (lambda ctx: StageResult()), depends_on=tuple(deps), **more
    )


def blocked(ctx):
    raise Blocked("inbox_empty", "nothing yet")


def test_discover_finds_every_module_in_name_order_and_ignores_those_without_stages(
    tmp_path, monkeypatch
):
    package = tmp_path / "toy_units"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    body = (
        "from capstone.pipeline.autopilot.stage import Stage, StageResult\n"
        "STAGES = (Stage('{0}', 'T', lambda ctx: StageResult()),)\n"
        "SECTIONS = {{'{0}_ops': {{}}}}\n"
        "def after_tick(ctx, report): pass\n"
    )
    for name in ("b_unit", "a_unit"):
        (package / f"{name}.py").write_text(body.format(name), encoding="utf-8")
    (package / "helper.py").write_text("X = 1\n", encoding="utf-8")  # no STAGES: ignored
    monkeypatch.syspath_prepend(str(tmp_path))
    import toy_units

    assert [s.name for s in stages.discover(toy_units)] == ["a_unit", "b_unit"]
    sections, hooks = ops.discover(toy_units)
    assert sorted(sections) == ["a_unit_ops", "b_unit_ops"] and len(hooks) == 2
    twice = types.SimpleNamespace(__path__=[str(package)], __name__="toy_units")
    (package / "c_unit.py").write_text("SECTIONS = {'a_unit_ops': {}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declared twice"):
        ops.discover(twice)


def test_the_real_package_discovers_a_list_of_stages():
    assert all(isinstance(s, Stage) for s in stages.discover())


def test_cli_tick_status_and_clear(tmp_path, capsys, monkeypatch):
    toys = [toy("first", blocked), toy("b", deps=["first"])]
    monkeypatch.setattr(stages, "discover", lambda: toys)
    monkeypatch.setattr(ops, "discover", lambda: ({}, []))
    text = '[autopilot]\nrun_root = "run"\npolicy_id = "ap-policy-v1"\n'
    config = tmp_path / "autopilot.toml"
    config.write_text(text, encoding="utf-8")
    assert cli.main(["tick", "--config", str(config)]) == 2
    out = capsys.readouterr()
    assert out.out == "" and len(out.err.strip().splitlines()) == 1
    assert out.err.startswith("autopilot: blocked first: inbox_empty: nothing yet (cleared by:")
    assert (tmp_path / "run" / STATE).is_file()  # run_root is relative to the config file
    assert cli.main(["status", "--config", str(config)]) == 2
    table = json.loads(capsys.readouterr().out)
    assert table["policy_id"] == "ap-policy-v1" and table["events"] == 3
    assert {row["stage"]: row["state"] for row in table["stages"]} == {
        "first": "blocked",
        "b": "pending",
    }
    assert cli.main(["clear", "--config", str(config), "--stage", "first"] + CLEAR) == 2
    assert "first' is blocked, not stopped" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # --stage, --reviewer and --reason belong to clear only
        cli.main(["tick", "--config", str(config), "--stage", "first"])
    capsys.readouterr()
    config.write_text(text + "\n[typo]\n", encoding="utf-8")
    assert cli.main(["tick", "--config", str(config)]) == 2
    assert capsys.readouterr().err == "autopilot: unknown section [typo]\n"


def test_status_alone_never_creates_the_run_root(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(stages, "discover", lambda: [toy("a")])
    monkeypatch.setattr(ops, "discover", lambda: ({}, []))
    config = tmp_path / "autopilot.toml"
    config.write_text('[autopilot]\nrun_root = "run"\npolicy_id = "p"\n', encoding="utf-8")
    assert cli.main(["status", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["events"] == 0
    assert not (tmp_path / "run").exists()


def test_a_spending_stage_goes_through_the_spend_guard_and_the_default_allows_no_calls(
    tmp_path, capsys, monkeypatch
):
    ran = []

    def run(ctx):
        ran.append(ctx.stage)
        return StageResult()

    monkeypatch.setattr(stages, "discover", lambda: [toy("freeze", run, spend=lambda ctx: 5)])
    monkeypatch.setattr(ops, "discover", lambda: ({}, []))
    config = tmp_path / "autopilot.toml"
    text = '[autopilot]\nrun_root = "run"\npolicy_id = "p"\n'
    config.write_text(text, encoding="utf-8")
    assert cli.main(["tick", "--config", str(config)]) == 2
    assert "spend_run_ceiling_exceeded" in capsys.readouterr().err and ran == []
    caller = tmp_path / "other"
    caller.mkdir()
    config = caller / "autopilot.toml"  # a new run root with the caller's own ceilings
    config.write_text(text + "[spend]\nrun_ceiling = 5\nglobal_ceiling = 5\n", encoding="utf-8")
    assert cli.main(["tick", "--config", str(config)]) == 0
    assert ran == ["freeze"]
    assert (caller / "run" / "autopilot" / "spend-guard.json").is_file()
    capsys.readouterr()
