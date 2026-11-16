"""`autopilot init` writes the starter set safely; `autopilot preflight` reports what is
missing before a first run. Fixtures only; nothing connects."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from capstone.pipeline.autopilot import __main__ as cli
from capstone.pipeline.autopilot import config as config_module
from capstone.pipeline.autopilot import preflight, scaffold

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "docs" / "examples" / "autopilot.example.toml"


def _toml(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def ready(tmp_path, *, edits=()):
    """init's starter set with every marker resolved and the world it names created."""
    folder = tmp_path / "run-config"
    scaffold.init(folder)
    config = folder / "autopilot.toml"
    text = re.sub(r"\s*# TODO\(person\):[^\n]*", "", config.read_text(encoding="utf-8"))
    for old, new in edits:
        assert old in text
        text = text.replace(old, new)
    config.write_text(text, encoding="utf-8")
    (folder / "runs").mkdir()
    registry = folder / "registry"
    (registry / ".git").mkdir(parents=True)
    (registry / "deploy").mkdir()
    (registry / "deploy" / "strategies.yml").write_text("strategies: []\n", encoding="utf-8")
    (folder / "archive" / "qualified").mkdir(parents=True)
    return config


def by_check(rows):
    out = {}
    for row in rows:
        out.setdefault(row.check, []).append(row)
    return out


# --- init ---------------------------------------------------------------------------


def test_init_writes_the_starter_set(tmp_path):
    written, kept = scaffold.init(tmp_path / "set")
    assert written == list(scaffold.FILES) and kept == []
    for name in scaffold.FILES:
        assert (tmp_path / "set" / name).read_bytes() == (scaffold.STARTER / name).read_bytes()


def test_init_never_overwrites(tmp_path):
    scaffold.init(tmp_path)
    mine = tmp_path / "autopilot.toml"
    mine.write_text("# mine\n", encoding="utf-8")
    written, kept = scaffold.init(tmp_path)
    assert written == [] and kept == ["autopilot.toml"]
    assert mine.read_text(encoding="utf-8") == "# mine\n"


def test_the_starter_config_validates_and_has_the_safe_defaults(tmp_path):
    scaffold.init(tmp_path)
    config = config_module.load(tmp_path / "autopilot.toml", [])
    assert config.core["real_model_calls"] is False
    assert config.sections["spend"]["run_ceiling"] == 0
    assert config.sections["spend"]["global_ceiling"] == 0


def test_the_starter_config_covers_the_example():
    example, starter = _toml(EXAMPLE), _toml(scaffold.STARTER / "autopilot.toml")
    assert set(example) == set(starter)
    for section, values in example.items():
        assert set(values) <= set(starter[section]), section


def test_every_value_a_person_must_choose_carries_a_marker():
    text = (scaffold.STARTER / "autopilot.toml").read_text(encoding="utf-8")
    marked = {line.split("=")[0].strip() for line in text.splitlines() if preflight.TODO in line}
    assert {"run_root", "policy_id", "checkout", "input_archive"} <= marked


def test_the_init_command(tmp_path, capsys):
    assert cli.main(["init", "--dir", str(tmp_path / "set")]) == 0
    out = capsys.readouterr().out
    assert out.count("wrote ") == len(scaffold.FILES) and "preflight --config" in out
    assert cli.main(["init", "--dir", str(tmp_path / "set")]) == 0
    assert capsys.readouterr().out.count("kept ") == len(scaffold.FILES)


def test_init_refuses_an_unreadable_starter_set(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scaffold, "STARTER", tmp_path / "absent")
    assert cli.main(["init", "--dir", str(tmp_path / "set")]) == 2
    assert "starter set is not readable" in capsys.readouterr().err


# --- preflight ------------------------------------------------------------------------


def test_a_fresh_starter_set_reports_every_marker_and_missing_prerequisite(tmp_path):
    scaffold.init(tmp_path)
    rows, config = preflight.run(tmp_path / "autopilot.toml")
    assert config is not None and all(not row.failed for row in rows if row.check == "config")
    checks = by_check(rows)
    assert len(checks["todo"]) == 4 and all(r.code == "todo_open" for r in checks["todo"])
    assert checks["registry"][0].code == "registry_not_git"
    assert checks["input_archive"][0].code == "path_missing"
    assert checks["path [autopilot] run_root"][0].code == "path_parent_missing"
    assert [r for r in rows if r.failed]


def test_a_resolved_world_is_ready(tmp_path):
    config = ready(tmp_path)
    rows, loaded = preflight.run(config)
    assert [row.line() for row in rows if row.failed] == []
    assert loaded is not None
    assert by_check(rows)["todo"][0].status == "ok"
    assert by_check(rows)["spend"][0].status == "ok"


def test_preflight_command_exit_codes_and_report_beside_the_config(tmp_path, capsys):
    config = ready(tmp_path)
    assert cli.main(["preflight", "--config", str(config)]) == 0
    report = json.loads((config.parent / "preflight.json").read_text(encoding="utf-8"))
    assert report["schema"] == preflight.SCHEMA and report["ready"] and report["failed"] == []
    assert all(row["fix"] for row in report["rows"])
    assert not (config.parent / "runs" / "first-run").exists()  # preflight never creates the run

    (tmp_path / "other").mkdir()
    scaffold.init(tmp_path / "other")
    assert cli.main(["preflight", "--config", str(tmp_path / "other" / "autopilot.toml")]) == 2
    captured = capsys.readouterr()
    assert "todo_open" in captured.out and "required not ok" in captured.err
    assert "fix: choose the value" in captured.out


def test_the_report_goes_under_the_run_root_once_it_exists(tmp_path):
    config = ready(tmp_path)
    run_root = config.parent / "runs" / "first-run"
    run_root.mkdir(parents=True)
    assert cli.main(["preflight", "--config", str(config)]) == 0
    assert (run_root / "ops" / "preflight.json").is_file()
    assert not (config.parent / "preflight.json").exists()


def test_the_registry_must_be_a_separate_git_clone_holding_the_registry_file(tmp_path):
    config = ready(tmp_path)
    folder = config.parent

    def registry_row():
        rows, _ = preflight.run(config)
        return by_check(rows)["registry"][0]

    (folder / "registry" / "deploy" / "strategies.yml").unlink()
    assert registry_row().code == "registry_file_missing"
    (folder / "registry" / ".git").rmdir()
    assert registry_row().code == "registry_not_git"
    inside = preflight.code_checkout() / "registry-clone"
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace('checkout = "registry"', f'checkout = "{inside.as_posix()}"'))
    assert registry_row().code == "registry_is_code_checkout"
    config.write_text(text.replace('checkout = "registry"', ""), encoding="utf-8")
    assert registry_row().code == "registry_not_set"


def test_spend_is_checked_against_real_model_calls(tmp_path):
    config = ready(tmp_path, edits=[("real_model_calls = false", "real_model_calls = true")])
    rows, loaded = preflight.run(config)
    assert loaded is None  # real model calls need a runs_dir: the loader refuses first
    assert by_check(rows)["config"][0].code == "config_invalid"

    config = ready(
        tmp_path / "b",
        edits=[
            ("real_model_calls = false", "real_model_calls = true"),
            ("accounting_roots = []", 'accounting_roots = []\nruns_dir = "runs"'),
        ],
    )
    spend = by_check(preflight.run(config)[0])["spend"][0]
    assert spend.code == "spend_inconsistent" and "run_ceiling 0" in spend.detail

    config = ready(
        tmp_path / "c",
        edits=[
            ("real_model_calls = false", "real_model_calls = true"),
            ("accounting_roots = []", 'accounting_roots = []\nruns_dir = "runs"'),
            ("run_ceiling = 0", "run_ceiling = 3"),
            ("global_ceiling = 0", "global_ceiling = 5"),
        ],
    )
    assert by_check(preflight.run(config)[0])["spend"][0].status == "ok"
    config = ready(tmp_path / "d", edits=[("run_ceiling = 0", "run_ceiling = 9")])
    assert "above global_ceiling" in by_check(preflight.run(config)[0])["spend"][0].detail


def test_an_invalid_or_unreadable_config_is_one_row_and_still_lists_markers(tmp_path):
    rows, config = preflight.run(tmp_path / "absent.toml")
    assert config is None and [(r.status, r.code) for r in rows] == [
        ("missing", "config_unreadable")
    ]
    bad = tmp_path / "bad.toml"
    bad.write_text('[autopilot]\nrun_root = "r"  # TODO(person): x\nbogus = 1\n', encoding="utf-8")
    rows, config = preflight.run(bad)
    assert config is None
    assert [r.code for r in rows] == ["config_invalid", "todo_open"]
    assert (
        "unknown key 'bogus'" in rows[0].detail
        and "missing required key 'policy_id'" in rows[0].detail
    )


def test_a_row_needs_a_known_status_and_code():
    with pytest.raises(ValueError, match="preflight row"):
        preflight.Row("x", "fine", "ok")
    with pytest.raises(ValueError, match="preflight row"):
        preflight.Row("x", "ok", "made_up")
    skipped = preflight.Checks()
    skipped.skip("c", "off")
    assert not skipped.rows[0].failed and skipped.rows[0].required is False


def test_template_files_a_config_names_are_scanned_for_markers(tmp_path):
    config = ready(tmp_path)
    template = config.parent / "t.json"
    template.write_text('{"a": "TODO(person): choose"}', encoding="utf-8")
    cfg = config_module.validate(
        {
            "autopilot": {"run_root": "r", "policy_id": "p"},
            "procedure": {"template": "t.json"},
        },
        [_stage_with_template()],
        path=config,
    )
    assert preflight._named_files(cfg) == [config, template.resolve()]
    checks = preflight.Checks()
    preflight._todos(checks, preflight._named_files(cfg))
    assert [r.code for r in checks.rows] == ["todo_open"] and checks.rows[0].detail.startswith(
        "t.json:1"
    )


def _stage_with_template():
    from capstone.pipeline.autopilot.config import Key
    from capstone.pipeline.autopilot.stage import Stage, StageResult

    return Stage(
        name="procedure",
        unit="T",
        run=lambda ctx: StageResult(),
        schema={"template": Key("path")},
    )
