"""Autopilot schedule rendering: task.xml and cron entries from `[schedule]`.

Toy stages only; nothing is registered or installed, nothing touches the network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import ops
from capstone.pipeline.autopilot.ops import common, schedule
from capstone.pipeline.autopilot.stage import Stage, StageResult

REPO = Path(__file__).resolve().parents[1]
SECTIONS, HOOKS = ops.discover()
TASK = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"


def toy():
    return Stage(name="a", unit="T", run=lambda ctx: StageResult())


def make_config(tmp_path, **sections):
    raw = {"autopilot": {"run_root": str(tmp_path / "run"), "policy_id": "p1"}, **sections}
    return cfg.validate(raw, [toy()], extra=SECTIONS, path=tmp_path / "autopilot.toml")


def test_schedule_section_has_defaults(tmp_path):
    config = make_config(tmp_path)
    assert config.sections["schedule"] == {"task_name": "capstone-autopilot", "time": "18:30"}


def test_schedule_renders_task_xml_and_cron_from_the_config(tmp_path):
    config = make_config(tmp_path, schedule={"task_name": "sched toy", "time": "06:05"})
    python = tmp_path / "venv" / "python.exe"
    doc = schedule.render(config, python=str(python), checkout=REPO)
    out = common.ops_dir(config.run_root) / "schedule"
    assert (out / "task.xml").read_bytes()[:2] == b"\xff\xfe"  # UTF-16 LE with BOM
    task = ET.parse(out / "task.xml").getroot()
    assert task.find(f"{TASK}RegistrationInfo/{TASK}URI").text == "\\sched toy"
    trigger = task.find(f"{TASK}Triggers/{TASK}CalendarTrigger")
    assert trigger.find(f"{TASK}StartBoundary").text.endswith("T06:05:00")
    assert trigger.find(f"{TASK}ScheduleByDay/{TASK}DaysInterval").text == "1"
    settings = task.find(f"{TASK}Settings")
    assert settings.find(f"{TASK}MultipleInstancesPolicy").text == "IgnoreNew"
    exec_ = task.find(f"{TASK}Actions/{TASK}Exec")
    assert exec_.find(f"{TASK}Command").text == "powershell.exe"
    arguments = exec_.find(f"{TASK}Arguments").text
    assert "run-tick.ps1" in arguments and str(python) in arguments
    assert str((tmp_path / "autopilot.toml").resolve()) in arguments
    assert doc["command"][0] == "powershell.exe" and doc["time"] == "06:05"
    cron = (out / "autopilot.cron").read_text(encoding="utf-8").splitlines()[-1]
    assert cron.startswith("5 6 * * * mkdir -p ") and " && cd " in cron
    assert "-m capstone.pipeline.autopilot tick --config" in cron
    assert cron.endswith("# sched toy") and "tick.log" in cron
    assert json.loads((out / "schedule.json").read_text("utf-8"))["task_name"] == "sched toy"


def test_the_cron_line_creates_the_log_folder_before_it_redirects(tmp_path):
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("no POSIX shell")
    config = make_config(tmp_path)
    # `true` stands in for the interpreter: the line only has to get as far as running it
    schedule.render(config, python="true", checkout=tmp_path, out=tmp_path / "rendered")
    out = tmp_path / "rendered"
    line = (out / "autopilot.cron").read_text(encoding="utf-8").splitlines()[-1]
    command = line.split(" * * * ", 1)[1].split(" # ", 1)[0]
    log = common.ops_dir(config.run_root) / "tick.log"
    assert not log.parent.exists()
    done = subprocess.run([sh, "-c", command], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert log.is_file()


def test_rendering_registers_nothing_and_writes_only_under_out(tmp_path):
    config = make_config(tmp_path)
    out = tmp_path / "elsewhere"
    schedule.render(config, python="python", checkout=REPO, out=out)
    assert sorted(p.name for p in out.iterdir()) == ["autopilot.cron", "schedule.json", "task.xml"]
    assert not (config.run_root / "autopilot" / "ops" / "schedule").exists()


@pytest.mark.parametrize(
    ("line", "needle"),
    [
        ('time = "25:00"', "is not HH:MM"),
        ('time = "6:5"', "is not HH:MM"),
        ('task_name = "a\\\\b"', "letters, digits"),
    ],
)
def test_schedule_refuses_a_bad_time_or_name_with_one_line(tmp_path, capsys, line, needle):
    text = f'[autopilot]\nrun_root = "run"\npolicy_id = "p1"\n[schedule]\n{line}\n'
    (tmp_path / "autopilot.toml").write_text(text, encoding="utf-8")
    assert schedule.main(["--config", str(tmp_path / "autopilot.toml")]) == 2
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and needle in err
    assert not (tmp_path / "run").exists()


def test_main_prints_the_document(tmp_path, capsys):
    (tmp_path / "autopilot.toml").write_text(
        '[autopilot]\nrun_root = "run"\npolicy_id = "p1"\n', encoding="utf-8"
    )
    out = tmp_path / "out"
    argv = ["--config", str(tmp_path / "autopilot.toml"), "--out", str(out)]
    assert schedule.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["time"] == "18:30"
    assert (out / "task.xml").is_file()


def test_the_tick_script_runs_the_tick_and_parses():
    script = REPO / "deploy" / "autopilot" / "run-tick.ps1"
    text = script.read_text(encoding="utf-8")
    assert "-m capstone.pipeline.autopilot tick --config $Config" in text
    assert "exit $code" in text
    assert "*>>" not in text  # appends UTF-16 under Windows PowerShell 5.1
    assert "Add-Content -Path $Log -Encoding utf8" in text.split("autopilot tick", 1)[1]
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("no PowerShell to parse the script")
    check = (
        "$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}',[ref]$null,[ref]$e);$e.Count"
    )
    done = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", check],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0 and done.stdout.strip() == "0", done.stdout
