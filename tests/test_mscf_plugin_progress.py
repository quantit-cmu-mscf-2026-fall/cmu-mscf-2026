"""Progress tracking for the pipeline view: `track`, the `pipeline_run.py` wrapper, stations.

Fixtures only. Progress files go to a temp folder through `MSCF_PROGRESS_DIR`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "mscf-plugin" / "skills" / "pipeline-view" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mscf_progress  # noqa: E402

STATIONS = json.loads((SCRIPTS / "stations.json").read_text(encoding="utf-8"))["stations"]


@pytest.fixture
def pdir(tmp_path, monkeypatch):
    path = tmp_path / "progress"
    monkeypatch.setenv("MSCF_PROGRESS_DIR", str(path))
    monkeypatch.delenv("MSCF_RUN_ID", raising=False)
    return path


def events(path: Path, run_id: str) -> list[dict]:
    lines = (path / f"{run_id}.events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_track_records_stations_progress_waiting_and_a_clean_finish(pdir):
    with mscf_progress.track("momentum-15-1") as run:
        meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
        assert meta["status"] == "running" and meta["name"] == "momentum-15-1"
        run.stage("data")
        run.wait("queued for a CPU")
        run.progress(1.7, "chunk 3")
    meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["exit_code"] == 0 and meta["ended"]
    evs = events(pdir, run.run_id)
    assert evs[0]["station"] == "data" and evs[1]["waiting"] is True
    assert evs[2]["pct"] == 1.0 and evs[2]["waiting"] is False  # clamped to 0..1


def test_track_marks_a_crash_but_not_a_clean_sys_exit(pdir):
    with pytest.raises(RuntimeError), mscf_progress.track("crashes") as run:
        raise RuntimeError("NaN spread")
    assert json.loads(run.meta_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert events(pdir, run.run_id)[-1]["note"] == "RuntimeError"  # the type, not "NaN spread"
    with pytest.raises(SystemExit), mscf_progress.track("exits") as run:
        raise SystemExit(0)
    assert json.loads(run.meta_path.read_text(encoding="utf-8"))["status"] == "done"


def test_track_turns_itself_off_when_the_directory_is_unwritable(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MSCF_PROGRESS_DIR", str(blocker / "progress"))
    with mscf_progress.track("still-runs") as run:
        run.stage("data")
    assert run.enabled is False


def test_pipeline_run_passes_the_exit_code_and_lets_the_script_attach(pdir):
    runner = SCRIPTS / "pipeline_run.py"
    child = (
        "import mscf_progress; r = mscf_progress.track('wrapped'); r.stage('signal'); "
        "raise SystemExit(3)"
    )
    code = subprocess.run(
        [sys.executable, str(runner), "--name", "wrapped", "--", sys.executable, "-c", child]
    ).returncode
    assert code == 3
    (meta_path,) = list(pdir.glob("*.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and meta["exit_code"] == 3 and meta["name"] == "wrapped"
    assert [e.get("station") for e in events(pdir, meta["run_id"])] == ["hypothesis", "signal"]


def test_the_wrapper_notes_a_script_name_never_the_command_arguments(pdir):
    runner = SCRIPTS / "pipeline_run.py"
    for command, note in (
        ([sys.executable, "-c", "print('sharpe 1.37 on AAPL')"], ""),
        ([sys.executable, "experiments/mom.py", "--secret-flag", "x"], "mom.py"),
    ):
        pdir_files = set(pdir.glob("*.json")) if pdir.is_dir() else set()
        subprocess.run([sys.executable, str(runner), "--name", "n", "--", *command])
        (meta_path,) = set(pdir.glob("*.json")) - pdir_files
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert events(pdir, meta["run_id"])[0]["note"] == note


def test_progress_files_stay_outside_the_repository_and_hold_no_host_or_session(pdir, monkeypatch):
    with mscf_progress.track("private") as run:
        run.stage("data")
    meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
    assert set(meta) == {
        "run_id",
        "name",
        "pid",
        "user",
        "started",
        "heartbeat",
        "status",
        "exit_code",
        "ended",
    }
    monkeypatch.delenv("MSCF_PROGRESS_DIR")
    assert ROOT not in mscf_progress.progress_dir().parents


def test_stations_are_the_ten_steps_of_the_pipeline_and_the_stage_tags_the_ledger_uses():
    assert [s["id"] for s in STATIONS][:5] == ["hypothesis", "data", "signal", "backtest", "ledger"]
    assert [s["ledger_stage"] for s in STATIONS[5:]] == [
        "screen",
        "robustness",
        "incremental",
        "final",
        "incubation",
    ]
    assert all({"id", "label", "detail", "icon"} <= set(s) for s in STATIONS)
    text = (SCRIPTS / "stations.json").read_text(encoding="utf-8")
    assert not re.search(r"caleb|ibkr|[A-Za-z]:\\\\", text, re.I)
