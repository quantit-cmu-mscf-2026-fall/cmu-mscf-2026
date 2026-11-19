"""Session capture and redaction behave as documented.

The capture hook runs on every agent stop, so its two contracts are absolute:
it must capture when given a valid payload, and it must never fail (exit != 0
or noise on stdout would disrupt the person working) no matter what it is fed.
Redaction is tested against clearly fake credential-shaped fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

STARTER = Path(__file__).resolve().parents[1]
CAPTURE = STARTER / "scripts" / "capture" / "session_capture.py"
SYNC = STARTER / "scripts" / "capture" / "sync_sessions.py"


def _fake_repo(tmp_path: Path) -> Path:
    """Copy the real capture script into a tmp repo tree so its repo-root
    resolution (parents[2]) lands inside tmp_path."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "capture").mkdir(parents=True)
    shutil.copyfile(CAPTURE, repo / "scripts" / "capture" / "session_capture.py")
    return repo


def test_capture_copies_and_indexes(tmp_path):
    repo = _fake_repo(tmp_path)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"role": "user"}\n')
    payload = {
        "session_id": "abcd1234efgh5678",
        "transcript_path": str(transcript),
        "hook_event_name": "Stop",
    }
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "capture" / "session_capture.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"USER": "tester", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    copied = repo / ".sessions" / "transcripts" / "tester-abcd1234.jsonl"
    assert copied.exists()
    assert copied.read_text() == '{"role": "user"}\n'
    index_lines = (repo / ".sessions" / "index.jsonl").read_text().splitlines()
    entry = json.loads(index_lines[-1])
    assert entry["session_id"] == "abcd1234efgh5678"
    assert entry["event"] == "Stop"


def test_capture_never_fails_on_garbage(tmp_path):
    repo = _fake_repo(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "capture" / "session_capture.py")],
        input="this is not json {",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def _load_sync():
    spec = importlib.util.spec_from_file_location("sync_sessions", SYNC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_redact_patterns():
    mod = _load_sync()
    fixtures = "\n".join(
        [
            "sk-" + "a" * 24,
            "ghp_" + "B" * 24,
            "AKIA" + "Z" * 16,
            "eyJ" + "x" * 24 + "." + "y" * 12 + "." + "z" * 12,
            "password = hunter2secretvalue",
        ]
    )
    clean, n = mod.redact(fixtures)
    assert n >= 5
    assert "hunter2secretvalue" not in clean
    assert "[REDACTED]" in clean

    benign = "the word token alone, and a plain sentence about backtests"
    same, zero = mod.redact(benign)
    assert same == benign
    assert zero == 0


def test_hook_commands_are_anchored_to_project_dir():
    """Hooks must not depend on the session's cwd (a `cd` would break them)."""
    settings = json.loads((STARTER / ".claude" / "settings.json").read_text())
    for event in settings["hooks"].values():
        for group in event:
            for hook in group["hooks"]:
                assert "$CLAUDE_PROJECT_DIR/" in hook["command"]


def test_placebo_ledger_is_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "experiments/placebo/runs.jsonl"],
        cwd=STARTER,
    )
    assert result.returncode == 0


def test_sync_keeps_real_and_placebo_ledgers_separate(tmp_path, monkeypatch):
    mod = _load_sync()
    repo, archive = tmp_path / "repo", tmp_path / "archive"
    (repo / "experiments" / "placebo").mkdir(parents=True)
    archive.mkdir()
    (repo / "experiments" / "runs.jsonl").write_text('"real fixture"\n', encoding="utf-8")
    (repo / "experiments" / "placebo" / "runs.jsonl").write_text(
        '"placebo fixture"\n', encoding="utf-8"
    )
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    monkeypatch.setattr(mod, "ARCHIVE", archive)
    monkeypatch.setenv("GITHUB_USER", "fixture")
    # No real archive Git commands, including pushes, execute in this test.
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0)
    )
    assert mod.main() == 0
    assert (archive / "ledger" / "fixture-runs.jsonl").read_text() == '"real fixture"\n'
    assert (
        archive / "ledger" / "placebo" / "fixture-runs.jsonl"
    ).read_text() == '"placebo fixture"\n'


def test_placebo_archive_cannot_overwrite_another_users_real_ledger(tmp_path, monkeypatch):
    mod = _load_sync()
    real_repo, placebo_repo, archive = tmp_path / "real", tmp_path / "placebo", tmp_path / "archive"
    (real_repo / "experiments").mkdir(parents=True)
    (placebo_repo / "experiments" / "placebo").mkdir(parents=True)
    archive.mkdir()
    (real_repo / "experiments" / "runs.jsonl").write_text("REAL-A\n", encoding="utf-8")
    (placebo_repo / "experiments" / "placebo" / "runs.jsonl").write_text(
        "PLACEBO-B\n", encoding="utf-8"
    )
    monkeypatch.setattr(mod, "ARCHIVE", archive)
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0)
    )
    monkeypatch.setattr(mod, "REPO_ROOT", real_repo)
    monkeypatch.setenv("GITHUB_USER", "fixture-placebo")
    assert mod.main() == 0
    monkeypatch.setattr(mod, "REPO_ROOT", placebo_repo)
    monkeypatch.setenv("GITHUB_USER", "fixture")
    assert mod.main() == 0
    assert (archive / "ledger" / "fixture-placebo-runs.jsonl").read_text() == "REAL-A\n"
    assert (archive / "ledger" / "placebo" / "fixture-runs.jsonl").read_text() == "PLACEBO-B\n"
