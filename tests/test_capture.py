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
