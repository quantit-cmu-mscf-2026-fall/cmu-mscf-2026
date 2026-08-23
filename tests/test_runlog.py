"""Tests for the experiment run ledger.

The ledger's length is the trial count behind every multiple-testing
correction, so two guarantees matter: every `log_run` call really lands in the
file with its fields intact, and a broken `git` never blocks logging.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from capstone import runlog

REPO_ROOT = Path(__file__).resolve().parent.parent

ENTRY_FIELDS = (
    "ts_utc",
    "user",
    "git_sha",
    "session_id",
    "name",
    "seed",
    "params",
    "metrics",
    "tags",
    "notes",
)


def test_log_run_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    runlog.log_run("alpha", params={"lookback": 20}, metrics={"sharpe": 1.1}, seed=7)
    runlog.log_run("alpha", params={"lookback": 60}, seed=8, tags=["sweep"])
    runlog.log_run("beta", notes="baseline")

    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    entries = [json.loads(line) for line in lines]
    for entry in entries:
        for field in ENTRY_FIELDS:
            assert field in entry, f"missing field {field!r}"
        # ts_utc must be timezone-aware — a naive timestamp would make trial
        # ordering ambiguous across contributors in different timezones.
        assert datetime.fromisoformat(entry["ts_utc"]).tzinfo is not None

    assert entries[0]["params"] == {"lookback": 20}
    assert entries[0]["metrics"] == {"sharpe": 1.1}
    assert entries[0]["seed"] == 7
    assert entries[1]["tags"] == ["sweep"]
    assert entries[2]["params"] == {}
    assert entries[2]["seed"] is None
    assert entries[2]["notes"] == "baseline"


def test_git_sha_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    def broken_run(*args, **kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr(runlog.subprocess, "run", broken_run)

    # A missing git must degrade the SHA, never raise out of log_run.
    entry = runlog.log_run("gamma")
    assert entry["git_sha"] == "unknown"


def test_stats_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    for seed in range(3):
        runlog.log_run("alpha", seed=seed)

    env = dict(os.environ, CAPSTONE_LEDGER_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "capstone.runlog", "stats"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    assert "total runs: 3" in result.stdout
