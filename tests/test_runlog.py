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
    "hypothesis",
    "source",
    "decision_ref",
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


def test_log_run_extended_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    entry = runlog.log_run(
        "alpha",
        params={"lookback": 20},
        metrics={"decision": "pass"},
        seed=7,
        hypothesis="BH screening retains the signal in the faint regime",
        source="machine-generated",
        decision_ref="docs/decisions.md",
        tags=["screening"],
    )

    assert entry["hypothesis"] == "BH screening retains the signal in the faint regime"
    assert entry["source"] == "machine-generated"
    assert entry["decision_ref"] == "docs/decisions.md"
    assert entry["metrics"]["decision"] == "pass"


def test_git_sha_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    def broken_run(*args, **kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr(runlog.subprocess, "run", broken_run)

    # A missing git must degrade the SHA, never raise out of log_run.
    entry = runlog.log_run("gamma")
    assert entry["git_sha"] == "unknown"


def test_digest_cli_groups_by_tag_and_decision_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    trial_a = runlog.log_run(
        "alpha",
        hypothesis="Keep the signal in the faint regime",
        source="machine-generated",
        decision_ref="docs/decisions.md",
        metrics={"decision": "pass"},
        tags=["screening"],
    )
    runlog.log_outcome(
        trial_ref=trial_a["run_id"],
        decision="pass",
        metrics={"decision": "pass", "sharpe": 1.1},
        source="machine-generated",
        decision_ref="docs/decisions.md",
        tags=["screening"],
    )
    trial_b = runlog.log_run(
        "beta",
        hypothesis="Reject nulls under strict control",
        source="literature:benjamini_hochberg+2024-01-01",
        decision_ref="docs/decisions.md",
        metrics={"decision": "nothing"},
        tags=["screening"],
    )
    runlog.log_outcome(
        trial_ref=trial_b["run_id"],
        decision="nothing",
        metrics={"decision": "nothing", "sharpe": -0.3},
        source="literature:benjamini_hochberg+2024-01-01",
        decision_ref="docs/decisions.md",
        tags=["screening"],
    )

    env = dict(os.environ, CAPSTONE_LEDGER_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "capstone.runlog", "digest"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    assert "screening" in result.stdout
    assert "Hypothesis" in result.stdout
    assert "Source" in result.stdout
    assert "Decision" in result.stdout
    assert "docs/decisions.md" in result.stdout
    assert "pass" in result.stdout
    assert "nothing" in result.stdout


def test_screened_candidate_has_trial_and_outcome_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    trial = runlog.log_run(
        "candidate_screen",
        params={"n_candidates": 2, "alpha": 0.05},
        seed=7,
        hypothesis="Low-volatility premium",
        source="economic-reasoning",
        tags=["backlog-screen"],
    )
    outcome = runlog.log_outcome(
        trial_ref=trial["run_id"],
        decision="nothing",
        metrics={"decision": "nothing", "sharpe": -0.41},
        source="economic-reasoning",
        hypothesis="Low-volatility premium",
        decision_ref="docs/decisions.md",
        tags=["backlog-screen"],
    )

    entries = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    trial_rows = [entry for entry in entries if entry.get("kind") == "trial"]
    outcome_rows = [entry for entry in entries if entry.get("kind") == "outcome"]
    assert len(trial_rows) == 1
    assert len(outcome_rows) == 1
    assert outcome_rows[0]["trial_ref"] == trial["run_id"]
    assert outcome_rows[0]["metrics"]["decision"] == "nothing"

    env = dict(os.environ, CAPSTONE_LEDGER_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "capstone.runlog", "digest"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    assert "Low-volatility premium" in result.stdout
    assert "economic-reasoning" in result.stdout
    assert "Decision: nothing" in result.stdout


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
