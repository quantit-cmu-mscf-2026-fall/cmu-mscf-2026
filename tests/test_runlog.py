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

import pytest

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


def test_read_entries_returns_what_was_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    assert runlog.read_entries() == []

    runlog.log_run("alpha", seed=1)
    runlog.log_run("beta", seed=2)
    entries = runlog.read_entries()
    assert [entry["name"] for entry in entries] == ["alpha", "beta"]
    assert [entry["seed"] for entry in entries] == [1, 2]


def test_trial_count_counts_all_or_one_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    for seed in range(3):
        runlog.log_run("alpha", seed=seed)
    runlog.log_run("beta")

    assert runlog.trial_count() == 4
    assert runlog.trial_count("alpha") == 3
    assert runlog.trial_count("beta") == 1


def test_trial_count_refuses_zero(tmp_path, monkeypatch):
    # A correction against zero trials is no correction; the usual cause is a
    # wrong ledger directory or a misspelled name, so it must fail loudly.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    with pytest.raises(LookupError, match="no trials"):
        runlog.trial_count()

    runlog.log_run("alpha")
    with pytest.raises(LookupError, match="'alhpa'"):
        runlog.trial_count("alhpa")


def test_read_entries_names_a_malformed_line_instead_of_skipping_it(tmp_path, monkeypatch):
    # A run interrupted mid-write leaves a truncated line. Skipping it would
    # drop a trial from every downstream count, which under-corrects and
    # invents discoveries, so it has to fail and say which line.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("alpha", seed=0)
    with (tmp_path / runlog.LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write('{"name": "beta", "seed": 1\n')
    runlog.log_run("gamma", seed=2)

    with pytest.raises(ValueError, match=r"runs\.jsonl:2 is not valid JSON"):
        runlog.read_entries()
    with pytest.raises(ValueError, match=r"runs\.jsonl:2 is not valid JSON"):
        runlog.trial_count()


def test_an_interrupted_write_does_not_swallow_the_next_entry(tmp_path, monkeypatch):
    # The truncated line usually has no newline. If the next entry were
    # appended onto it, deleting the line the error names would drop a
    # complete trial along with the fragment.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    path = tmp_path / runlog.LEDGER_FILENAME
    runlog.log_run("alpha", seed=0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"name": "beta", "seed": 1')
    runlog.log_run("gamma", seed=2)

    with pytest.raises(ValueError, match=r"runs\.jsonl:2 is not valid JSON"):
        runlog.read_entries()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1] == '{"name": "beta", "seed": 1'
    assert json.loads(lines[2])["name"] == "gamma"

    # Deleting the named line, as the error says, keeps every complete trial.
    path.write_text("\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8")
    assert [e["name"] for e in runlog.read_entries()] == ["alpha", "gamma"]


def test_a_bare_string_tag_filter_is_refused(tmp_path, monkeypatch):
    # set("synthetic") is a set of letters: as an exclude it matches nothing
    # and returns the unscoped count with no error.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("pbo-calibration", tags=["synthetic"])
    runlog.log_run("mom-sweep")

    with pytest.raises(TypeError, match="exclude_tags takes a list"):
        runlog.trial_count(exclude_tags="synthetic")
    with pytest.raises(TypeError, match="include_tags takes a list"):
        runlog.trial_count(include_tags="synthetic")
    assert runlog.trial_count(exclude_tags=["synthetic"]) == 1


def test_blank_lines_are_not_trials(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("alpha")
    with (tmp_path / runlog.LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write("\n   \n")

    assert runlog.trial_count() == 1


def test_tags_keep_methodology_runs_out_of_a_strategy_family(tmp_path, monkeypatch):
    # The case this exists for: the ledger holds real strategy trials next to
    # calibration and simulation runs, which `CLAUDE.md` requires us to log but
    # which never competed for any strategy's result. Counting them only
    # inflates m. Without a tag filter there is no way to leave them out.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    for lookback in (20, 60, 120):
        runlog.log_run("mom-sweep", params={"lookback": lookback})
    runlog.log_run("carry-sweep")
    for seed in range(7):
        runlog.log_run("pbo-calibration", tags=["synthetic", "calibration"], seed=seed)

    assert runlog.trial_count() == 11  # everything, methodology included
    assert runlog.trial_count(exclude_tags=["synthetic"]) == 4  # the real trials
    assert runlog.trial_count(include_tags=["synthetic"]) == 7
    assert runlog.trial_count("mom-sweep") == 3

    # Name and tag filters compose, and an empty intersection still refuses to
    # answer 0 rather than letting a correction run against no trials.
    with pytest.raises(LookupError, match="excluding tags"):
        runlog.trial_count("pbo-calibration", exclude_tags=["synthetic"])


def test_an_entry_without_a_name_is_counted_but_matches_no_name(tmp_path, monkeypatch):
    # So the per-name counts need not sum to the total. Documented, and pinned
    # here because a reader will otherwise assume they do.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("alpha")
    with (tmp_path / runlog.LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seed": 9, "params": {}}) + "\n")

    assert runlog.trial_count() == 2
    assert runlog.trial_count("alpha") == 1


def test_ledger_path_is_the_file_log_run_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    assert runlog.ledger_path() == tmp_path / "runs.jsonl"
    runlog.log_run("alpha", seed=0)
    assert runlog.ledger_path().exists()


def test_a_repeated_trial_counts_twice(tmp_path, monkeypatch):
    # Each entry was logged before its result was seen, so a re-run is a trial too.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    for _ in range(3):
        runlog.log_run("sweep", params={"lookback": 20}, seed=0)
    assert runlog.trial_count("sweep") == 3
