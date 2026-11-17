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


def test_a_bare_string_tag_is_one_tag_when_logged(tmp_path, monkeypatch):
    # The other side of the filter check above: log_run(tags="strategy") used
    # to store the string, which set() splits into letters, so include_tags
    # dropped the entry (an under-count) and exclude_tags kept it.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    entry = runlog.log_run("pbo-calibration", tags="synthetic")
    runlog.log_run("mom-sweep", tags="strategy")
    runlog.log_run("carry-sweep", tags=["strategy"])

    assert entry["tags"] == ["synthetic"]
    assert runlog.read_entries()[0]["tags"] == ["synthetic"]
    assert runlog.trial_count(include_tags=["strategy"]) == 2
    assert runlog.trial_count(exclude_tags=["synthetic"]) == 2


def test_a_string_tag_already_in_the_ledger_still_filters(tmp_path, monkeypatch):
    # Entries written before log_run normalised strings hold the bare string.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    with (tmp_path / runlog.LEDGER_FILENAME).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": "old-calibration", "tags": "synthetic"}) + "\n")
        fh.write(json.dumps({"name": "old-sweep", "tags": "strategy"}) + "\n")

    assert runlog.trial_count(include_tags=["strategy"]) == 1
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


# ---------------------------------------------------------------------------
# Corrections: the ledger is append-only, so a run that should not count is
# excluded by a later "ledger-correction" entry, never deleted.


def _correction(ledger_name, excludes, reason="duplicates"):
    return runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"ledger_name": ledger_name, "excludes": excludes, "reason": reason},
    )


def test_trial_count_honours_a_correction(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    first = [runlog.log_run("sweep", params={"factor_id": f}) for f in ("a", "b")]
    runlog.log_run("sweep", params={"factor_id": "a", "variant_id": "a-1"})
    runlog.log_run("other", params={"factor_id": "a"})
    assert runlog.trial_count("sweep") == 3

    _correction(
        "sweep", [{"ts_utc": e["ts_utc"], "factor_id": e["params"]["factor_id"]} for e in first]
    )
    assert runlog.trial_count("sweep") == 1
    assert runlog.trial_count("other") == 1
    # The correction is bookkeeping, not a trial, and nothing was deleted.
    assert runlog.trial_count() == 2
    assert len(runlog.read_entries()) == 5


def test_a_correction_matches_on_the_timestamp_and_every_param_it_names(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep", params={"variant_id": "v1", "lookback": 20})
    runlog.log_run("sweep", params={"variant_id": "v2"})
    _correction("sweep", [{"ts_utc": run["ts_utc"], "variant_id": "v3"}])  # wrong param
    _correction("other", [{"ts_utc": run["ts_utc"]}])  # another experiment's runs
    _correction("sweep", [{"ts_utc": "2001-01-01T00:00:00+00:00"}])  # no such run
    assert runlog.trial_count("sweep") == 2

    _correction("sweep", [{"ts_utc": run["ts_utc"], "variant_id": "v1", "lookback": 20}])
    assert runlog.trial_count("sweep") == 1


def test_corrections_compose_with_the_tag_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep", params={"seed_set": 1}, tags=["real"])
    runlog.log_run("sweep", params={"seed_set": 2}, tags=["real"])
    runlog.log_run("sweep", params={"seed_set": 3}, tags=["synthetic"])
    _correction("sweep", [{"ts_utc": run["ts_utc"], "seed_set": 1}])
    assert runlog.trial_count("sweep", include_tags=["real"]) == 1
    assert runlog.trial_count("sweep", exclude_tags=["real"]) == 1
    assert runlog.trial_count(exclude_tags=["synthetic"]) == 1


def test_a_correction_that_excludes_everything_still_refuses_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep")
    _correction("sweep", [{"ts_utc": run["ts_utc"]}])
    with pytest.raises(LookupError):
        runlog.trial_count("sweep")


@pytest.mark.parametrize(
    "params",
    [
        {"excludes": [{"ts_utc": "x"}], "reason": "r"},  # no ledger_name
        {"ledger_name": "sweep", "excludes": [{"factor_id": "a"}], "reason": "r"},  # no ts_utc
        {"ledger_name": "sweep", "excludes": {"ts_utc": "x"}, "reason": "r"},  # not a list
    ],
)
def test_a_malformed_correction_raises(tmp_path, monkeypatch, params):
    # Ignoring it would quietly keep runs someone meant to exclude, or (worse,
    # with a guessed reading) drop runs nobody meant to.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("sweep")
    runlog.log_run(runlog.CORRECTION_NAME, params=params)
    with pytest.raises(ValueError, match="correction"):
        runlog.trial_count("sweep")


def test_stats_counts_trials_after_corrections(tmp_path, monkeypatch, capsys):
    # CLAUDE.md sends people to `stats` for the trial count, so it must agree
    # with `trial_count`.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runs = [runlog.log_run("alpha", seed=seed) for seed in range(3)]
    runlog.log_run("beta")
    _correction("alpha", [{"ts_utc": runs[0]["ts_utc"]}])
    runlog.main(["stats"])
    out = capsys.readouterr().out
    assert "total runs: 5" in out
    assert "trials (corrections applied): 3" in out
    assert "  alpha: 2" in out and "  beta: 1" in out
    assert runlog.CORRECTION_NAME not in out


# ---------------------------------------------------------------------------
# Review of #94: a correction must say why, can be retracted, and a seen result
# cannot leave the count invisibly.


def _retraction(*targets, reason="the correction excluded the wrong runs"):
    return runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"retracts": [t["ts_utc"] for t in targets], "reason": reason},
    )


def _ten_seen_runs():
    return [
        runlog.log_run("mom", params={"lookback": i}, metrics={"sharpe": i / 10}) for i in range(10)
    ]


def _exclude_nine_losers(runs):
    return _correction(
        "mom", [{"ts_utc": r["ts_utc"], "lookback": r["params"]["lookback"]} for r in runs[:9]]
    )


@pytest.mark.parametrize("reason", [None, "", "   ", 7, ["why"]])
def test_a_correction_needs_a_non_empty_reason(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep")
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"ledger_name": "sweep", "excludes": [{"ts_utc": run["ts_utc"]}], "reason": reason},
    )
    with pytest.raises(ValueError, match="reason"):
        runlog.trial_count("sweep")


def test_a_correction_without_any_reason_key_is_refused(tmp_path, monkeypatch):
    # Nine of ten seen runs dropped, no explanation: the "best of N as one" case.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runs = _ten_seen_runs()
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"ledger_name": "mom", "excludes": [{"ts_utc": r["ts_utc"]} for r in runs[:9]]},
    )
    with pytest.raises(ValueError, match="reason"):
        runlog.trial_count("mom")


def test_a_correction_can_be_retracted(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runs = [runlog.log_run("sweep", params={"i": i}) for i in range(4)]
    wrong = _correction("sweep", [{"ts_utc": r["ts_utc"]} for r in runs[:3]])
    assert runlog.trial_count("sweep") == 1
    _retraction(wrong)
    assert runlog.trial_count("sweep") == 4  # the mistake no longer lowers m
    assert len(runlog.read_entries()) == 6  # and nothing was rewritten


def test_a_retraction_can_be_retracted_in_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep")
    runlog.log_run("sweep", params={"i": 1})
    first = _correction("sweep", [{"ts_utc": run["ts_utc"]}])
    undo = _retraction(first)
    assert runlog.trial_count("sweep") == 2
    _retraction(undo, reason="the first correction was right after all")
    assert runlog.trial_count("sweep") == 1


def test_a_retraction_names_an_earlier_correction(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep")
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"retracts": [run["ts_utc"]], "reason": "r"},  # a run, not a correction
    )
    with pytest.raises(ValueError, match="no earlier correction"):
        runlog.trial_count("sweep")


@pytest.mark.parametrize("retracts", ["x", [], [3], {"ts_utc": "x"}])
def test_a_malformed_retraction_raises(tmp_path, monkeypatch, retracts):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("sweep")
    runlog.log_run(runlog.CORRECTION_NAME, params={"retracts": retracts, "reason": "r"})
    with pytest.raises(ValueError, match="retracts"):
        runlog.trial_count("sweep")


def test_one_entry_cannot_both_retract_and_exclude(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep")
    first = _correction("sweep", [{"ts_utc": run["ts_utc"]}])
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={
            "retracts": [first["ts_utc"]],
            "ledger_name": "sweep",
            "excludes": [],
            "reason": "r",
        },
    )
    with pytest.raises(ValueError, match="not both"):
        runlog.trial_count("sweep")


def test_excluding_seen_runs_warns_and_can_be_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    _exclude_nine_losers(_ten_seen_runs())
    with pytest.warns(UserWarning, match="9 run"):
        assert runlog.trial_count("mom") == 1
    assert runlog.trial_count("mom", count_seen_exclusions=True) == 10


def test_excluding_unseen_runs_does_not_warn(tmp_path, monkeypatch, recwarn):
    # No metrics logged: no result existed to be seen, so a duplicate is just dropped.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runs = [runlog.log_run("sweep", params={"i": i}) for i in range(3)]
    _correction("sweep", [{"ts_utc": r["ts_utc"]} for r in runs[:2]])
    assert runlog.trial_count("sweep") == 1
    assert runlog.trial_count("sweep", count_seen_exclusions=True) == 1
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_the_seen_warning_is_scoped_to_the_counted_family(tmp_path, monkeypatch, recwarn):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    seen = runlog.log_run("loud", metrics={"sharpe": 1.0})
    runlog.log_run("loud", metrics={"sharpe": 0.5})
    runlog.log_run("quiet")
    _correction("loud", [{"ts_utc": seen["ts_utc"]}])
    assert runlog.trial_count("quiet") == 1
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_stats_shows_what_was_excluded_and_what_was_seen(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runs = _ten_seen_runs()
    dups = [runlog.log_run("dup") for _ in range(3)]
    _exclude_nine_losers(runs)
    wrong = _correction("dup", [{"ts_utc": dups[0]["ts_utc"]}])
    _retraction(wrong)
    _correction("dup", [{"ts_utc": "2001-01-01T00:00:00+00:00"}])  # matches no run
    runlog.main(["stats"])
    out = capsys.readouterr().out
    assert "trials (corrections applied): 4" in out  # mom 1 + dup 3
    assert "excluded by corrections: 9" in out
    assert "of which results already recorded (seen): 9" in out
    assert "trials if seen exclusions were counted: 13" in out
    assert "correction entries: 4 (1 retracted)" in out
    assert "correction items matching no earlier run: 1" in out
    assert "  mom: 9 (9)" in out
    assert "  dup: 3" in out and "dup: 0" not in out


def test_a_correction_matches_the_exact_json_value(tmp_path, monkeypatch):
    # `1 == True` in Python; in JSON they are different values.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    run = runlog.log_run("sweep", params={"flag": True, "w": 20})
    runlog.log_run("sweep")
    _correction("sweep", [{"ts_utc": run["ts_utc"], "flag": 1}])
    _correction("sweep", [{"ts_utc": run["ts_utc"], "w": 20.0}])
    _correction("sweep", [{"ts_utc": run["ts_utc"], "w": True}])
    assert runlog.trial_count("sweep") == 2
    _correction("sweep", [{"ts_utc": run["ts_utc"], "flag": True, "w": 20}])
    assert runlog.trial_count("sweep") == 1


def test_a_correction_only_applies_to_earlier_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    path = tmp_path / runlog.LEDGER_FILENAME
    stamp = "2026-01-01T00:00:00+00:00"
    other = {"name": "sweep", "ts_utc": "2026-01-02T00:00:00+00:00", "params": {}}
    target = {"name": "sweep", "ts_utc": stamp, "params": {}}
    correction = {
        "name": runlog.CORRECTION_NAME,
        "ts_utc": "2026-01-03T00:00:00+00:00",
        "params": {"ledger_name": "sweep", "excludes": [{"ts_utc": stamp}], "reason": "r"},
    }
    path.write_text("\n".join(json.dumps(e) for e in (correction, target, other)) + "\n")
    assert runlog.trial_count("sweep") == 2  # written before the run, so it cannot exclude it
    path.write_text("\n".join(json.dumps(e) for e in (target, correction, other)) + "\n")
    assert runlog.trial_count("sweep") == 1


@pytest.mark.parametrize("ledger_name", [None, "", ["sweep"], {"a": 1}, 3])
def test_a_correction_needs_a_string_ledger_name(tmp_path, monkeypatch, ledger_name):
    # None used to pass and exclude every unnamed run; a list raised TypeError.
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("sweep")
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"ledger_name": ledger_name, "excludes": [{"ts_utc": "x"}], "reason": "r"},
    )
    with pytest.raises(ValueError, match="ledger_name"):
        runlog.trial_count("sweep")


@pytest.mark.parametrize("ts_utc", [["t"], {"a": 1}, 3, None, ""])
def test_a_correction_item_needs_a_string_ts_utc(tmp_path, monkeypatch, ts_utc):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("sweep")
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"ledger_name": "sweep", "excludes": [{"ts_utc": ts_utc}], "reason": "r"},
    )
    with pytest.raises(ValueError, match="ts_utc"):
        runlog.trial_count("sweep")


def test_a_malformed_correction_names_the_file_and_line(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    runlog.log_run("sweep")
    runlog.log_run("sweep")
    runlog.log_run(
        runlog.CORRECTION_NAME, params={"ledger_name": "sweep", "excludes": {}, "reason": "r"}
    )
    with pytest.raises(ValueError) as caught:
        runlog.trial_count("sweep")
    assert f"{tmp_path / runlog.LEDGER_FILENAME}:3 " in str(caught.value)


def test_an_unnamed_run_is_not_excluded_by_a_null_ledger_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    path = tmp_path / runlog.LEDGER_FILENAME
    path.write_text(json.dumps({"ts_utc": "U1", "params": {}}) + "\n")
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={"ledger_name": None, "excludes": [{"ts_utc": "U1"}], "reason": "r"},
    )
    with pytest.raises(ValueError, match="ledger_name"):
        runlog.trial_count()
