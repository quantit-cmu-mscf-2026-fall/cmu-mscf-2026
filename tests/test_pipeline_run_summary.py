"""`pipeline.run_summary`: usage, cost estimate and the slide-ready summary of a search.

A scripted model client, synthetic rows, a temporary ledger. No model, no network, no data.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace

import pytest
from test_deepen import ScriptedEditor
from test_factor_evaluation import fake_rows
from test_pipeline_search_run import PERIOD, Loader
from test_pipeline_search_steps import EDITS, ScriptedClient, _config

from capstone import gate, runlog
from capstone.pipeline import manifest, run_summary
from capstone.pipeline.run_summary import run_real_search, summarize_workdir


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    base = tmp_path_factory.mktemp("summary")
    client = ScriptedClient()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CAPSTONE_LEDGER_DIR", str(base / "ledger"))
        summary = run_real_search(
            _config(base, **PERIOD),
            client=client,
            editor=ScriptedEditor(*EDITS),
            loader=Loader(fake_rows(n_stocks=12, n_days=150)),
            source_version="fx",
            workdir=base / "run",
        )
        entries = runlog.read_entries()
    return dict(summary=summary, client=client, entries=entries, workdir=base / "run")


def test_the_summary_counts_calls_trials_and_variants(ran):
    summary = ran["summary"]
    assert summary["model_calls"] == ran["client"].calls > 0
    assert summary["trial_count"] >= summary["n_variants"] > 0
    assert summary["families"] and summary["deepen"]["rounds"] == 3
    assert summary["stage3"] == manifest.STAGE3_UNMEASURED
    assert summary["thresholds_source"] in {"config", "code-defaults"}


def test_the_summary_prices_the_recorded_usage(ran):
    usage = ran["summary"]["usage"]
    assert sum(m["calls"] for m in usage["by_model"].values()) == ran["client"].calls
    assert sum(m["input_tokens"] for m in usage["by_model"].values()) == 10 * ran["client"].calls
    assert set(usage["unpriced"]) <= set(usage["by_model"])


def test_the_summary_lists_each_paper_and_the_factors_it_stored(ran):
    papers = ran["summary"]["papers"]
    assert [p["title"] for p in papers] == ["alpha", "beta"]
    assert all(p["hypotheses"] == 1 and p["factors"] >= 1 for p in papers)
    assert sum(p["factors"] for p in papers) == ran["summary"]["seed_factors"]


def test_the_summary_counts_the_deepen_children_by_status(ran):
    children = ran["summary"]["deepen"]["children"]
    assert children and sum(children.values()) >= ran["summary"]["deepen"]["rounds"]
    assert all(d["by"] == "memory" for d in ran["summary"]["deepen"]["decisions"])


def test_the_summary_ranks_the_variants_by_net_sharpe(ran):
    variants = ran["summary"]["variants"]
    sharpes = [v["sharpe_net"] for v in variants["top"]]
    assert sharpes == sorted(sharpes, reverse=True) and len(sharpes) <= run_summary.TOP_VARIANTS
    assert {"variant_id", "hold", "neutral", "expression", "sharpe_gross"} <= set(
        variants["top"][0]
    )
    assert variants["count"] == ran["summary"]["n_variants"]


def test_the_run_registers_the_gate_config_and_never_looks(ran):
    names = [e["name"] for e in ran["entries"]]
    assert gate.REGISTRATION_NAME in names and gate.LOOK_NAME not in names
    assert ran["summary"]["holdout_opened"] is False
    assert ran["summary"]["gate_config_hash"]


def test_the_run_writes_its_usage_and_summary_files(ran):
    workdir = ran["workdir"]
    lines = (workdir / run_summary.USAGE_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == ran["summary"]["model_calls"]
    on_disk = json.loads((workdir / run_summary.RUN_SUMMARY_FILE).read_text(encoding="utf-8"))
    assert on_disk == ran["summary"]


def test_the_summary_rebuilt_from_the_workdir_equals_the_runs(ran, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(ran["workdir"].parent / "ledger"))
    assert summarize_workdir(ran["workdir"]) == ran["summary"]


def test_a_look_in_the_ledger_for_this_gate_config_shows_in_the_summary(ran, monkeypatch, tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    source = ran["workdir"].parent / "ledger" / runlog.LEDGER_FILENAME
    look = {"name": gate.LOOK_NAME, "tags": [], "config": ran["summary"]["gate_config_hash"]}
    unnamed = {"kind": "decision", "tags": []}  # an entry without a name must not break it
    lines = [source.read_text(encoding="utf-8"), json.dumps(unnamed) + "\n"]
    (ledger / runlog.LEDGER_FILENAME).write_text("".join(lines), encoding="utf-8")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(ledger))
    assert summarize_workdir(ran["workdir"])["holdout_opened"] is False
    with (ledger / runlog.LEDGER_FILENAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(look) + "\n")
    assert summarize_workdir(ran["workdir"])["holdout_opened"] is True


def test_an_edited_summary_frame_is_refused(ran, monkeypatch, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(ran["workdir"], copy)
    frame = copy / manifest.SUMMARY_FILE
    frame.write_bytes(frame.read_bytes() + b"x")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(ran["workdir"].parent / "ledger"))
    with pytest.raises(ValueError, match="changed since the search"):
        summarize_workdir(copy)


def test_the_decisions_carry_the_agents_reason_and_evidence():
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE moves (id INTEGER PRIMARY KEY, action TEXT, from_id TEXT, reason TEXT)"
    )
    agent = {"by": "agent", "motif": "m", "reason": "why", "evidence_used": "e1"}
    rows = [("deepen", "p1", json.dumps(agent)), ("bond", "p2", "{}")]
    con.executemany("INSERT INTO moves (action, from_id, reason) VALUES (?, ?, ?)", rows)
    assert run_summary._decisions(con) == [
        {"by": "agent", "parent": "p1", "motif": "m", "reason": "why", "evidence_used": "e1"}
    ]


def test_usage_is_written_even_when_the_search_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    config = _config(tmp_path, **PERIOD)
    config = replace(config, papers=str(_failing_papers(tmp_path)))
    with pytest.raises(RuntimeError):
        run_real_search(
            config,
            client=ScriptedClient(),
            editor=ScriptedEditor(*EDITS),
            loader=Loader(fake_rows(n_stocks=12, n_days=150)),
            workdir=tmp_path / "run",
        )
    lines = (tmp_path / "run" / run_summary.USAGE_FILE).read_text(encoding="utf-8").splitlines()
    assert lines and not (tmp_path / "run" / run_summary.RUN_SUMMARY_FILE).exists()


def _failing_papers(tmp_path):
    path = tmp_path / "failing.toml"
    path.write_text(
        '[[papers]]\nkey = "k:0"\ntitle = "alpha"\nabstract = "a"\n\n'
        '[[papers]]\nkey = "k:1"\ntitle = "FAIL"\nabstract = "a"\n',
        encoding="utf-8",
    )
    return path


def test_a_workdir_that_holds_a_search_is_refused_before_any_call(ran):
    client = ScriptedClient()
    with pytest.raises(FileExistsError, match="already holds a search"):
        run_real_search(
            _config(ran["workdir"].parent, **PERIOD), client=client, workdir=ran["workdir"]
        )
    assert client.calls == 0
