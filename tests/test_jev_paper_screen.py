"""Jev paper screen: questions, parsing, metrics, cache resume, call cap, ledger, statements.

Fake HTTP post only; no API calls.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from capstone.analysis.outputs import write_outputs

SCRIPT = Path(__file__).resolve().parents[1] / "research" / "jev_paper_screen.py"
_spec = importlib.util.spec_from_file_location("jev_paper_screen", SCRIPT)
jps = importlib.util.module_from_spec(_spec)
sys.modules["jev_paper_screen"] = jps
_spec.loader.exec_module(jps)

CFG = jps.load_config()
TAGS = list(CFG["taxonomy"])


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("TYPESAFE_API_KEY", "K")
    return tmp_path / "ledger" / "runs.jsonl"


def _questions():
    return jps.build_questions(CFG, jps.load_definitions(TAGS))


def _response(methods=3.0, findings=1.0, tag_p=0.2, top_p=0.1):
    def score(x):
        return {
            "type": "score", "score": x, "confidence": 0.9,
            "probabilities": {"0": 0.2, "3": 0.5, "4": top_p, "5": top_p},
        }  # fmt: skip

    answers = {
        "methods_relevance": score(methods),
        "findings_relevance": score(findings),
    }
    for t in TAGS:
        answers[f"tag_{t}"] = {"type": "noul", "noul": tag_p}
    return {
        "model": jps.MODEL,
        "answers": answers,
        "usage": {"input_tokens": 1000, "output_tokens": 5},
    }


# -- questions -------------------------------------------------------------


def test_questions_come_from_config_rubric_and_taxonomy():
    q = _questions()
    levels = jps.rubric_levels(CFG["rubric"])
    assert len(levels) == 6
    assert q["methods_relevance"]["type"] == "score"
    assert len(q["methods_relevance"]["criteria"]) == 6
    methods5, findings5 = levels[5]
    assert methods5.startswith("LLM/agentic generation") and "FINDINGS" not in methods5
    assert findings5.startswith("documented cross-sectional")
    assert q["methods_relevance"]["criteria"][5] == methods5
    assert q["findings_relevance"]["criteria"][5] == findings5
    assert levels[1] == ("tangential", "tangential")  # unmarked levels are sent verbatim to both
    assert [k for k in q if k.startswith("tag_")] == [f"tag_{t}" for t in TAGS]
    assert all(q[f"tag_{t}"]["type"] == "noul" and q[f"tag_{t}"]["instructions"] for t in TAGS)
    assert CFG["team_description"].split()[0] in q["methods_relevance"]["instructions"]


def test_tag_definitions_must_match_the_taxonomy(tmp_path):
    bad = tmp_path / "t.toml"
    bad.write_text('[tags]\nsurvey = "Is it a survey?"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differ from taxonomy"):
        jps.load_definitions(TAGS, bad)


def test_state_has_only_title_and_abstract_and_marks_a_missing_abstract():
    state = jps.build_state({"title": " T ", "abstract": None, "relevance": 5, "haiku_tags": ["x"]})
    assert state == {"title": "T", "abstract": "(no abstract available)"}


# -- parsing ---------------------------------------------------------------


def test_parse_reads_scores_and_tag_probabilities():
    out = jps.parse_response(_response(methods=3.5, findings=1.0, tag_p=0.3, top_p=0.25), TAGS)
    assert out["methods"] == {"expected": 3.5, "p_high": 0.5}
    assert out["findings"]["expected"] == 1.0
    assert out["tags"] == dict.fromkeys(TAGS, 0.3)


@pytest.mark.parametrize("break_it", ["missing_tag", "bad_noul", "score_range", "no_probs"])
def test_parse_rejects_malformed_answers(break_it):
    r = _response()
    if break_it == "missing_tag":
        del r["answers"][f"tag_{TAGS[0]}"]
    elif break_it == "bad_noul":
        r["answers"][f"tag_{TAGS[0]}"]["noul"] = 1.5
    elif break_it == "score_range":
        r["answers"]["methods_relevance"]["score"] = 7
    else:
        del r["answers"]["findings_relevance"]["probabilities"]
    with pytest.raises(jps.JevParseError):
        jps.parse_response(r, TAGS)


# -- metrics ---------------------------------------------------------------


def test_rank_correlations_on_constructed_cases():
    assert jps.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert jps.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert jps.kendall([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    assert jps.spearman([1, 1, 1], [1, 2, 3]) != jps.spearman([1, 1, 1], [1, 2, 3])  # nan
    assert jps.auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert jps.auroc([0.5, 0.5], [True, False]) == 0.5


def test_tag_metrics_on_a_constructed_case():
    truth = pd.DataFrame({"a": [True, True, False, False], "b": [False, False, False, False]})
    p = pd.DataFrame({"a": [0.9, 0.4, 0.6, 0.1], "b": [0.9, 0.1, 0.1, 0.1]})
    sweep = jps.tag_sweep(p, truth)
    row = sweep[(sweep.tag == "a") & (sweep.threshold == 0.5)].iloc[0]
    assert (row.tp, row.fp, row.fn) == (1, 1, 1)
    assert row.precision == 0.5 and row.recall == 0.5 and row.f1 == 0.5
    summary = jps.tag_summary(sweep, p, truth).set_index("tag")
    assert summary.loc["a", "f1_best"] > summary.loc["a", "f1_0.5"]
    assert summary.loc["a", "best_threshold"] == 0.2  # ties go to the lowest
    assert summary.loc["b", "n_haiku"] == 0 and pd.isna(summary.loc["b", "f1_0.5"])
    assert jps.macro(summary["f1_0.5"]) == 0.5  # b has no Haiku positives and is excluded


def test_rank_of_counts_strictly_higher_and_ties():
    keys = pd.Series(["a", "b", "c", "d"])
    vals = pd.Series([5, 3, 3, 1])
    assert jps.rank_of(vals, "a", keys) == (1, 1)
    assert jps.rank_of(vals, "c", keys) == (2, 2)


# -- sample, run, cache, ledger --------------------------------------------


def _db(tmp_path, per_level=4):
    path = tmp_path / "papers.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE papers (key TEXT, title TEXT, abstract TEXT, published TEXT, status TEXT, "
        "relevance INTEGER, tags TEXT)"
    )
    rows = [("adopt1", "Adopted", "abs", "2025-01-01", "adopted", 5, json.dumps(["survey"]))]
    for level in range(6):
        for i in range(per_level):
            tag = json.dumps([TAGS[(level + i) % len(TAGS)]])
            rows.append(
                (f"s{level}_{i}", f"T{level}{i}", f"A{level}{i}", "2025", "screened", level, tag)
            )
    rows.append(("f1", "Filtered", "x", "2025", "filtered", None, None))
    con.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def test_sample_takes_all_adopted_and_n_per_level_seeded(tmp_path):
    papers = jps.load_papers(_db(tmp_path))
    assert all(p["relevance"] is not None for p in papers)
    a = jps.sample_papers(papers, 2, seed=0)
    assert a == jps.sample_papers(papers, 2, seed=0)
    assert a != jps.sample_papers(papers, 2, seed=1)
    assert sum(p["status"] == "adopted" for p in a) == 1
    assert sorted(p["relevance"] for p in a if p["status"] == "screened") == [
        lvl for lvl in range(6) for _ in range(2)
    ]


def test_database_is_opened_read_only(tmp_path):
    db = _db(tmp_path)
    before = db.read_bytes()
    jps.load_papers(db)
    assert db.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError):
        con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
        con.execute("DELETE FROM papers")


def _varied_post(calls):
    def post(url, headers, body):
        calls.append(body)
        n = len(calls)
        return _response(methods=n % 6, findings=(n * 2) % 6, tag_p=0.7 if n % 2 else 0.1)

    return post


def test_run_logs_before_the_first_call_and_sends_only_the_pinned_model(tmp_path, ledger):
    papers = jps.sample_papers(jps.load_papers(_db(tmp_path)), 1, 0)
    seen = []

    def post(url, headers, body):
        seen.append((ledger.exists() and len(ledger.read_text().splitlines()), body))
        return _response()

    jps.run(papers, _questions(), TAGS, tmp_path / "out", "K", 50, post=post, sleep=lambda s: None)
    assert all(n == 1 for n, _ in seen)  # the one ledger entry exists before every call
    assert len(ledger.read_text().splitlines()) == 1
    body = seen[0][1]
    assert body["model"] == "jev-1.14.0" and set(body["state"]) == {"title", "abstract"}


def test_cache_resumes_without_new_calls_and_refuses_over_the_cap(tmp_path):
    papers = jps.sample_papers(jps.load_papers(_db(tmp_path)), 2, 0)
    out, q = tmp_path / "out", _questions()
    calls: list = []
    with pytest.raises(SystemExit, match="refusing"):
        jps.run(papers, q, TAGS, out, "K", len(papers) - 1, post=_varied_post(calls))
    assert calls == []
    first = jps.run(papers, q, TAGS, out, "K", 100, post=_varied_post(calls))
    assert len(calls) == len(papers) == len(first)
    again = jps.run(
        papers, q, TAGS, out, "K", 0, post=_varied_post(calls)
    )  # cap 0: nothing pending
    assert len(calls) == len(papers) and [r["key"] for r in again] == [r["key"] for r in first]


def test_changed_questions_invalidate_the_cache(tmp_path):
    papers = jps.sample_papers(jps.load_papers(_db(tmp_path)), 1, 0)
    out, calls = tmp_path / "out", []
    jps.run(papers, _questions(), TAGS, out, "K", 100, post=_varied_post(calls))
    changed = _questions()
    changed["tag_survey"]["instructions"] = "Is it a survey, honestly?"
    with pytest.raises(SystemExit, match="refusing"):
        jps.run(papers, changed, TAGS, out, "K", 0, post=_varied_post(calls))


def test_retryable_status_is_retried_and_others_raise(tmp_path):
    papers = jps.sample_papers(jps.load_papers(_db(tmp_path)), 1, 0)[:1]
    attempts, naps = [], []

    def flaky(url, headers, body):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 529 if len(attempts) == 1 else 429, "x", {}, None)
        return _response()

    jps.run(papers, _questions(), TAGS, tmp_path / "o1", "K", 5, post=flaky, sleep=naps.append)
    assert len(attempts) == 3 and naps == [1.0, 2.0]

    def denied(url, headers, body):
        raise urllib.error.HTTPError(url, 401, "no", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        jps.run(papers, _questions(), TAGS, tmp_path / "o2", "K", 5, post=denied, sleep=naps.append)


def test_parse_errors_are_not_cached_and_are_retried(tmp_path):
    papers = jps.sample_papers(jps.load_papers(_db(tmp_path)), 1, 0)[:1]
    out = tmp_path / "out"
    rows = jps.run(papers, _questions(), TAGS, out, "K", 5, post=lambda *a: {"answers": {}})
    assert rows == []
    good = jps.run(papers, _questions(), TAGS, out, "K", 5, post=lambda *a: _response())
    assert len(good) == 1


# -- outputs ---------------------------------------------------------------


def test_main_writes_sourced_statements_and_the_disagreement_sheet(tmp_path, ledger):
    db = _db(tmp_path, per_level=6)
    out = tmp_path / "out"
    calls: list = []
    path = jps.main(
        ["--db", str(db), "--n-per-level", "6", "--out", str(out)],
        post=_varied_post(calls),
        sleep=lambda s: None,
    )
    assert path == out and len(calls) == 37
    stats = json.loads((out / "stats.json").read_text())
    text = (out / "STATEMENTS.md").read_text()
    for source in set(__import__("re").findall(r"`stats.json: ([^`]+)`", text)):
        assert source in stats
    for key in ("relevance.spearman", "tags.macro_f1_0.5", "adopted.ranks", "cost.usd"):
        assert key in stats
    assert "not ground truth" in text and "drafts" in text
    sheet = pd.read_csv(out / "disagreements.csv", encoding="utf-8-sig")
    assert len(sheet) == jps.DISAGREEMENTS
    assert sheet.human_relevance.isna().all()
    assert sheet.jev_minus_haiku.abs().is_monotonic_decreasing
    assert len(ledger.read_text().splitlines()) == 1


def test_unsourced_statement_is_refused_by_write_outputs(tmp_path):
    from capstone.analysis.outputs import Statement

    with pytest.raises(ValueError, match="not in stats"):
        write_outputs(tmp_path / "x", title="t", stats={}, statements=[Statement("s", "nope")])
