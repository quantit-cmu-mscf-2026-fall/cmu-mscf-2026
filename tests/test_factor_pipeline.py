"""Sources, the run loop and the command line, with a fake model and a fake paperlog."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from capstone.factors import __main__ as cli
from capstone.factors import store
from capstone.factors.llm import FactorConfig
from capstone.factors.pipeline import run
from capstone.factors.sources import FileSource, PaperlogSource

HYPOTHESIS = {
    "observation": "Stocks that fell over a week tend to rebound.",
    "knowledge": "Short-term reversal.",
    "justification": "Liquidity providers are paid to absorb order imbalances.",
    "specification": "Minus the 5-day return, scaled by volume trend.",
    "falsification_condition": "No negative relation between past 5-day and next-day returns.",
    "kind": "market",
}
FACTOR = {
    "expression": "-(close / shift(close, 5) - 1) * ts_rank(volume, 20)",
    "rationale": "reversal, stronger on heavy volume",
}
CONFIG = FactorConfig(model="test-model", hypotheses_per_paper=1, factors_per_hypothesis=1)


class ScriptedClient:
    """Answers by tool name; a paper whose title contains "FAIL" makes extraction raise."""

    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        tool = kwargs["tools"][0]["name"]
        if tool == "record_hypotheses":
            if "FAIL" in kwargs["messages"][0]["content"]:
                raise RuntimeError("model unavailable")
            answer = {"hypotheses": [HYPOTHESIS]}
        else:
            answer = {"factors": [FACTOR]}
        block = SimpleNamespace(type="tool_use", name=tool, input=answer)
        return SimpleNamespace(content=[block])


class ListSource:
    def __init__(self, papers):
        self._papers = papers
        self.marked: list[str] = []

    def papers(self):
        return self._papers

    def done(self, keys):
        self.marked.extend(keys)


def test_file_source_reads_papers_and_requires_key_and_title(tmp_path):
    good = tmp_path / "papers.toml"
    good.write_text('[[papers]]\nkey = "arxiv:1"\ntitle = "T"\nabstract = "A"\n')
    assert FileSource(good).papers() == [{"key": "arxiv:1", "title": "T", "abstract": "A"}]

    bad = tmp_path / "bad.toml"
    bad.write_text('[[papers]]\ntitle = "No key"\n')
    with pytest.raises(ValueError, match="key"):
        FileSource(bad).papers()


def test_example_papers_file_is_valid():
    papers = FileSource("config/papers.example.toml").papers()
    assert {p["key"] for p in papers} == {"arxiv:2502.16789", "arxiv:2505.11122"}
    assert all(len(p["abstract"]) > 500 for p in papers)


def test_run_stores_the_chain_and_marks_only_finished_papers(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    source = ListSource(
        [
            {"key": "arxiv:ok", "title": "Works", "authors": ["A", "B"], "published": "2025-01-01"},
            {"key": "arxiv:bad", "title": "FAIL here"},
        ]
    )
    summary = run(con, source, ScriptedClient(), CONFIG)

    assert source.marked == ["arxiv:ok"]
    assert summary.papers_done == ["arxiv:ok"]
    assert "model unavailable" in summary.papers_failed["arxiv:bad"]
    assert summary.outcomes["stored"] == 1

    (factor,) = store.factors(con)
    (edge,) = store.lineage(con, factor["id"])
    assert edge["paper_key"] == "arxiv:ok"
    citation = con.execute("SELECT citation FROM papers WHERE key = 'arxiv:ok'").fetchone()[0]
    assert citation == "A et al., 2025"


def test_paperlog_source_claims_papers_through_its_api(monkeypatch):
    calls = {}

    def pending(min_relevance, limit, db_path):
        calls["pending"] = (min_relevance, limit, db_path)
        return [{"key": "arxiv:9", "title": "T"}]

    def mark_ingested(keys, db_path):
        calls["ingested"] = (keys, db_path)
        return len(keys)

    api = ModuleType("paperlog.api")
    api.pending, api.mark_ingested = pending, mark_ingested
    package = ModuleType("paperlog")
    package.api = api
    monkeypatch.setitem(sys.modules, "paperlog", package)
    monkeypatch.setitem(sys.modules, "paperlog.api", api)

    source = PaperlogSource(db_path="p.db", min_relevance=3, limit=5)
    assert source.papers() == [{"key": "arxiv:9", "title": "T"}]
    source.done([])
    assert "ingested" not in calls
    source.done(["arxiv:9"])
    assert calls == {"pending": (3, 5, "p.db"), "ingested": (["arxiv:9"], "p.db")}


def test_paperlog_source_says_what_to_do_when_paperlog_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "paperlog", None)
    with pytest.raises(ImportError, match="file source"):
        PaperlogSource()


def test_cli_run_list_lineage_and_stats(tmp_path, monkeypatch, capsys):
    papers = tmp_path / "papers.toml"
    papers.write_text('[[papers]]\nkey = "arxiv:1"\ntitle = "Reversal"\nabstract = "A"\n')
    config = tmp_path / "factors.toml"
    config.write_text(
        'model = "test-model"\nhypotheses_per_paper = 1\nfactors_per_hypothesis = 1\n'
    )
    db = str(tmp_path / "factors.db")
    monkeypatch.setattr(cli, "make_client", ScriptedClient)

    assert (
        cli.main(["--db", db, "run", "--source", "file", str(papers), "--config", str(config)]) == 0
    )
    assert "1 stored" in capsys.readouterr().out

    assert cli.main(["--db", db, "list"]) == 0
    fid = capsys.readouterr().out.split()[0]

    assert cli.main(["--db", db, "lineage", fid]) == 0
    out = capsys.readouterr().out
    assert "Reversal (arxiv:1)" in out and "reversal, stronger on heavy volume" in out

    assert cli.main(["--db", db, "stats"]) == 0
    assert "factors 1" in capsys.readouterr().out
    assert cli.main(["--db", db, "lineage", "0000000000000000"]) == 1


def _fake_paperlog(monkeypatch, rows):
    api = ModuleType("paperlog.api")
    api.pending = lambda **kw: pytest.fail("a keyed source must not read the queue")
    api.get = lambda key, db_path=None: rows.get(key)
    api.mark_ingested = lambda keys, db_path=None: len(keys)
    package = ModuleType("paperlog")
    package.api = api
    monkeypatch.setitem(sys.modules, "paperlog", package)
    monkeypatch.setitem(sys.modules, "paperlog.api", api)


def test_a_keyed_paperlog_source_takes_exactly_that_paper(monkeypatch):
    _fake_paperlog(monkeypatch, {"arxiv:1": {"key": "arxiv:1", "title": "One"}})
    assert PaperlogSource(keys=["arxiv:1"]).papers() == [{"key": "arxiv:1", "title": "One"}]
    with pytest.raises(KeyError, match="paperlog add"):
        PaperlogSource(keys=["arxiv:missing"]).papers()


def test_a_keyed_file_source_picks_one_paper():
    (paper,) = FileSource("config/papers.example.toml", keys=["arxiv:2505.11122"]).papers()
    assert paper["title"].startswith("Navigating the Alpha Jungle")
    with pytest.raises(KeyError):
        FileSource("config/papers.example.toml", keys=["arxiv:0000.00000"]).papers()


def test_method_hypotheses_are_stored_but_never_become_factors(tmp_path):
    class MethodOnly(ScriptedClient):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            if kwargs["tools"][0]["name"] == "record_hypotheses":
                response.content[0].input = {"hypotheses": [{**HYPOTHESIS, "kind": "method"}]}
            return response

    con = store.connect(tmp_path / "factors.db")
    summary = run(con, ListSource([{"key": "arxiv:m", "title": "M"}]), MethodOnly(), CONFIG)

    assert summary.hypotheses == {"method": 1}
    assert store.proposals(con) == []
    assert [h["kind"] for h in store.hypotheses(con)] == ["method"]


def test_usage_is_recorded_per_paper_even_when_the_paper_fails(tmp_path):
    class Metered(ScriptedClient):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.usage = SimpleNamespace(input_tokens=1000, output_tokens=200)
            return response

    class FailsOnFactors(Metered):
        def create(self, **kwargs):
            if kwargs["tools"][0]["name"] == "record_factors":
                raise RuntimeError("overloaded")
            return super().create(**kwargs)

    con = store.connect(tmp_path / "factors.db")
    summary = run(con, ListSource([{"key": "arxiv:ok", "title": "A"}]), Metered(), CONFIG)
    run(con, ListSource([{"key": "arxiv:bad", "title": "B"}]), FailsOnFactors(), CONFIG)

    assert summary.usage == {"calls": 2, "input_tokens": 2000, "output_tokens": 400}
    rows = {r["paper_key"]: tuple(r)[3:6] for r in con.execute("SELECT * FROM usage")}
    assert rows == {"arxiv:ok": (2, 2000, 400), "arxiv:bad": (1, 1000, 200)}


def test_a_store_from_before_kinds_gains_the_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE papers (key TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT, "
        "citation TEXT, source TEXT, added_at TEXT NOT NULL);"
        "CREATE TABLE hypotheses (id TEXT PRIMARY KEY, paper_key TEXT NOT NULL, "
        "observation TEXT NOT NULL, knowledge TEXT NOT NULL, justification TEXT NOT NULL, "
        "specification TEXT NOT NULL, falsification_condition TEXT NOT NULL, model TEXT, "
        "prompt_version TEXT, created_at TEXT NOT NULL);"
        "INSERT INTO papers VALUES ('p', 'P', '', '', '', 'now');"
        "INSERT INTO hypotheses VALUES ('h', 'p', 'o', 'k', 'j', 's', 'f', 'm', 'v1', 'now');"
    )
    old.commit()
    old.close()

    con = store.connect(path)
    assert [h["kind"] for h in store.hypotheses(con)] == ["market"]


def test_methods_prints_method_hypotheses_as_markdown(tmp_path, capsys):
    db = str(tmp_path / "factors.db")
    con = store.connect(db)
    store.add_paper(
        con, "arxiv:m", "Methods Paper", url="https://arxiv.org/abs/m", citation="X, 2025"
    )
    store.add_hypothesis(
        con, store.Hypothesis(paper_key="arxiv:m", **{**HYPOTHESIS, "kind": "method"})
    )
    store.add_hypothesis(con, store.Hypothesis(paper_key="arxiv:m", **HYPOTHESIS))
    con.close()

    assert cli.main(["--db", db, "methods"]) == 0
    out = capsys.readouterr().out
    assert "## Methods Paper (X, 2025)" in out
    assert out.count("- **Claim:**") == 1
    assert cli.main(["--db", db, "methods", "--paper", "arxiv:other"]) == 1
