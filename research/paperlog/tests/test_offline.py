"""Offline tests: no network, fake LLM client. Run with: python -m pytest -q"""

import json
import types
from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path

import yaml
from paperlog import api, cli, collect, dedupe, prefilter
from paperlog import db as dbm
from paperlog.models import Paper
from paperlog.screen import save
from paperlog.sources import openalex

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
CFG = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_ids_and_keys():
    p = Paper(title="X", doi="https://doi.org/10.48550/arXiv.2412.20138")
    assert p.arxiv_id == "2412.20138" and p.key == "arxiv:2412.20138"
    q = Paper(title="X", arxiv_id="http://arxiv.org/abs/2412.20138v3")
    assert q.key == p.key


def test_batch_dedupe_merges_sources_and_countries():
    a = Paper(
        title="TradingAgents: Multi-Agents LLM Financial Trading Framework",
        arxiv_id="2412.20138v1",
        sources=["arxiv"],
    )
    b = Paper(
        title="TradingAgents - multi-agents LLM financial trading framework.",
        abstract="longer abstract text",
        countries=["us"],
        sources=["openalex"],
    )
    c = Paper(title="Something else entirely", sources=["s2"])
    out = dedupe.dedupe_batch([a, b, c])
    assert len(out) == 2
    m = [p for p in out if p.arxiv_id][0]
    assert m.sources == ["arxiv", "openalex"] and m.countries == ["US"]


def test_openalex_parsing_chinese_record():
    w = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/abc",
        "display_name": "基于多智能体的股票研究框架",
        "publication_date": "2026-08-01",
        "abstract_inverted_index": {"本文": [0], "提出": [1]},
        "authorships": [
            {
                "author": {"display_name": "张三"},
                "institutions": [{"display_name": "Tsinghua University", "country_code": "CN"}],
                "countries": ["CN"],
            }
        ],
        "language": "zh",
        "primary_location": {
            "landing_page_url": "https://x.cn/1",
            "source": {"display_name": "金融研究"},
        },
    }
    p = openalex.to_paper(w, "openalex-zh:test")
    assert p.abstract == "本文 提出" and p.countries == ["CN"] and p.venue == "金融研究"
    assert prefilter.passes(p.title, p.abstract, CFG["prefilter"])


def test_prefilter_rejects_unrelated():
    assert not prefilter.passes("Protein folding with diffusion", "biology", CFG["prefilter"])


class FakeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kw):
        paper = json.loads(kw["messages"][0]["content"])
        rel = 5 if "Trading" in paper["title"] or "股票" in paper["title"] else 1
        block = types.SimpleNamespace(
            type="tool_use",
            input={
                "relevance": rel,
                "tags": ["multi_agent_orchestration"],
                "title_en": "Multi-agent stock research framework"
                if "股票" in paper["title"]
                else paper["title"],
                "summary_en": "A test summary.",
                "lookahead_bias": "not_addressed",
                "markets": ["US equities"],
            },
        )
        return types.SimpleNamespace(
            content=[block], usage=types.SimpleNamespace(input_tokens=900, output_tokens=200)
        )


def test_full_run_with_mocked_sources(tmp_path, monkeypatch):
    fake_papers = [
        Paper(
            title="TradingAgents: Multi-Agents LLM Financial Trading Framework",
            abstract="LLM agents for stock trading",
            arxiv_id="2412.20138",
            published="2026-09-01",
            url="https://arxiv.org/abs/2412.20138",
            sources=["arxiv"],
            language="en",
        ),
        Paper(
            title="基于多智能体的股票研究框架",
            abstract="大模型 智能体 股票",
            doi="10.1/abc",
            published="2026-08-20",
            countries=["CN"],
            language="zh",
            sources=["openalex"],
        ),
        Paper(
            title="Protein folding with diffusion",
            abstract="biology",
            doi="10.9/zzz",
            sources=["openalex"],
        ),
    ]
    monkeypatch.setattr(cli.arxiv, "fetch", lambda *a, **k: fake_papers[:1])
    monkeypatch.setattr(cli.openalex, "fetch", lambda *a, **k: fake_papers[1:])
    monkeypatch.setattr(cli.s2, "fetch_citations", lambda *a, **k: [])
    monkeypatch.setattr(cli.s2, "fetch_references", lambda *a, **k: [])
    monkeypatch.setattr(cli, "anchors_for_run", lambda *a, **k: ([], []))
    import paperlog.screen as sc

    orig_init = sc.Screener.__init__
    monkeypatch.setattr(
        sc.Screener, "__init__", lambda self, cfg, client=None: orig_init(self, cfg, FakeClient())
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    dbp, dd, csvp = str(tmp_path / "p.db"), str(tmp_path / "d"), str(tmp_path / "p.csv")
    args = Namespace(
        db=dbp, since="2026-01-01", days=None, target=None, no_screen=False, digest_dir=dd, csv=csvp
    )
    cli.cmd_run(args, CFG)

    con = dbm.connect(dbp)
    rows = {r["key"]: r for r in con.execute("SELECT * FROM papers")}
    assert rows["doi:10.9/zzz"]["status"] == "filtered"
    assert rows["arxiv:2412.20138"]["relevance"] == 5
    assert rows["doi:10.1/abc"]["title_en"] == "Multi-agent stock research framework"

    import glob

    md = open(glob.glob(dd + "/*.md")[0], encoding="utf-8").read()
    assert "original in zh" in md and "look-ahead bias not addressed" in md
    print(md)

    # second run: same papers again -> nothing new, sightings merged, digest kept
    cli.cmd_run(args, CFG)
    assert con.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3
    assert "original in zh" in open(glob.glob(dd + "/*.md")[0], encoding="utf-8").read()

    cli.set_review(con, "doi:10.1/abc", "adopted", 5, None)
    con.commit()
    cli.cmd_stats(Namespace(db=dbp), CFG)


def test_api_reads_and_marks(tmp_path, monkeypatch):
    from paperlog import api

    dbp = str(tmp_path / "a.db")
    con = dbm.connect(dbp)
    for i, (rel, tags) in enumerate(
        [(5, ["valuation"]), (2, ["survey"]), (4, ["tool_use", "valuation"])]
    ):
        k = dbm.insert(con, Paper(title=f"Paper {i}", doi=f"10.1/{i}", published="2026-09-01"), 1)
        save(
            con,
            k,
            {
                "relevance": rel,
                "tags": tags,
                "title_en": f"Paper {i}",
                "summary_en": "s",
                "lookahead_bias": "unclear",
            },
            "m",
        )
    con.commit()
    monkeypatch.setenv("PAPERLOG_DB", dbp)
    assert [p["key"] for p in api.papers(min_relevance=4)] == ["doi:10.1/0", "doi:10.1/2"]
    assert len(api.papers(tags=["valuation"])) == 2
    api.mark("doi:10.1/2", status="adopted", notes="experiment exp-014")
    assert [p["key"] for p in api.adopted()] == ["doi:10.1/2"]
    assert api.get("doi:10.1/2")["human_notes"] == "experiment exp-014"


def test_config_found_from_elsewhere(tmp_path, monkeypatch):
    from paperlog.paths import config_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PAPERLOG_CONFIG", raising=False)
    assert yaml.safe_load(open(config_path(), encoding="utf-8"))["screening"]["model"]


def test_bad_key_stops_after_first_failure(tmp_path, monkeypatch, capsys):
    import anthropic
    import httpx2 as httpx
    import paperlog.screen as sc

    class BadKeyClient:
        calls = 0

        def __init__(self):
            self.messages = self

        def create(self, **kw):
            BadKeyClient.calls += 1
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.AuthenticationError(
                "invalid x-api-key", response=httpx.Response(401, request=req), body=None
            )

    papers = [
        Paper(title=f"LLM agent for stock trading {i}", abstract="agent stock", doi=f"10.2/{i}")
        for i in range(10)
    ]
    monkeypatch.setattr(cli.arxiv, "fetch", lambda *a, **k: papers)
    monkeypatch.setattr(cli.openalex, "fetch", lambda *a, **k: [])
    monkeypatch.setattr(cli.s2, "fetch_citations", lambda *a, **k: [])
    monkeypatch.setattr(cli.s2, "fetch_references", lambda *a, **k: [])
    monkeypatch.setattr(cli, "anchors_for_run", lambda *a, **k: ([], []))
    orig_init = sc.Screener.__init__
    monkeypatch.setattr(
        sc.Screener, "__init__", lambda self, cfg, client=None: orig_init(self, cfg, BadKeyClient())
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    args = Namespace(
        db=str(tmp_path / "b.db"),
        since="2026-01-01",
        days=None,
        target=None,
        no_screen=False,
        digest_dir=str(tmp_path / "d"),
        csv="",
    )
    cli.cmd_run(args, CFG)
    assert BadKeyClient.calls == 1
    assert "rejected the API key" in capsys.readouterr().out


def test_ladder_widens_then_stops():
    plans = collect.ladder(CFG["collection"], "2026-09-07")
    assert plans[0].level == 0 and plans[0].since == "2026-09-07"
    assert plans[1].since < plans[0].since  # window doubles backwards
    assert any(p.depth > 1 for p in plans)  # then goes deeper
    assert plans[-1].exploration and plans[-1].loosen_prefilter
    floor = min(p.since for p in plans)
    assert (
        floor
        >= (date.today() - timedelta(days=CFG["collection"]["max_lookback_days"] + 1)).isoformat()
    )


def _mock_sources(monkeypatch, per_level):
    """arXiv returns a different slice of papers at each level."""
    state = {"calls": 0}

    def fake_arxiv(cfg, since, depth=1, extra=None, log=print):
        i = state["calls"]
        state["calls"] += 1
        n = per_level[i] if i < len(per_level) else 0
        # each level also re-returns everything from earlier levels (as the real APIs do)
        total = sum(per_level[: i + 1])
        return (
            [
                Paper(
                    title=f"LLM agent stock trading study {j}",
                    abstract="llm agent stock market",
                    arxiv_id=f"2609.{10000 + j}",
                    published="2026-09-10",
                    sources=["arxiv"],
                )
                for j in range(total)
            ]
            if n or total
            else []
        )

    monkeypatch.setattr(cli.arxiv, "fetch", fake_arxiv)
    monkeypatch.setattr(cli.openalex, "fetch", lambda *a, **k: [])
    monkeypatch.setattr(cli.s2, "fetch_citations", lambda *a, **k: [])
    monkeypatch.setattr(cli.s2, "fetch_references", lambda *a, **k: [])
    monkeypatch.setattr(cli, "anchors_for_run", lambda *a, **k: ([], []))
    return state


def test_escalates_until_target_met(tmp_path, monkeypatch, capsys):
    state = _mock_sources(monkeypatch, [3, 6, 20])
    args = Namespace(
        db=str(tmp_path / "e.db"),
        since="2026-09-01",
        days=None,
        target=10,
        no_screen=True,
        digest_dir=str(tmp_path / "d"),
        csv="",
    )
    cli.cmd_run(args, CFG)
    assert state["calls"] == 3  # stopped as soon as the target was met
    con = dbm.connect(args.db)
    assert con.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 29
    assert "Level 2" in capsys.readouterr().out


def test_no_escalation_when_first_level_is_enough(tmp_path, monkeypatch):
    state = _mock_sources(monkeypatch, [50])
    args = Namespace(
        db=str(tmp_path / "f.db"),
        since="2026-09-01",
        days=None,
        target=10,
        no_screen=True,
        digest_dir=str(tmp_path / "d"),
        csv="",
    )
    cli.cmd_run(args, CFG)
    assert state["calls"] == 1


def test_escalation_stops_on_diminishing_returns(tmp_path, monkeypatch, capsys):
    state = _mock_sources(monkeypatch, [2, 1])  # level 1 adds 1 new paper, below min_yield 5
    args = Namespace(
        db=str(tmp_path / "g.db"),
        since="2026-09-01",
        days=None,
        target=100,
        no_screen=True,
        digest_dir=str(tmp_path / "d"),
        csv="",
    )
    cli.cmd_run(args, CFG)
    assert state["calls"] == 2
    assert "stopping escalation" in capsys.readouterr().out


def test_rescreen_records_history_and_movement(tmp_path, monkeypatch, capsys):
    import paperlog.screen as sc

    dbp = str(tmp_path / "h.db")
    con = dbm.connect(dbp)
    k = dbm.insert(
        con, Paper(title="Agentic equity research", doi="10.5/a", abstract="llm agent stock"), 1
    )
    save(
        con,
        k,
        {
            "relevance": 2,
            "tags": [],
            "title_en": "t",
            "summary_en": "s",
            "lookahead_bias": "unclear",
        },
        "m",
        1,
    )
    con.commit()

    class Rescorer:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            block = types.SimpleNamespace(
                type="tool_use",
                input={
                    "relevance": 5,
                    "tags": ["valuation"],
                    "title_en": "t",
                    "summary_en": "rescored under v2",
                    "lookahead_bias": "controlled",
                },
            )
            return types.SimpleNamespace(
                content=[block], usage=types.SimpleNamespace(input_tokens=10, output_tokens=5)
            )

    orig = sc.Screener.__init__
    monkeypatch.setattr(
        sc.Screener, "__init__", lambda self, cfg, client=None: orig(self, cfg, Rescorer())
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["screening"]["rubric_version"] = 2
    cli.cmd_rescreen(Namespace(db=dbp, all=False, filtered=False, limit=10), cfg)

    assert "2 -> 5" in capsys.readouterr().out
    monkeypatch.setenv("PAPERLOG_DB", dbp)
    hist = api.history(k)
    assert [h["relevance"] for h in hist] == [2, 5]
    assert [h["rubric_version"] for h in hist] == [1, 2]


def test_pending_and_ingest_claim(tmp_path, monkeypatch):
    dbp = str(tmp_path / "i.db")
    con = dbm.connect(dbp)
    for i in range(3):
        k = dbm.insert(con, Paper(title=f"Paper {i}", doi=f"10.6/{i}"), 1)
        save(
            con,
            k,
            {
                "relevance": 5 - i,
                "tags": [],
                "title_en": f"Paper {i}",
                "summary_en": "s",
                "lookahead_bias": "unclear",
            },
            "m",
            1,
        )
    con.commit()
    monkeypatch.setenv("PAPERLOG_DB", dbp)
    first = api.pending(min_relevance=4)
    assert [p["key"] for p in first] == ["doi:10.6/0", "doi:10.6/1"]
    api.mark_ingested([p["key"] for p in first])
    assert api.pending(min_relevance=4) == []  # idempotent: not handed out twice
    assert len(api.pending(min_relevance=0)) == 1


def test_add_paper_by_arxiv_url(tmp_path, monkeypatch, capsys):
    import paperlog.screen as sc

    monkeypatch.setattr(
        cli.resolve,
        "resolve",
        lambda ident: (
            Paper(
                title="TradingAgents",
                abstract="llm agent stock trading",
                arxiv_id="2412.20138",
                s2_id="abc",
                sources=["s2"],
            ),
            "arxiv id",
        ),
    )
    orig = sc.Screener.__init__
    monkeypatch.setattr(
        sc.Screener, "__init__", lambda self, cfg, client=None: orig(self, cfg, FakeClient())
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    dbp = str(tmp_path / "j.db")
    args = Namespace(
        db=dbp,
        identifier="https://arxiv.org/abs/2412.20138",
        adopt=True,
        anchor=False,
        rating=5,
        notes="found on X",
        no_screen=False,
    )
    key = cli.cmd_add(args, CFG)
    assert key == "arxiv:2412.20138"
    con = dbm.connect(dbp)
    row = con.execute("SELECT * FROM papers WHERE key=?", (key,)).fetchone()
    assert row["status"] == "adopted" and row["human_rating"] == 5 and row["relevance"] == 5
    # adopted papers become citation anchors for future runs
    assert con.execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 1
    # adding it again doesn't duplicate
    cli.cmd_add(Namespace(**{**vars(args), "no_screen": True, "adopt": False, "rating": None}), CFG)
    assert con.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert "Already in the log" in capsys.readouterr().out


def test_filtered_papers_revisited_when_prefilter_loosens(tmp_path):
    dbp = str(tmp_path / "k.db")
    con = dbm.connect(dbp)
    k = dbm.insert(con, Paper(title="Multi-agent planning with memory", abstract="agents only"), 1)
    # The strict rule (a method term AND a finance term), whatever the shipped config uses.
    cfg = {**yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["prefilter"], "mode": "both"}
    assert cli.classify(con, [k], cfg, loosen=False) == []  # no finance term -> filtered
    assert con.execute("SELECT status FROM papers WHERE key=?", (k,)).fetchone()[0] == "filtered"
    revived = cli.revisit_filtered(con, cfg, loosen=True)  # exploration level
    assert revived == [k]
    assert con.execute("SELECT status FROM papers WHERE key=?", (k,)).fetchone()[0] == "new"


def test_screening_prompt_describes_this_team():
    from paperlog.screen import Screener

    s = Screener(CFG["screening"], client=FakeClient())
    assert "agentic alpha-discovery system" in s.system
    assert "equity research pipeline" not in s.system
    generic = Screener({**CFG["screening"], "team_description": ""}, client=FakeClient())
    assert "a quantitative research team." in generic.system


def test_add_resolves_an_openalex_work_id(monkeypatch):
    from paperlog import resolve

    work = {
        "id": "https://openalex.org/W42",
        "display_name": "Alpha mining with LLMs",
        "publication_date": "2026-09-01",
        "language": "en",
    }
    seen = {}

    def fake_get(url, params=None, headers=None):
        seen["url"] = url
        return types.SimpleNamespace(json=lambda: work)

    monkeypatch.setattr(openalex, "get", fake_get)
    paper, how = resolve.resolve("https://openalex.org/W42")
    assert how == "openalex id" and paper.title == "Alpha mining with LLMs"
    assert seen["url"].endswith("/works/W42")


def test_arxiv_id_falls_back_to_arxiv_when_semantic_scholar_fails(monkeypatch):
    from paperlog import resolve
    from paperlog.sources import arxiv

    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2308.00016v1</id><published>2023-07-31T00:00:00Z</published>
<title>Alpha-GPT: Human-AI Interactive Alpha Mining</title><summary>LLM alpha mining.</summary>
<author><name>A. Author</name></author></entry></feed>"""
    monkeypatch.setattr(resolve.s2, "by_id", lambda pid: None)  # e.g. rate-limited (429)
    monkeypatch.setattr(arxiv, "get", lambda url, params=None: types.SimpleNamespace(text=feed))
    paper, how = resolve.resolve("2308.00016")
    assert how == "arxiv id (via arXiv)"
    assert paper.title.startswith("Alpha-GPT") and paper.published == "2023-07-31"
    assert paper.key == "arxiv:2308.00016"



def test_shipped_prefilter_keeps_classic_finance_papers():
    cfg = CFG["prefilter"]
    assert prefilter.passes(
        "Returns to Buying Winners and Selling Losers",
        "Strategies which buy past winners and sell past losers earn positive returns.",
        cfg,
    )
    assert prefilter.passes(
        "Common risk factors in the returns on stocks and bonds",
        "Three stock-market factors: market, size and book-to-market equity.",
        cfg,
    )
    assert not prefilter.passes("Protein folding with diffusion", "biology", cfg)
