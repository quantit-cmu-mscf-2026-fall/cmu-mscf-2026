"""Papers side by side (`capstone.factors.paper_pool`).

A thread-safe fake model with seeded random delays answers from the request alone,
so any difference between concurrency 1 and N would come from the pool. No model,
no network, no market data.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from capstone.factors import paper_pool, store
from capstone.factors.llm import FactorConfig
from capstone.factors.sources import FileSource

PAPERS = 6
SHARED = "rank(volume) - rank(ts_mean(volume, 10))"  # proposed by every paper: duplicates
CONFIG = FactorConfig(alignment_model="fake-judge", hypotheses_per_paper=2)


def _hypothesis(key: str, n: int, kind: str = "market") -> dict:
    return {
        "observation": f"{key}: observation {n}",
        "knowledge": "k",
        "justification": "j",
        "specification": f"spec {key} {n}",
        "falsification_condition": "f",
        "kind": kind,
    }


def _response(tool: str, answer: dict, tokens: int) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name=tool, input=answer)
    return SimpleNamespace(
        content=[block], usage=SimpleNamespace(input_tokens=tokens, output_tokens=1)
    )


class DelayedFake:
    """Answers depend only on the request; each call sleeps a seeded random delay."""

    def __init__(self, seed: int = 0, max_delay: float = 0.02):
        self.seed, self.max_delay = seed, max_delay
        self.messages = self
        self.lock = threading.Lock()
        self.sent: list[dict] = []

    def create(self, **kwargs):
        text = json.dumps(kwargs, sort_keys=True, default=str)
        digest = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        time.sleep(random.Random(self.seed + digest).random() * self.max_delay)
        with self.lock:
            self.sent.append(kwargs)
        tool = kwargs["tools"][0]["name"]
        user = kwargs["messages"][0]["content"]
        if tool == "record_hypotheses":
            key = json.loads(user)["title"]
            if "broken" in key:
                raise RuntimeError("the fake model is down for this paper")
            hyps = [_hypothesis(key, 0), _hypothesis(key, 1, "method" if "m" in key else "market")]
            return _response(tool, {"hypotheses": hyps}, len(user))
        if tool == "record_factors":
            spec = user.split("specification: ")[1].split("\n")[0]
            tag = int(hashlib.sha256(spec.encode()).hexdigest()[:4], 16) % 7 + 2
            if "did not parse" in user:  # the repair round
                exprs = [f"-ts_sum(returns, {tag})"]
            else:
                exprs = [SHARED, f"ts_mean(close, {tag}) / close", "rank(((("]
                if "overused" in kwargs["system"]:
                    exprs[1] = f"-delta(close, {tag}) / close"
            factors = [{"expression": e, "rationale": "fake"} for e in exprs]
            return _response(tool, {"factors": factors}, len(user))
        if tool == "record_alignment":
            answer = {"variables": True, "direction": "volume" not in user, "horizon": True}
            return _response(tool, {**answer, "reason": "fake"}, len(user))
        raise AssertionError(f"no answer for {tool}")


def _papers_file(tmp_path, n=PAPERS, broken=()):
    lines = []
    for i in range(n):
        title = f"paper{i}" + ("-broken" if i in broken else "") + ("-m" if i % 3 == 0 else "")
        lines += ["[[papers]]", f'key = "k:{i}"', f'title = "{title}"', 'abstract = "a"', ""]
    path = tmp_path / "papers.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _dump(con: sqlite3.Connection) -> dict:
    """Every table's rows in rowid order, timestamps left out."""
    out = {}
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in sorted(t for t in tables if t != "sqlite_sequence"):
        rows = con.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
        out[table] = [
            {k: r[k] for k in r.keys() if k not in ("created_at", "added_at")} for r in rows
        ]
    return out


def _run(tmp_path, name, concurrency, *, seed=0, broken=()):
    con = store.connect(tmp_path / f"{name}.db")
    fake = DelayedFake(seed)
    source = FileSource(_papers_file(tmp_path, broken=broken))
    summary = paper_pool.run_isolated(con, source, fake, CONFIG, concurrency)
    dump = _dump(con)
    con.close()
    return dict(summary=summary, dump=dump, fake=fake)


def test_the_store_and_usage_are_identical_at_concurrency_1_and_n(tmp_path):
    one = _run(tmp_path, "one", 1, seed=1)
    many = _run(tmp_path, "many", PAPERS, seed=2)  # other delays, other completion order
    assert one["dump"] == many["dump"]
    assert vars(one["summary"]) == vars(many["summary"])
    assert len(one["fake"].sent) == len(many["fake"].sent)
    assert one["dump"]["usage"]  # per-paper usage rows are stored, in paper order
    # the run did what the test means to cover
    statuses = Counter(p["status"] for p in one["dump"]["proposals"])
    assert statuses["stored"] and statuses["duplicate"] and statuses["rejected"]
    assert any("parse error" in p["reason"] for p in one["dump"]["proposals"])
    assert one["summary"].outcomes == statuses
    assert len(one["summary"].papers_done) == PAPERS


def test_a_failed_paper_is_reported_in_its_place_at_any_concurrency(tmp_path):
    one = _run(tmp_path, "one", 1, broken=(2,))
    many = _run(tmp_path, "many", PAPERS, seed=5, broken=(2,))
    assert one["dump"] == many["dump"]
    assert list(one["summary"].papers_failed) == ["k:2"] == list(many["summary"].papers_failed)
    assert one["summary"].papers_done == [f"k:{i}" for i in range(PAPERS) if i != 2]
    papers = [p["key"] for p in one["dump"]["papers"]]
    assert papers == [f"k:{i}" for i in range(PAPERS)]  # the failed paper's row stays


def test_every_paper_sees_the_store_as_the_paper_phase_found_it(tmp_path):
    """The isolated meaning: paper k's prompts don't depend on papers before it, so
    each paper's first factor request is the one it makes alone."""
    many = _run(tmp_path, "many", PAPERS)
    alone = []
    for i in range(PAPERS):
        sub = tmp_path / f"alone{i}"
        sub.mkdir()
        con = store.connect(sub / "s.db")
        fake = DelayedFake(max_delay=0)
        path = _papers_file(sub)
        source = FileSource(path, keys=[f"k:{i}"])
        paper_pool.run_isolated(con, source, fake, CONFIG, 1)
        alone += [k for k in fake.sent if k["tools"][0]["name"] == "record_factors"]
        con.close()
    together = [k for k in many["fake"].sent if k["tools"][0]["name"] == "record_factors"]
    key = lambda k: json.dumps(k, sort_keys=True)  # noqa: E731
    assert sorted(map(key, together)) == sorted(map(key, alone))


def test_concurrency_resolves_from_the_machine_and_the_operator_cap(monkeypatch):
    monkeypatch.delenv("CAPSTONE_MAX_WORKERS", raising=False)
    cores = paper_pool.executor.detect().cores
    assert paper_pool.resolve_concurrency(0, 1000) == cores
    assert paper_pool.resolve_concurrency(0, 3) == min(3, cores)
    assert paper_pool.resolve_concurrency(5, 3) == 3
    monkeypatch.setenv("CAPSTONE_MAX_WORKERS", "2")
    assert paper_pool.resolve_concurrency(0, 1000) == min(2, cores)
    with pytest.raises(ValueError):
        paper_pool.resolve_concurrency(-1, 3)
