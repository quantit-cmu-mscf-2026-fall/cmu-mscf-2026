"""The deepen move with learned memory search: no network, no key, no market data.

A fake client stands in for Opus, a fake editor scripts the edits, and a fake
scorer gives each factor a fixed quality and independent values.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from capstone.factors import store
from capstone.factors.deepen import (
    MOTIF_WORDS,
    BudgetExhausted,
    Draft,
    ModelEditor,
    Pool,
    Scored,
    applicable,
    correlations,
    edit_prompt,
)
from capstone.factors.llm import FactorConfig
from capstone.factors.memory import PRODUCIBLE_MOTIFS
from capstone.factors.tree import identity_key, parse

CONFIG = FactorConfig(model="test-model", alignment_model="", max_repairs=1)


class FakeScorer:
    """Q from a table (default 0.15); values independent per factor, from its key."""

    def __init__(self, qualities: dict[str, float] | None = None, fail: set[str] = frozenset()):
        self.qualities = {identity_key(parse(k)): v for k, v in (qualities or {}).items()}
        self.fail = {identity_key(parse(e)) for e in fail}
        self.calls = 0

    def __call__(self, tree):
        self.calls += 1
        key = identity_key(tree)
        if key in self.fail:
            return None
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        values = np.random.default_rng(seed).standard_normal(500)
        return Scored(self.qualities.get(key, 0.15), values)


class ScriptedEditor:
    """Returns the next scripted list of expressions, whatever it is asked."""

    def __init__(self, *rounds: list[str]):
        self.rounds = list(rounds)
        self.asked: list[tuple] = []

    def drafts(self, parent, motif, n, lineage, vetoed):
        self.asked.append((parent, motif, n, list(lineage), list(vetoed)))
        out = []
        for expression in self.rounds.pop(0)[:n]:
            try:
                out.append(Draft(expression, "r", parse(expression)))
            except Exception as exc:  # noqa: BLE001 - the error text is the point
                out.append(Draft(expression, "r", None, str(exc)))
        return out


class FakeClient:
    """Answers each call with the next scripted list of expressions."""

    def __init__(self, *answers: list[str]):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        edits = [{"expression": e, "rationale": "r"} for e in self.answers.pop(0)]
        block = SimpleNamespace(
            type="tool_use", name=kwargs["tools"][0]["name"], input={"edits": edits}
        )
        return SimpleNamespace(
            content=[block], usage=SimpleNamespace(input_tokens=100, output_tokens=50)
        )


SEEDS = ["ts_mean(returns, 21)", "ts_std(volume, 63)", "rank(close / open)"]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger" / "runs.jsonl"


@pytest.fixture
def con(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "paper:1", "A paper")
    hid = store.add_hypothesis(
        con,
        store.Hypothesis(
            paper_key="paper:1",
            observation="o",
            knowledge="k",
            justification="j",
            specification="s",
            falsification_condition="f",
        ),
    )
    for e in SEEDS:
        store.add_factor(con, parse(e), hid)
    yield con
    con.close()


# ---------------------------------------------------------------------------
# The pool


def _pool(capacity=50):
    return Pool(CONFIG.__class__(**{**CONFIG.__dict__, "memory_pool_capacity": capacity}))


def test_admission_rules():
    rng = np.random.default_rng(0)
    pool = _pool()
    base = rng.standard_normal(1000)
    pool.add("m1", 0.15, base)
    fresh = rng.standard_normal(1000)
    assert pool.judge(0.15, 10, fresh).status == "admitted"
    assert pool.judge(0.20, 10, fresh).status == "high_quality"
    assert pool.judge(0.10, 10, fresh).status == "admitted"
    assert pool.judge(0.0999, 10, fresh).status == "rejected"
    assert pool.judge(0.15, 31, fresh).status == "rejected"
    close = base + 0.3 * rng.standard_normal(1000)  # corr about 0.96
    assert pool.judge(0.30, 10, close).status == "rejected"


def test_a_full_pool_replaces_its_weakest_member_only_for_a_better_child():
    rng = np.random.default_rng(1)
    pool = _pool(capacity=2)
    pool.add("m1", 0.15, rng.standard_normal(1000))
    pool.add("m2", 0.12, rng.standard_normal(1000))
    assert pool.judge(0.11, 10, rng.standard_normal(1000)).reason == "pool full"
    values = rng.standard_normal(1000)
    decision = pool.judge(0.13, 10, values)
    assert decision.replaces == "m2"
    pool.add("c", 0.13, values, decision)
    assert set(pool.quality) == {"m1", "c"}
    with pytest.raises(ValueError):
        pool.add("x", 0.0, values, pool.judge(0.0, 10, values))


def test_rho_max_follows_membership():
    rng = np.random.default_rng(2)
    pool = _pool()
    base = rng.standard_normal(1000)
    pool.add("m1", 0.15, base)
    assert pool.rho_max("m1") == 0.0
    pool.add("m2", 0.12, -base)
    assert pool.rho_max("m1") == pytest.approx(1.0)
    pool.remove("m2")
    assert pool.rho_max("m1") == 0.0


def test_vectorized_correlations_match_pairwise_ones():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5_000) * 1e6 + 3e7
    x[rng.random(5_000) < 0.05] = np.nan
    rows = []
    for k in range(5):
        y = 0.2 * k * x + rng.standard_normal(5_000) * 1e6
        y[rng.random(5_000) < 0.1] = np.nan
        rows.append(y)
    rows += [np.full(5_000, 2.0), np.full(5_000, np.nan)]
    for row, value in zip(rows, correlations(x, np.stack(rows)), strict=True):
        both = np.isfinite(x) & np.isfinite(row)
        if both.sum() < 2 or np.ptp(row[both]) == 0:
            assert value == 0.0
        else:
            assert value == pytest.approx(np.corrcoef(x[both], row[both])[0, 1], abs=1e-12)


def test_which_edits_a_parent_offers():
    assert not applicable(parse("high / low"), "window_rescale")
    assert applicable(parse("ts_mean(close, 5)"), "window_rescale")
    assert not applicable(parse("rank(cap)"), "operator_sub")
    assert all(
        applicable(parse("rank(cap)"), m)
        for m in PRODUCIBLE_MOTIFS
        if m != "operator_sub" and m != "window_rescale"
    )


# ---------------------------------------------------------------------------
# Opus writes the edits


def test_every_motif_has_plain_words():
    assert set(MOTIF_WORDS) == set(PRODUCIBLE_MOTIFS)


def test_prompt_carries_parent_lineage_motif_and_vetoes():
    text = edit_prompt(
        parse("ts_mean(returns, 5)"), "nesting", 3, ["returns", "ts_mean(returns, 5)"], ["other"]
    )
    assert "Parent: ts_mean(returns, 5)" in text
    assert "returns -> ts_mean(returns, 5)" in text
    assert MOTIF_WORDS["nesting"] in text and "other (" in text and "Write 3 different" in text


def test_drafts_are_repaired_and_exactly_n():
    client = FakeClient(
        ["ts_mean(returns, 21)", "ts_mean(returns,", "x", "extra(1)", "too many"],
        ["ts_std(returns, 5)", "ts_sum(returns, 63)"],
    )
    usage: Counter = Counter()
    editor = ModelEditor(client, CONFIG, max_calls=10, usage=usage)
    drafts = editor.drafts(parse("returns"), "nesting", 4, ["returns"], [])
    assert [d.expression for d in drafts] == [
        "ts_mean(returns, 21)",
        "ts_std(returns, 5)",
        "ts_sum(returns, 63)",
        "extra(1)",  # still broken after max_repairs = 1
    ]
    assert drafts[3].tree is None and drafts[3].error
    assert usage["calls"] == 2 and usage["input_tokens"] == 200
    assert client.calls[0]["model"] == "test-model" and client.calls[0]["tools"][0]["strict"]


def test_the_call_limit_is_hard():
    editor = ModelEditor(FakeClient(["rank(returns)"]), CONFIG, max_calls=1, usage=Counter())
    editor.drafts(parse("returns"), "rank_switch", 1, [], [])
    with pytest.raises(BudgetExhausted):
        editor.drafts(parse("returns"), "rank_switch", 1, [], [])
