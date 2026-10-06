"""The deepen move with learned memory search: no network, no key, no market data.

A fake client stands in for Opus, a fake editor scripts the edits, and a fake
scorer gives each factor a fixed quality and independent values.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from capstone.factors import store
from capstone.factors.deepen import (
    MOTIF_WORDS,
    BudgetExhausted,
    DeepenAgent,
    Deepener,
    Draft,
    ModelEditor,
    Pool,
    Scored,
    applicable,
    correlations,
    edit_prompt,
)
from capstone.factors.llm import FactorConfig
from capstone.factors.memory import PRODUCIBLE_MOTIFS, memory_events
from capstone.factors.tree import factor_id, identity_key, parse

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


def deepener(con, editor, scorer=None, *, children=5, seed=0, parents=SEEDS):
    cfg = replace(CONFIG, memory_children_per_parent=children)
    d = Deepener(con, cfg, scorer or FakeScorer(), editor, run_id="test", seed=seed)
    for e in parents:
        assert d.add_parent(factor_id(parse(e)))
    return d


def _entries(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------
# One round


def test_a_round_records_move_lineage_events_and_one_ledger_trial_per_score(con, ledger):
    editor = ScriptedEditor(["ts_mean(returns, 5)", "log(cap)", "returns +", "-ts_sum(volume, 10)"])
    d = deepener(con, editor, children=4)
    outcomes = d.step()
    assert [o.status for o in outcomes].count("invalid") == 1  # "returns +" does not parse
    assert d.evaluations == 4
    moves = con.execute("SELECT action, from_id, reason FROM moves").fetchall()
    assert len(moves) == 1 and moves[0]["action"] == "deepen"
    parent_id, record = moves[0]["from_id"], json.loads(moves[0]["reason"])
    motif = record["motif"]
    assert record["by"] == "memory"  # no agent: the memory selected
    refined = [p for p in store.proposals(con) if p["edge_type"] == "refined"]
    assert len(refined) == 4 and {p["parent_factor_id"] for p in refined} == {parent_id}
    events = memory_events(con, "test")
    assert len(events) == 4 and {e["motif_intended"] for e in events} == {motif}
    scored = [o for o in outcomes if o.quality is not None]
    entries = _entries(ledger)
    assert scored and len(entries) == len(scored)  # one trial per score, no more
    assert {e["params"]["factor_id"] for e in entries} == {o.factor_id for o in scored}
    assert all(e["tags"] == ["learned-memory-search", "deepen"] for e in entries)


def test_screening_rejections_are_not_scored(con, ledger):
    # Window variants of a stored factor fail the store-originality check.
    editor = ScriptedEditor(["ts_mean(returns, 63)", "ts_mean(returns, 252)"])
    scorer = FakeScorer()
    d = deepener(con, editor, scorer, children=2, parents=["ts_mean(returns, 21)"])
    calls_before = scorer.calls
    outcomes = d.step()
    assert all(o.status == "rejected" and "not original" in o.reason for o in outcomes)
    assert scorer.calls == calls_before and _entries(ledger) == []


def test_pool_decisions_set_the_memory_status(con, ledger):
    scorer = FakeScorer({"log(cap)": 0.25, "sign(volume)": 0.05}, fail={"ts_max(cap, 5)"})
    editor = ScriptedEditor(["log(cap)", "sign(volume)", "ts_max(cap, 5)"])
    d = deepener(con, editor, scorer, children=3)
    by_expr = {o.expression: o for o in d.step()}
    assert by_expr["log(cap)"].status == "high_quality"
    low = by_expr["sign(volume)"]
    assert low.status == "rejected" and "Q 0.050" in low.reason
    assert by_expr["ts_max(cap, 5)"].status == "invalid"
    assert factor_id(parse("log(cap)")) in d.pool.quality  # a future parent
    assert len(_entries(ledger)) == 2  # the unscorable child is not a trial


def test_an_admitted_child_carries_its_lineage(con, ledger):
    editor = ScriptedEditor(["log(cap)"])
    d = deepener(con, editor, children=1)
    d.step()
    chain = d.lineage(factor_id(parse("log(cap)")))
    assert [parse(e) for e in chain] == [editor.asked[0][0], parse("log(cap)")]


def test_repeated_failures_veto_an_edit_and_the_prompt_says_so(con, ledger):
    # One parent, so every round's memory is about the same kind of parent.
    # Round 1 runs at u0; rounds 2 and 3 at u1 (selected once, then twice).
    editor = ScriptedEditor(*[["returns +"] * 5] * 3)
    d = deepener(con, editor, parents=["ts_mean(returns, 21)"])
    for _ in range(3):
        d.step()
    round2_motif, round3_vetoes = editor.asked[1][1], editor.asked[2][4]
    assert round2_motif in round3_vetoes  # five failures of five: vetoed, and said so
    assert editor.asked[2][1] != round2_motif  # and not chosen again


def test_the_same_seed_makes_the_same_choices(tmp_path, ledger):
    def run(name):
        c = store.connect(tmp_path / f"{name}.db")
        store.add_paper(c, "paper:1", "A paper")
        hid = store.add_hypothesis(c, store.Hypothesis("paper:1", "o", "k", "j", "s", "f"))
        for e in SEEDS:
            store.add_factor(c, parse(e), hid)
        editor = ScriptedEditor(*[["log(cap)", "sign(volume)"]] * 4)
        d = deepener(c, editor, children=2, seed=3)
        for _ in range(4):
            d.step()
        return [(a[1], a[0]) for a in editor.asked]

    assert run("a") == run("b")


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


def test_window_rescale_is_skipped_while_screening_rejects_every_window_variant(con, ledger):
    d = deepener(con, ScriptedEditor(), parents=["ts_mean(returns, 21)"])
    view = d._views()[0]
    assert not d._allowed(view, "window_rescale")
    d.config = replace(d.config, max_store_share=0.9)
    assert d._allowed(view, "window_rescale")


def test_frequent_subtrees_count_only_what_the_edit_added(con, ledger):
    # Make ts_std(volume, t) and ts_mean(cap, t) the store's frequent structures.
    hid = con.execute("SELECT id FROM hypotheses").fetchone()[0]
    for e in [
        "ts_std(volume, 5) * high",
        "ts_std(volume, 10) - low",
        "ts_mean(cap, 5) / open",
        "ts_mean(cap, 21) + high",
    ]:
        store.add_factor(con, parse(e), hid)
    editor = ScriptedEditor(["log(ts_std(volume, 63))", "ts_std(volume, 63) - ts_mean(cap, 3)"])
    d = deepener(con, editor, children=2, parents=["ts_std(volume, 63)"])
    keeps, adds = d.step()
    assert "frequent subtree" not in keeps.reason  # inherited from the parent: allowed
    assert adds.status == "rejected" and "frequent subtree" in adds.reason  # newly added


# ---------------------------------------------------------------------------
# The agent decides, the memory advises


class AgentClient:
    """Answers record_deepen calls with scripted decisions; repairs with scripted edits."""

    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(
            type="tool_use", name=kwargs["tools"][0]["name"], input=self.answers.pop(0)
        )
        return SimpleNamespace(
            content=[block], usage=SimpleNamespace(input_tokens=1000, output_tokens=500)
        )


def _decision(parent, edit, edits, reason="nesting beat expectations here"):
    return {
        "parent_id": factor_id(parse(parent)),
        "edit": edit,
        "reason": reason,
        "evidence_used": "nesting: +0.05 at confidence 0.55",
        "edits": [{"expression": e, "rationale": "r"} for e in edits],
    }


def _agent_deepener(con, client, children=2):
    agent = DeepenAgent(client, CONFIG, max_calls=10, usage=Counter())
    d = deepener(con, ScriptedEditor(), children=children)
    d.agent = agent
    return d


def test_the_agent_decides_and_its_reason_is_recorded_before_scoring(con, ledger):
    client = AgentClient(
        _decision("rank(close / open)", "nesting", ["ts_mean(rank(close / open), 5)", "log(cap)"])
    )
    d = _agent_deepener(con, client)
    outcomes = d.step()
    assert len(client.calls) == 1  # decision and edits in one call
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Parent " in prompt and "allowed edits:" in prompt and "edit | tried" in prompt
    move = con.execute("SELECT from_id, reason FROM moves").fetchone()
    record = json.loads(move["reason"])
    assert move["from_id"] == factor_id(parse("rank(close / open)"))
    assert record["by"] == "agent" and record["motif"] == "nesting"
    assert record["reason"] and record["evidence_used"]
    assert {o.motif_intended for o in outcomes} == {"nesting"}


def test_a_vetoed_edit_is_refused_then_asked_again(con, ledger):
    d = _agent_deepener(con, None)
    parent = factor_id(parse("rank(close / open)"))
    ctx = next(v.context for v in d._views() if v.id == parent)
    for _ in range(6):  # nesting keeps failing for this kind of parent: vetoed
        d.memory.update(
            {
                "context": ctx,
                "motif_realized": None,
                "motif_intended": "nesting",
                "status": "invalid",
                "residual": None,
            }
        )
    client = AgentClient(
        _decision("rank(close / open)", "nesting", ["ts_mean(rank(close / open), 5)"]),
        _decision(
            "rank(close / open)",
            "normalization",
            ["log(rank(close / open))", "-rank(close / open)"],
        ),
    )
    d.agent.client = client
    # Before the round, the parent's allowed edits already exclude the vetoed one.
    assert "nesting" not in next(c for c in d.candidates() if c.view.id == parent).allowed
    outcomes = d.step()
    assert len(client.calls) == 2
    assert "refused" in client.calls[1]["messages"][0]["content"]
    assert {o.motif_intended for o in outcomes} == {"normalization"}


def test_two_refusals_fall_back_to_the_memorys_selection(con, ledger):
    bad = _decision("rank(close / open)", "window_rescale", ["rank(close / open)"])  # not offered
    unknown = {**bad, "parent_id": "nope", "edit": "nesting"}
    d = _agent_deepener(con, AgentClient(bad, unknown))
    d.editor = ScriptedEditor(["log(cap)", "sign(volume)"])
    d.step()
    record = json.loads(con.execute("SELECT reason FROM moves").fetchone()["reason"])
    assert record["by"] == "memory" and d.refusals == 1
    assert len(d.editor.asked) == 1  # the editor wrote the fallback round


def test_the_agent_spends_from_the_same_call_limit(con, ledger):
    client = AgentClient(_decision("rank(close / open)", "nesting", ["log(cap)", "x("]))
    agent = DeepenAgent(client, CONFIG, max_calls=1, usage=Counter())
    d = deepener(con, ScriptedEditor(), children=2)
    d.agent = agent
    with pytest.raises(BudgetExhausted):  # the repair call would be the second
        d.step()
