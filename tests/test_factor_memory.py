"""Learned memory search's core (AlphaMemo Eq. 2-11): labels, context, memory, decisions."""

from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import replace

import pytest

from capstone.factors import store
from capstone.factors.llm import FactorConfig, load_config
from capstone.factors.memory import (
    MOTIFS,
    PRODUCIBLE_MOTIFS,
    STATUSES,
    Baselines,
    MemoryState,
    PairStats,
    ParentView,
    action_score,
    edit_motif,
    event_motif,
    evidence,
    evidence_table,
    gate,
    ledger_score,
    locate_edit,
    memory_events,
    parameter_bin,
    parent_context,
    preprocess,
    record_event,
    reduce_events,
    select_action,
    veto_gate,
    vetoed,
    warmup,
)
from capstone.factors.tree import Const, identity_key, parse, unparse

CONFIG = FactorConfig()


def motif(parent: str, child: str) -> str:
    return edit_motif(parse(parent), parse(child))


# ---------------------------------------------------------------------------
# Preprocessing and diff


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 0), (5, 0), (6, 1), (21, 1), (22, 2), (63, 2), (64, 3), (252, 3), (253, 4), (1260, 4)],
)
def test_parameter_bin_edges(value, expected):
    assert parameter_bin(value) == expected


def test_constants_bin_by_magnitude_and_keep_their_sign():
    assert parameter_bin(-30.0) == 2
    assert preprocess(parse("close * 0.5")) == preprocess(parse("close * 3"))
    assert preprocess(Const(-3.0)) == Const(-5.0)


def test_windows_in_one_bin_preprocess_the_same():
    assert preprocess(parse("ts_mean(close, 6)")) == preprocess(parse("ts_mean(close, 21)"))
    assert preprocess(parse("ts_mean(close, 21)")) != preprocess(parse("ts_mean(close, 22)"))


@pytest.mark.parametrize(
    ("wrapped", "plain"),
    [
        ("--close", "close"),
        ("----close", "close"),
        ("---close", "-close"),
        ("rank(rank(close))", "rank(close)"),
        ("rank(rank(rank(close)))", "rank(close)"),
        ("ts_mean(--volume, 5) / rank(rank(close))", "ts_mean(volume, 5) / rank(close)"),
    ],
)
def test_redundant_wrappers_are_removed(wrapped, plain):
    assert identity_key(preprocess(parse(wrapped))) == identity_key(preprocess(parse(plain)))


def test_preprocessing_is_canonical_and_still_parses():
    a = preprocess(
        parse("(volume + close) * ts_corr(returns, close, 10) + shift(shift(cap, 3), 3)")
    )
    b = preprocess(parse("shift(cap, 6) + ts_corr(close, returns, 12) * (close + volume)"))
    assert a == b
    assert preprocess(parse(unparse(a))) == a


def test_shifts_fold_before_binning():
    # 3 + 3 = 6 is in the 6-21 bin; binning each 3 first would give 5 + 5.
    assert preprocess(parse("shift(shift(close, 3), 3)")) == preprocess(parse("shift(close, 21)"))


def test_equivalent_trees_have_no_edit():
    assert locate_edit(parse("close + volume"), parse("volume + close")) is None
    assert locate_edit(parse("ts_mean(close, 10)"), parse("ts_mean(close, 20)")) is None


def test_diff_descends_to_the_smallest_differing_pair():
    p, c = locate_edit(
        parse("ts_mean(close, 5) / ts_mean(volume, 21)"),
        parse("ts_mean(close, 5) / ts_mean(rank(volume), 21)"),
    )
    assert (unparse(p), unparse(c)) == ("volume", "rank(volume)")


def test_diff_matches_commutative_operands_as_a_multiset():
    p, c = locate_edit(parse("returns + volume + cap"), parse("returns + volume + close"))
    assert (unparse(p), unparse(c)) == ("cap", "close")


# ---------------------------------------------------------------------------
# Edit motifs: one case per label, top level and nested

CASES = [
    ("close", "rank(close)", "rank_switch"),
    ("rank(close)", "close", "rank_switch"),
    ("rank(close)", "ts_rank(close, 10)", "rank_switch"),
    ("zscore(close, 20)", "rank(close)", "rank_switch"),
    ("ts_mean(close, 5) - volume", "ts_mean(rank(close), 5) - volume", "rank_switch"),
    ("close", "close * volume", "interaction"),
    ("close", "volume / close", "interaction"),
    ("close", "close - ts_mean(close, 20)", "interaction"),
    ("close + volume", "close + volume + returns", "interaction"),
    ("rank(returns) * cap", "rank(returns + volume) * cap", "interaction"),
    ("ts_mean(close, 5)", "ts_mean(close, 63)", "window_rescale"),
    ("shift(returns, 1)", "shift(returns, 30)", "window_rescale"),
    ("close * 2", "close * 100", "window_rescale"),
    ("rank(ts_std(returns, 21)) * cap", "rank(ts_std(returns, 252)) * cap", "window_rescale"),
    ("ts_mean(close, 20)", "ts_std(close, 20)", "operator_sub"),
    ("ts_mean(close, 20)", "ts_max(close, 60)", "operator_sub"),
    ("close + volume", "close * volume", "operator_sub"),
    ("ts_corr(close, volume, 10)", "ts_cov(close, volume, 10)", "operator_sub"),
    ("rank(close - open) * cap", "rank(close / open) * cap", "operator_sub"),
    ("ts_mean(close, 5)", "ts_mean(open, 5)", "feature_swap"),
    ("returns + volume + cap", "returns + volume + close", "feature_swap"),
    ("ts_corr(returns, mkt_return, 63)", "ts_corr(returns, volume, 63)", "feature_swap"),
    ("returns", "ts_mean(returns, 5)", "nesting"),
    ("volume / shares", "decay_linear(volume / shares, 10)", "nesting"),
    ("returns", "ts_corr(returns, mkt_return, 63)", "nesting"),
    ("rank(returns) - cap", "rank(ts_sum(returns, 5)) - cap", "nesting"),
    ("ts_mean(returns, 5)", "returns", "nesting"),
    ("ts_corr(returns, mkt_return, 63)", "returns", "nesting"),
    ("returns", "shift(returns, 1)", "temporal_shift"),
    ("shift(close, 5)", "delta(close, 5)", "temporal_shift"),
    ("ts_mean(close, 5)", "delta(close, 5)", "temporal_shift"),
    ("rank(close) * volume", "rank(delta(close, 1)) * volume", "temporal_shift"),
    ("shift(returns, 1)", "returns", "temporal_shift"),
    ("delta(close, 5)", "close", "temporal_shift"),
    ("returns", "zscore(returns, 20)", "normalization"),
    ("log(volume)", "volume", "normalization"),
    ("returns", "-returns", "normalization"),
    ("-returns", "returns", "normalization"),
    ("log(volume)", "abs(volume)", "normalization"),
    ("ts_mean(returns, 5) / cap", "ts_mean(sign(returns), 5) / cap", "normalization"),
    ("close + volume + returns", "close + volume", "other"),
    ("ts_mean(close, 10)", "ts_mean(close, 20)", "other"),
    ("close - volume", "open - shares", "other"),
    ("close", "returns * volume", "other"),
    ("rank(close)", "rank(rank(close))", "other"),
]


@pytest.mark.parametrize(("parent", "child", "label"), CASES)
def test_edit_motif(parent, child, label):
    assert motif(parent, child) == label


def test_every_producible_label_has_a_case_and_condition_gate_none():
    assert {label for _, _, label in CASES} == set(PRODUCIBLE_MOTIFS)
    assert MOTIFS[0] == "condition_gate" and len(PRODUCIBLE_MOTIFS) == 9


def test_rule_order_breaks_ties():
    assert motif("rank(close)", "zscore(close, 20)") == "rank_switch"  # not normalization
    assert motif("close", "ts_rank(close, 20)") == "rank_switch"  # not nesting
    assert motif("ts_mean(close, 5)", "ts_std(close, 252)") == "operator_sub"


# ---------------------------------------------------------------------------
# Parent context


def test_context_key_format():
    tree = parse("ts_sum(returns, 5) / cap")
    assert parent_context(tree, quality=0.15, times_selected=1) == "returns+size|q2|d0|u1"


@pytest.mark.parametrize(
    ("expression", "groups"),
    [
        ("close - open + high - low", "price"),
        ("volume / shares", "volume"),
        ("cap * returns", "returns+size"),
        ("ts_cov(returns, mkt_return, 63) * volume", "market+returns+volume"),
    ],
)
def test_context_field_groups(expression, groups):
    assert parent_context(parse(expression), quality=0.0, times_selected=0).split("|")[0] == groups


@pytest.mark.parametrize(
    ("quality", "band"),
    [
        (None, "unscored"),
        (math.nan, "unscored"),
        (0.0499, "q0"),
        (0.05, "q1"),
        (0.0999, "q1"),
        (0.10, "q2"),
        (0.1999, "q2"),
        (0.20, "q3"),
        (-0.25, "q3"),
    ],
)
def test_context_quality_bin_edges(quality, band):
    assert parent_context(parse("returns"), quality=quality, times_selected=0).split("|")[1] == band


@pytest.mark.parametrize(
    ("nodes", "band"),
    [(1, "d0"), (5, "d0"), (6, "d1"), (10, "d1"), (11, "d2"), (20, "d2"), (21, "d3")],
)
def test_context_size_bin_edges(nodes, band):
    tree = parse("-" * (nodes - 1) + "returns")  # one node per minus sign
    assert parent_context(tree, quality=0.0, times_selected=0).split("|")[2] == band


@pytest.mark.parametrize(
    ("times", "band"), [(0, "u0"), (1, "u1"), (2, "u1"), (3, "u2"), (5, "u2"), (6, "u3")]
)
def test_context_usage_bin_edges(times, band):
    assert parent_context(parse("returns"), quality=0.0, times_selected=times).endswith(band)


# ---------------------------------------------------------------------------
# Memory events and statistics


@pytest.fixture
def con(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    yield con
    con.close()


def event(con, status, quality=None, baseline=None, *, motif="nesting", realized="", **kw):
    return record_event(
        con,
        run_id=kw.pop("run_id", "run-a"),
        iteration=kw.pop("iteration", 0),
        parent_id="p1",
        context=kw.pop("context", "returns|q2|d0|u0"),
        motif_intended=motif,
        motif_realized=motif if realized == "" else realized,
        status=status,
        child_expression="ts_mean(returns, 5)",
        quality=quality,
        baseline=baseline,
        **kw,
    )


def test_events_store_the_residual_and_read_back_in_order(con):
    event(con, "admitted", 0.25, 0.10, iteration=0)
    event(con, "invalid", realized=None, iteration=1)
    rows = memory_events(con)
    assert [r["iteration"] for r in rows] == [0, 1]
    assert rows[0]["residual"] == pytest.approx(0.15)
    assert rows[1]["residual"] is None
    assert memory_events(con, run_id="other") == []


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "stored"},
        {"status": "admitted", "motif": "rewrite"},
        {"status": "admitted", "quality": 0.2},
        {"status": "admitted", "baseline": 0.1},
    ],
)
def test_bad_events_are_refused(con, bad):
    with pytest.raises(ValueError):
        event(con, **bad)
    assert memory_events(con) == []


def test_a_failed_parse_counts_under_the_intended_motif():
    assert event_motif({"motif_realized": None, "motif_intended": "nesting"}) == "nesting"
    assert event_motif({"motif_realized": "other", "motif_intended": "nesting"}) == "other"


def test_statistics_by_hand(con):
    # Residuals 0.1, 0.3, -0.1: mean 0.1, sample sd 0.2. Two failures of four.
    for status, q in [("admitted", 0.2), ("high_quality", 0.4), ("rejected", 0.0)]:
        event(con, status, q, 0.1)
    event(con, "invalid", realized=None)
    stats = MemoryState.from_events(memory_events(con)).stats("returns|q2|d0|u0", "nesting")
    assert (stats.n, stats.attempts, stats.failures) == (3, 4, 2)
    assert stats.mean == pytest.approx(0.1) and stats.std == pytest.approx(0.2)
    assert stats.failure_rate == pytest.approx(3 / 6)  # Beta(3, 3)


def test_an_unseen_pair_has_the_prior():
    stats = MemoryState().stats("price|unscored|d0|u0", "feature_swap")
    assert (stats.n, stats.attempts, stats.failure_rate) == (0, 0, 0.5)
    assert math.isnan(stats.std)


def test_online_welford_equals_batch_reduction(con):
    rng = random.Random(0)
    contexts = ["returns|q1|d0|u0", "price+volume|q2|d1|u1", "size|unscored|d0|u0"]
    online = MemoryState()
    for i in range(600):
        status = rng.choice(STATUSES)
        scored = status != "invalid"
        online.update(
            event(
                con,
                status,
                rng.gauss(0.1, 0.08) if scored else None,
                rng.uniform(0.0, 0.2) if scored else None,
                motif=rng.choice(PRODUCIBLE_MOTIFS),
                realized="" if scored else None,
                context=rng.choice(contexts),
                iteration=i,
            )
        )
    batch = reduce_events(memory_events(con))
    assert online.pairs.keys() == batch.keys()
    for key, want in batch.items():
        got = online.pairs[key]
        assert (got.n, got.attempts, got.failures) == (want.n, want.attempts, want.failures)
        assert got.mean == pytest.approx(want.mean, abs=1e-12)
        assert got.m2 == pytest.approx(want.m2, abs=1e-12)


def test_older_stores_gain_the_memory_table(tmp_path):
    path = tmp_path / "factors.db"
    store.connect(path).close()
    old = sqlite3.connect(path)
    old.execute("DROP TABLE memory_events")  # a store from before the table existed
    old.commit()
    old.close()
    con = store.connect(path)
    event(con, "admitted", 0.2, 0.1)
    assert len(memory_events(con)) == 1
    con.close()


# ---------------------------------------------------------------------------
# Decisions


def stats_of(residuals, failures=0):
    s = PairStats()
    for r in residuals:
        s.add("admitted", r)
    for _ in range(failures):
        s.add("invalid", None)
    return s


def test_settings_load_from_the_config_file():
    cfg = load_config()
    assert (cfg.memory_lambda_max, cfg.memory_warmup_start, cfg.memory_warmup_length) == (
        0.05,
        0,
        200,
    )
    assert (cfg.memory_kappa, cfg.memory_veto_confidence, cfg.memory_veto_failure_rate) == (
        5.0,
        0.3,
        0.7,
    )


@pytest.mark.parametrize(
    "bad", [{"memory_warmup_length": 0}, {"memory_kappa": 0.0}, {"memory_veto_failure_rate": 1.5}]
)
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValueError):
        FactorConfig(**bad)


def test_baseline_is_the_parent_quality_until_the_context_has_history():
    b = Baselines()
    assert b.baseline("returns|q2|d0|u0", 0.15) == 0.15
    b.add("returns|q2|d0|u0", 0.10)
    b.add("returns|q2|d0|u0", 0.30)
    b.add("price|q0|d0|u0", 0.90)  # another context does not count
    assert b.baseline("returns|q2|d0|u0", 0.15) == pytest.approx(0.20)


def test_gate_by_hand():
    # n = 3, mu = 0.1, sigma = 0.2: 3/8 * min(1, 0.1 / 0.200001).
    assert gate(stats_of([0.1, 0.3, -0.1]), 5.0, 1e-6) == pytest.approx(0.1875, rel=1e-4)
    assert gate(stats_of([0.2] * 5), 5.0, 1e-6) == pytest.approx(0.5)  # sigma 0: saturates
    assert gate(stats_of([0.5]), 5.0, 1e-6) == 0.0  # closed below two residuals


def test_veto_gate_counts_attempts_until_two_are_scored():
    assert veto_gate(stats_of([], failures=5), 5.0, 1e-6) == pytest.approx(0.5)
    s = stats_of([0.1, 0.3, -0.1], failures=10)
    assert veto_gate(s, 5.0, 1e-6) == gate(s, 5.0, 1e-6)


@pytest.mark.parametrize(
    ("t", "start", "expected"),
    [(0, 0, 0.0), (50, 0, 0.0125), (200, 0, 0.05), (900, 0, 0.05), (150, 50, 0.025)],
)
def test_warmup_by_hand(t, start, expected):
    assert warmup(t, replace(CONFIG, memory_warmup_start=start)) == pytest.approx(expected)


def test_scores_by_hand():
    assert ledger_score(0.2, 0.5, 3) == pytest.approx(0.2 * 0.5 / 2)
    expected = math.log(0.05 + 1e-6) + 0.05 * 0.5 * 0.1
    assert action_score(0.05, 0.05, 0.5, 0.1, 1e-6) == pytest.approx(expected)


def test_veto_needs_both_confidence_and_failures():
    assert vetoed(stats_of([], failures=4), CONFIG)  # gate 4/9 > 0.3, rate 5/6 > 0.7
    assert not vetoed(stats_of([], failures=1), CONFIG)  # rate 2/3 < 0.7
    assert not vetoed(stats_of([], failures=2), CONFIG)  # gate 2/7 < 0.3
    assert not vetoed(stats_of([0.2] * 50), CONFIG)  # successes never veto


PARENTS = [
    ParentView("a", "returns|q2|d0|u0", 0.15, 0.0, 0),
    ParentView("b", "volume|q1|d0|u0", 0.12, 0.0, 0),
]


def _fail(memory, context, motif, times=10):
    for _ in range(times):
        memory.update(
            {
                "context": context,
                "motif_realized": None,
                "motif_intended": motif,
                "status": "invalid",
                "residual": None,
            }
        )


def test_selection_without_memory_follows_the_ledger_and_breaks_ties_in_order():
    action = select_action(PARENTS, MemoryState(), 0, CONFIG)
    assert (action.parent_id, action.motif) == ("a", PRODUCIBLE_MOTIFS[0])


def test_a_vetoed_action_is_never_chosen():
    memory = MemoryState()
    _fail(memory, "returns|q2|d0|u0", "rank_switch")
    action = select_action(PARENTS, memory, 0, CONFIG)
    assert (action.parent_id, action.motif) == ("a", PRODUCIBLE_MOTIFS[1])


def test_everything_vetoed_selects_nothing():
    memory = MemoryState()
    for p in PARENTS:
        for m in PRODUCIBLE_MOTIFS:
            _fail(memory, p.context, m)
    assert select_action(PARENTS, memory, 0, CONFIG) is None


def test_selection_skips_pairs_that_cannot_be_made():
    action = select_action(PARENTS, MemoryState(), 0, CONFIG, lambda p, m: m != "rank_switch")
    assert action.motif == PRODUCIBLE_MOTIFS[1]


def test_positive_memory_lifts_a_weaker_parent_only_after_warmup():
    memory = MemoryState()
    for _ in range(20):  # nesting did much better than expected for b's kind of parent
        memory.update(
            {
                "context": "volume|q1|d0|u0",
                "motif_realized": "nesting",
                "motif_intended": "nesting",
                "status": "admitted",
                "residual": 0.5,
            }
        )
    cfg = replace(CONFIG, memory_lambda_max=1.0)
    assert select_action(PARENTS, memory, 0, cfg).parent_id == "a"  # lambda_0 = 0
    action = select_action(PARENTS, memory, 200, cfg)
    assert (action.parent_id, action.motif) == ("b", "nesting")


def test_an_rng_breaks_exact_ties_at_random():
    picks = {
        select_action(PARENTS, MemoryState(), 0, CONFIG, rng=random.Random(i)).motif
        for i in range(40)
    }
    assert len(picks) > 3  # not always the first motif
    assert all(
        select_action(PARENTS, MemoryState(), 0, CONFIG, rng=random.Random(i)).parent_id == "a"
        for i in range(10)
    )  # only exact ties are random: the better parent still wins


# ---------------------------------------------------------------------------
# The evidence the agent reads


def _event(context, motif, status, residual):
    return {
        "context": context,
        "motif_realized": motif if status != "invalid" else None,
        "motif_intended": motif,
        "status": status,
        "residual": residual,
    }


def test_evidence_gives_each_edit_a_verdict():
    ctx = "returns|q2|d0|u0"
    memory = MemoryState()
    for r in [0.2, 0.2, 0.3, 0.25, 0.2]:  # nesting beat expectations, consistently
        memory.update(_event(ctx, "nesting", "admitted", r))
    for r in [-0.1, -0.12, -0.09, -0.11]:  # interaction fell short, consistently
        memory.update(_event(ctx, "interaction", "rejected", r))
    for _ in range(5):  # rank_switch never even scored
        memory.update(_event(ctx, "rank_switch", "invalid", None))
    memory.update(_event(ctx, "feature_swap", "admitted", 0.4))  # one result: not enough
    rows = {r.motif: r for r in evidence(memory, ctx, CONFIG)}
    assert rows["nesting"].verdict == "beats expectations"
    assert rows["interaction"].verdict == "vetoed: keeps failing"  # 4 of 4 failed
    assert rows["rank_switch"].verdict == "vetoed: keeps failing"
    assert rows["feature_swap"].verdict == "too little evidence"
    assert rows["operator_sub"].verdict == "untried"
    nesting = rows["nesting"]
    stats = memory.stats(ctx, "nesting")
    assert (nesting.attempts, nesting.scored) == (5, 5)
    assert nesting.mean == pytest.approx(stats.mean)
    assert nesting.confidence == pytest.approx(gate(stats, 5.0, 1e-6))
    assert rows["rank_switch"].mean is None


def test_evidence_falls_short_when_trusted_and_negative():
    ctx = "price|q1|d1|u0"
    memory = MemoryState()
    for r in [-0.05, -0.06, -0.05, -0.04, -0.05]:  # scored, admitted, but below expectation
        memory.update(_event(ctx, "operator_sub", "admitted", r))
    (row,) = evidence(memory, ctx, CONFIG, motifs=["operator_sub"])
    assert row.verdict == "falls short of expectations"


def test_evidence_table_lists_every_edit_once():
    rows = evidence(MemoryState(), "size|unscored|d0|u0", CONFIG)
    table = evidence_table(rows)
    assert table.splitlines()[0].startswith("edit | tried")
    assert len(table.splitlines()) == 2 + len(PRODUCIBLE_MOTIFS)
    assert all(m in table for m in PRODUCIBLE_MOTIFS) and "untried" in table


@pytest.mark.parametrize(
    ("quality", "band"),
    [(-3.0, "q0"), (-0.01, "q0"), (0.0, "q1"), (0.99, "q1"), (1.0, "q2"), (1.645, "q3")],
)
def test_signed_quality_bins_put_losers_at_the_bottom(quality, band):
    # The evaluation scorer's z: negative loses after costs, so it can't share
    # a bin with a winner the way a negative ICIR does.
    key = parent_context(
        parse("returns"),
        quality=quality,
        times_selected=0,
        quality_edges=(0.0, 1.0, 1.645),
        signed_quality=True,
    )
    assert key.split("|")[1] == band


def test_quality_edges_must_increase():
    from dataclasses import replace

    from capstone.factors.llm import load_config

    with pytest.raises(ValueError, match="strictly increasing"):
        replace(load_config(), memory_quality_edges=(0.1, 0.1, 0.2))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(load_config(), memory_quality_edges=())
