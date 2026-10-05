"""Learned memory search, part A: edit motifs, parent context, and the memory store."""

from __future__ import annotations

import math

import pytest

from capstone.factors import memory, store
from capstone.factors.tree import parse


@pytest.mark.parametrize(
    ("parent", "child", "motif"),
    [
        ("-ts_sum(returns, 5)", "-ts_sum(returns, 5)", "same"),
        ("-ts_sum(returns, 5)", "-ts_sum(returns, 10)", "window"),
        ("-ts_sum(returns, 5)", "rank(-ts_sum(returns, 5))", "wrap:rank"),
        ("ts_sum(returns, 5)", "-ts_sum(returns, 5)", "wrap:neg"),
        ("-ts_sum(returns, 5)", "-ts_sum(returns, 5) * log(cap)", "add_term:*"),
        ("-ts_sum(returns, 5) * log(cap)", "-ts_sum(returns, 5)", "drop_term"),
        ("ts_mean(close, 5)", "ts_mean(volume, 5)", "swap_field"),
        ("ts_mean(close, 5)", "ts_std(close, 5)", "swap_func"),
        ("close / volume", "close - volume", "swap_op"),
        ("rank(close)", "close", "unwrap"),
        ("ts_mean(close, 5)", "ts_corr(close, volume, 21)", "rewrite"),
        # The change is found however deep it sits.
        ("rank(ts_mean(close, 5) / volume)", "rank(ts_mean(close, 5) / shares)", "swap_field"),
        ("rank(ts_mean(close, 5) / volume)", "rank(ts_mean(close, 63) / volume)", "window"),
    ],
)
def test_edit_motifs(parent, child, motif):
    assert memory.edit_motif(parse(parent), parse(child)) == motif


def test_parent_context_buckets_fields_quality_depth_and_usage():
    tree = parse("-ts_sum(returns, 5) * log(cap)")
    ctx = memory.parent_context(tree, quality=-0.12, depth=2, children=4)
    assert ctx.key == "returns+size|q2|d2|u2"  # |ICIR| 0.12 is in [0.10, 0.20)
    fresh = memory.parent_context(parse("rank(close / volume)"), quality=None, depth=0, children=0)
    assert fresh.key == "price+volume|unscored|d0|u0"
    deep = memory.parent_context(tree, quality=0.5, depth=9, children=40)
    assert (deep.quality, deep.depth, deep.usage) == ("q3", 3, 3)


@pytest.fixture
def con(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    yield con
    con.close()


def test_events_reduce_to_residual_moments_and_a_failure_posterior(con):
    ctx = memory.parent_context(parse("-ts_sum(returns, 5)"), quality=0.12, depth=1, children=1)
    scored = [(0.18, 0.12), (0.10, 0.12), (0.20, 0.12)]  # residuals 0.06, -0.02, 0.08
    for quality, baseline in scored:
        memory.record(
            con,
            parent_id="p",
            child_expression="c",
            context=ctx,
            motif="wrap:rank",
            status="high_quality",
            quality=quality,
            baseline=baseline,
        )
    for status in ("rejected", "invalid"):
        memory.record(
            con, parent_id="p", child_expression="c", context=ctx, motif="wrap:rank", status=status
        )
    stats = memory.memory_stats(con)[(ctx.key, "wrap:rank")]
    residuals = [0.06, -0.02, 0.08]
    assert stats.n == 3 and stats.attempts == 5 and stats.failures == 2
    assert stats.mean == pytest.approx(sum(residuals) / 3)
    mean = sum(residuals) / 3
    assert stats.variance == pytest.approx(sum((r - mean) ** 2 for r in residuals) / 2)
    assert (stats.failure_alpha, stats.failure_beta) == (3.0, 4.0)
    assert stats.failure_rate == pytest.approx(3 / 7)


def test_unscored_pairs_have_no_residual_moments(con):
    ctx = memory.parent_context(parse("close"), quality=None, depth=0, children=0)
    memory.record(
        con, parent_id="p", child_expression="c", context=ctx, motif="window", status="rejected"
    )
    stats = memory.memory_stats(con)[(ctx.key, "window")]
    assert stats.n == 0 and math.isnan(stats.mean) and math.isnan(stats.variance)
    assert stats.failure_rate == pytest.approx(2 / 3)


def test_unknown_status_is_rejected(con):
    ctx = memory.parent_context(parse("close"), quality=None, depth=0, children=0)
    with pytest.raises(ValueError, match="status"):
        memory.record(
            con, parent_id="p", child_expression="c", context=ctx, motif="window", status="great"
        )
