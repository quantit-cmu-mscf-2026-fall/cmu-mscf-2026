"""Tests for capstone.alpha_gpt.dsl."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from capstone.alpha_gpt.dsl import (
    Call,
    Const,
    DSLError,
    FieldRef,
    GroupRef,
    Limits,
    canonical,
    evaluate,
    parse,
)
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel

FIELDS = ("open", "high", "low", "close", "volume", "vwap", "returns")
GROUPS = ("sector",)


def _parse(expression: str, **kwargs):
    kwargs.setdefault("fields", FIELDS)
    kwargs.setdefault("groups", GROUPS)
    return parse(expression, **kwargs)


@pytest.fixture(scope="module")
def panel():
    return make_ohlcv_panel(n_dates=60, n_assets=8, n_sectors=3, seed=0)


class TestValid:
    def test_simple_call_tree(self):
        assert _parse("ts_mean(close, 5)") == Call("ts_mean", (FieldRef("close"), Const(5)))

    @pytest.mark.parametrize(
        "expression",
        [
            "neg(returns)",
            "normed_rank(div(volume, ts_mean(volume, 20)))",
            "cwise_mul(sign(returns), ts_zscore_scale(volume, 20))",
            "grouped_demean(ts_delta(close, 5), sector)",
            "ts_corr(close, volume, 10)",
            "minus(1, div(close, shift(close, 5)))",
            "greater(returns, -0.02)",
            "add(close, 1.5)",
        ],
    )
    def test_canonical_round_trip(self, expression):
        node = _parse(expression)
        assert _parse(canonical(node)) == node

    def test_whitespace_does_not_matter(self):
        assert canonical(_parse("  ts_mean( close ,5 ) ")) == "ts_mean(close, 5)"

    def test_commutative_arguments_are_sorted(self):
        assert _parse("add(open, close)") == _parse("add(close, open)")
        assert canonical(_parse("cwise_mul(volume, returns)")) == "cwise_mul(returns, volume)"

    def test_non_commutative_arguments_keep_their_order(self):
        assert _parse("minus(open, close)") != _parse("minus(close, open)")

    def test_negative_literal(self):
        node = _parse("cwise_mul(returns, -1.5)")
        assert Const(-1.5) in node.args

    def test_group_argument(self):
        node = _parse("grouped_demean(returns, sector)")
        assert node.args[1] == GroupRef("sector")


class TestRejected:
    @pytest.mark.parametrize(
        ("expression", "message"),
        [
            ("", "empty"),
            ("ts_mean(close, 5) ts_mean", "not a valid expression"),
            ("foo(close)", "unknown operator 'foo'"),
            ('__import__("os")', "unknown operator '__import__'"),
            ('__import__("os").system("x")', "direct operator calls"),
            ("ts_mean(price, 5)", "unknown field 'price'"),
            ("close / open", "infix arithmetic"),
            ("add(close, open) + 1", "infix arithmetic"),
            ("close > open", "use greater or less"),
            ("close and open", "and/or"),
            ("close.shift(-1)", "direct operator calls"),
            ("neg(close.values)", "attribute access"),
            ("neg(close[0])", "indexing"),
            ("lambda: 1", "Lambda is not allowed"),
            ("close if open else high", "IfExp is not allowed"),
            ('neg("close")', "strings are not allowed"),
            ("-close", "unary minus"),
            ("ts_mean(x=close, d=5)", "keyword arguments"),
            ("neg(*close)", "starred"),
            ("ts_mean(close)", r"takes 2 arguments ts_mean\(x, d\), got 1"),
            ("ts_mean(close, True)", "True/False"),
            ("add(close, True)", "True/False"),
            ("ts_mean(close, 0)", r"window must be an integer literal in \[2, 126\]"),
            ("ts_mean(close, 1)", "window must"),
            ("ts_mean(close, -5)", "window must"),
            ("ts_mean(close, 5.0)", "window must"),
            ("ts_mean(close, 999)", "window must"),
            ("ts_mean(close, volume)", "window must"),
            ("shift(close, -1)", "negative lags would read the future"),
            ("shift(close, 0)", "lag must"),
            ("ts_mean(1.0, 5)", "numeric literal is not allowed here"),
            ("1.0", "numeric literal is not allowed here"),
            ("add(1, 2)", "at least one field or operator call"),
            ("add(close, 1e999)", "finite"),
            ("add(close, 1e7)", "magnitude"),
            ("neg(sector)", "group 'sector' can only be used as the group argument"),
            ("grouped_demean(close, close)", "expected a group name"),
        ],
    )
    def test_invalid_expression(self, expression, message):
        with pytest.raises(DSLError, match=message) as info:
            _parse(expression)
        assert info.value.expression == expression

    def test_grouped_operator_without_groups(self):
        with pytest.raises(DSLError, match="no groups"):
            _parse("grouped_demean(close, sector)", groups=())

    def test_depth_limit(self):
        expression = "neg(" * 8 + "close" + ")" * 8
        with pytest.raises(DSLError, match="depth 9 exceeds 8"):
            _parse(expression)
        _parse("neg(" * 7 + "close" + ")" * 7)

    def test_node_limit(self):
        with pytest.raises(DSLError, match="7 nodes, more than 5"):
            _parse("add(add(close, open), add(high, low))", limits=Limits(max_nodes=5))

    def test_length_limit(self):
        with pytest.raises(DSLError, match="longer than 20 characters"):
            _parse("ts_mean(close, 5)    add", limits=Limits(max_chars=20))

    def test_non_string_is_a_caller_bug(self):
        with pytest.raises(TypeError):
            parse(42, fields=FIELDS)


class TestEvaluate:
    def test_matches_direct_computation(self, panel):
        returns = panel.fields["returns"]
        out = evaluate(_parse("neg(ts_mean(returns, 5))"), panel)
        pd.testing.assert_frame_equal(out, -returns.rolling(5, min_periods=5).mean())

    def test_literal_on_the_left(self, panel):
        close = panel.fields["close"]
        pd.testing.assert_frame_equal(evaluate(_parse("minus(1, close)"), panel), 1 - close)

    def test_group_operator_uses_panel_groups(self, panel):
        out = evaluate(_parse("grouped_demean(returns, sector)"), panel)
        sectors = panel.groups["sector"]
        for label in sectors.unique():
            members = sectors.index[sectors == label]
            assert out[members].sum(axis=1).abs().max() < 1e-12

    def test_never_uses_eval_or_exec(self, panel, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("eval/exec must never be used")

        monkeypatch.setattr(builtins, "eval", forbidden)
        monkeypatch.setattr(builtins, "exec", forbidden)
        node = _parse("normed_rank(ts_corr(close, volume, 10))")
        assert evaluate(node, panel).shape == (60, 8)
