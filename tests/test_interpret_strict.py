"""Strict-domain mode and the dated cross-sectional universe of the interpreter."""

import numpy as np
import pandas as pd
import pytest

from capstone.factors.interpret import EvaluationError, evaluate
from capstone.factors.tree import parse


def frame(rows):
    return pd.DataFrame(rows, columns=[101, 102, 103, 104][: len(rows[0])], dtype=float)


@pytest.mark.parametrize(
    "expression",
    [
        "sign((close - 25) / (close - 11))",
        "rank((close - 25) / (close - 11))",
        "scale((close - 25) / (close - 11))",
        "1 / (1 / (close - 11))",
        "sign(close * (1e308 * 1e308))",
        "close + (1 / 0)",
    ],
)
def test_undefined_intermediate_cannot_become_a_valid_score(expression):
    fields = {"close": frame([[11.0, 21.0, 31.0, 41.0]])}
    row = evaluate(parse(expression), fields, strict_domains=True).iloc[0]
    assert pd.isna(row.loc[101])


def test_strict_mode_keeps_defined_neighbours_of_an_undefined_value():
    fields = {"close": frame([[11.0, 21.0, 31.0, 41.0]])}
    strict = evaluate(parse("sign((close - 25) / (close - 11))"), fields, strict_domains=True)
    assert pd.isna(strict.iloc[0, 0])
    assert strict.iloc[0, 1:].tolist() == [-1.0, 1.0, 1.0]


def test_strict_scale_uses_finite_normalization_before_outer_signal():
    fields = {"close": frame([[1e308, 1e308]]), "cap": frame([[1.0, 2.0]])}
    node = parse("scale(close) + cap")
    assert evaluate(node, fields).iloc[0].tolist() == [1.0, 2.0]
    assert evaluate(node, fields, strict_domains=True).iloc[0].tolist() == [1.5, 2.5]


@pytest.mark.parametrize(
    "expression", ["close + log(2)", "close + abs(2)", "close + ts_mean(2, 2)"]
)
def test_strict_function_constant_arguments_broadcast_explicitly(expression):
    fields = {"close": frame([[1.0, 2.0], [3.0, 4.0]])}
    node = parse(expression)
    with pytest.raises(AttributeError):
        evaluate(node, fields)
    expected = fields["close"].copy()
    expected += np.log(2) if "log" in expression else 2
    if "ts_mean" in expression:
        expected.iloc[0] = np.nan
    pd.testing.assert_frame_equal(evaluate(node, fields, strict_domains=True), expected)


def test_strict_correlation_avoids_internal_variance_product_overflow():
    fields = {"close": frame([[1.0e150, 2.0e150], [1.1e150, 2.1e150]])}
    node = parse("ts_corr(close, close, 2)")
    assert evaluate(node, fields).iloc[-1].tolist() == [0.0, 0.0]
    assert evaluate(node, fields, strict_domains=True).iloc[-1].tolist() == pytest.approx(
        [1.0, 1.0]
    )


@pytest.mark.parametrize(
    "expression",
    [
        "close + cap",
        "close / cap",
        "sign(close - 15)",
        "ts_mean(close, 2)",
        "ts_std(close, 2)",
        "rank(close)",
        "ts_corr(close, cap, 2)",
        "signed_power(close, 2)",
        "zscore(close, 2)",
        "winsorize(close)",
        "scale(close)",
    ],
)
def test_strict_mode_retains_finite_supported_numerics(expression):
    rng = np.random.default_rng(0)
    fields = {
        "close": pd.DataFrame(rng.uniform(5, 30, (6, 4))),
        "cap": pd.DataFrame(rng.uniform(1, 9, (6, 4))),
    }
    node = parse(expression)
    pd.testing.assert_frame_equal(
        evaluate(node, fields, strict_domains=True), evaluate(node, fields)
    )


def test_cross_sections_use_dated_members_and_keep_historical_exits():
    close = pd.DataFrame(
        [[1.0, 3.0, 2.0, 1e9], [1.0, 7.0, np.nan, 1e9]], columns=[101, 102, 103, 104]
    )
    mask = pd.DataFrame(
        [[True, True, True, False], [True, True, False, False]], columns=close.columns
    )
    values = evaluate(
        parse("ts_mean(rank(close), 2)"),
        {"close": close},
        strict_domains=True,
        cross_sectional_universe=mask,
    )
    # Day 0 ranks are (1/3, 1, 2/3), including the later-exited 103; day 1 ranks are (1/2, 1).
    assert values.iloc[1, :2].tolist() == pytest.approx([5 / 12, 1.0])


def test_a_member_without_a_value_blanks_that_dates_cross_section():
    close = pd.DataFrame([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]], columns=[1, 2, 3])
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    values = evaluate(
        parse("rank(close)"), {"close": close}, strict_domains=True, cross_sectional_universe=mask
    )
    assert values.iloc[0].isna().all()
    assert values.iloc[1].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_the_universe_needs_strict_mode_matching_axes_and_known_membership():
    close = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    node = parse("rank(close)")
    with pytest.raises(EvaluationError, match="strict mode and matching axes"):
        evaluate(node, {"close": close}, cross_sectional_universe=mask)
    with pytest.raises(EvaluationError, match="strict mode and matching axes"):
        evaluate(
            node, {"close": close}, strict_domains=True, cross_sectional_universe=mask.iloc[:1]
        )
    unknown = mask.astype(object)
    unknown.iloc[0, 0] = None
    with pytest.raises(EvaluationError, match="explicitly known"):
        evaluate(node, {"close": close}, strict_domains=True, cross_sectional_universe=unknown)
