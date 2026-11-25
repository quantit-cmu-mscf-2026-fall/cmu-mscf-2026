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


def test_a_member_without_a_value_leaves_the_other_members_ranks_alone():
    close = pd.DataFrame([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]], columns=[1, 2, 3])
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    values = evaluate(
        parse("rank(close)"), {"close": close}, strict_domains=True, cross_sectional_universe=mask
    )
    assert values.iloc[0].tolist()[:2] == pytest.approx([0.5, 1.0])
    assert np.isnan(values.iloc[0, 2])
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


@pytest.mark.parametrize("direction", [1, -1])
def test_strict_near_perfect_correlation_retains_defined_values(direction):
    close = pd.DataFrame(np.random.default_rng(0).lognormal(4, 3e-5, (1000, 3)))
    fields = {"close": close * direction, "cap": close * 1e9}
    values = evaluate(parse("ts_corr(cap, close, 10)"), fields, strict_domains=True)
    assert values.iloc[:9].isna().all().all()
    assert values.iloc[9:].notna().all().all()
    # pandas 2's rolling moments lose a few ppm on this tiny-volatility panel.
    np.testing.assert_allclose(values.iloc[9:], direction, rtol=0, atol=1e-5)
    assert (values.iloc[9:].abs() <= 1).all().all()


@pytest.mark.parametrize(
    "expression",
    ["ts_corr(shares, close, 20)", "ts_corr(close, shares, 20)", "zscore(shares, 20)"],
)
def test_a_constant_window_after_a_level_step_has_no_strict_score(expression):
    shares = pd.DataFrame({101: [3e8] * 5 + [3e8 + 21000] * 25})
    close = pd.DataFrame({101: np.arange(30, dtype=float) + 10})
    values = evaluate(parse(expression), {"shares": shares, "close": close}, strict_domains=True)
    assert values.iloc[19:24].notna().all().all()
    assert values.iloc[24:].isna().all().all()


def test_strict_ema_does_not_carry_a_value_over_an_undefined_input_day():
    close = frame([[12], [13], [11], [14], [15], [16]])
    node = parse("ema(1 / (close - 11), 2)")
    default = evaluate(node, {"close": close})
    strict = evaluate(node, {"close": close}, strict_domains=True)
    assert np.isfinite(default.iloc[2, 0])
    assert pd.isna(strict.iloc[2, 0])
    pd.testing.assert_frame_equal(strict.drop(index=2), default.drop(index=2), check_exact=True)


@pytest.mark.parametrize("expression", ["zscore(close, 2)", "winsorize(close)"])
def test_strict_undefined_moments_do_not_survive_as_finite_scores(expression):
    close = frame([[1e200], [2e200]]) if expression.startswith("zscore") else frame([[1]])
    values = evaluate(parse(expression), {"close": close}, strict_domains=True)
    assert values.isna().all().all()


def test_strict_pair_constant_second_argument_is_broadcast():
    close = frame([[1, 2], [2, 4], [3, 6]])
    values = evaluate(parse("ts_cov(close, 2, 2)"), {"close": close}, strict_domains=True)
    assert values.iloc[0].isna().all()
    assert (values.iloc[1:] == 0).all().all()


@pytest.mark.parametrize("kind", ["series", "array", "nonboolean"])
def test_invalid_universe_types_and_values_raise_the_documented_error(kind):
    close = frame([[1, 2], [3, 4]])
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    if kind == "series":
        mask = mask.iloc[:, 0]
    elif kind == "array":
        mask = mask.to_numpy()
    else:
        mask = mask.astype(int) * 2
    with pytest.raises(EvaluationError):
        evaluate(
            parse("rank(close)"),
            {"close": close},
            strict_domains=True,
            cross_sectional_universe=mask,
        )


@pytest.mark.parametrize(
    "expression",
    [
        "ts_corr(cap, close, 10)",
        "zscore(cap, 10)",
        "ema(close, 3)",
        "rank(close)",
        "scale(close)",
        "winsorize(close)",
        "ts_mean(rank(close), 3)",
    ],
)
def test_strict_numerical_fixes_keep_history_bit_identical(expression):
    close = pd.DataFrame(np.random.default_rng(7).lognormal(4, 0.02, (50, 3)))
    fields = {"close": close, "cap": close * 1e9}
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    node = parse(expression)
    options = {"strict_domains": True, "cross_sectional_universe": mask}
    expected = evaluate(node, fields, **options)
    changed = {name: value.copy() for name, value in fields.items()}
    for value in changed.values():
        value.iloc[30:] *= 1e80
    future_mask = mask.copy()
    future_mask.iloc[30:] = False
    after = evaluate(node, changed, strict_domains=True, cross_sectional_universe=future_mask)
    pd.testing.assert_frame_equal(expected.iloc[:30], after.iloc[:30], check_exact=True)
    for end in (15, 29, 39):
        truncated = evaluate(
            node,
            {name: value.iloc[: end + 1] for name, value in fields.items()},
            strict_domains=True,
            cross_sectional_universe=mask.iloc[: end + 1],
        )
        pd.testing.assert_frame_equal(expected.iloc[[end]], truncated.iloc[[-1]], check_exact=True)
