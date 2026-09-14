"""Tests for capstone.alpha_gpt.operators."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from capstone.alpha_gpt.operators import OPERATORS

DATES = pd.bdate_range("2020-01-01", periods=5)


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES[: len(values)], columns=["A", "B", "C"])


X = _frame(
    [
        [1.0, 3.0, 2.0],
        [2.0, 1.0, 4.0],
        [3.0, 2.0, 6.0],
        [4.0, 5.0, 8.0],
        [5.0, 4.0, 10.0],
    ]
)


def _arguments(spec, rng, n_dates=40, n_assets=6):
    """Random but valid arguments for any operator, keyed by argument kind."""
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"S{i}" for i in range(n_assets)]
    args = []
    for kind in spec.args:
        if kind in ("series", "value"):
            values = rng.normal(1.0, 1.0, size=(n_dates, n_assets))
            values[rng.random(values.shape) < 0.05] = np.nan
            args.append(pd.DataFrame(values, index=dates, columns=assets))
        elif kind == "window":
            args.append(5)
        elif kind == "lag":
            args.append(2)
        else:
            args.append(pd.Series(["g1", "g1", "g2", "g2", "g2", "g3"], index=assets))
    return args


class TestRegistry:
    def test_names_match_keys_and_docs_exist(self):
        assert len(OPERATORS) == 25
        for name, spec in OPERATORS.items():
            assert spec.name == name
            assert spec.doc.strip()

    @pytest.mark.parametrize("name", sorted(OPERATORS))
    def test_arity_matches_implementation(self, name):
        spec = OPERATORS[name]
        params = inspect.signature(spec.fn).parameters
        assert len(params) == len(spec.args)

    def test_signature_text(self):
        assert OPERATORS["ts_corr"].signature == "ts_corr(x, y, d)"
        assert OPERATORS["shift"].signature == "shift(x, n)"
        assert OPERATORS["grouped_demean"].signature == "grouped_demean(x, group)"


class TestCausality:
    @pytest.mark.parametrize("name", sorted(OPERATORS))
    def test_changing_the_future_never_changes_the_past(self, name):
        spec = OPERATORS[name]
        rng = np.random.default_rng(0)
        args = _arguments(spec, rng)
        cutoff = 25

        perturbed = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                changed = arg.copy()
                changed.iloc[cutoff + 1 :] = rng.normal(
                    50.0, 20.0, size=changed.iloc[cutoff + 1 :].shape
                )
                perturbed.append(changed)
            else:
                perturbed.append(arg)

        before = spec(*args).iloc[: cutoff + 1]
        after = spec(*perturbed).iloc[: cutoff + 1]
        pd.testing.assert_frame_equal(before, after)

    @pytest.mark.parametrize("name", sorted(OPERATORS))
    def test_output_is_aligned_and_has_no_infinity(self, name):
        spec = OPERATORS[name]
        args = _arguments(spec, np.random.default_rng(1))
        # Zeros exercise div/log/zscore edge cases.
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                arg.iloc[3] = 0.0
        out = spec(*args)
        assert out.shape == (40, 6)
        assert not np.isinf(out.to_numpy()).any()


class TestTimeSeries:
    def test_shift_and_delta(self):
        assert OPERATORS["shift"](X, 1).iloc[1].tolist() == [1.0, 3.0, 2.0]
        assert OPERATORS["ts_delta"](X, 2).iloc[2].tolist() == [2.0, -1.0, 4.0]
        assert OPERATORS["ts_delta"](X, 2).iloc[1].isna().all()

    def test_rolling_statistics_need_a_full_window(self):
        mean = OPERATORS["ts_mean"](X, 3)
        assert mean.iloc[1].isna().all()
        assert mean.iloc[2].tolist() == [2.0, 2.0, 4.0]
        assert OPERATORS["ts_min"](X, 2).iloc[1].tolist() == [1.0, 1.0, 2.0]
        assert OPERATORS["ts_max"](X, 2).iloc[1].tolist() == [2.0, 3.0, 4.0]
        assert OPERATORS["ts_std"](X, 2).iloc[1, 0] == pytest.approx(np.std([1, 2], ddof=1))

    def test_ts_rank_is_percentile_of_today_in_window(self):
        # Column B over rows 0..2 is [3, 1, 2]: today's 2 is the middle of three.
        assert OPERATORS["ts_rank"](X, 3).iloc[2, 1] == pytest.approx(2 / 3)

    def test_ts_zscore_scale(self):
        out = OPERATORS["ts_zscore_scale"](X, 3)
        assert out.iloc[2, 0] == pytest.approx((3 - 2) / 1.0)

    def test_ts_decayed_linear_weights_today_most(self):
        out = OPERATORS["ts_decayed_linear"](X, 3)
        assert out.iloc[1].isna().all()
        assert out.iloc[2, 0] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)

    def test_ts_decayed_linear_shorter_than_window(self):
        assert OPERATORS["ts_decayed_linear"](X.iloc[:2], 3).isna().all().all()

    def test_ts_corr_of_a_series_with_itself_is_one(self):
        out = OPERATORS["ts_corr"](X, X, 3)
        assert out.iloc[4].tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_ts_corr_constant_window_is_missing_not_infinite(self):
        flat = _frame([[1.0, 1.0, 1.0]] * 5)
        assert OPERATORS["ts_corr"](flat, X, 3).iloc[4].isna().all()


class TestCrossSectional:
    def test_normed_rank(self):
        assert OPERATORS["normed_rank"](X).iloc[0].tolist() == pytest.approx([1 / 3, 1.0, 2 / 3])

    def test_zscore_scale_rows_have_zero_mean(self):
        out = OPERATORS["zscore_scale"](X)
        assert out.mean(axis=1).abs().max() < 1e-12

    def test_zscore_scale_constant_row_is_missing(self):
        flat = _frame([[2.0, 2.0, 2.0]])
        assert OPERATORS["zscore_scale"](flat).isna().all().all()

    def test_winsorize_clips_an_outlier_to_mean_plus_three_std(self):
        row = [0.0] * 19 + [100.0]
        wide = pd.DataFrame([row], index=DATES[:1], columns=[f"S{i}" for i in range(20)])
        out = OPERATORS["winsorize_scale"](wide)
        expected = np.mean(row) + 3 * np.std(row, ddof=1)
        assert out.iloc[0, -1] == pytest.approx(expected)
        assert out.iloc[0, 0] == 0.0

    def test_grouped_demean_sums_to_zero_within_groups(self):
        group = pd.Series({"A": "g1", "B": "g1", "C": "g2"})
        out = OPERATORS["grouped_demean"](X, group)
        assert out["C"].abs().max() == 0.0
        assert (out["A"] + out["B"]).abs().max() < 1e-12


class TestElementWise:
    def test_arithmetic_with_literals(self):
        assert OPERATORS["add"](X, 1).iloc[0].tolist() == [2.0, 4.0, 3.0]
        assert OPERATORS["minus"](1, X).iloc[0].tolist() == [0.0, -2.0, -1.0]
        assert OPERATORS["cwise_mul"](X, X).iloc[1].tolist() == [4.0, 1.0, 16.0]

    def test_div_by_zero_is_missing(self):
        zeros = X * 0.0
        assert OPERATORS["div"](X, zeros).isna().all().all()
        assert OPERATORS["div"](X, 0).isna().all().all()
        assert OPERATORS["div"](X, 2).iloc[0].tolist() == [0.5, 1.5, 1.0]

    def test_log_non_positive_is_missing(self):
        out = OPERATORS["log"](X - 2.0)
        assert out.iloc[0].isna().tolist() == [True, False, True]

    def test_unary(self):
        signed = X - 3.0
        assert OPERATORS["neg"](X).iloc[0].tolist() == [-1.0, -3.0, -2.0]
        assert OPERATORS["abs"](signed).iloc[0].tolist() == [2.0, 0.0, 1.0]
        assert OPERATORS["sign"](signed).iloc[0].tolist() == [-1.0, 0.0, -1.0]
        assert OPERATORS["relu"](signed).iloc[0].tolist() == [0.0, 0.0, 0.0]

    def test_comparisons_are_zero_one_and_keep_missing(self):
        with_nan = X.copy()
        with_nan.iloc[0, 0] = np.nan
        assert OPERATORS["greater"](with_nan, 2).iloc[0].tolist()[1:] == [1.0, 0.0]
        assert np.isnan(OPERATORS["greater"](with_nan, 2).iloc[0, 0])
        assert OPERATORS["less"](X, 2).iloc[0].tolist() == [1.0, 0.0, 0.0]

    def test_literal_only_arguments_are_a_caller_bug(self):
        with pytest.raises(TypeError, match="DataFrame"):
            OPERATORS["add"](1, 2)
