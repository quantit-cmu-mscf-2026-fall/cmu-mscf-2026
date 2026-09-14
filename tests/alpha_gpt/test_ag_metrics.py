"""Tests for capstone.alpha_gpt.metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from capstone.alpha_gpt.metrics import forward_returns, rank_ic_series, split_metrics
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel
from capstone.cv import average_uniqueness, label_end_times


@pytest.fixture(scope="module")
def null_panel():
    return make_ohlcv_panel(n_dates=400, n_assets=40, pattern="none", seed=0)


@pytest.fixture(scope="module")
def reversal_panel():
    return make_ohlcv_panel(n_dates=400, n_assets=80, pattern="reversal", strength=0.06, seed=2)


class TestRankIC:
    def test_target_is_next_day_return(self, null_panel):
        returns = null_panel.fields["returns"]
        cheat = returns.shift(-1)  # impossible inside the DSL; only here to pin alignment
        ic = rank_ic_series(cheat, returns).dropna()
        # Only the last date is unscorable: it has no next-day return.
        assert len(ic) == len(returns) - 1
        assert ic.index[-1] == returns.index[-2]
        assert ic.min() == pytest.approx(1.0)

    def test_same_day_return_has_no_ic_on_null_data(self, null_panel):
        returns = null_panel.fields["returns"]
        ic = rank_ic_series(returns, returns).dropna()
        assert abs(ic.mean() / ic.std() * math.sqrt(len(ic))) < 3

    def test_pairs_are_masked_before_ranking(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        signal = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [0.0] * 4], index=dates)
        returns = pd.DataFrame([[0.0] * 4, [1.0, 2.0, np.nan, 4.0]], index=dates)
        ic = rank_ic_series(signal, returns, min_assets=3)
        assert ic.iloc[0] == pytest.approx(1.0)

    def test_dates_with_too_few_assets_are_missing(self, null_panel):
        returns = null_panel.fields["returns"]
        signal = returns.copy()
        signal.iloc[10, 5:] = np.nan
        ic = rank_ic_series(signal, returns, min_assets=20)
        assert np.isnan(ic.iloc[10])
        assert not np.isnan(ic.iloc[11])

    def test_constant_signal_is_missing_not_an_error(self, null_panel):
        returns = null_panel.fields["returns"]
        flat = returns * 0.0 + 1.0
        assert rank_ic_series(flat, returns).isna().all()


class TestSplitMetrics:
    def test_planted_reversal_is_significant_on_validation(self, reversal_panel):
        returns = reversal_panel.fields["returns"]
        spec = SplitSpec.from_fractions(reversal_panel.dates)
        metrics = split_metrics(
            -returns, returns, spec.dates(reversal_panel.dates, "valid"), split="valid"
        )
        assert metrics.ic_tstat > 3
        assert metrics.ic_pvalue < 0.01
        assert metrics.ls_sharpe > 1
        assert metrics.split == "valid"

    def test_sign_of_long_short_follows_the_signal(self, null_panel):
        returns = null_panel.fields["returns"]
        dates = null_panel.dates[5:300]
        good = split_metrics(returns.shift(-1), returns, dates, split="train")
        bad = split_metrics(-returns.shift(-1), returns, dates, split="train")
        assert good.ic_mean == pytest.approx(1.0)
        assert good.ls_sharpe > 10 > -10 > bad.ls_sharpe

    def test_data_after_the_split_cannot_change_its_metrics(self, null_panel):
        returns = null_panel.fields["returns"]
        signal = -returns.rolling(5).mean()
        dates = null_panel.dates[50:250]
        before = split_metrics(signal, returns, dates, split="valid")

        # Everything from two days after the last signal date onward is "the future":
        # the day right after is needed to score the last signal.
        cutoff = null_panel.dates.get_loc(dates[-1]) + 2
        rng = np.random.default_rng(9)
        signal_future, returns_future = signal.copy(), returns.copy()
        signal_future.iloc[cutoff:] = rng.normal(size=signal_future.iloc[cutoff:].shape)
        returns_future.iloc[cutoff:] = rng.normal(size=returns_future.iloc[cutoff:].shape)
        after = split_metrics(signal_future, returns_future, dates, split="valid")

        assert after == before

    def test_short_split_reports_ic_but_no_annualised_long_short(self, null_panel):
        returns = null_panel.fields["returns"]
        metrics = split_metrics(-returns, returns, null_panel.dates[10:40], split="valid")
        assert metrics.n_days == 30
        assert not math.isnan(metrics.ic_tstat)
        assert math.isnan(metrics.ls_sharpe)
        assert math.isnan(metrics.ls_sharpe_pvalue)

    def test_coverage_counts_defined_cells(self, null_panel):
        returns = null_panel.fields["returns"]
        signal = returns.rolling(20).mean()
        dates = null_panel.dates[:100]
        metrics = split_metrics(signal, returns, dates, split="train")
        assert metrics.coverage == pytest.approx(80 / 100)

    def test_unweighted_n_eff_is_the_day_count(self, null_panel):
        returns = null_panel.fields["returns"]
        metrics = split_metrics(-returns, returns, null_panel.dates[10:200], split="train")
        assert metrics.n_eff == metrics.n_days == 190

    def test_as_flat_prefixes_and_replaces_nan(self, null_panel):
        returns = null_panel.fields["returns"]
        metrics = split_metrics(-returns, returns, null_panel.dates[10:40], split="valid")
        flat = metrics.as_flat("valid")
        assert "split" not in flat and "valid_split" not in flat
        assert flat["valid_n_days"] == 30
        assert flat["valid_ls_sharpe"] is None


class TestHorizonAndWeights:
    def test_horizon_one_is_the_next_day_return(self, null_panel):
        returns = null_panel.fields["returns"]
        pd.testing.assert_frame_equal(forward_returns(returns, 1), returns.shift(-1))

    def test_multi_day_forward_return_compounds(self, null_panel):
        returns = null_panel.fields["returns"]
        forward = forward_returns(returns, 3)
        expected = (1 + returns.iloc[11:14, 0]).prod() - 1
        assert forward.iloc[10, 0] == pytest.approx(expected)
        assert forward.iloc[-3:].isna().all().all()

    def test_masked_return_blanks_every_label_that_spans_it(self, null_panel):
        returns = null_panel.fields["returns"].copy()
        returns.iloc[50] = np.nan
        forward = forward_returns(returns, 5)
        assert forward.iloc[45:50].isna().all().all()
        assert forward.iloc[44].notna().all() and forward.iloc[50].notna().all()

    @pytest.mark.parametrize("bad", [0, True, 2.5])
    def test_invalid_horizon(self, bad, null_panel):
        with pytest.raises(ValueError, match="horizon"):
            forward_returns(null_panel.fields["returns"], bad)

    def test_cheating_signal_at_longer_horizon_has_ic_one(self, null_panel):
        returns = null_panel.fields["returns"]
        ic = rank_ic_series(forward_returns(returns, 5), returns, horizon=5).dropna()
        assert ic.min() == pytest.approx(1.0)

    def test_unit_weights_reproduce_the_unweighted_statistics(self, null_panel):
        returns = null_panel.fields["returns"]
        signal = -returns.rolling(3).mean()
        dates = null_panel.dates[10:300]
        plain = split_metrics(signal, returns, dates, split="train")
        ones = pd.Series(1.0, index=null_panel.dates)
        weighted = split_metrics(signal, returns, dates, split="train", weights=ones)
        assert weighted.as_flat("m") == pytest.approx(plain.as_flat("m"))

    def test_constant_weights_shrink_t_by_their_square_root(self, null_panel):
        returns = null_panel.fields["returns"]
        signal = -returns.rolling(3).mean()
        dates = null_panel.dates[10:300]
        plain = split_metrics(signal, returns, dates, split="train")
        fifth = split_metrics(
            signal, returns, dates, split="train", weights=pd.Series(0.2, index=null_panel.dates)
        )
        assert fifth.ic_mean == pytest.approx(plain.ic_mean)
        assert fifth.n_eff == pytest.approx(0.2 * plain.n_days)
        assert fifth.ic_tstat == pytest.approx(plain.ic_tstat * math.sqrt(0.2))

    def test_uniqueness_weights_deflate_overlapping_label_t_stats(self, reversal_panel):
        returns = reversal_panel.fields["returns"]
        dates = reversal_panel.dates
        weights = average_uniqueness(label_end_times(dates, 5), dates)
        scored = dates[20:300]
        plain = split_metrics(-returns, returns, scored, split="train", horizon=5)
        weighted = split_metrics(
            -returns, returns, scored, split="train", horizon=5, weights=weights
        )
        assert weighted.n_eff == pytest.approx(plain.n_days / 5)
        assert abs(weighted.ic_tstat) < abs(plain.ic_tstat)

    def test_weights_must_cover_every_scored_date(self, null_panel):
        returns = null_panel.fields["returns"]
        partial = pd.Series(1.0, index=null_panel.dates[:100])
        with pytest.raises(ValueError, match="cover every scored date"):
            split_metrics(-returns, returns, null_panel.dates[50:150], split="t", weights=partial)
