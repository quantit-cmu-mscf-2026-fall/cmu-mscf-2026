"""Tests for capstone.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.backtest import (
    BacktestSummary,
    backtest_frame,
    candidate_returns,
    run_backtest,
    summarize,
    sweep,
    to_weights,
)
from capstone.synth import make_panel


def _random_panel(n_dates: int = 10, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.standard_normal((n_dates, n_assets)), index=dates, columns=assets)


def test_to_weights_demean_and_gross():
    signal = _random_panel()
    weights = to_weights(signal, demean=True, gross=1.0)

    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-9)
    assert np.allclose(weights.abs().sum(axis=1), 1.0, atol=1e-9)
    assert weights.index.equals(signal.index)
    assert weights.columns.equals(signal.columns)


def test_to_weights_respects_gross_target():
    signal = _random_panel(seed=2)
    weights = to_weights(signal, demean=True, gross=2.5)
    assert np.allclose(weights.abs().sum(axis=1), 2.5, atol=1e-9)


def test_to_weights_all_zero_row_is_safe():
    signal = _random_panel(n_dates=5, n_assets=4)
    signal.iloc[2] = 0.0
    weights = to_weights(signal, demean=True, gross=1.0)

    assert (weights.iloc[2] == 0.0).all()
    assert np.isfinite(weights.to_numpy()).all()


def test_to_weights_all_nan_row_is_safe():
    signal = _random_panel(n_dates=5, n_assets=4)
    signal.iloc[1] = np.nan
    weights = to_weights(signal, demean=True, gross=1.0)

    assert (weights.iloc[1] == 0.0).all()
    assert np.isfinite(weights.to_numpy()).all()


def test_lookahead_discrimination():
    """Removing `.shift(1)` in run_backtest should make this test fail.

    A signal set to next-period's return is genuine foresight and should
    produce a large positive mean. A signal set to the *contemporaneous*
    return would be look-ahead if positions were not shifted, but with the
    shift in place it carries no informational edge over the following
    period and its mean should sit near zero, far below the foresight case.
    """
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=120)
    assets = [f"A{i}" for i in range(6)]
    returns = pd.DataFrame(rng.standard_normal((120, 6)) * 0.01, index=dates, columns=assets)

    foresight = run_backtest(returns.shift(-1), returns)
    contemporaneous = run_backtest(returns, returns)

    foresight_mean = foresight.dropna().mean()
    contemporaneous_mean = contemporaneous.dropna().mean()

    assert foresight_mean > 0
    assert foresight_mean > 20 * abs(contemporaneous_mean)


def _constant_book(n_dates: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long A / short B every day, on flat prices: the only P&L is trading cost."""
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    signal = pd.DataFrame({"A": 1.0, "B": -1.0}, index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    return signal, returns


def test_first_period_is_nan_not_a_zero_return():
    # Nothing is held on the first date, so it is not a period with a return.
    signal, returns = _constant_book()
    strategy = run_backtest(signal, returns)
    assert np.isnan(strategy.iloc[0])
    assert not strategy.iloc[1:].isna().any()
    assert summarize(strategy).n_obs == len(signal) - 1


def test_building_the_book_is_charged():
    # Opening a gross-1.0 book is 1.0 of turnover, paid once, when the position
    # is first held; holding it unchanged costs nothing after that.
    signal, returns = _constant_book()
    strategy = run_backtest(signal, returns, cost_bps=100.0)
    assert strategy.iloc[1] == pytest.approx(-0.01)
    assert (strategy.iloc[2:] == 0.0).all()


def test_days_with_no_position_are_nan_not_flat_days():
    # #25: a lookback's warm-up holds nothing, so those dates are not periods
    # with a return; counting them as 0.0 inflated n_obs and the hit-rate base.
    signal, returns = _constant_book(n_dates=120)
    signal.iloc[:20] = np.nan  # 20-day warm-up: no signal, no position
    strategy = run_backtest(signal, returns)
    assert strategy.iloc[:21].isna().all()  # warm-up plus the one-day lag
    assert not strategy.iloc[21:].isna().any()
    assert summarize(strategy).n_obs == len(signal) - 21


def test_closing_the_book_keeps_its_cost():
    # Going flat is a trade: the exit date carries its cost rather than NaN,
    # and only the dates after it, with nothing held, are NaN.
    signal, returns = _constant_book()
    signal.iloc[40:] = np.nan
    strategy = run_backtest(signal, returns, cost_bps=100.0)
    assert strategy.iloc[41] == pytest.approx(-0.01)  # unwinding 1.0 of gross
    assert strategy.iloc[42:].isna().all()


def test_flat_signal_days_are_zero_not_nan():
    # #25: a day with a real signal but zero net weight is a flat day, 0.0, not
    # a missing one. A row equal across assets demeans to an empty book;
    # dropping those days would inflate a timing overlay's Sharpe.
    signal = _random_panel(n_dates=30, n_assets=5, seed=4)
    returns = _random_panel(n_dates=30, n_assets=5, seed=5) * 0.01
    signal.iloc[10:15] = 1.0  # five equal rows: real signal, zero weight
    strategy = run_backtest(signal, returns, cost_bps=10.0)
    assert np.isnan(strategy.iloc[0])
    assert not strategy.iloc[1:].isna().any()
    assert strategy.iloc[11] < 0  # closing the book is charged
    assert (strategy.iloc[12:16] == 0.0).all()


def test_summarize_short_series_raises():
    returns = pd.Series(np.random.default_rng(0).standard_normal(30) * 0.01)
    with pytest.raises(ValueError):
        summarize(returns)


def test_summarize_returns_backtest_summary_with_sane_bounds():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.standard_normal(300) * 0.01)
    summary = summarize(returns, freq="daily")

    assert isinstance(summary, BacktestSummary)
    assert summary.n_obs == 300
    assert summary.max_drawdown <= 0.0
    assert 0.0 <= summary.hit_rate <= 1.0


def test_summarize_zero_vol_gives_nan_sharpe_not_error():
    returns = pd.Series([0.0] * 100)
    summary = summarize(returns, freq="daily")
    assert summary.vol == 0.0
    assert np.isnan(summary.sharpe)


def test_cost_bps_reduces_mean_return():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=150)
    assets = [f"A{i}" for i in range(8)]
    returns = pd.DataFrame(rng.standard_normal((150, 8)) * 0.01, index=dates, columns=assets)
    signal = pd.DataFrame(rng.standard_normal((150, 8)), index=dates, columns=assets)

    free = run_backtest(signal, returns, cost_bps=0.0)
    costly = run_backtest(signal, returns, cost_bps=50.0)

    assert costly.dropna().mean() < free.dropna().mean()


class TestCandidateReturns:
    def test_each_column_is_that_candidates_backtest(self):
        panel = make_panel(n_dates=80, n_assets=6, n_candidates=4, n_real=1, seed=0)
        matrix = candidate_returns(panel, cost_bps=5.0)
        assert list(matrix.columns) == list(panel.truth.index)
        for name in panel.truth.index:
            expected = run_backtest(panel.signals[name], panel.returns, cost_bps=5.0)
            pd.testing.assert_series_equal(
                matrix[name], expected.iloc[1:], check_names=False, check_freq=False
            )

    def test_drops_the_untraded_first_date_and_leaves_no_gaps(self):
        panel = make_panel(n_dates=80, n_assets=6, n_candidates=3, seed=0)
        matrix = candidate_returns(panel)
        assert matrix.index[0] == panel.returns.index[1]
        assert len(matrix) == 79
        assert not matrix.isna().any().any()

    def test_agrees_with_sweep(self):
        panel = make_panel(n_dates=120, n_assets=6, n_candidates=3, seed=0)
        sharpe = candidate_returns(panel).apply(lambda col: summarize(col).sharpe)
        swept = sweep(panel)["sharpe"]
        # sweep keeps the zero first row; the difference is one observation.
        np.testing.assert_allclose(sharpe.to_numpy(), swept.to_numpy(), rtol=0.05)


# ---------------------------------------------------------------------------
# Per-asset costs and holding periods (backtest_frame)


def _panel(n_dates=60, n_assets=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_dates)
    cols = [f"a{j}" for j in range(n_assets)]
    signal = pd.DataFrame(rng.standard_normal((n_dates, n_assets)), dates, cols)
    returns = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_assets)), dates, cols)
    return signal, returns


def test_a_table_of_one_rate_matches_the_flat_rate():
    signal, returns = _panel()
    flat = backtest_frame(signal, returns, cost_bps=7.0, demean=True, gross=1.0)
    table = pd.DataFrame(7.0, index=signal.index, columns=signal.columns)
    per_asset = backtest_frame(signal, returns, cost_bps=table, demean=True, gross=1.0)
    pd.testing.assert_frame_equal(flat, per_asset)


def test_each_asset_pays_its_own_rate_on_the_trade_date():
    signal, returns = _panel(n_dates=5, n_assets=2)
    signal.loc[:] = [1.0, -1.0]  # one fixed book: the only trade builds it
    rate = pd.DataFrame([[10.0, 30.0]] * 5, index=signal.index, columns=signal.columns)
    rate.iloc[0] = [20.0, 40.0]  # the book is built at the first close
    frame = backtest_frame(signal, returns, cost_bps=rate, demean=True, gross=1.0)
    # Weights +0.5 / -0.5, bought at the first date's rates, charged on the second.
    assert frame["cost"].iloc[1] == pytest.approx(0.5 * 20e-4 + 0.5 * 40e-4)
    assert (frame["cost"].iloc[2:] == 0.0).all()
    np.testing.assert_allclose(frame["net"], frame["gross"] - frame["cost"])


def test_a_missing_rate_pays_that_days_median():
    signal, returns = _panel(n_dates=5, n_assets=3)
    signal.loc[:] = [1.0, 0.0, -1.0]
    rate = pd.DataFrame(10.0, index=signal.index, columns=signal.columns)
    rate.iloc[0] = [np.nan, 20.0, 40.0]
    frame = backtest_frame(signal, returns, cost_bps=rate, demean=True, gross=1.0)
    assert frame["cost"].iloc[1] == pytest.approx(0.5 * 30e-4 + 0.5 * 40e-4)


def test_holding_trades_only_on_rebalance_dates():
    signal, returns = _panel(n_dates=60)
    daily = backtest_frame(signal, returns, cost_bps=10.0, demean=True, gross=1.0)
    held = backtest_frame(signal, returns, cost_bps=10.0, demean=True, gross=1.0, hold=10)
    traded = held["turnover"].fillna(0) > 0
    # A rebalance at date t (every 10th from the first) is charged at t + 1.
    assert set(np.flatnonzero(traded)) <= {1, 11, 21, 31, 41, 51}
    assert held["turnover"].sum() < daily["turnover"].sum()
    with pytest.raises(ValueError, match="hold"):
        backtest_frame(signal, returns, cost_bps=10.0, demean=True, gross=1.0, hold=0)


def test_a_warm_up_is_left_out_when_holding_too():
    signal, returns = _panel(n_dates=40)
    signal.iloc[:12] = np.nan  # no signal for 12 dates
    held = backtest_frame(signal, returns, cost_bps=10.0, demean=True, gross=1.0, hold=5)
    # The first rebalance with a signal is date 15; its book is held from 16.
    assert held["net"].iloc[:16].isna().all()
    assert held["net"].iloc[16:].notna().all()
