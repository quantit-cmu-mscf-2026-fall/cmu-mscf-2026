"""Tests for capstone.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.backtest import BacktestSummary, run_backtest, summarize, to_weights


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
