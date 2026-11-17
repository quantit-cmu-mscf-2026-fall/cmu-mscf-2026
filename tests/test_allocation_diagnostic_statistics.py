import numpy as np
import pandas as pd
import pytest

from capstone.pipeline.allocation_diagnostic_statistics import (
    concentration_diagnostics,
    forecast_audit,
    pool_evidence,
    risk_sharpe_decomposition,
)
from capstone.pipeline.reference_inputs import POOL


def monthly_panel():
    dates = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    values = np.random.default_rng(404).normal(0.003, 0.025, (len(dates), len(POOL)))
    return pd.DataFrame(values, index=dates, columns=POOL)


def forecast_fixture():
    panel = monthly_panel()
    dates = pd.date_range("2000-01-01", periods=6, freq="MS")
    weights = {name: 1 / len(POOL) for name in POOL}
    factor = panel.loc[dates].mean(axis=1)
    wealth = 100000 * (1 + factor).cumprod()
    history = pd.DataFrame(
        {"equity": wealth, "return": factor, "factor_pnl": wealth / (1 + factor) * factor}
    )
    decisions = [{"as_of": str(at.date()), "weights": weights} for at in dates[[0, 3]]]
    return panel, decisions, history


def test_sharpe_decomposition_identity_and_constant_series():
    index = pd.date_range("1996-01-01", periods=24, freq="MS")
    equal = pd.DataFrame({"excess": np.tile([-0.01, 0.02], 12)}, index=index)
    candidate = equal * 1.5 + 0.001
    result = risk_sharpe_decomposition(candidate, equal)
    assert result["identity_residual"] == pytest.approx(0, abs=1e-15)
    assert result["annual_excess_mean_difference"] == pytest.approx(0.042)
    assert result["volatility_term"] + result["sharpe_term"] == pytest.approx(0.042)
    assert result == risk_sharpe_decomposition(candidate.iloc[::-1], equal.iloc[::-1])
    flat = equal * 0
    undefined = risk_sharpe_decomposition(candidate, flat)
    assert undefined["equal_weight"]["sharpe"] is None
    assert undefined["identity_residual"] is None
    with pytest.raises(ValueError, match="identical"):
        risk_sharpe_decomposition(candidate, equal.iloc[:-1])


def test_pool_periods_block_intervals_permutation_and_flat_series():
    panel = monthly_panel()
    result = pool_evidence(panel)
    assert len(result["overall"]) == 5
    assert list(result["five_year_periods"]) == [
        "1996-2000",
        "2001-2005",
        "2006-2010",
        "2011-2015",
        "2016-2020",
    ]
    assert result["overall"]["BAB"]["months"] == 300
    assert result["five_year_periods"]["1996-2000"]["BAB"]["months"] == 60
    assert result["formal_admission"] is False
    assert result == pool_evidence(panel.iloc[::-1, ::-1])
    flat = pool_evidence(panel * 0)
    assert flat["overall"]["HML"]["sharpe"] is None
    assert flat["overall"]["HML"]["annual_excess_mean_95_marginal_interval"] == [0, 0]
    with pytest.raises(ValueError, match="complete"):
        pool_evidence(panel.drop(pd.Timestamp("2000-01-01")))


def test_concentration_signed_contributions_and_cash():
    history = pd.DataFrame(
        {
            "weight_BAB": [0.4, 0],
            "weight_HML": [0.4, 0],
            "factor_pnl_BAB": [15, 0],
            "factor_pnl_HML": [-5, 0],
        },
        index=pd.date_range("1996-01-01", periods=2, freq="MS"),
    )
    result = concentration_diagnostics(history)
    assert result["monthly"][0]["normalized_max_weight"] == 0.5
    assert result["monthly"][0]["effective_number"] == 2
    assert result["monthly"][1]["effective_number"] is None
    assert result["BAB_signed_pnl_fraction"] == 1.5
    assert result["cash_only_months"] == 1
    assert result == concentration_diagnostics(history.iloc[::-1, ::-1])
    history["factor_pnl_HML"] = [-15, 0]
    assert concentration_diagnostics(history)["BAB_signed_pnl_fraction"] is None
    history.loc[history.index[0], "weight_HML"] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        concentration_diagnostics(history)


def test_forecasts_prior_only_and_actual_scoring_independent():
    panel, decisions, history = forecast_fixture()
    result = forecast_audit(panel, decisions, history)
    first = result["forecasts"][0]
    expected = panel.loc[panel.index < "2000-01-01"].tail(120).mean().mean()
    assert first["raw_monthly_mean"] == pytest.approx(expected)
    assert first["half_shrunk_monthly_mean"] == pytest.approx(0.5 * expected)
    assert first["training_observations"] == 120
    assert first["history_end"] == "1999-12-01"
    prior = panel.loc[panel.index < "2000-01-01"].tail(120)
    weights = np.full(len(POOL), 1 / len(POOL))
    expected_variance = weights @ prior.cov().to_numpy() @ weights
    assert first["sample_covariance_annual_volatility"] == pytest.approx(
        np.sqrt(expected_variance * 12)
    )
    assert result["scored_quarters"] == 2
    score = result["following_quarter_scores"][0]
    expected_realized = panel.loc["2000-01-01":"2000-03-01"].mean(axis=1).sum()
    assert score["actual_quarter_arithmetic_factor_pnl_fraction"] == pytest.approx(
        expected_realized
    )
    assert score["raw_quarter_mean_prediction"] == pytest.approx(3 * expected)
    assert score["actual_three_month_annualized_scatter"] == pytest.approx(
        panel.loc["2000-01-01":"2000-03-01"].mean(axis=1).std() * np.sqrt(12)
    )
    changed = panel.copy()
    changed.loc[changed.index >= "2000-01-01"] += 0.1
    changed_result = forecast_audit(changed, decisions, history)
    assert first == changed_result["forecasts"][0]
    assert changed_result["forecasts"][1] != result["forecasts"][1]
    assert score != changed_result["following_quarter_scores"][0]
    changed_history = history.copy()
    changed_history["factor_pnl"] *= 2
    new_scores = forecast_audit(panel, decisions, changed_history)
    assert result["forecasts"] == new_scores["forecasts"]
    assert result["quarter_arithmetic_rmse"] == new_scores["quarter_arithmetic_rmse"]
    assert (
        score["actual_quarter_arithmetic_factor_pnl_fraction"]
        == new_scores["following_quarter_scores"][0][
            "actual_quarter_arithmetic_factor_pnl_fraction"
        ]
    )
    assert (
        score["actual_path_quarter_arithmetic_factor_pnl_fraction"]
        != new_scores["following_quarter_scores"][0][
            "actual_path_quarter_arithmetic_factor_pnl_fraction"
        ]
    )
    assert result == forecast_audit(panel.iloc[::-1, ::-1], decisions[::-1], history.iloc[::-1])


def test_forecast_flat_history_incomplete_quarter_and_invalid_inputs():
    panel, decisions, history = forecast_fixture()
    panel *= 0
    history["return"] = 0.0
    history["equity"] = 100000.0
    history["factor_pnl"] = 0.0
    result = forecast_audit(panel, decisions, history)
    assert result["quarter_arithmetic_rmse"] == {"raw": 0, "half_shrunk": 0, "zero_mean": 0}
    assert result["forecasts"][0]["shrunk_covariance_annual_volatility"] == 0
    partial = forecast_audit(panel.loc[:"2000-05-01"], decisions, history.iloc[:-1])
    assert partial["scored_quarters"] == 1
    assert partial["following_quarter_scores"][1]["status"] == "incomplete_following_quarter"
    early = [{"as_of": "1991-01-01", "weights": decisions[0]["weights"]}]
    assert forecast_audit(panel, early, history)["forecasts"][0]["status"] == (
        "insufficient_training_history"
    )
    initial = [{"as_of": "1996-01-01", "weights": decisions[0]["weights"]}]
    initial_forecast = forecast_audit(panel, initial, history)["forecasts"][0]
    assert initial_forecast["status"] == "forecast_created"
    assert initial_forecast["training_observations"] == 72
    with pytest.raises(ValueError, match="unique"):
        forecast_audit(panel, decisions * 2, history)
    broken = panel.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        forecast_audit(broken, decisions, history)
