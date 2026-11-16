import numpy as np
import pandas as pd
import pytest

from capstone.strategy.robust_allocation import (
    AllocationPolicy,
    optimize_allocation,
    shrunk_covariance,
)


def policy(**kwargs):
    return AllocationPolicy(
        periods=12, lookback=120, minimum=60, window=36, step=12, holding_periods=3, **kwargs
    )


def observations():
    rng = np.random.default_rng(404)
    q, _ = np.linalg.qr(rng.normal(size=(120, 3)) - 0.5)
    q, _ = np.linalg.qr(q - q.mean(axis=0))
    x = q * np.sqrt(120) * 0.02 + np.array([0.004, 0.008, -0.01])
    return pd.DataFrame(
        x, columns=["a", "b", "bad"], index=pd.date_range("1990-01-01", periods=120, freq="MS")
    )


def test_covariance_matches_independent_outer_product_calculation():
    x = observations().to_numpy()[:65]
    y = x - x.mean(axis=0)
    outer = np.array([np.outer(row, row) for row in y])
    sample = outer.mean(axis=0)
    target = np.eye(3) * np.trace(sample) / 3
    delta = np.square(sample - target).sum()
    beta = np.square(outer - sample).sum() / len(x) ** 2
    rho = min(1, beta / delta)
    actual, shrink = shrunk_covariance(x)
    np.testing.assert_allclose(actual, (1 - rho) * sample + rho * target, atol=1e-15)
    assert shrink == pytest.approx(rho)
    assert np.linalg.eigvalsh(actual).min() >= 0


def test_known_sharpe_optimum_selects_members_and_weights():
    result = optimize_allocation(
        observations(), "2000-01-01", mode="shrunk_max_sharpe", policy=policy()
    )
    assert result["status"] == "allocated"
    assert result["weights"]["bad"] == 0
    assert result["weights"]["a"] == pytest.approx(1 / 3, abs=1e-5)
    assert result["weights"]["b"] == pytest.approx(2 / 3, abs=1e-5)


@pytest.mark.parametrize("mode", ["equal_weight", "shrunk_max_sharpe", "median_window_sharpe"])
def test_future_and_order_invariance_and_duplicate_budget(mode):
    frame = observations()
    before = optimize_allocation(frame, "1998-01-01", mode=mode, policy=policy())
    changed = frame.copy()
    changed.loc[changed.index >= "1998-01-01"] = 99
    assert (
        optimize_allocation(changed.iloc[::-1, ::-1], "1998-01-01", mode=mode, policy=policy())
        == before
    )
    duplicate = frame.assign(copy=frame.a)
    after = optimize_allocation(duplicate, "1998-01-01", mode=mode, policy=policy())
    assert after["weights"]["a"] + after["weights"]["copy"] == pytest.approx(
        before["weights"]["a"], abs=1e-7
    )
    assert after["weights"]["b"] == pytest.approx(before["weights"]["b"], abs=1e-7)


@pytest.mark.parametrize("mode", ["shrunk_max_sharpe", "median_window_sharpe"])
def test_no_edge_and_inadequate_history_hold_cash(mode):
    frame = observations().abs() * -1
    result = optimize_allocation(frame, "2000-01-01", mode=mode, policy=policy())
    assert result["status"] == "no_positive_training_edge_cash"
    assert result["cash_weight"] == 1 and sum(result["weights"].values()) == 0
    assert (
        optimize_allocation(frame.iloc[:10], "2000-01-01", mode=mode, policy=policy())["status"]
        == "insufficient_history_cash"
    )
    flat = frame * 0
    assert optimize_allocation(flat, "2000-01-01", mode=mode, policy=policy())["cash_weight"] == 1


def test_risk_limit_and_costs_affect_allocations():
    frame = observations()
    result = optimize_allocation(
        frame, "2000-01-01", mode="shrunk_max_sharpe", policy=policy(annual_risk_limit=0.01)
    )
    assert result["estimated_annual_volatility"] <= 0.01000001
    assert 0 < sum(result["weights"].values()) < 1
    costly = optimize_allocation(
        frame, "2000-01-01", mode="shrunk_max_sharpe", policy=policy(ongoing_bps=100)
    )
    assert costly["cash_weight"] == 1


def test_missing_inputs_rejected():
    frame = observations()
    frame.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        optimize_allocation(frame, "2000-01-01", mode="equal_weight", policy=policy())


def test_median_uses_portfolio_windows_rather_than_component_sharpes():
    frame = observations()
    result = optimize_allocation(frame, "2000-01-01", mode="median_window_sharpe", policy=policy())
    w = pd.Series(result["weights"]).reindex(frame.columns).to_numpy()
    ratios = []
    for end in range(120, 35, -12):
        x = frame.iloc[end - 36 : end].to_numpy()
        covariance, _ = shrunk_covariance(x)
        ratios.append(0.5 * x.mean(axis=0) @ w / np.sqrt(w @ covariance @ w))
    assert result["training_objective"] == pytest.approx(np.median(ratios))
