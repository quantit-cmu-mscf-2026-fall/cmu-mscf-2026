"""Synthetic discriminating checks; no historical study inputs are consulted."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.strategy.robust_allocation import (
    AllocationPolicy,
    AllocationPolicyV2,
    optimize_allocation,
    shrunk_covariance,
)


def policy(version=2, **kwargs):
    cls = AllocationPolicyV2 if version == 2 else AllocationPolicy
    return cls(
        periods=12, lookback=120, minimum=60, window=36, step=12, holding_periods=3, **kwargs
    )


def frame(values):
    values = np.asarray(values).reshape(120, -1)
    return pd.DataFrame(
        values,
        index=pd.date_range("1990-01-01", periods=120, freq="MS"),
        columns=[f"s{i}" for i in range(values.shape[1])],
    )


def regime(seed=108, count=2):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.003, 0.015, (10, count)).repeat(12, axis=0)
    return frame(rng.normal(0, 0.03, (120, count)) + means)


def independent_score(x, weights, p, mode, previous=None):
    w = np.asarray(weights)
    previous = np.zeros(len(w)) if previous is None else np.asarray(previous)
    drag = (
        p.ongoing_bps / 10000 * w.sum()
        + p.switch_bps / 10000 * abs(w - previous).sum() / p.holding_periods
    )
    windows = [x[end - p.window : end] for end in range(len(x), p.window - 1, -p.step)]
    if mode == "shrunk_max_sharpe":
        windows = [x]
    return float(
        np.median(
            [
                (p.mean_shrink * z.mean(axis=0) @ w - drag)
                / max(np.sqrt(max(w @ shrunk_covariance(z)[0] @ w, 0)), 1e-12)
                for z in windows
            ]
        )
    )


@pytest.mark.parametrize("mode", ["shrunk_max_sharpe", "median_window_sharpe"])
def test_feasible_unchanged_prior_does_not_false_reject_to_cash(mode):
    x = frame(np.tile([-0.30, 0.30], 60) + 0.0002)
    p = policy(switch_bps=6)
    cov = shrunk_covariance(x.to_numpy())[0]
    keep = p.annual_risk_limit / np.sqrt(cov[0, 0] * p.periods)
    old = optimize_allocation(
        x, "2000-01-01", mode=mode, policy=policy(version=1, switch_bps=6), previous={"s0": keep}
    )
    assert old["status"] == "no_positive_training_edge_cash"
    actual = optimize_allocation(x, "2000-01-01", mode=mode, policy=p, previous={"s0": keep})
    assert actual["status"] == "allocated"
    assert actual["weights"]["s0"] == keep
    assert actual["solver"]["feasible_prior_included"]
    assert actual["training_objective"] > 0
    assert actual["training_objective"] == pytest.approx(
        independent_score(x.to_numpy(), [keep], p, mode, [keep]), abs=1e-14
    )


@pytest.mark.parametrize("mode", ["shrunk_max_sharpe", "median_window_sharpe"])
def test_returned_coefficients_are_exactly_scored_with_risk_costs_and_prior(mode):
    x = regime(404)
    p = policy(annual_risk_limit=0.025, ongoing_bps=1, switch_bps=6)
    prior = {"s0": 0.06, "s1": 0.05}
    actual = optimize_allocation(x, "2000-01-01", mode=mode, policy=p, previous=prior)
    assert actual["status"] == "allocated"
    weights = np.array(list(actual["weights"].values()))
    assert weights.min() >= 0 and weights.sum() <= 1 + 1e-12
    risk = np.sqrt(weights @ shrunk_covariance(x.to_numpy())[0] @ weights * p.periods)
    assert risk <= p.annual_risk_limit + 1e-12
    assert actual["estimated_annual_volatility"] == pytest.approx(risk)
    assert actual["training_objective"] == pytest.approx(
        independent_score(x.to_numpy(), weights, p, mode, [0.06, 0.05]), abs=1e-13
    )
    assert actual["solver"]["global_optimum_guaranteed"] is False


def test_seed_108_median_local_search_miss_is_found_without_history_tuning():
    x = regime()
    old = optimize_allocation(x, "2000-01-01", mode="median_window_sharpe", policy=policy(1))
    actual = optimize_allocation(x, "2000-01-01", mode="median_window_sharpe", policy=policy())
    assert actual["training_objective"] > old["training_objective"] + 0.01
    assert actual["training_objective"] >= 0.2514
    for a in np.linspace(0, 1, 11):
        assert (
            actual["training_objective"]
            >= independent_score(x.to_numpy(), [a, 1 - a], policy(), "median_window_sharpe") - 1e-12
        )
    assert actual["solver"]["lattice_points_evaluated"] == 11
    assert actual["solver"]["lattice_step"] == 0.10
    assert actual["solver"]["global_optimum_guaranteed"] is False


def test_smooth_known_optimum_and_zero_weight_selection_are_preserved():
    rng = np.random.default_rng(404)
    q, _ = np.linalg.qr(rng.normal(size=(120, 3)) - 0.5)
    q, _ = np.linalg.qr(q - q.mean(axis=0))
    x = frame(q * np.sqrt(120) * 0.02 + np.array([0.004, 0.008, -0.01]))
    actual = optimize_allocation(x, "2000-01-01", mode="shrunk_max_sharpe", policy=policy())
    assert actual["weights"]["s0"] == pytest.approx(1 / 3, abs=1e-5)
    assert actual["weights"]["s1"] == pytest.approx(2 / 3, abs=1e-5)
    assert actual["weights"]["s2"] == 0


@pytest.mark.parametrize("mode", ["shrunk_max_sharpe", "median_window_sharpe"])
def test_every_declared_feasible_lattice_point_is_no_better_than_returned_score(mode):
    x = regime(407, 3)
    p = policy(annual_risk_limit=0.02, switch_bps=6, ongoing_bps=1)
    prior = {"s0": 0.02, "s1": 0.02, "s2": 0.02}
    actual = optimize_allocation(x, "2000-01-01", mode=mode, policy=p, previous=prior)
    covariance = shrunk_covariance(x.to_numpy())[0]
    all_scores = []
    for a in range(11):
        for b in range(11 - a):
            q = np.array([a, b, 10 - a - b]) / 10
            risk = np.sqrt(q @ covariance @ q * p.periods)
            w = q * min(1, p.annual_risk_limit / max(risk, 1e-12))
            all_scores.append(independent_score(x.to_numpy(), w, p, mode, list(prior.values())))
    assert actual["solver"]["lattice_points_evaluated"] == len(all_scores) == 66
    assert actual["solver"]["best_lattice_objective"] == pytest.approx(max(all_scores))
    assert actual["solver"]["best_feasible_objective"] >= max(all_scores) - 1e-12


def test_bounded_lattice_reports_skip_instead_of_global_optimum_claim():
    x = regime(404, 8)
    actual = optimize_allocation(x, "2000-01-01", mode="shrunk_max_sharpe", policy=policy())
    solver = actual["solver"]
    assert solver["lattice_points_required"] == 19448
    assert solver["maximum_lattice_points"] == 10000
    assert solver["lattice_points_evaluated"] == 0
    assert solver["lattice_status"] == "skipped_exceeds_point_bound"
    assert solver["global_optimum_guaranteed"] is False


@pytest.mark.parametrize("mode", ["equal_weight", "shrunk_max_sharpe", "median_window_sharpe"])
def test_future_and_row_column_permutations_do_not_change_v2_decisions(mode):
    x = regime(404)
    expected = optimize_allocation(x, "1998-01-01", mode=mode, policy=policy())
    x.loc[x.index >= "1998-01-01"] = 999
    actual = optimize_allocation(x.iloc[::-1, ::-1], "1998-01-01", mode=mode, policy=policy())
    assert actual == expected


@pytest.mark.parametrize("mode", ["shrunk_max_sharpe", "median_window_sharpe"])
def test_bad_or_flat_candidates_choose_cash_without_inventing_cash_sharpe(mode):
    for x in [-regime(404).abs(), regime(404) * 0]:
        actual = optimize_allocation(x, "2000-01-01", mode=mode, policy=policy())
        assert actual["status"] == "no_positive_training_edge_cash"
        assert sum(actual["weights"].values()) == 0
        assert actual["training_objective"] is None
        assert actual["cash_comparison"]["sharpe"] is None


def test_feasible_prior_preserves_unequal_duplicate_holdings_without_fictitious_switch():
    x = frame(np.tile([-0.30, 0.30], 60) + 0.0002)
    x["copy"] = x.s0
    p = policy(switch_bps=6)
    keep = p.annual_risk_limit / np.sqrt(shrunk_covariance(x[["s0"]].to_numpy())[0][0, 0] * 12)
    prior = {"s0": keep * 0.9, "copy": keep * 0.1}
    actual = optimize_allocation(
        x, "2000-01-01", mode="shrunk_max_sharpe", policy=p, previous=prior
    )
    assert actual["weights"] == prior
    assert actual["solver"]["selected_candidate"] == "prior_no_trade"


def test_feasible_lattice_remains_usable_if_all_local_searches_fail(monkeypatch):
    monkeypatch.setattr(
        "capstone.strategy.robust_allocation.minimize",
        lambda *a, **kw: SimpleNamespace(success=False),
    )
    actual = optimize_allocation(
        regime(), "2000-01-01", mode="median_window_sharpe", policy=policy()
    )
    assert actual["status"] == "allocated"
    assert actual["solver"]["local_successes"] == 0
    assert actual["solver"]["selected_candidate"] == "lattice"


def test_legacy_policy_serialization_and_frozen_identity_are_unchanged():
    p = policy(1)
    assert asdict(p) == {
        "periods": 12,
        "lookback": 120,
        "minimum": 60,
        "window": 36,
        "step": 12,
        "mean_shrink": 0.5,
        "annual_risk_limit": 0.1,
        "ongoing_bps": 0,
        "switch_bps": 0,
        "holding_periods": 3,
    }
    x = frame(np.tile([-0.01, 0.03], 60)).rename(columns={"s0": "a"})
    actual = optimize_allocation(x, "2000-01-01", mode="equal_weight", policy=p)
    assert actual["schema_version"] == 3
    assert "solver" not in actual
    assert actual["id"] == "0cc521f24acf7b831010ad4f7def98786c0cb9f248f4f83188505ed7d94bc601"


def test_new_policy_explicit_version_and_boundary_validation():
    assert asdict(policy())["policy_version"] == 2
    with pytest.raises(ValueError, match="version"):
        AllocationPolicyV2(policy_version=True)
    x = regime()
    x.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        optimize_allocation(x, "2000-01-01", mode="shrunk_max_sharpe", policy=policy())
