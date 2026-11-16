"""Contract of the incubation monitor: an anytime-valid test of decay.

H0: the strategy still earns its expected per-period mean. The monitor's
e-process must cross 1/alpha with probability at most alpha at any time under
H0, detect a decayed edge within a stated number of periods, be deterministic,
resume exactly from its serialised state and skip periods with no position.
"""

from __future__ import annotations

import builtins
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from capstone.incubation.monitor import (
    MonitorConfig,
    MonitorState,
    e_process,
    monitor,
)
from capstone.synth import make_return_matrix

ALPHA = 0.05
MEAN = 0.2  # expected per-period mean, in units of the scale: a per-period Sharpe of 0.2
SCALE = 1.0
N_PERIODS = 1_000
N_PATHS = 2_000
# Three Monte Carlo standard errors of a rejection rate at alpha over N_PATHS paths.
MC_TOL = 3 * np.sqrt(ALPHA * (1 - ALPHA) / N_PATHS)


def _config(**overrides) -> MonitorConfig:
    params = {"expected_mean": MEAN, "scale": SCALE, "alpha": ALPHA, "horizon": N_PERIODS}
    params.update(overrides)
    return MonitorConfig(**params)


def _shocks(seed: int, n_paths: int = N_PATHS, **kwargs) -> np.ndarray:
    """Unit-variance, mean-zero shocks, periods x paths, from the shared generator."""
    matrix = make_return_matrix(n_obs=N_PERIODS, n_candidates=n_paths, vol=1.0, seed=seed, **kwargs)
    return matrix.returns.to_numpy()


def _crossing_rate(config: MonitorConfig, returns: np.ndarray, within: int | None = None) -> float:
    e = e_process(config, returns[:within])
    return float((e.max(axis=0) >= 1 / config.alpha).mean())


# --- Ville's inequality: size under H0 over the whole run ---------------------


@pytest.mark.parametrize(
    ("label", "kwargs", "seed"),
    [("normal", {}, 101), ("t5", {"t_df": 5.0}, 102)],
)
def test_ever_crossing_under_the_null_is_at_most_alpha(label, kwargs, seed):
    returns = MEAN + SCALE * _shocks(seed, **kwargs)
    rate = _crossing_rate(_config(), returns)
    assert rate <= ALPHA + MC_TOL, f"{label}: crossed on {rate:.4f} of paths"


def test_a_mean_above_expectation_is_also_the_null():
    # H0 is composite (mean >= expected); a better-than-expected strategy is never retired
    # more often than the boundary case. The scale bounds the second moment about the
    # expected mean, so it carries the extra mean too.
    returns = MEAN + 0.1 + _shocks(103)
    rate = _crossing_rate(_config(scale=float(np.sqrt(1 + 0.1**2))), returns)
    assert rate <= ALPHA + MC_TOL


def test_ar1_autocorrelation_robustness_note():
    # Serial correlation violates the martingale assumption. Measured, not guaranteed:
    # unadjusted AR(1) 0.2 stays within 2 alpha; inflating the scale by the long-run
    # standard deviation factor sqrt((1 + phi) / (1 - phi)) restores alpha.
    phi = 0.2
    returns = MEAN + _shocks(104, ar1=phi)
    raw = _crossing_rate(_config(), returns)
    adjusted = _crossing_rate(_config(scale=float(np.sqrt((1 + phi) / (1 - phi)))), returns)
    assert raw <= 2 * ALPHA
    assert adjusted <= ALPHA + MC_TOL


def test_ever_crossing_under_correlated_nulls_is_at_most_alpha():
    # Paths that share a common factor (the shared generator's `rho`) are each a valid null,
    # so the size per path is unchanged, but their crossings move together and one dataset's
    # rate is noisy. Pool independent datasets and judge the mean against its own spread.
    config = _config(horizon=500)
    rates = []
    for seed in range(40):
        matrix = make_return_matrix(
            n_obs=500, n_candidates=100, vol=1.0, rho=0.5, seed=1_000 + seed
        )
        rates.append(_crossing_rate(config, MEAN + matrix.returns.to_numpy()))
    rates = np.asarray(rates)
    spread = rates.std(ddof=1) / np.sqrt(len(rates))
    assert rates.mean() <= ALPHA + 3 * max(spread, 1e-3), f"mean crossing rate {rates.mean():.4f}"
    assert rates.std(ddof=1) > 0  # the correlation is really there: datasets differ


# --- Power: a decayed edge is retired ------------------------------------------


def test_an_edge_decayed_to_zero_is_retired_within_500_periods():
    returns = 0.0 + _shocks(201, n_paths=1_000)
    assert _crossing_rate(_config(), returns, within=500) >= 0.80


def test_an_edge_turned_negative_is_retired_within_150_periods():
    returns = -MEAN + _shocks(202, n_paths=1_000)
    assert _crossing_rate(_config(), returns, within=150) >= 0.90


# --- Determinism and exact resume ----------------------------------------------


def _path(seed: int = 8, n: int = 600, mean: float = MEAN) -> np.ndarray:
    return mean + np.random.default_rng(seed).standard_normal(n)


def test_same_input_gives_the_same_e_values():
    returns = _path()
    first = e_process(_config(), returns)
    second = e_process(_config(), returns.copy())
    assert np.array_equal(first, second)


def test_one_at_a_time_updates_match_the_batch_e_process():
    config = _config(horizon=10_000)
    returns = _path()
    state = MonitorState.start(config)
    sequential = []
    for r in returns:
        state = state.update(r)
        sequential.append(state.e_value)
    np.testing.assert_allclose(sequential, e_process(config, returns), rtol=1e-12, atol=0)


def test_serialise_midway_and_resume_matches_an_uninterrupted_run():
    config = _config(horizon=10_000)
    returns = _path()
    whole = monitor(returns, MonitorState.start(config))

    first = monitor(returns[:250], MonitorState.start(config))
    restored = MonitorState.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored == first
    resumed = monitor(returns[250:], restored)

    assert resumed == whole
    assert resumed.e_value == whole.e_value


# --- Periods with no position ---------------------------------------------------


def test_nan_periods_are_skipped_not_treated_as_zero():
    config = _config(horizon=10_000)
    returns = _path(n=200)
    gappy = returns.copy()
    gappy[[3, 50, 51, 120]] = np.nan

    skipped = monitor(gappy, MonitorState.start(config))
    dropped = monitor(gappy[~np.isnan(gappy)], MonitorState.start(config))
    zeroed = monitor(np.nan_to_num(gappy, nan=0.0), MonitorState.start(config))

    assert skipped.log_wealth == dropped.log_wealth
    assert skipped.n_obs == dropped.n_obs == 196
    assert skipped.n_periods == 200
    assert zeroed.log_wealth != skipped.log_wealth

    e = e_process(config, gappy)
    assert e[3] == e[2]  # a gap carries the e-value forward
    assert np.array_equal(e[~np.isnan(gappy)], e_process(config, gappy[~np.isnan(gappy)]))


def test_the_horizon_counts_observed_periods_only():
    config = _config(horizon=5)
    state = monitor([MEAN, np.nan, MEAN, np.nan, MEAN, MEAN], MonitorState.start(config))
    assert state.decision == "continue"
    assert state.n_obs == 4
    assert monitor([MEAN], state).decision == "keep"


# --- Decisions --------------------------------------------------------------------


def test_retire_when_the_e_value_reaches_one_over_alpha_and_stop_there():
    config = _config(horizon=10_000)
    returns = np.full(400, -3.0)
    state = monitor(returns, MonitorState.start(config))
    assert state.decision == "retire"
    assert state.e_value >= 1 / ALPHA
    assert state.n_periods < 400  # monitoring stops at the crossing
    e = e_process(config, returns)
    assert state.n_periods == int(np.argmax(e >= 1 / ALPHA)) + 1


def test_keep_at_the_horizon_without_rejection():
    state = monitor(np.full(30, MEAN + 1.0), MonitorState.start(_config(horizon=30)))
    assert state.decision == "keep"
    assert state.e_value < 1 / ALPHA


def test_continue_before_the_horizon_without_rejection():
    state = monitor(np.full(10, MEAN), MonitorState.start(_config(horizon=30)))
    assert state.decision == "continue"
    assert MonitorState.start(_config()).decision == "continue"
    assert MonitorState.start(_config()).e_value == 1.0


def test_updating_after_a_final_decision_raises():
    state = monitor(np.full(30, MEAN), MonitorState.start(_config(horizon=30)))
    with pytest.raises(ValueError, match="keep"):
        state.update(MEAN)


# --- Inputs and configuration ----------------------------------------------------


def test_any_returns_source_is_accepted():
    config = _config(horizon=10_000)
    values = list(_path(n=50))
    index = pd.date_range("2021-01-31", periods=50, freq="ME")
    from_list = monitor(values, MonitorState.start(config))
    from_series = monitor(pd.Series(values, index=index), MonitorState.start(config))
    from_array = monitor(np.asarray(values), MonitorState.start(config))
    assert from_list == from_series == from_array


def test_infinite_returns_raise():
    with pytest.raises(ValueError, match="finite"):
        MonitorState.start(_config()).update(np.inf)


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"scale": 0.0},
        {"scale": np.nan},
        {"horizon": 0},
        {"expected_mean": np.nan},
        {"bets": ()},
        {"bets": (0.5, -0.1)},
    ],
)
def test_bad_configuration_raises(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_config_is_frozen_and_state_rejects_a_mismatched_dict():
    config = _config()
    with pytest.raises(AttributeError):
        config.alpha = 0.1  # type: ignore[misc]
    data = MonitorState.start(config).to_dict()
    data["log_wealth"] = data["log_wealth"][:-1]
    with pytest.raises(ValueError, match="bets"):
        MonitorState.from_dict(data)


def test_config_from_backtest_uses_its_mean_and_standard_deviation():
    backtest = 0.01 + 0.02 * np.random.default_rng(3).standard_normal(500)
    backtest[10] = np.nan
    config = MonitorConfig.from_backtest(backtest, alpha=0.1, horizon=60)
    clean = backtest[~np.isnan(backtest)]
    assert config.expected_mean == pytest.approx(clean.mean(), rel=1e-12)
    assert config.scale == pytest.approx(clean.std(ddof=1), rel=1e-12)
    assert (config.alpha, config.horizon) == (0.1, 60)


def test_the_monitor_reads_no_files(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the monitor opened a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "open", refuse)
    monkeypatch.setattr(pd, "read_parquet", refuse)
    config = _config(horizon=10_000)
    state = monitor(_path(n=100), MonitorState.start(config))
    MonitorState.from_dict(state.to_dict())
    e_process(config, _path(n=100))
