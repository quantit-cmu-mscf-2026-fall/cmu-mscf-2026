"""Contract of the confirming e-process: has an incubating strategy shown an edge live?

H0: no edge, E[r_t | past] <= 0. The confirming e-process must cross 1/alpha
with probability at most alpha at any time under H0 (normal, t(5), and AR(1)
with the scale inflated by the long-run sd factor), so an incubation-only
strategy with no edge is promoted at most alpha of the time and retired
otherwise by the horizon; its power must grow with the edge; it must be
deterministic and resume exactly from its serialised state. Accepted
strategies keep the decay-only monitor (tests/test_incubation.py).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from capstone.incubation.confirm import (
    CONFIRM_BETS,
    ConfirmConfig,
    ConfirmState,
    confirm,
    confirm_e_process,
    decide_paths,
)
from capstone.incubation.monitor import MonitorConfig, MonitorState, monitor
from capstone.synth import make_return_matrix

ALPHA = 0.05
YEAR = 252
HORIZON = 10 * YEAR  # the null tests look over ten years
N_PATHS = 2_000
# Three Monte Carlo standard errors of a rejection rate at alpha over N_PATHS paths.
MC_TOL = 3 * np.sqrt(ALPHA * (1 - ALPHA) / N_PATHS)


def _config(**overrides) -> ConfirmConfig:
    params = {"scale": 1.0, "alpha": ALPHA, "horizon": HORIZON}
    params.update(overrides)
    return ConfirmConfig(**params)


def _decay(expected_mean: float = 0.0, horizon: int = HORIZON, **overrides) -> MonitorConfig:
    return MonitorConfig(expected_mean=expected_mean, scale=1.0, horizon=horizon, **overrides)


def _shocks(seed: int, n_obs: int = HORIZON, n_paths: int = N_PATHS, **kwargs) -> np.ndarray:
    """Unit-variance, mean-zero shocks, periods x paths, from the shared generator."""
    matrix = make_return_matrix(n_obs=n_obs, n_candidates=n_paths, vol=1.0, seed=seed, **kwargs)
    return matrix.returns.to_numpy()


def _promote_rate(config: ConfirmConfig, returns: np.ndarray) -> float:
    e = confirm_e_process(config, returns)
    return float((e.max(axis=0) >= 1 / config.alpha).mean())


# --- Ville's inequality: false promotion under H0 over the horizon ---------------


@pytest.mark.parametrize(
    ("label", "kwargs", "seed"),
    [("normal", {}, 301), ("t5", {"t_df": 5.0}, 302)],
)
def test_false_promotion_under_no_edge_is_at_most_alpha(label, kwargs, seed):
    rate = _promote_rate(_config(), _shocks(seed, **kwargs))
    print(f"\n{label}: false promotion within {HORIZON} periods {rate:.4f}")
    assert rate <= ALPHA + MC_TOL, f"{label}: promoted {rate:.4f} of null paths"


def test_a_negative_edge_is_also_the_null():
    # H0 is composite (mean <= 0): a losing strategy is promoted no more often.
    returns = -0.05 + _shocks(303)
    assert _promote_rate(_config(), returns) <= ALPHA + MC_TOL


def test_ar1_null_with_the_long_run_scale_is_at_most_alpha():
    # Serial correlation breaks the martingale assumption; the caller inflates the
    # scale by the long-run sd factor sqrt((1 + phi) / (1 - phi)), as for decay.
    phi = 0.2
    returns = _shocks(304, ar1=phi)
    raw = _promote_rate(_config(), returns)
    adjusted = _promote_rate(_config(scale=float(np.sqrt((1 + phi) / (1 - phi)))), returns)
    print(f"\nAR(1) {phi}: false promotion {raw:.4f} unadjusted, {adjusted:.4f} adjusted")
    assert adjusted <= ALPHA + MC_TOL


def test_false_promotion_under_correlated_nulls_is_at_most_alpha():
    # Paths sharing a common factor (`rho`) are each a valid null, so the size per path is
    # unchanged but their crossings move together. Pool independent datasets and judge the
    # mean rate against its own between-dataset spread.
    config = _config(horizon=YEAR)
    rates = []
    for seed in range(40):
        matrix = make_return_matrix(
            n_obs=YEAR, n_candidates=100, vol=1.0, rho=0.5, seed=2_000 + seed
        )
        rates.append(_promote_rate(config, matrix.returns.to_numpy()))
    rates = np.asarray(rates)
    spread = rates.std(ddof=1) / np.sqrt(len(rates))
    assert rates.mean() <= ALPHA + 3 * max(spread, 1e-3), f"promoted {rates.mean():.4f}"
    assert rates.std(ddof=1) > 0  # the datasets differ: the correlation is really there


def test_null_paths_are_retired_by_the_horizon_and_promoted_at_most_alpha():
    # The combined incubation-only decision on a zero-edge market: decay (at an
    # expected mean of 0, as the gate's haircut leaves these) or confirmation, else
    # retire at the horizon. Nothing is left undecided.
    returns = _shocks(305, n_paths=1_000)
    decisions, n_obs = decide_paths(_config(), _decay(), returns)
    assert set(np.unique(decisions)) <= {"promote", "retire"}
    retire = float((decisions == "retire").mean())
    print(f"\nnull: retired {retire:.4f}, promoted {1 - retire:.4f} within {HORIZON}")
    assert retire >= 1 - ALPHA - 3 * np.sqrt(ALPHA * (1 - ALPHA) / 1_000)
    assert (n_obs <= HORIZON).all()


# --- Power: a real edge is promoted ------------------------------------------------

POWER_YEARS = (1, 5, 10, 20)
POWER_SHARPES = (0.0, 0.5, 1.0, 2.0)  # annualised; daily mean = Sharpe / sqrt(252)


def test_power_grows_with_the_edge_and_the_horizon():
    shocks = _shocks(401, n_obs=max(POWER_YEARS) * YEAR, n_paths=500)
    config = _config(horizon=max(POWER_YEARS) * YEAR)
    table = {}
    for sharpe in POWER_SHARPES:
        e = confirm_e_process(config, sharpe / np.sqrt(YEAR) + shocks)
        crossed = np.maximum.accumulate(e >= 1 / ALPHA, axis=0)
        table[sharpe] = [float(crossed[y * YEAR - 1].mean()) for y in POWER_YEARS]
    print("\npromote rate within " + " / ".join(f"{y}y" for y in POWER_YEARS))
    for sharpe, rates in table.items():
        print(f"  Sharpe {sharpe}: " + " / ".join(f"{r:.3f}" for r in rates))
    for i in range(len(POWER_YEARS)):
        column = [table[s][i] for s in POWER_SHARPES]
        assert column == sorted(column)  # monotone in the edge (same shocks)
    for rates in table.values():
        assert rates == sorted(rates)  # and in the horizon
    assert table[0.0][-1] <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / 500)
    assert table[1.0][3] >= 0.80  # Sharpe 1.0 within 20 years
    assert table[2.0][1] >= 0.70  # Sharpe 2.0 within 5 years
    assert table[2.0][1] > table[1.0][1] > table[0.5][1]


# --- The incubation-only decisions -----------------------------------------------


def _path(seed: int = 8, n: int = 900, mean: float = 0.0) -> np.ndarray:
    return mean + np.random.default_rng(seed).standard_normal(n)


def test_promote_when_the_confirming_e_value_reaches_one_over_alpha():
    state = confirm(np.full(2_000, 0.5), ConfirmState.start(_config(), _decay()))
    assert state.decision == "promote" and state.reason == "confirmed"
    assert state.e_value >= 1 / ALPHA
    e = confirm_e_process(_config(), np.full(2_000, 0.5))
    assert state.n_obs == int(np.argmax(e >= 1 / ALPHA)) + 1


def test_retire_when_the_decay_process_crosses():
    state = confirm(np.full(400, -3.0), ConfirmState.start(_config(), _decay()))
    assert state.decision == "retire" and state.reason == "decayed"
    assert state.decay.decision == "retire"
    assert state.n_obs < 400


def test_retire_at_the_horizon_without_confirmation():
    config, decay = _config(horizon=30), _decay(horizon=30)
    state = confirm(np.zeros(40), ConfirmState.start(config, decay))
    assert state.decision == "retire" and state.reason == "not confirmed"
    assert state.n_obs == 30
    early = confirm(np.zeros(10), ConfirmState.start(config, decay))
    assert early.decision == "continue" and early.reason == ""
    with pytest.raises(ValueError, match="retire"):
        state.update(0.0)


def test_the_decay_process_is_the_monitor_on_the_same_returns():
    returns = _path(mean=-0.05)
    decay = _decay(expected_mean=0.01, horizon=10_000)
    state = confirm(returns, ConfirmState.start(_config(horizon=10_000), decay))
    assert state.decay == monitor(returns[: state.n_periods], MonitorState.start(decay))


def test_the_decay_horizon_must_be_the_confirm_horizon():
    with pytest.raises(ValueError, match="horizon"):
        ConfirmState.start(_config(horizon=100), _decay(horizon=252))


def test_one_at_a_time_updates_match_the_batch_e_process_and_decide_paths():
    config = _config(horizon=10_000)
    returns = _path(mean=0.05)
    state = ConfirmState.start(config, _decay(horizon=10_000))
    sequential = []
    for r in returns:
        state = state.update(r)
        sequential.append(state.e_value)
    np.testing.assert_allclose(sequential, confirm_e_process(config, returns), rtol=1e-12, atol=0)

    paths = np.column_stack([_path(seed=s, n=300, mean=m) for s, m in enumerate([0, 0.3, -1, 0])])
    paths[[5, 6, 100], 3] = np.nan
    config, decay = _config(horizon=250), _decay(horizon=250)
    decisions, n_obs = decide_paths(config, decay, paths)
    for j in range(paths.shape[1]):
        one = confirm(paths[:, j], ConfirmState.start(config, decay))
        assert (decisions[j], n_obs[j]) == (one.decision, one.n_obs)


def test_serialise_midway_and_resume_matches_an_uninterrupted_run():
    config, decay = _config(horizon=10_000), _decay(expected_mean=0.01, horizon=10_000)
    returns = _path(mean=0.02)
    whole = confirm(returns, ConfirmState.start(config, decay))

    first = confirm(returns[:300], ConfirmState.start(config, decay))
    restored = ConfirmState.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored == first
    resumed = confirm(returns[300:], restored)
    assert resumed == whole
    assert resumed.e_value == whole.e_value


def test_a_decided_state_round_trips_too():
    state = confirm(np.full(2_000, 0.5), ConfirmState.start(_config(), _decay()))
    assert ConfirmState.from_dict(json.loads(json.dumps(state.to_dict()))) == state


def test_same_input_gives_the_same_e_values():
    returns = _path()
    assert np.array_equal(
        confirm_e_process(_config(), returns), confirm_e_process(_config(), returns.copy())
    )


def test_nan_periods_are_skipped_and_do_not_count_towards_the_horizon():
    config, decay = _config(horizon=5), _decay(horizon=5)
    state = confirm([0.0, np.nan, 0.0, np.nan, 0.0, 0.0], ConfirmState.start(config, decay))
    assert state.decision == "continue"
    assert (state.n_obs, state.n_periods) == (4, 6)
    assert confirm([0.0], state).decision == "retire"
    gappy = _path(n=200)
    gappy[[3, 50]] = np.nan
    e = confirm_e_process(_config(), gappy)
    assert e[3] == e[2]


def test_any_returns_source_is_accepted():
    values = list(_path(n=50))
    start = ConfirmState.start(_config(horizon=100), _decay(horizon=100))
    index = pd.date_range("2021-01-31", periods=50, freq="ME")
    assert (
        confirm(values, start)
        == confirm(pd.Series(values, index=index), start)
        == confirm(np.asarray(values), start)
    )


def test_infinite_returns_raise():
    with pytest.raises(ValueError, match="finite"):
        ConfirmState.start(_config(), _decay()).update(np.inf)


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"scale": 0.0},
        {"scale": np.nan},
        {"horizon": 0},
        {"bets": ()},
        {"bets": (0.5, -0.1)},
    ],
)
def test_bad_configuration_raises(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_config_is_frozen_and_state_rejects_a_bad_dict():
    config = _config()
    with pytest.raises(AttributeError):
        config.alpha = 0.1  # type: ignore[misc]
    assert config.bets == CONFIRM_BETS
    data = ConfirmState.start(config, _decay()).to_dict()
    short = {**data, "log_wealth": data["log_wealth"][:-1]}
    with pytest.raises(ValueError, match="bets"):
        ConfirmState.from_dict(short)
    with pytest.raises(ValueError, match="decision"):
        ConfirmState.from_dict({**data, "decision": "keep"})
