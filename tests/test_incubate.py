"""Run an incubation over a returns series: the scale, the synthetic live series, the decision.

Synthetic returns only; nothing is logged, registered or traded. Which strategies are
incubated, for how long and who acts on the decision are policy questions for the team.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from capstone.incubate import IncubationSettings, incubate, live_continuation, long_run_sd_factor

SETTINGS = IncubationSettings(horizon=60, confirm_horizon=120)
SD = 0.01


def iid(n=1000, mean=0.0, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, SD, n))


def ar1(phi, n=5000, seed=0):
    noise = np.random.default_rng(seed).normal(0.0, SD, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return pd.Series(x)


class TestSettings:
    def test_defaults_are_the_testbed_values(self):
        s = IncubationSettings()
        assert (s.alpha, s.horizon, s.confirm_alpha, s.confirm_horizon) == (0.05, 252, 0.05, 1260)

    @pytest.mark.parametrize(
        "bad",
        [
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"confirm_alpha": 1.5},
            {"horizon": 0},
            {"confirm_horizon": 2.5},
            {"block": True},
        ],
    )
    def test_bad_values_are_refused(self, bad):
        with pytest.raises(ValueError):
            IncubationSettings(**bad)


class TestScale:
    def test_independent_returns_are_not_inflated(self):
        assert long_run_sd_factor(iid(5000)) == pytest.approx(1.0, abs=0.1)

    def test_positive_autocorrelation_inflates_the_scale_like_an_ar1(self):
        phi = 0.5
        expected = np.sqrt((1 + phi) / (1 - phi))  # sqrt(3)
        assert long_run_sd_factor(ar1(phi)) == pytest.approx(expected, rel=0.2)

    def test_negative_autocorrelation_never_shrinks_the_scale(self):
        assert long_run_sd_factor(ar1(-0.5)) == 1.0

    @pytest.mark.parametrize("x", [[], [0.0] * 50, [0.01]])
    def test_degenerate_series_get_the_neutral_factor(self, x):
        assert long_run_sd_factor(pd.Series(x)) == 1.0

    def test_missing_values_are_ignored(self):
        x = iid(500)
        with_gaps = pd.concat([x, pd.Series([np.nan] * 20)], ignore_index=True)
        assert long_run_sd_factor(with_gaps) == long_run_sd_factor(x)


class TestLiveContinuation:
    def test_it_keeps_the_dispersion_and_recentres_on_the_requested_mean(self):
        x = iid(2000, mean=0.003)
        live = live_continuation(x, mean=0.0, periods=1500, seed=1, block=20)
        assert len(live) == 1500
        assert live.mean() == pytest.approx(0.0, abs=0.0006)
        assert live.std() == pytest.approx(x.std(), rel=0.1)

    def test_it_is_deterministic_for_a_seed_and_differs_across_seeds(self):
        x = iid(500)
        a = live_continuation(x, mean=0.0, periods=100, seed=3, block=10)
        assert (a == live_continuation(x, mean=0.0, periods=100, seed=3, block=10)).all()
        assert (a != live_continuation(x, mean=0.0, periods=100, seed=4, block=10)).any()

    def test_blocks_keep_the_serial_correlation(self):
        x = ar1(0.6, n=4000)
        live = pd.Series(live_continuation(x, mean=0.0, periods=4000, seed=0, block=50))
        assert live.autocorr(1) > 0.4
        assert pd.Series(live_continuation(x, mean=0.0, periods=4000, seed=0, block=1)).autocorr(
            1
        ) == pytest.approx(0.0, abs=0.1)


class TestIncubate:
    def test_the_monitor_runs_at_the_gates_mean_and_the_inflated_scale(self):
        search = ar1(0.4, n=3000)
        live = live_continuation(search, mean=0.0, periods=SETTINGS.horizon, seed=0, block=20)
        result = incubate(
            SETTINGS, family="f", expected_mean=0.001, scale=SD, search_returns=search, live=live
        )
        assert result.family == "f" and result.confirm is None
        assert result.config.expected_mean == 0.001
        assert result.lr_factor >= 1.0
        assert result.config.scale == pytest.approx(SD * result.lr_factor)
        assert (result.config.alpha, result.config.horizon) == (SETTINGS.alpha, SETTINGS.horizon)
        assert result.decision == result.state.decision
        assert result.decision in ("continue", "retire", "keep")

    def test_a_decayed_strategy_is_retired(self):
        search = iid(2000, mean=0.002)
        live = np.full(SETTINGS.horizon, -0.03)
        result = incubate(
            SETTINGS,
            family="f",
            expected_mean=0.002,
            scale=SD,
            search_returns=search,
            live=live,
        )
        assert result.decision == "retire"

    def test_a_strategy_with_its_expected_edge_is_not_retired(self):
        search = iid(2000, mean=0.002)
        live = live_continuation(search, mean=0.002, periods=SETTINGS.horizon, seed=1, block=20)
        result = incubate(
            SETTINGS, family="f", expected_mean=0.002, scale=SD, search_returns=search, live=live
        )
        assert result.decision != "retire"

    def test_confirming_runs_both_processes_to_the_confirm_horizon(self):
        search = iid(2000, mean=0.002)
        live = live_continuation(
            search, mean=0.0, periods=SETTINGS.confirm_horizon, seed=2, block=20
        )
        result = incubate(
            SETTINGS,
            family="f",
            expected_mean=0.0,
            scale=SD,
            search_returns=search,
            live=live,
            confirming=True,
        )
        assert result.confirm is not None
        assert result.confirm.config.horizon == SETTINGS.confirm_horizon
        assert result.confirm.config.alpha == SETTINGS.confirm_alpha
        assert result.confirm.config.scale == result.config.scale
        assert result.config.horizon == SETTINGS.confirm_horizon  # the decay monitor too
        assert result.state == result.confirm.decay
        assert result.decision == result.confirm.decision
        assert result.decision in ("continue", "promote", "retire")

    def test_with_no_edge_the_confirmed_decision_is_rarely_promote(self):
        # The anytime-valid test promotes a null strategy with probability at most alpha.
        promoted = 0
        for seed in range(20):
            search = iid(1000, seed=seed)
            live = live_continuation(
                search, mean=0.0, periods=SETTINGS.confirm_horizon, seed=seed, block=20
            )
            result = incubate(
                SETTINGS,
                family="f",
                expected_mean=0.0,
                scale=SD,
                search_returns=search,
                live=live,
                confirming=True,
            )
            promoted += result.decision == "promote"
        assert promoted <= 3  # alpha = 0.05 of 20 worlds, with room for sampling noise

    def test_a_strong_edge_is_promoted(self):
        search = iid(2000, mean=0.01)
        live = live_continuation(
            search, mean=0.01, periods=SETTINGS.confirm_horizon, seed=3, block=20
        )
        result = incubate(
            SETTINGS,
            family="f",
            expected_mean=0.005,
            scale=SD,
            search_returns=search,
            live=live,
            confirming=True,
        )
        assert result.decision == "promote"

    def test_it_is_pure_and_the_result_is_immutable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
        search = iid(500)
        live = live_continuation(search, mean=0.0, periods=20, seed=0, block=5)
        result = incubate(
            SETTINGS, family="f", expected_mean=0.0, scale=SD, search_returns=search, live=live
        )
        assert not (tmp_path / "ledger").exists()  # logs nothing
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.family = "g"
