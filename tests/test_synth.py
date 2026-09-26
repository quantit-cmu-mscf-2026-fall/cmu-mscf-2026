"""Tests for the synthetic generator.

The generator is the control arm: if it does not plant what it says it plants,
every measurement made against it is meaningless. These tests check the ground
truth is actually true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.backtest import run_backtest, summarize
from capstone.synth import (
    bootstrap_from_real,
    candidate_frames,
    make_panel,
    make_return_matrix,
)


class TestPanelShape:
    def test_counts_candidates_not_columns(self):
        # signals has MultiIndex columns (candidate, asset), so its width is
        # candidates x assets. Reporting that width as the candidate count
        # overstates it by a factor of n_assets.
        panel = make_panel(n_dates=200, n_assets=10, n_candidates=7, seed=0)
        assert panel.n_candidates == 7
        assert panel.signals.shape[1] == 70
        assert "7 candidates" in repr(panel)

    def test_truth_partitions_candidates(self):
        panel = make_panel(n_dates=200, n_assets=5, n_candidates=20, n_real=6, seed=0)
        assert panel.n_true == 6
        assert panel.n_null == 14
        assert panel.n_true + panel.n_null == panel.n_candidates

    def test_returns_and_signals_share_an_index(self):
        panel = make_panel(n_dates=150, n_assets=5, n_candidates=3, seed=0)
        for _, frame, _ in candidate_frames(panel):
            assert frame.index.equals(panel.returns.index)
            assert list(frame.columns) == list(panel.returns.columns)


class TestGroundTruth:
    def test_planted_signals_actually_predict(self):
        # The core claim of the module. If real signals do not outperform null
        # ones, the "known truth" is not known and nothing built on it holds.
        panel = make_panel(
            n_dates=2520,
            n_assets=50,
            n_candidates=40,
            n_real=20,
            effect_size=0.05,
            seed=7,
        )
        real, null = [], []
        for _name, signal, is_real in candidate_frames(panel):
            sharpe = summarize(run_backtest(signal, panel.returns)).sharpe
            (real if is_real else null).append(sharpe)

        assert np.mean(real) > np.mean(null) + 1.0, (
            f"planted signals (mean SR {np.mean(real):.2f}) barely beat nulls "
            f"(mean SR {np.mean(null):.2f}); the generator is not planting anything"
        )

    def test_null_panel_has_no_real_signals(self):
        panel = make_panel(n_dates=500, n_assets=10, n_candidates=50, n_real=0, seed=0)
        assert panel.n_true == 0
        assert not panel.truth.any()

    def test_effect_size_scales_predictive_power(self):
        def mean_sharpe(effect: float) -> float:
            panel = make_panel(
                n_dates=1260,
                n_assets=30,
                n_candidates=12,
                n_real=12,
                effect_size=effect,
                seed=3,
            )
            return float(
                np.mean(
                    [
                        summarize(run_backtest(sig, panel.returns)).sharpe
                        for _, sig, _ in candidate_frames(panel)
                    ]
                )
            )

        assert mean_sharpe(0.08) > mean_sharpe(0.02)

    def test_signals_do_not_leak_the_contemporaneous_return(self):
        # A planted signal must lead the return, not coincide with it. If the
        # generator leaked the current return, a backtest with NO shift would
        # look just as good as the correctly-shifted one.
        panel = make_panel(
            n_dates=1260,
            n_assets=20,
            n_candidates=5,
            n_real=5,
            effect_size=0.05,
            seed=1,
        )
        for _, signal, _ in candidate_frames(panel):
            correlations = signal.corrwith(panel.returns).abs()
            assert correlations.max() < 0.2


class TestReproducibility:
    def test_same_seed_reproduces_the_panel(self):
        a = make_panel(n_dates=100, n_assets=5, n_candidates=4, n_real=2, seed=42)
        b = make_panel(n_dates=100, n_assets=5, n_candidates=4, n_real=2, seed=42)
        pd.testing.assert_frame_equal(a.returns, b.returns)
        pd.testing.assert_series_equal(a.truth, b.truth)

    def test_different_seeds_differ(self):
        a = make_panel(n_dates=100, n_assets=5, n_candidates=4, seed=1)
        b = make_panel(n_dates=100, n_assets=5, n_candidates=4, seed=2)
        assert not a.returns.equals(b.returns)

    def test_params_record_the_settings(self):
        panel = make_panel(n_dates=100, n_assets=5, n_candidates=4, n_real=1, seed=9)
        assert panel.params["seed"] == 9
        assert panel.params["n_real"] == 1


class TestValidation:
    def test_rejects_more_real_than_candidates(self):
        with pytest.raises(ValueError):
            make_panel(n_candidates=5, n_real=6)

    @pytest.mark.parametrize("effect", [-0.1, 1.0, 1.5])
    def test_rejects_invalid_effect_size(self, effect):
        with pytest.raises(ValueError):
            make_panel(effect_size=effect)


class TestBootstrap:
    def test_preserves_shape_and_is_all_null(self):
        source = make_panel(n_dates=300, n_assets=8, n_candidates=2, seed=0).returns
        panel = bootstrap_from_real(source, n_candidates=15, seed=0)
        assert panel.returns.shape == source.shape
        assert panel.n_candidates == 15
        assert panel.n_true == 0

    def test_rejects_empty_returns(self):
        with pytest.raises(ValueError):
            bootstrap_from_real(pd.DataFrame(), n_candidates=5)


def _annual_sharpe(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() / returns.std() * np.sqrt(252)


def _mean_pairwise_corr(returns: pd.DataFrame) -> float:
    corr = returns.corr().to_numpy()
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


class TestReturnMatrix:
    def test_shape_truth_and_params(self):
        matrix = make_return_matrix(n_obs=300, n_candidates=40, n_real=7, seed=0)
        assert matrix.returns.shape == (300, 40)
        assert list(matrix.returns.columns) == list(matrix.truth.index)
        assert (matrix.n_true, matrix.n_null, matrix.n_candidates) == (7, 33, 40)
        assert matrix.params["n_real"] == 7 and matrix.params["seed"] == 0

    def test_same_seed_reproduces_and_different_seeds_differ(self):
        kwargs = dict(n_obs=200, n_candidates=10, n_real=2, rho=0.3, ar1=0.2, garch=(0.1, 0.8))
        first = make_return_matrix(**kwargs, seed=5)
        pd.testing.assert_frame_equal(first.returns, make_return_matrix(**kwargs, seed=5).returns)
        assert not first.returns.equals(make_return_matrix(**kwargs, seed=6).returns)

    def test_planted_sharpe_is_hit_and_nulls_are_centred_on_zero(self):
        # 500 candidates x 10 years: each Sharpe has standard error ~0.32, so the
        # group means are pinned to within a few hundredths.
        matrix = make_return_matrix(n_candidates=500, n_real=50, sharpe_real=1.0, seed=0)
        sharpe = _annual_sharpe(matrix.returns)
        assert sharpe[matrix.truth].mean() == pytest.approx(1.0, abs=0.15)
        assert sharpe[~matrix.truth].mean() == pytest.approx(0.0, abs=0.06)
        # The spread of null Sharpes is what the multiple-testing bar is built on.
        assert sharpe[~matrix.truth].std() == pytest.approx(np.sqrt(252 / 2520), rel=0.1)

    @pytest.mark.parametrize(
        "options",
        [{}, {"rho": 0.5}, {"ar1": 0.4}, {"t_df": 5}, {"garch": (0.1, 0.85)}],
        ids=["plain", "rho", "ar1", "t", "garch"],
    )
    def test_every_option_keeps_the_target_volatility(self, options):
        matrix = make_return_matrix(n_candidates=50, vol=0.02, seed=1, **options)
        assert matrix.returns.std().mean() == pytest.approx(0.02, rel=0.05)

    def test_rho_sets_the_pairwise_correlation(self):
        assert _mean_pairwise_corr(make_return_matrix(n_candidates=50, seed=0).returns) < 0.02
        correlated = make_return_matrix(n_candidates=50, rho=0.4, seed=0).returns
        assert _mean_pairwise_corr(correlated) == pytest.approx(0.4, abs=0.05)

    def test_ar1_sets_the_lag_one_autocorrelation(self):
        def mean_autocorr(options):
            returns = make_return_matrix(n_candidates=50, seed=0, **options).returns
            return float(returns.apply(lambda col: col.autocorr()).mean())

        assert abs(mean_autocorr({})) < 0.02
        assert mean_autocorr({"ar1": 0.3}) == pytest.approx(0.3, abs=0.03)
        assert mean_autocorr({"ar1": -0.3}) == pytest.approx(-0.3, abs=0.03)

    def test_t_df_fattens_the_tails(self):
        plain = make_return_matrix(n_candidates=50, seed=0).returns
        fat = make_return_matrix(n_candidates=50, t_df=5, seed=0).returns
        assert abs(float(plain.kurt().mean())) < 0.2
        assert float(fat.kurt().mean()) > 2.0

    def test_garch_clusters_volatility(self):
        # Squared returns are autocorrelated under GARCH (theory: ~0.18 at lag 1
        # for alpha=0.1, beta=0.85) and not otherwise.
        def squared_autocorr(options):
            returns = make_return_matrix(n_candidates=50, seed=0, **options).returns
            return float((returns**2).apply(lambda col: col.autocorr()).mean())

        assert abs(squared_autocorr({})) < 0.02
        assert squared_autocorr({"garch": (0.1, 0.85)}) > 0.1

    def test_options_do_not_move_null_means(self):
        # A property that shifted the mean would plant signal in the nulls. rho is
        # left out: a shared zero-mean factor cannot shift the mean, but it moves
        # every candidate's realised Sharpe together, so the cross-candidate
        # average stops being a precise check. Here each Sharpe has standard
        # error ~0.43 (ar1 inflates it), so the average of 300 has ~0.025.
        matrix = make_return_matrix(n_candidates=300, ar1=0.3, t_df=5, garch=(0.1, 0.85), seed=0)
        assert _annual_sharpe(matrix.returns).mean() == pytest.approx(0.0, abs=0.1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_obs": 1},
            {"n_candidates": 0},
            {"n_real": 5, "n_candidates": 4},
            {"n_real": -1},
            {"vol": 0.0},
            {"rho": 1.0},
            {"rho": -0.1},
            {"ar1": 1.0},
            {"t_df": 2.0},
            {"garch": (0.5, 0.5)},
            {"garch": (-0.1, 0.5)},
            {"periods_per_year": 0},
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            make_return_matrix(**kwargs)
