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
from capstone.synth import bootstrap_from_real, candidate_frames, make_panel


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


class TestCandidateFrames:
    def test_yields_all_candidates(self):
        panel = make_panel(n_dates=100, n_assets=5, n_candidates=10, seed=0)
        names = [name for name, _, _ in candidate_frames(panel)]
        assert names == list(panel.truth.index)

    def test_real_flag_matches_truth(self):
        panel = make_panel(n_dates=100, n_assets=5, n_candidates=10, n_real=4, seed=0)
        for name, _frame, is_real in candidate_frames(panel):
            assert is_real == panel.truth[name]
