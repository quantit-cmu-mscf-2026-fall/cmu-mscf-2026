"""Tests that the stationary bootstrap draws what it says it draws.

Every resampling test built on top of it (Reality Check, Romano-Wolf, bootstrap
Sharpe intervals) inherits its mistakes: blocks of the wrong length understate
dependence, unequal date weights bias the mean, and resampling columns on
different dates erases the correlation the snooping tests exist to respect.
"""

from __future__ import annotations

import numpy as np
import pytest

from capstone.bootstrap import stationary_bootstrap_indices
from capstone.synth import make_return_matrix


def _new_block_share(idx: np.ndarray) -> float:
    """Share of steps that do not continue the previous date's block."""
    n_obs = idx.shape[1]
    continues = idx[:, 1:] == (idx[:, :-1] + 1) % n_obs
    return float(1.0 - continues.mean())


def _lag1_autocorr(values: np.ndarray) -> float:
    """Mean lag-1 autocorrelation across the columns of a dates x candidates array."""
    centred = values - values.mean(axis=0)
    return float(np.mean((centred[1:] * centred[:-1]).sum(axis=0) / (centred**2).sum(axis=0)))


def _mean_pairwise_corr(values: np.ndarray) -> float:
    corr = np.corrcoef(values, rowvar=False)
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


class TestShapeAndRange:
    def test_shape_and_values_are_row_indices(self):
        idx = stationary_bootstrap_indices(250, 10.0, 40, seed=0)
        assert idx.shape == (40, 250)
        assert np.issubdtype(idx.dtype, np.integer)
        assert idx.min() >= 0 and idx.max() < 250

    def test_accepts_an_integer_block_length(self):
        assert stationary_bootstrap_indices(50, 5, 3, seed=0).shape == (3, 50)


class TestReproducibility:
    def test_same_seed_reproduces_and_different_seeds_differ(self):
        first = stationary_bootstrap_indices(300, 8.0, 20, seed=5)
        np.testing.assert_array_equal(first, stationary_bootstrap_indices(300, 8.0, 20, seed=5))
        assert not np.array_equal(first, stationary_bootstrap_indices(300, 8.0, 20, seed=6))

    def test_replicates_differ_from_each_other(self):
        idx = stationary_bootstrap_indices(300, 8.0, 20, seed=0)
        assert len({row.tobytes() for row in idx}) == 20


class TestBlocks:
    @pytest.mark.parametrize("mean_block", [1.0, 5.0, 25.0])
    def test_blocks_restart_at_rate_one_over_mean_block(self, mean_block):
        # A restart lands on the next date by chance with probability 1/n_obs,
        # which looks like a continuation. ~200k steps, so the share is known
        # to well under the 5% tolerance at every block length here.
        n_obs = 500
        idx = stationary_bootstrap_indices(n_obs, mean_block, 400, seed=0)
        expected = (1.0 / mean_block) * (1.0 - 1.0 / n_obs)
        assert _new_block_share(idx) == pytest.approx(expected, rel=0.05)

    def test_long_blocks_wrap_past_the_last_date(self):
        # With restarts all but impossible, each replicate is one block: a
        # rotation of the sample that runs off the end and continues at 0.
        n_obs = 100
        idx = stationary_bootstrap_indices(n_obs, 1e9, 30, seed=0)
        rotation = np.tile(np.arange(n_obs), (30, 1))
        np.testing.assert_array_equal((idx - idx[:, :1]) % n_obs, rotation)

    def test_every_date_is_drawn_equally_often_including_the_ends(self):
        # Without the wrap, the first dates could only be reached by a block
        # starting exactly there and would be drawn about 1/mean_block as often.
        n_obs, n_boot = 200, 5000
        idx = stationary_bootstrap_indices(n_obs, 10.0, n_boot, seed=0)
        counts = np.bincount(idx.ravel(), minlength=n_obs)
        assert counts[:10].mean() == pytest.approx(n_boot, rel=0.15)
        assert counts[-10:].mean() == pytest.approx(n_boot, rel=0.15)


class TestDependenceSurvives:
    """Applied to a matrix, one index row per replicate keeps what the snooping tests need."""

    def test_autocorrelation_survives_blocks_and_not_single_days(self):
        # Only lag-1 pairs that straddle a block boundary lose their
        # dependence, a share 1/mean_block of them: 0.3 * (1 - 1/20) = 0.285.
        values = make_return_matrix(n_obs=2520, n_candidates=20, ar1=0.3, seed=0).returns.to_numpy()

        def resampled_autocorr(mean_block: float) -> float:
            idx = stationary_bootstrap_indices(len(values), mean_block, 50, seed=1)
            return float(np.mean([_lag1_autocorr(values[row]) for row in idx]))

        assert resampled_autocorr(20.0) == pytest.approx(0.285, abs=0.03)
        assert resampled_autocorr(1.0) == pytest.approx(0.0, abs=0.02)

    def test_cross_candidate_correlation_survives(self):
        values = make_return_matrix(n_obs=1000, n_candidates=20, rho=0.5, seed=0).returns.to_numpy()
        idx = stationary_bootstrap_indices(len(values), 10.0, 50, seed=1)
        resampled = np.mean([_mean_pairwise_corr(values[row]) for row in idx])
        assert resampled == pytest.approx(_mean_pairwise_corr(values), abs=0.02)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_obs": 1},
            {"n_obs": 0},
            {"n_obs": 100.0},
            {"n_obs": True},
            {"mean_block": 0.0},
            {"mean_block": -5.0},
            {"mean_block": 0.5},
            {"mean_block": float("nan")},
            {"mean_block": float("inf")},
            {"mean_block": True},
            {"n_boot": 0},
            {"n_boot": -1},
            {"n_boot": 10.5},
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs):
        args = {"n_obs": 100, "mean_block": 5.0, "n_boot": 10, **kwargs}
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(**args, seed=0)
