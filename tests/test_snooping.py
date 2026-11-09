"""Tests that White's Reality Check says "nothing here" when nothing is there.

The null cases are the point: on a matrix of pure noise the best of many
candidates always looks good, and the p-value has to know that, including when
candidates are correlated, autocorrelated, fat-tailed or volatility-clustered.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from capstone.bootstrap import stationary_bootstrap_indices
from capstone.snooping import RealityCheckResult, white_reality_check
from capstone.synth import make_return_matrix

ALPHA = 0.05
N_NULL_SIMS = 300
# A true 5% test stays under this about 99.9% of the time over N_NULL_SIMS runs.
NULL_CEILING = ALPHA + 3 * math.sqrt(ALPHA * (1 - ALPHA) / N_NULL_SIMS)

# The data and the bootstrap must not share a seed: both would read the same
# random stream, the resampling would depend on the data, and the rate is off.
BOOTSTRAP_SEED_OFFSET = 1_000_000

CASES = {
    "iid": {},
    "rho": {"rho": 0.5},
    "ar1": {"ar1": 0.3},
    "t_df": {"t_df": 4.0},
    "garch": {"garch": (0.1, 0.85)},
}


@pytest.fixture
def matrix() -> pd.DataFrame:
    return make_return_matrix(n_obs=120, n_candidates=6, seed=3).returns


def _null_rejection_rate(options: dict) -> float:
    # Blocks must span the dependence. At ar1=0.3 and block 10 the test is mildly
    # liberal (6.7% over 1000 other seeds, against 5.7% at block 20).
    mean_block = 20.0 if "ar1" in options else 10.0
    rejected = 0
    for sim in range(N_NULL_SIMS):
        returns = make_return_matrix(n_obs=500, n_candidates=20, seed=sim, **options).returns
        result = white_reality_check(
            returns, mean_block=mean_block, n_boot=199, seed=BOOTSTRAP_SEED_OFFSET + sim
        )
        rejected += result.pvalue <= ALPHA
    return rejected / N_NULL_SIMS


class TestOutput:
    def test_shapes_and_types(self, matrix):
        result = white_reality_check(matrix, n_boot=50)
        assert isinstance(result, RealityCheckResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert result.best_candidate in matrix.columns
        assert result.mean_relative_performance.index.equals(matrix.columns)
        assert result.bootstrap_statistics.shape == (50,)

    def test_result_is_frozen(self, matrix):
        result = white_reality_check(matrix, n_boot=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.pvalue = 0.0

    def test_does_not_modify_its_input(self, matrix):
        before = matrix.copy()
        white_reality_check(matrix, n_boot=5)
        pd.testing.assert_frame_equal(matrix, before)

    def test_single_candidate_is_allowed(self, matrix):
        assert white_reality_check(matrix.iloc[:, :1], n_boot=5).best_candidate == matrix.columns[0]


class TestMethodology:
    def test_matches_the_formulas_by_hand(self, matrix):
        n_obs = len(matrix)
        benchmark = pd.Series(np.linspace(-1e-3, 1e-3, n_obs), index=matrix.index)
        result = white_reality_check(matrix, benchmark=benchmark, mean_block=4.0, n_boot=30, seed=9)

        relative = matrix.to_numpy() - benchmark.to_numpy()[:, np.newaxis]
        means = relative.mean(axis=0)
        statistic = np.sqrt(n_obs) * means.max()
        idx = stationary_bootstrap_indices(n_obs, 4.0, 30, seed=9)
        replicates = [np.sqrt(n_obs) * (relative - means)[row].mean(axis=0).max() for row in idx]
        pvalue = (1 + np.sum(np.array(replicates) >= statistic)) / 31

        assert result.statistic == pytest.approx(statistic)
        assert result.best_candidate == matrix.columns[means.argmax()]
        np.testing.assert_allclose(result.mean_relative_performance, means)
        np.testing.assert_allclose(result.bootstrap_statistics, replicates)
        assert result.pvalue == pytest.approx(pvalue)

    def test_float_benchmark_equals_a_constant_series(self, matrix):
        constant = pd.Series(2e-4, index=matrix.index)
        from_float = white_reality_check(matrix, benchmark=2e-4, n_boot=30)
        from_series = white_reality_check(matrix, benchmark=constant, n_boot=30)
        assert from_float.statistic == pytest.approx(from_series.statistic)
        assert from_float.pvalue == from_series.pvalue

    def test_null_distribution_ignores_each_candidates_mean(self, matrix):
        # Centring makes the bootstrap blind to the means: without it, a
        # candidate's own edge would inflate its null and hide the signal.
        shifts = np.arange(matrix.shape[1]) * 1e-3
        shifted = white_reality_check(matrix + shifts, n_boot=30)
        original = white_reality_check(matrix, n_boot=30)
        np.testing.assert_allclose(shifted.bootstrap_statistics, original.bootstrap_statistics)

    def test_pvalue_is_never_zero(self, matrix):
        result = white_reality_check(matrix + 1.0, n_boot=20)
        assert result.pvalue == pytest.approx(1 / 21)


class TestReproducibility:
    def test_same_seed_reproduces_and_different_seeds_differ(self, matrix):
        first = white_reality_check(matrix, n_boot=40, seed=1)
        again = white_reality_check(matrix, n_boot=40, seed=1)
        other = white_reality_check(matrix, n_boot=40, seed=2)
        np.testing.assert_array_equal(first.bootstrap_statistics, again.bootstrap_statistics)
        assert first.pvalue == again.pvalue
        assert not np.array_equal(first.bootstrap_statistics, other.bootstrap_statistics)


class TestNullCalibration:
    @pytest.mark.parametrize("options", CASES.values(), ids=CASES.keys())
    def test_rejects_at_about_alpha_when_nothing_is_real(self, options):
        rate = _null_rejection_rate(options)
        assert rate <= NULL_CEILING
        # A test that never rejects would pass the ceiling and find nothing.
        assert rate >= 0.01


class TestPower:
    @pytest.mark.parametrize("options", CASES.values(), ids=CASES.keys())
    def test_finds_planted_candidates(self, options):
        n_sims = 40
        rejected = best_is_real = 0
        for sim in range(n_sims):
            data = make_return_matrix(
                n_obs=1260, n_candidates=20, n_real=3, sharpe_real=3.0, seed=sim, **options
            )
            result = white_reality_check(data.returns, n_boot=199, seed=BOOTSTRAP_SEED_OFFSET + sim)
            rejected += result.pvalue <= ALPHA
            best_is_real += bool(data.truth[result.best_candidate])
        assert rejected / n_sims >= 0.9
        assert best_is_real / n_sims >= 0.9


class TestValidation:
    def test_rejects_non_dataframe(self, matrix):
        with pytest.raises(ValueError, match="DataFrame"):
            white_reality_check(matrix.to_numpy())

    @pytest.mark.parametrize(
        "bad",
        [
            pd.DataFrame(),
            pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=5)),
            pd.DataFrame(columns=["a", "b"], dtype=float),
        ],
    )
    def test_rejects_empty(self, bad):
        with pytest.raises(ValueError, match="non-empty"):
            white_reality_check(bad)

    def test_rejects_a_single_observation(self, matrix):
        with pytest.raises(ValueError, match="2 observations"):
            white_reality_check(matrix.iloc[:1])

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_performance(self, matrix, bad_value):
        bad = matrix.copy()
        bad.iloc[3, 2] = bad_value
        with pytest.raises(ValueError, match="NaN or infinite"):
            white_reality_check(bad)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    def test_rejects_non_finite_benchmark_series(self, matrix, bad_value):
        bad = pd.Series(0.0, index=matrix.index)
        bad.iloc[3] = bad_value
        with pytest.raises(ValueError, match="NaN or infinite"):
            white_reality_check(matrix, benchmark=bad)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, True, "spy", None])
    def test_rejects_bad_scalar_benchmark(self, matrix, bad):
        with pytest.raises(ValueError, match="benchmark"):
            white_reality_check(matrix, benchmark=bad)

    def test_rejects_misaligned_benchmark(self, matrix):
        short = pd.Series(0.0, index=matrix.index[:-1])
        with pytest.raises(ValueError, match="same index"):
            white_reality_check(matrix, benchmark=short)
        reordered = pd.Series(0.0, index=matrix.index[::-1])
        with pytest.raises(ValueError, match="same index"):
            white_reality_check(matrix, benchmark=reordered)

    @pytest.mark.parametrize("n_boot", [0, -1, 10.5, True])
    def test_rejects_bad_n_boot(self, matrix, n_boot):
        with pytest.raises(ValueError, match="n_boot"):
            white_reality_check(matrix, n_boot=n_boot)

    @pytest.mark.parametrize("mean_block", [0.5, 0.0, float("nan"), float("inf"), True])
    def test_rejects_bad_mean_block(self, matrix, mean_block):
        with pytest.raises(ValueError, match="mean_block"):
            white_reality_check(matrix, mean_block=mean_block)
