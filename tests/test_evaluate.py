"""Tests that the error-control claims in `evaluate` actually hold.

These are not smoke tests. Each one is built so that breaking the guarantee it
protects turns it red — if you weaken a correction, something here should fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from capstone.evaluate import (
    average_correlation,
    benjamini_hochberg,
    benjamini_yekutieli,
    bh_adjusted,
    bonferroni,
    by_adjusted,
    deflated_sharpe_ratio,
    estimate_pi0,
    evidence_profile,
    expected_max_sharpe,
    false_discovery_rate,
    haircut_sharpe,
    holm,
    holm_adjusted,
    implied_independent_trials,
    min_track_record_length,
    newey_west_lags,
    power,
    probabilistic_sharpe_ratio,
    sharpe_pvalue,
    sharpe_test,
    sharpe_variance,
    storey_qvalues,
)
from capstone.synth import make_return_matrix


def _uniform_pvalues(n: int, seed: int = 0) -> pd.Series:
    """P-values under a true null are uniform on [0, 1] by construction."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(size=n), index=[f"c{i:04d}" for i in range(n)])


class TestBenjaminiHochberg:
    def test_controls_fdr_under_the_global_null(self):
        # With every hypothesis null, BH at alpha should reject nothing in the
        # large majority of replications. The FWER-like behaviour under the
        # global null is what makes the procedure trustworthy.
        rejections = [
            int(benjamini_hochberg(_uniform_pvalues(200, seed), alpha=0.05).sum())
            for seed in range(50)
        ]
        false_positive_runs = sum(1 for r in rejections if r > 0)
        assert false_positive_runs <= 6, (
            f"{false_positive_runs}/50 runs rejected under the global null; "
            "expected roughly 5% of runs"
        )

    def test_realised_fdr_is_near_alpha_with_real_signals(self):
        # 20 strong signals among 200 candidates. Averaged over replications the
        # realised false-discovery proportion must sit at or below alpha.
        alpha = 0.10
        realised = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            names = [f"c{i:04d}" for i in range(200)]
            truth = pd.Series(False, index=names)
            truth.iloc[:20] = True

            pvalues = pd.Series(rng.uniform(size=200), index=names)
            # Strong signals produce very small p-values.
            pvalues.iloc[:20] = rng.uniform(0, 1e-4, size=20)

            rejected = benjamini_hochberg(pvalues, alpha=alpha)
            fdr = false_discovery_rate(rejected, truth)
            if not np.isnan(fdr):
                realised.append(fdr)

        assert realised, "no replication rejected anything; the test is inert"
        assert np.mean(realised) <= alpha * 1.5

    def test_is_step_up_not_pointwise(self):
        # The step-up rule rejects everything at or below the LARGEST passing
        # rank, including p-values that fail their own threshold.
        #
        # Sorted p:      0.005   0.030   0.036   0.900
        # Threshold:     0.0125  0.025   0.0375  0.050
        # Passes own?    yes     NO      yes     no
        #
        # 'b' fails its own threshold but sits below the largest passing rank
        # ('c'), so a correct step-up sweeps it in. A pointwise implementation
        # returns {a, c} and this test goes red — which is the point of it.
        pvalues = pd.Series([0.005, 0.030, 0.036, 0.9], index=["a", "b", "c", "d"])
        rejected = benjamini_hochberg(pvalues, alpha=0.05)
        assert set(rejected[rejected].index) == {"a", "b", "c"}

    def test_nan_pvalues_are_never_rejected(self):
        pvalues = pd.Series([0.001, np.nan, 0.002], index=["a", "b", "c"])
        rejected = benjamini_hochberg(pvalues, alpha=0.05)
        assert not rejected["b"]

    def test_empty_and_all_nan_inputs(self):
        assert int(benjamini_hochberg(pd.Series(dtype=float)).sum()) == 0
        all_nan = pd.Series([np.nan, np.nan], index=["a", "b"])
        assert int(benjamini_hochberg(all_nan).sum()) == 0

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_alpha(self, alpha):
        with pytest.raises(ValueError):
            benjamini_hochberg(_uniform_pvalues(10), alpha=alpha)


class TestBonferroni:
    def test_is_more_conservative_than_bh(self):
        # Bonferroni bounds the chance of ANY false rejection, so it can never
        # reject more than BH at the same alpha.
        for seed in range(20):
            pvalues = _uniform_pvalues(100, seed)
            pvalues.iloc[:5] = pvalues.iloc[:5] / 1000
            assert int(bonferroni(pvalues).sum()) <= int(benjamini_hochberg(pvalues).sum())

    def test_threshold_is_alpha_over_family_size(self):
        pvalues = pd.Series([0.004, 0.006], index=["a", "b"])
        rejected = bonferroni(pvalues, alpha=0.01)
        assert bool(rejected["a"]) and not bool(rejected["b"])

    def test_nan_counts_toward_family_size(self):
        # A candidate that failed to produce a p-value was still a trial.
        # 0.007 discriminates: it clears alpha/1 = 0.01 (NaN dropped) but not
        # alpha/2 = 0.005 (NaN counted). A p-value below 0.005 would be
        # rejected either way and prove nothing.
        with_nan = pd.Series([0.007, np.nan], index=["a", "b"])
        assert not bool(bonferroni(with_nan, alpha=0.01)["a"])
        without_nan = pd.Series([0.007], index=["a"])
        assert bool(bonferroni(without_nan, alpha=0.01)["a"])


class TestSharpeInference:
    def test_pvalue_is_small_for_a_strong_sharpe(self):
        assert sharpe_pvalue(2.0, n_obs=2520) < 0.01

    def test_pvalue_is_large_for_a_weak_sharpe(self):
        assert sharpe_pvalue(0.05, n_obs=252) > 0.5

    def test_pvalue_shrinks_with_more_observations(self):
        assert sharpe_pvalue(1.0, n_obs=2520) < sharpe_pvalue(1.0, n_obs=252)

    def test_nan_sharpe_propagates(self):
        assert np.isnan(sharpe_pvalue(float("nan"), n_obs=252))

    def test_expected_max_grows_with_trials(self):
        # This is the whole lesson: searching harder raises the bar.
        assert expected_max_sharpe(1000, 1260) > expected_max_sharpe(10, 1260)

    def test_expected_max_is_zero_for_a_single_trial(self):
        assert expected_max_sharpe(1, 1260) == 0.0


class TestDeflatedSharpe:
    def test_falls_as_trials_increase(self):
        few = deflated_sharpe_ratio(1.5, n_trials=10, n_obs=1260)
        many = deflated_sharpe_ratio(1.5, n_trials=10_000, n_obs=1260)
        assert many < few

    def test_penalises_negative_skew_and_fat_tails(self):
        clean = deflated_sharpe_ratio(1.5, n_trials=100, n_obs=1260)
        ugly = deflated_sharpe_ratio(1.5, n_trials=100, n_obs=1260, skew=-1.5, kurtosis=8.0)
        assert ugly < clean

    def test_a_lucky_best_of_many_does_not_survive(self):
        # A Sharpe at exactly the expected maximum of the null distribution
        # carries no evidence of skill: the probability should sit near 0.5,
        # nowhere near a 0.95 retention bar.
        n_trials, n_obs = 1000, 1260
        lucky = expected_max_sharpe(n_trials, n_obs)
        assert deflated_sharpe_ratio(lucky, n_trials, n_obs) < 0.6

    def test_nan_propagates(self):
        assert np.isnan(deflated_sharpe_ratio(float("nan"), 10, 1260))


class TestFalseDiscoveryRate:
    def test_undefined_when_nothing_was_rejected(self):
        # Reporting 0.0 here would make a procedure that never fires look
        # perfectly precise. The denominator is empty; the rate is undefined.
        rejected = pd.Series([False, False], index=["a", "b"])
        truth = pd.Series([True, False], index=["a", "b"])
        assert np.isnan(false_discovery_rate(rejected, truth))

    def test_counts_only_false_rejections(self):
        rejected = pd.Series([True, True, False], index=["a", "b", "c"])
        truth = pd.Series([True, False, True], index=["a", "b", "c"])
        assert false_discovery_rate(rejected, truth) == 0.5

    def test_raises_when_ground_truth_is_missing(self):
        rejected = pd.Series([True], index=["a"])
        truth = pd.Series([True], index=["z"])
        with pytest.raises(KeyError):
            false_discovery_rate(rejected, truth)


class TestPower:
    def test_zero_when_nothing_is_found(self):
        rejected = pd.Series([False, False], index=["a", "b"])
        truth = pd.Series([True, True], index=["a", "b"])
        assert power(rejected, truth) == 0.0

    def test_undefined_when_there_is_nothing_to_find(self):
        rejected = pd.Series([True], index=["a"])
        truth = pd.Series([False], index=["a"])
        assert np.isnan(power(rejected, truth))

    def test_a_procedure_that_rejects_nothing_has_no_power(self):
        # The reason FDR must never be reported alone.
        names = [f"c{i}" for i in range(100)]
        truth = pd.Series([i < 10 for i in range(100)], index=names)
        never_fires = pd.Series(False, index=names)
        assert np.isnan(false_discovery_rate(never_fires, truth))
        assert power(never_fires, truth) == 0.0


class TestPaperExamples:
    """Each number here is printed in the paper cited; the functions must reproduce it."""

    def test_lo_2002_table_1_iid_standard_errors(self):
        # Table 1: SE = sqrt((1 + SR^2 / 2) / T), SR per period.
        for sr, t, printed in [(0.5, 12, 0.306), (1.0, 60, 0.158), (3.0, 500, 0.105)]:
            assert np.sqrt((1 + sr**2 / 2) / t) == pytest.approx(printed, abs=5e-4)
        returns = pd.Series(np.r_[np.full(50, 2.0), np.full(50, 0.0)])  # SR exactly 1
        assert sharpe_variance(returns, method="normal") == pytest.approx(1.5)

    def test_bailey_2012_psr_example(self):
        # Section 3: monthly SR 0.458 over two years gives 0.982 assuming
        # normality, 0.913 with the track record's skew and kurtosis, and 0.953
        # over three years (0.9535 unrounded).
        sharpe = 0.458 * np.sqrt(12)
        fat = dict(skew=-2.448, kurtosis=10.164, periods_per_year=12)
        normal = probabilistic_sharpe_ratio(sharpe, 24, periods_per_year=12)
        assert normal == pytest.approx(0.982, abs=5e-4)
        assert probabilistic_sharpe_ratio(sharpe, 24, **fat) == pytest.approx(0.913, abs=5e-4)
        assert probabilistic_sharpe_ratio(sharpe, 36, **fat) == pytest.approx(0.953, abs=1e-3)

    @pytest.mark.parametrize(
        "periods_per_year, skew, kurtosis, years",
        [
            (252, 0.0, 3.0, 2.73),
            (52, 0.0, 3.0, 2.83),
            (12, 0.0, 3.0, 3.24),
            (12, -0.72, 5.78, 4.99),
        ],
    )
    def test_bailey_2012_min_track_record_examples(self, periods_per_year, skew, kurtosis, years):
        # Section 5: annualised SR 2 against 1 at 95%, daily, weekly and
        # monthly, then monthly with the HFR index's skew and kurtosis.
        observations = min_track_record_length(
            2.0, benchmark=1.0, skew=skew, kurtosis=kurtosis, periods_per_year=periods_per_year
        )
        assert observations / periods_per_year == pytest.approx(years, abs=5e-3)

    def test_bailey_2014_dsr_example(self):
        # "A numerical example": N=100, V[SR_n]=1/2 (annualised), T=1250,
        # skew -3, kurtosis 10, SR 2.5, 250 periods a year.
        kwargs = dict(periods_per_year=250, trials_sharpe_variance=0.5)
        threshold = expected_max_sharpe(100, 1250, **kwargs)
        assert threshold / np.sqrt(250) == pytest.approx(0.1132, abs=5e-5)
        dsr = deflated_sharpe_ratio(2.5, 100, 1250, skew=-3, kurtosis=10, **kwargs)
        assert dsr == pytest.approx(0.9004, abs=5e-5)
        # 0.9505 after only N=46 trials, and with normal returns 0.9505 after N=88.
        at_46 = deflated_sharpe_ratio(2.5, 46, 1250, skew=-3, kurtosis=10, **kwargs)
        assert at_46 == pytest.approx(0.9505, abs=5e-5)
        assert deflated_sharpe_ratio(2.5, 88, 1250, **kwargs) == pytest.approx(0.9505, abs=5e-5)

    def test_default_trial_variance_is_the_iid_null_one(self):
        # Without a measured variance the old behaviour holds: V[SR_n] = 1/T
        # per period. On the paper's example that is far too lenient.
        assert expected_max_sharpe(100, 1250, 250) == pytest.approx(
            expected_max_sharpe(100, 1250, 250, trials_sharpe_variance=250 / 1250)
        )
        lenient = deflated_sharpe_ratio(2.5, 100, 1250, skew=-3, kurtosis=10, periods_per_year=250)
        assert lenient > 0.95


def _rejection_rate(method: str, n_obs: int = 1000, n_candidates: int = 1000, **options) -> float:
    returns = make_return_matrix(n_obs=n_obs, n_candidates=n_candidates, seed=0, **options).returns
    pvalues = [sharpe_test(returns[name], method=method).pvalue for name in returns.columns]
    return float(np.mean(np.array(pvalues) < 0.05))


class TestSharpeVarianceCalibration:
    """Null calibration: at a nominal 5%, how often does each method reject a true null?

    1000 null candidates per case, so a correct 5% test lands in about
    [0.036, 0.064]. The bounds leave a little more room than that: across ten
    seeds the calibrated cases ranged 0.040-0.069 and the broken ones started
    at 0.128.
    """

    @pytest.mark.parametrize("method", ["normal", "nonnormal", "hac"])
    def test_every_method_is_calibrated_on_iid_normal_nulls(self, method):
        assert 0.025 <= _rejection_rate(method) <= 0.08

    def test_nonnormal_is_calibrated_under_fat_tails(self):
        assert 0.025 <= _rejection_rate("nonnormal", t_df=5) <= 0.08

    def test_autocorrelation_breaks_the_iid_methods_and_hac_repairs_most_of_it(self):
        # AR(1) at 0.3 inflates the variance of the mean by (1+0.3)/(1-0.3).
        # Mertens has no autocorrelation term, so it over-rejects as badly as
        # the normal formula; this is the claim in sharpe_variance's docstring.
        assert _rejection_rate("normal", ar1=0.3) > 0.12
        assert _rejection_rate("nonnormal", ar1=0.3) > 0.12
        assert _rejection_rate("hac", ar1=0.3) < 0.10

    def test_negative_autocorrelation_makes_the_iid_methods_timid(self):
        assert _rejection_rate("nonnormal", ar1=-0.3) < 0.02
        assert _rejection_rate("hac", ar1=-0.3) > 0.025

    def test_hac_holds_up_under_volatility_clustering(self):
        assert _rejection_rate("hac", ar1=0.3, garch=(0.1, 0.85)) < 0.10

    def test_hac_finds_planted_signal(self):
        # Power: Sharpe 1.0 over ten years with AR(1) 0.3 is about 2.3 standard errors.
        matrix = make_return_matrix(n_candidates=200, n_real=200, sharpe_real=1.0, ar1=0.3, seed=0)
        pvalues = [sharpe_test(matrix.returns[c]).pvalue for c in matrix.returns.columns]
        assert np.mean(np.array(pvalues) < 0.05) > 0.5


class TestSharpeVariance:
    def test_hac_with_no_lags_is_exactly_mertens(self):
        returns = make_return_matrix(n_obs=500, n_candidates=1, n_real=1, t_df=5, seed=3).returns
        series = returns.iloc[:, 0]
        assert sharpe_variance(series, "hac", lags=0) == pytest.approx(
            sharpe_variance(series, "nonnormal"), rel=1e-12
        )

    def test_nonnormal_matches_the_skew_kurtosis_formula(self):
        returns = make_return_matrix(n_obs=800, n_candidates=1, n_real=1, t_df=5, seed=1).returns
        x = returns.iloc[:, 0].to_numpy()
        sr = x.mean() / x.std()
        expected = 1 - stats.skew(x) * sr + (stats.kurtosis(x, fisher=False) - 1) / 4 * sr**2
        assert sharpe_variance(returns.iloc[:, 0], "nonnormal") == pytest.approx(expected)

    def test_default_lags_follow_the_newey_west_rule(self):
        assert newey_west_lags(100) == 4
        assert newey_west_lags(2520) == 8

    def test_zero_variance_gives_nan(self):
        assert np.isnan(sharpe_variance(pd.Series([0.01] * 100)))

    def test_rejects_bad_arguments(self):
        series = pd.Series(np.random.default_rng(0).standard_normal(100))
        with pytest.raises(ValueError):
            sharpe_variance(series, method="bootstrap")
        with pytest.raises(ValueError):
            sharpe_variance(series, lags=-1)
        with pytest.raises(ValueError):
            sharpe_variance(pd.Series([0.01]))

    def test_sharpe_test_psr_agrees_with_its_pvalue(self):
        # The one-sided PSR and the two-sided p-value come from the same z.
        returns = make_return_matrix(n_candidates=1, n_real=1, sharpe_real=0.8, seed=2).returns
        result = sharpe_test(returns.iloc[:, 0])
        assert result.sharpe > 0
        assert 1 - result.psr == pytest.approx(result.pvalue / 2, abs=2e-3)


class TestTrackRecordAndPSR:
    def test_min_track_record_is_infinite_below_the_benchmark(self):
        assert min_track_record_length(0.5, benchmark=1.0) == float("inf")

    def test_psr_at_min_track_record_is_the_confidence_level(self):
        n = min_track_record_length(1.5, benchmark=0.5, skew=-0.5, kurtosis=6.0)
        psr = probabilistic_sharpe_ratio(1.5, n, benchmark=0.5, skew=-0.5, kurtosis=6.0)
        assert psr == pytest.approx(0.95, abs=1e-9)

    def test_variance_override_is_used(self):
        assert probabilistic_sharpe_ratio(1.0, 500, variance=4.0) < probabilistic_sharpe_ratio(
            1.0, 500
        )


class TestIndependentTrials:
    def test_endpoints(self):
        assert implied_independent_trials(100, 0.0) == 100
        assert implied_independent_trials(100, 1.0) == 1
        assert implied_independent_trials(100, -0.2) == 100

    def test_average_correlation_recovers_rho(self):
        returns = make_return_matrix(n_candidates=40, rho=0.4, seed=0).returns
        assert average_correlation(returns) == pytest.approx(0.4, abs=0.05)


def _correlated_pvalues(
    n: int, n_real: int, rho: float, seed: int, *, shift: float = 3.5, mirrored: bool = False
) -> tuple[pd.Series, pd.Series]:
    """One-sided p-values of equicorrelated normal statistics, with known truth.

    Positively correlated one-sided tests are Benjamini & Yekutieli's Case 1:
    PRDS, so BH itself is covered. `mirrored` negates every other statistic,
    which puts negative correlations in the family and leaves that case.
    """
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho) * rng.standard_normal() + np.sqrt(1 - rho) * rng.standard_normal(n)
    if mirrored:
        z[1::2] *= -1
    truth = np.zeros(n, dtype=bool)
    truth[:n_real] = True
    z[truth] += shift
    names = [f"c{i:04d}" for i in range(n)]
    return pd.Series(stats.norm.sf(z), index=names), pd.Series(truth, index=names)


def _mean_fdp(selector, rho: float, *, mirrored: bool = False, reps: int = 300) -> float:
    fdps = []
    for seed in range(reps):
        pvalues, truth = _correlated_pvalues(200, 20, rho, seed, mirrored=mirrored)
        rejected = selector(pvalues)
        fdps.append(
            0.0 if not rejected.any() else float((rejected & ~truth).sum() / rejected.sum())
        )
    return float(np.mean(fdps))


class TestHolm:
    def test_steps_down_and_stops_at_the_first_failure(self):
        # m = 4. Thresholds at alpha = 0.05: 0.0125, 0.0167, 0.025, 0.05.
        # 'c' fails 0.025 and Holm stops there, so 'd' (0.04 < 0.05) is not
        # rejected even though it would pass its own threshold.
        pvalues = pd.Series([0.01, 0.015, 0.03, 0.04], index=["a", "b", "c", "d"])
        assert set(holm(pvalues)[holm(pvalues)].index) == {"a", "b"}

    def test_never_rejects_fewer_than_bonferroni(self):
        for seed in range(20):
            pvalues = _uniform_pvalues(100, seed)
            pvalues.iloc[:8] = pvalues.iloc[:8] / 500
            assert holm(pvalues).sum() >= bonferroni(pvalues).sum()

    def test_controls_familywise_error_under_correlation(self):
        # Any false rejection counts. 400 global-null families at rho 0.5.
        errors = sum(
            bool(holm(_correlated_pvalues(200, 0, 0.5, seed)[0]).any()) for seed in range(400)
        )
        assert errors / 400 <= 0.07

    def test_nan_counts_toward_the_family_and_is_never_rejected(self):
        # With the NaN, m = 2 and the first threshold is 0.005.
        assert not holm(pd.Series([0.007, np.nan], index=["a", "b"]), alpha=0.01).any()
        assert holm(pd.Series([0.004, np.nan], index=["a", "b"]), alpha=0.01)["a"]


class TestBenjaminiYekutieli:
    def test_is_bh_at_alpha_over_the_harmonic_sum(self):
        # m = 4: sum(1/i) = 25/12, so the thresholds are 0.024 * i / 4:
        # 0.006, 0.012, 0.018, 0.024. 'c' passes at 0.0175 and fails at 0.0181.
        pvalues = pd.Series([0.005, 0.011, 0.0175, 0.5], index=["a", "b", "c", "d"])
        assert set(benjamini_yekutieli(pvalues)[benjamini_yekutieli(pvalues)].index) == {
            "a",
            "b",
            "c",
        }
        pvalues["c"] = 0.0181
        assert set(benjamini_yekutieli(pvalues)[benjamini_yekutieli(pvalues)].index) == {"a", "b"}

    def test_never_rejects_more_than_bh(self):
        for seed in range(30):
            pvalues, _ = _correlated_pvalues(300, 30, 0.3, seed)
            assert benjamini_yekutieli(pvalues).sum() <= benjamini_hochberg(pvalues).sum()

    def test_controls_fdr_under_positive_correlation(self):
        assert _mean_fdp(benjamini_yekutieli, rho=0.5) <= 0.05

    def test_controls_fdr_with_negative_correlations(self):
        # Mirror-image candidates are outside BH's theorem but inside BY's.
        # (In these simulations BH also stays near 0.03-0.05 here; BY's
        # guarantee is what it buys, not a difference this setup shows.)
        assert _mean_fdp(benjamini_yekutieli, rho=0.5, mirrored=True) <= 0.05

    def test_bh_is_covered_for_one_sided_positively_correlated_tests(self):
        # Theorem 1.2: BH controls FDR at (m0 / m) alpha under PRDS.
        assert _mean_fdp(benjamini_hochberg, rho=0.5) <= 0.05 * 180 / 200 + 0.01


class TestPi0:
    def test_near_one_when_every_candidate_is_null(self):
        assert estimate_pi0(_uniform_pvalues(2000)) == pytest.approx(1.0, abs=0.07)

    def test_recovers_the_null_share(self):
        pvalues, _ = _correlated_pvalues(2000, 400, rho=0.0, seed=0, shift=5.0)
        assert estimate_pi0(pvalues) == pytest.approx(0.8, abs=0.06)
        assert estimate_pi0(pvalues, "bootstrap") == pytest.approx(0.8, abs=0.06)

    def test_lambda_zero_is_one(self):
        assert estimate_pi0(_uniform_pvalues(100), lambda_=0.0) == 1.0

    def test_is_capped_at_one(self):
        assert estimate_pi0(pd.Series([0.9, 0.95, 0.99]), lambda_=0.5) == 1.0

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            estimate_pi0(_uniform_pvalues(10), lambda_=1.0)
        with pytest.raises(ValueError):
            estimate_pi0(pd.Series([np.nan]))


class TestStoreyQValues:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
    def test_fdr_version_at_lambda_zero_is_benjamini_hochberg(self, alpha):
        pvalues, _ = _correlated_pvalues(500, 60, 0.2, seed=3)
        q = storey_qvalues(pvalues, lambda_=0.0, pfdr=False)
        pd.testing.assert_series_equal(q <= alpha, benjamini_hochberg(pvalues, alpha))

    def test_monotone_in_p_and_bounded(self):
        pvalues, _ = _correlated_pvalues(500, 60, 0.2, seed=4)
        q = storey_qvalues(pvalues)
        by_p = q[pvalues.sort_values().index].to_numpy()
        assert (np.diff(by_p) >= -1e-15).all()
        assert ((q > 0) & (q <= 1)).all()

    def test_pfdr_q_values_are_at_least_the_fdr_ones(self):
        pvalues, _ = _correlated_pvalues(300, 30, 0.0, seed=5)
        assert (storey_qvalues(pvalues, pfdr=True) >= storey_qvalues(pvalues) - 1e-15).all()

    def test_more_power_than_bh_when_many_candidates_are_real(self):
        # With pi0 = 0.6, BH's implicit pi0 = 1 is conservative.
        gains = []
        for seed in range(20):
            pvalues, _ = _correlated_pvalues(500, 200, 0.0, seed, shift=2.5)
            gains.append(
                (storey_qvalues(pvalues) <= 0.05).sum() - benjamini_hochberg(pvalues).sum()
            )
        assert np.mean(gains) > 0

    def test_nan_stays_nan_and_zero_p_is_finite(self):
        q = storey_qvalues(pd.Series([0.0, np.nan, 0.5, 0.01], index=list("abcd")))
        assert np.isnan(q["b"]) and np.isfinite(q["a"]) and q["a"] <= q["d"]


class TestOneSidedScreening:
    def test_pvalue_greater_is_calibrated_and_halves_the_two_sided_one(self):
        returns = make_return_matrix(n_obs=1000, n_candidates=1000, seed=0).returns
        results = [sharpe_test(returns[c]) for c in returns.columns]
        one_sided = np.array([r.pvalue_greater for r in results])
        assert 0.025 <= np.mean(one_sided < 0.05) <= 0.075
        positive = [r for r in results if r.sharpe > 0]
        assert all(r.pvalue_greater == pytest.approx(r.pvalue / 2) for r in positive)

    def test_bh_and_by_on_correlated_candidates_end_to_end(self):
        # The whole route: correlated strategy returns -> one-sided HAC p-values
        # -> selection, scored against the planted truth.
        fdps = {"bh": [], "by": []}
        for seed in range(10):
            matrix = make_return_matrix(
                n_obs=750, n_candidates=200, n_real=20, sharpe_real=2.5, rho=0.5, seed=seed
            )
            pvalues = pd.Series(
                {c: sharpe_test(matrix.returns[c]).pvalue_greater for c in matrix.returns.columns}
            )
            for name, selector in [("bh", benjamini_hochberg), ("by", benjamini_yekutieli)]:
                fdr = false_discovery_rate(selector(pvalues), matrix.truth)
                fdps[name].append(0.0 if np.isnan(fdr) else fdr)
        assert np.mean(fdps["bh"]) <= 0.08
        assert np.mean(fdps["by"]) <= 0.05


class TestAdjustedPValues:
    """Graded scores: each adjusted p-value is the level at which its procedure rejects."""

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
    def test_thresholding_reproduces_every_decision(self, alpha):
        # Two independent implementations must agree: the decision rules and
        # their adjusted p-values. NaNs included, since the rules count them
        # differently (Holm and BY in m, BH not).
        for seed in range(25):
            pvalues, _ = _correlated_pvalues(150, 30, 0.3, seed)
            pvalues.iloc[[5, 60]] = np.nan
            for adjusted, rule in [
                (holm_adjusted, holm),
                (bh_adjusted, benjamini_hochberg),
                (by_adjusted, benjamini_yekutieli),
            ]:
                pd.testing.assert_series_equal(
                    (adjusted(pvalues) <= alpha), rule(pvalues, alpha), check_names=False
                )

    def test_hand_computed_values(self):
        pvalues = pd.Series([0.01, 0.04, 0.03, 0.2], index=list("abcd"))
        # Holm: running max of (m - i + 1) p over the sorted p-values.
        assert holm_adjusted(pvalues).to_dict() == pytest.approx(
            {"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.2}
        )
        # BH: running min from the top of m p / i.
        assert bh_adjusted(pvalues).to_dict() == pytest.approx(
            {"a": 0.04, "c": 0.0533333, "b": 0.0533333, "d": 0.2}
        )
        assert by_adjusted(pvalues).to_dict() == pytest.approx(
            (bh_adjusted(pvalues) * 25 / 12).clip(upper=1).to_dict()
        )

    def test_ordering_between_procedures(self):
        pvalues, _ = _correlated_pvalues(300, 40, 0.3, seed=1)
        holm_p, by_p, bh_p = holm_adjusted(pvalues), by_adjusted(pvalues), bh_adjusted(pvalues)
        assert (pvalues <= bh_p + 1e-15).all()
        assert (bh_p <= by_p + 1e-15).all()
        assert (bh_p <= holm_p + 1e-15).all()
        assert ((holm_p <= 1) & (by_p <= 1)).all()

    def test_bh_adjusted_is_the_q_value_at_lambda_zero(self):
        pvalues, _ = _correlated_pvalues(200, 20, 0.0, seed=2)
        pd.testing.assert_series_equal(
            bh_adjusted(pvalues), storey_qvalues(pvalues, lambda_=0.0), check_names=False
        )

    def test_empty_and_nan(self):
        empty = pd.Series(dtype=float)
        assert holm_adjusted(empty).empty and bh_adjusted(empty).empty and by_adjusted(empty).empty
        adjusted = holm_adjusted(pd.Series([0.01, np.nan], index=["a", "b"]))
        assert adjusted["a"] == pytest.approx(0.02) and np.isnan(adjusted["b"])


class TestEvidenceProfile:
    def test_columns_order_and_monotone_strength(self):
        pvalues, truth = _correlated_pvalues(200, 20, 0.2, seed=0)
        profile = evidence_profile(pvalues)
        assert list(profile.columns) == ["p", "holm", "by", "bh", "q"]
        assert profile["p"].is_monotonic_increasing
        # Planted candidates are the strongest on every graded score.
        assert profile.index[:10].isin(truth[truth].index).all()

    def test_every_procedure_on_real_sharpe_pvalues(self):
        # The values the pipeline will actually pass: one-sided HAC p-values
        # of correlated strategy returns. The planted candidates lead every column.
        planted = make_return_matrix(
            n_obs=750, n_candidates=200, n_real=10, sharpe_real=3.0, rho=0.5, seed=0
        )
        pvalues = pd.Series(
            {c: sharpe_test(planted.returns[c]).pvalue_greater for c in planted.returns.columns}
        )
        profile = evidence_profile(pvalues)
        real = planted.truth[planted.truth].index
        for column in ["holm", "by", "bh", "q"]:
            strongest = profile[column].sort_values(kind="stable").index[:5]
            assert strongest.isin(real).all(), column


class TestCorrelationBreaksStorey:
    """Storey's estimates assume independence; correlated candidates break them, BH holds.

    Global-null families of 100 one-sided tests with pairwise correlation 0.5,
    400 of them: how often does each column call anything at 5%?
    """

    @staticmethod
    def _any_discovery_rate(column: str, rho: float) -> float:
        score = {"holm": holm_adjusted, "bh": bh_adjusted, "by": by_adjusted}.get(
            column, storey_qvalues
        )
        hits = [
            bool((score(_correlated_pvalues(100, 0, rho, seed)[0]) <= 0.05).any())
            for seed in range(400)
        ]
        return float(np.mean(hits))

    def test_bh_holm_and_by_keep_their_level(self):
        for column in ["holm", "bh", "by"]:
            assert self._any_discovery_rate(column, rho=0.5) <= 0.07, column

    def test_q_values_are_too_permissive_under_correlation(self):
        # Measured over four seed ranges: q 0.095-0.14 against BH 0.025-0.043
        # under correlation, and 0.035-0.063 for q when candidates are independent.
        assert self._any_discovery_rate("q", rho=0.0) <= 0.08
        correlated_q = self._any_discovery_rate("q", rho=0.5)
        assert correlated_q > 0.075
        assert correlated_q > 2 * self._any_discovery_rate("bh", rho=0.5)

    def test_pi0_collapses_under_correlation(self):
        pi0 = np.array(
            [estimate_pi0(_correlated_pvalues(100, 0, 0.5, seed)[0]) for seed in range(400)]
        )
        assert pi0.mean() < 0.85
        assert (pi0 < 0.8).mean() > 0.25


# --------------------------------------------------------------------------- #
# Haircut Sharpe: Harvey & Liu (2015) -- QUANTIT-51, QUANTIT-52
# --------------------------------------------------------------------------- #
# `haircut_sharpe` follows the authors' own MATLAB, not a reading of the paper.
# `scripts/haircut_reference.py` is a literal transcription of `Haircut_SR.m`
# and `sample_random_multests.m`; the values below were produced by running it
# at seed 0 with WW = 2000 (`python scripts/haircut_reference.py`).
#
# What is exact and what is not:
#
#   * `n_monthly_obs`, `pvalue` and everything Bonferroni are CLOSED FORM. They
#     must agree to floating-point precision; a tolerance there would hide a
#     convention error.
#   * Holm and BHY are medians over a simulated family, so they depend on the
#     random stream. The reference draws a full m_tot x m_tot `mvnrnd` and
#     slices; `haircut_sharpe` draws the same distribution through its
#     one-factor representation. Across 30 seeds at WW in {500, 1000, 2000} the
#     measured spread of the surviving Sharpe is sd <= 0.0003 (Holm) and
#     <= 0.0045 (BHY), with a total range span under 0.018. HAIRCUT_ATOL = 0.02
#     clears that on both sides; for scale, the NaN-padding approximation this
#     replaced was out by 0.12 on the same case, so the tolerance is still an
#     order of magnitude tighter than the error it has to detect.
HAIRCUT_ATOL = 0.02

# label, kwargs for haircut_sharpe, then the reference's N / p_raw / adjusted
# Sharpes. `n_trials` is our ledger total; the reference's num_test = n_trials-1.
HAIRCUT_REFERENCE = [
    (
        "monthly, 240 obs, SR 1.0, M=100",
        {"sharpe": 1.0, "n_obs": 240, "n_trials": 101, "frequency": "monthly"},
        {
            "n_monthly_obs": 240,
            "pvalue": 1.1975058908264558e-05,
            "pvalue_bonferroni": 0.0011975058908264558,
            "sharpe_bonferroni": 0.7331736399018519,
            "sharpe_holm": 0.7372469386728714,
            "sharpe_bhy": 0.7605725858413966,
        },
    ),
    (
        "monthly, 240 obs, SR 1.0, M=315",
        {"sharpe": 1.0, "n_obs": 240, "n_trials": 316, "frequency": "monthly"},
        {
            "n_monthly_obs": 240,
            "pvalue": 1.1975058908264558e-05,
            "pvalue_bonferroni": 0.0037721435561033356,
            "sharpe_bonferroni": 0.6541227192699938,
            "sharpe_holm": 0.6588395495840108,
            "sharpe_bhy": 0.739384506502619,
        },
    ),
    (
        "monthly, 120 obs, SR 0.75, M=315",
        {"sharpe": 0.75, "n_obs": 120, "n_trials": 316, "frequency": "monthly"},
        {
            "n_monthly_obs": 120,
            "pvalue": 0.019310820496796444,
            "pvalue_bonferroni": 1.0,
            "sharpe_bonferroni": 0.0,
            "sharpe_holm": 0.0,
            "sharpe_bhy": 0.15108679381138554,
        },
    ),
    (
        "daily, 2520 obs, SR 1.5, M=200",
        {
            "sharpe": 1.5,
            "n_obs": 2520,
            "n_trials": 201,
            "frequency": "daily",
            "avg_correlation": 0.4,
        },
        {
            "n_monthly_obs": 84,
            "pvalue": 0.00015291570299047486,
            "pvalue_bonferroni": 0.03058314059809497,
            "sharpe_bonferroni": 0.8315392511067888,
            "sharpe_holm": 0.844569644584858,
            "sharpe_bhy": 0.995820532226379,
        },
    ),
    (
        "monthly, 240 obs, SR 1.0, M=100, rho=0.1",
        {
            "sharpe": 1.0,
            "n_obs": 240,
            "n_trials": 101,
            "frequency": "monthly",
            "autocorrelation": 0.1,
        },
        {
            "n_monthly_obs": 240,
            "pvalue": 6.148219093260465e-05,
            "pvalue_bonferroni": 0.006148219093260465,
            "sharpe_bonferroni": 0.6181278107351466,
            "sharpe_holm": 0.6235760885229816,
            "sharpe_bhy": 0.657764912798989,
        },
    ),
    (
        "monthly, 240 obs, SR 0.3, M=315",
        {"sharpe": 0.3, "n_obs": 240, "n_trials": 316, "frequency": "monthly"},
        {
            "n_monthly_obs": 240,
            "pvalue": 0.18098583978783234,
            "pvalue_bonferroni": 1.0,
            "sharpe_bonferroni": 0.0,
            "sharpe_holm": 0.0,
            "sharpe_bhy": 0.0008803790037889812,
        },
    ),
]


def _haircut_ids():
    return [label for label, _, _ in HAIRCUT_REFERENCE]


class TestHaircutReferenceComparison:
    """`haircut_sharpe` against a literal transcription of the authors' MATLAB."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [(k, e) for _, k, e in HAIRCUT_REFERENCE],
        ids=_haircut_ids(),
    )
    def test_closed_form_quantities_match_exactly(self, kwargs, expected):
        # Frequency conversion, the two-sided t p-value and Bonferroni involve
        # no simulation. Any disagreement here is a convention error, not noise.
        out = haircut_sharpe(seed=0, **kwargs)
        row = out.iloc[0]

        assert out.attrs["n_monthly_obs"] == expected["n_monthly_obs"]
        assert row["pvalue"] == pytest.approx(expected["pvalue"], rel=1e-10)
        assert row["pvalue_bonferroni"] == pytest.approx(expected["pvalue_bonferroni"], rel=1e-10)
        assert row["sharpe_bonferroni"] == pytest.approx(
            expected["sharpe_bonferroni"], rel=1e-10, abs=1e-12
        )

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [(k, e) for _, k, e in HAIRCUT_REFERENCE],
        ids=_haircut_ids(),
    )
    def test_simulated_adjustments_match_within_monte_carlo_error(self, kwargs, expected):
        out = haircut_sharpe(seed=0, n_simulations=2000, **kwargs)
        row = out.iloc[0]

        assert row["sharpe_holm"] == pytest.approx(expected["sharpe_holm"], abs=HAIRCUT_ATOL)
        assert row["sharpe_bhy"] == pytest.approx(expected["sharpe_bhy"], abs=HAIRCUT_ATOL)

    @pytest.mark.parametrize("seed", [1, 7, 12345])
    def test_the_match_does_not_depend_on_the_committed_seed(self, seed):
        # The reference values were produced at seed 0. If they only reproduce
        # at seed 0, the tolerance is wrong rather than the implementation right.
        _, kwargs, expected = HAIRCUT_REFERENCE[0]
        row = haircut_sharpe(seed=seed, n_simulations=2000, **kwargs).iloc[0]

        assert row["sharpe_holm"] == pytest.approx(expected["sharpe_holm"], abs=HAIRCUT_ATOL)
        assert row["sharpe_bhy"] == pytest.approx(expected["sharpe_bhy"], abs=HAIRCUT_ATOL)


class TestHaircutFollowsTheAuthorsConventions:
    """The conventions that differ from the rest of this module, pinned."""

    def test_the_pvalue_is_two_sided_on_a_t_distribution(self):
        out = haircut_sharpe(1.0, 240, n_trials=101, frequency="monthly", seed=0)
        n_monthly = out.attrs["n_monthly_obs"]
        statistic = 1.0 / np.sqrt(12) * np.sqrt(n_monthly)

        expected = 2.0 * stats.t.sf(statistic, n_monthly - 1)
        assert out.iloc[0]["pvalue"] == pytest.approx(expected, rel=1e-12)

        # Not the normal, and not one-sided: both would be a different method.
        assert out.iloc[0]["pvalue"] != pytest.approx(2.0 * stats.norm.sf(statistic), rel=1e-6)
        assert out.iloc[0]["pvalue"] != pytest.approx(
            stats.t.sf(statistic, n_monthly - 1), rel=1e-6
        )

    @pytest.mark.parametrize(
        ("frequency", "n_obs", "n_monthly"),
        [
            ("daily", 2520, 84),  # a 360-day year, not 252 and not 365
            ("daily", 360, 12),
            ("weekly", 520, 120),
            ("monthly", 240, 240),
            ("quarterly", 80, 240),
            ("annual", 20, 240),
        ],
    )
    def test_observations_are_converted_to_months_on_a_360_day_year(
        self, frequency, n_obs, n_monthly
    ):
        out = haircut_sharpe(
            1.0, n_obs, n_trials=101, frequency=frequency, seed=0, n_simulations=200
        )
        assert out.attrs["n_monthly_obs"] == n_monthly

    def test_the_360_day_year_stays_inside_this_function(self):
        # The rest of the module annualises at 252. This is the regression test
        # for someone "tidying up" the inconsistency in the wrong direction.
        assert sharpe_pvalue(1.0, 2520) == pytest.approx(
            2.0 * stats.norm.sf(1.0 / np.sqrt(252) * np.sqrt(2520))
        )

    def test_a_non_annualised_sharpe_is_annualised_first(self):
        annual = haircut_sharpe(
            1.0, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200
        )
        periodic = haircut_sharpe(
            1.0 / np.sqrt(12),
            240,
            n_trials=101,
            frequency="monthly",
            annualized=False,
            seed=0,
            n_simulations=200,
        )
        assert periodic.iloc[0]["sharpe_annualized"] == pytest.approx(
            annual.iloc[0]["sharpe_annualized"]
        )

    def test_autocorrelation_shrinks_the_sharpe_before_testing(self):
        plain = haircut_sharpe(
            1.0, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200
        )
        serial = haircut_sharpe(
            1.0,
            240,
            n_trials=101,
            frequency="monthly",
            autocorrelation=0.1,
            seed=0,
            n_simulations=200,
        )
        assert serial.iloc[0]["sharpe_annualized"] < plain.iloc[0]["sharpe_annualized"]
        assert serial.iloc[0]["pvalue"] > plain.iloc[0]["pvalue"]

    def test_autocorrelation_does_nothing_at_annual_frequency(self):
        # As in the authors' code: the annual branch returns SR untouched.
        kwargs = {"n_obs": 20, "n_trials": 101, "frequency": "annual", "seed": 0}
        plain = haircut_sharpe(1.0, n_simulations=200, **kwargs)
        serial = haircut_sharpe(1.0, autocorrelation=0.4, n_simulations=200, **kwargs)
        assert serial.iloc[0]["sharpe_annualized"] == pytest.approx(
            plain.iloc[0]["sharpe_annualized"]
        )

    def test_all_four_of_the_authors_adjustments_are_reported(self):
        out = haircut_sharpe(1.0, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200)
        for name in ("bonferroni", "holm", "bhy", "average"):
            for prefix in ("pvalue", "sharpe", "haircut"):
                assert f"{prefix}_{name}" in out.columns

        row = out.iloc[0]
        assert row["pvalue_average"] == pytest.approx(
            (row["pvalue_bonferroni"] + row["pvalue_holm"] + row["pvalue_bhy"]) / 3
        )

    def test_the_haircut_is_the_share_of_the_sharpe_given_up(self):
        row = haircut_sharpe(
            1.0, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200
        ).iloc[0]
        expected = (row["sharpe_annualized"] - row["sharpe_bonferroni"]) / row["sharpe_annualized"]
        assert row["haircut_bonferroni"] == pytest.approx(expected)
        assert 0.0 <= row["haircut_bonferroni"] <= 1.0


class TestHaircutTrialCountConvention:
    """QUANTIT-52 and the M / M+1 split, which is where an off-by-one hides."""

    def test_num_test_is_one_less_than_the_ledger_total(self):
        out = haircut_sharpe(1.0, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200)
        assert out.attrs["n_trials"] == 101
        assert out.attrs["num_test"] == 100

    def test_bonferroni_uses_num_test_not_the_family_size(self):
        # The authors' `p_BON = min(M*p_val,1)` uses M even though Holm and BHY
        # use M+1. Reproducing that inconsistency is the point: with n_trials
        # 101 the multiplier must be 100, not 101.
        out = haircut_sharpe(1.0, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200)
        row = out.iloc[0]

        assert row["pvalue_bonferroni"] == pytest.approx(100 * row["pvalue"], rel=1e-12)
        assert row["pvalue_bonferroni"] != pytest.approx(101 * row["pvalue"], rel=1e-9)

    def test_one_more_trial_shifts_bonferroni_by_exactly_one_multiplier(self):
        kwargs = {
            "n_obs": 240,
            "frequency": "monthly",
            "seed": 0,
            "n_simulations": 200,
        }
        small = haircut_sharpe(1.0, n_trials=101, **kwargs).iloc[0]
        large = haircut_sharpe(1.0, n_trials=102, **kwargs).iloc[0]

        assert large["pvalue_bonferroni"] / small["pvalue_bonferroni"] == pytest.approx(
            101 / 100, rel=1e-10
        )

    def test_a_bigger_search_takes_more_off(self):
        surviving = [
            haircut_sharpe(
                1.0,
                240,
                n_trials=m,
                frequency="monthly",
                seed=0,
                n_simulations=500,
            ).iloc[0]["sharpe_bonferroni"]
            for m in (11, 51, 101, 316, 1001)
        ]
        assert surviving == sorted(surviving, reverse=True)
        assert surviving[0] > surviving[-1]

    def test_trial_count_is_read_from_the_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
        from capstone.runlog import log_run

        for seed in range(25):
            log_run("mom-sweep", params={"lookback": seed}, seed=seed)

        from_ledger = haircut_sharpe(1.0, 240, frequency="monthly", seed=0, n_simulations=200)
        explicit = haircut_sharpe(
            1.0, 240, n_trials=25, frequency="monthly", seed=0, n_simulations=200
        )

        # 25 logged trials: the strategy under test plus 24 others.
        assert from_ledger.attrs["n_trials"] == 25
        assert from_ledger.attrs["num_test"] == 24
        pd.testing.assert_frame_equal(from_ledger, explicit)

    def test_an_experiment_name_scopes_the_count(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
        from capstone.runlog import log_run

        for seed in range(10):
            log_run("mom-sweep", seed=seed)
        for seed in range(4):
            log_run("carry-sweep", seed=seed)

        kwargs = {"n_obs": 240, "frequency": "monthly", "seed": 0, "n_simulations": 200}
        assert haircut_sharpe(1.0, **kwargs).attrs["n_trials"] == 14
        scoped = haircut_sharpe(1.0, experiment="carry-sweep", **kwargs)
        assert scoped.attrs["n_trials"] == 4
        assert scoped.attrs["num_test"] == 3

    def test_an_empty_ledger_raises_instead_of_applying_no_haircut(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

        with pytest.raises(LookupError, match="no trials"):
            haircut_sharpe(1.0, 240, frequency="monthly")


class TestHaircutSimulationControls:
    """The simulated family is reproducible and its size is the caller's choice."""

    def test_the_same_seed_gives_the_same_answer(self):
        kwargs = {
            "n_obs": 240,
            "n_trials": 101,
            "frequency": "monthly",
            "n_simulations": 500,
        }
        first = haircut_sharpe(1.0, seed=42, **kwargs)
        second = haircut_sharpe(1.0, seed=42, **kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_agree_within_monte_carlo_error(self):
        kwargs = {
            "n_obs": 240,
            "n_trials": 101,
            "frequency": "monthly",
            "n_simulations": 1000,
        }
        values = [haircut_sharpe(1.0, seed=s, **kwargs).iloc[0]["sharpe_bhy"] for s in range(6)]
        assert max(values) - min(values) < HAIRCUT_ATOL

    def test_the_repetition_count_is_configurable_and_barely_moves_the_answer(self):
        kwargs = {"n_obs": 240, "n_trials": 101, "frequency": "monthly", "seed": 0}
        few = haircut_sharpe(1.0, n_simulations=200, **kwargs)
        many = haircut_sharpe(1.0, n_simulations=2000, **kwargs)

        assert few.attrs["n_simulations"] == 200
        assert many.attrs["n_simulations"] == 2000
        assert few.iloc[0]["sharpe_bhy"] == pytest.approx(
            many.iloc[0]["sharpe_bhy"], abs=HAIRCUT_ATOL
        )

    def test_an_unset_seed_still_runs(self):
        out = haircut_sharpe(
            1.0, 240, n_trials=101, frequency="monthly", seed=None, n_simulations=200
        )
        assert out.attrs["seed"] is None
        assert np.isfinite(out.iloc[0]["sharpe_bhy"])

    def test_the_average_correlation_selects_the_simulated_distribution(self):
        kwargs = {
            "n_obs": 240,
            "n_trials": 316,
            "frequency": "monthly",
            "seed": 0,
            "n_simulations": 1000,
        }
        low = haircut_sharpe(1.0, avg_correlation=0.0, **kwargs)
        high = haircut_sharpe(1.0, avg_correlation=0.8, **kwargs)

        assert low.attrs["avg_correlation"] == 0.0
        assert high.attrs["avg_correlation"] == 0.8
        # Different simulated families, so different BHY medians.
        assert low.iloc[0]["sharpe_bhy"] != high.iloc[0]["sharpe_bhy"]


class TestHaircutMultipleCandidates:
    def test_several_candidates_share_one_simulated_family(self):
        sharpes = pd.Series({"strong": 1.2, "fair": 0.8, "weak": 0.35})
        out = haircut_sharpe(
            sharpes, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=500
        )

        assert list(out.index) == ["strong", "fair", "weak"]  # sorted by p-value
        assert out["pvalue"].is_monotonic_increasing
        assert (out["sharpe_bonferroni"] <= out["sharpe_annualized"]).all()

    def test_a_non_positive_sharpe_is_not_testable(self):
        # The authors' p_val = 2*(1 - tcdf(T, N-1)) uses T, not |T|, so it
        # exceeds 1 for a negative Sharpe and their method has nothing to say.
        sharpes = pd.Series({"good": 1.0, "flat": 0.0, "bad": -0.4, "missing": np.nan})
        out = haircut_sharpe(
            sharpes, 240, n_trials=101, frequency="monthly", seed=0, n_simulations=200
        )

        for name in ("flat", "bad", "missing"):
            assert np.isnan(out.loc[name, "pvalue"])
            assert np.isnan(out.loc[name, "sharpe_bhy"])
            assert np.isnan(out.loc[name, "haircut_bhy"])
        assert np.isfinite(out.loc["good", "sharpe_bhy"])
        assert out.loc["good", "sharpe"] == 1.0  # the input is still reported


class TestHaircutGuardrails:
    def test_rejects_an_unknown_frequency(self):
        with pytest.raises(ValueError, match="unknown frequency"):
            haircut_sharpe(1.0, 240, n_trials=101, frequency="fortnightly")

    def test_rejects_a_trial_count_below_two(self):
        with pytest.raises(ValueError, match="must be at least 2"):
            haircut_sharpe(1.0, 240, n_trials=1, frequency="monthly")

    def test_rejects_too_few_observations(self):
        with pytest.raises(ValueError, match="n_obs must be at least 1"):
            haircut_sharpe(1.0, 0, n_trials=101, frequency="monthly")

    def test_rejects_a_sample_too_short_to_convert_to_two_months(self):
        with pytest.raises(ValueError, match="needs at least 2"):
            haircut_sharpe(1.0, 30, n_trials=101, frequency="daily")

    def test_rejects_a_non_positive_simulation_count(self):
        with pytest.raises(ValueError, match="n_simulations must be at least 1"):
            haircut_sharpe(1.0, 240, n_trials=101, frequency="monthly", n_simulations=0)

    def test_rejects_an_impossible_autocorrelation(self):
        with pytest.raises(ValueError, match="autocorrelation must be in"):
            haircut_sharpe(1.0, 240, n_trials=101, frequency="monthly", autocorrelation=1.0)
