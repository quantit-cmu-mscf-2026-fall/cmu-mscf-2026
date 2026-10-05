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
        # The p-values and the PSR use the same sqrt(T - 1) (Lauren's #29
        # review): with T in one and T - 1 in the other they differed slightly.
        returns = make_return_matrix(n_candidates=1, n_real=1, sharpe_real=0.8, seed=2).returns
        for benchmark in (0.0, 0.5):
            result = sharpe_test(returns.iloc[:, 0], benchmark=benchmark)
            assert result.pvalue_greater == pytest.approx(1 - result.psr, abs=1e-12)
        result = sharpe_test(returns.iloc[:, 0])
        assert result.sharpe > 0
        assert 1 - result.psr == pytest.approx(result.pvalue / 2, abs=1e-12)


class TestTrackRecordAndPSR:
    def test_min_track_record_is_infinite_below_the_benchmark(self):
        assert min_track_record_length(0.5, benchmark=1.0) == float("inf")

    def test_min_track_record_is_nan_when_the_variance_is_not_positive(self):
        # 1 - skew*SR + (kurtosis-1)/4*SR^2 = 1 - 15 + 4.5 < 0 at SR = 3 per period.
        sharpe = 3.0 * np.sqrt(252)
        assert np.isnan(min_track_record_length(sharpe, skew=5.0))
        assert np.isnan(probabilistic_sharpe_ratio(sharpe, 100, skew=5.0))
        assert np.isnan(min_track_record_length(1.0, variance=0.0))

    def test_min_track_record_is_nan_for_a_nan_sharpe(self):
        assert np.isnan(min_track_record_length(float("nan")))

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

    def test_average_correlation_leaves_out_undefined_pairs(self):
        # A constant trial has no correlation with anything. Counting its pairs
        # as 0 pulled the average toward independence (Lauren's #29 review).
        returns = make_return_matrix(n_candidates=10, rho=0.4, seed=0).returns
        with_constant = returns.assign(flat=0.0)
        assert average_correlation(with_constant) == pytest.approx(average_correlation(returns))
        flat = pd.DataFrame({"a": [0.0] * 5, "b": [1.0] * 5})
        assert np.isnan(average_correlation(flat))

    def test_fractional_trial_counts_are_accepted(self):
        # implied_independent_trials returns a float; it feeds n_trials.
        n = implied_independent_trials(100, 0.3)
        assert n == pytest.approx(70.3)
        between = expected_max_sharpe(n, 1000)
        assert expected_max_sharpe(70, 1000) < between < expected_max_sharpe(71, 1000)
        assert 0 <= deflated_sharpe_ratio(1.5, n, 1000) <= 1


class TestDeflatedSharpeNullCalibration:
    def test_best_of_autocorrelated_nulls_passes_rarely_with_both_corrections(self):
        # Lauren's #29 review: DSR's null calibration on autocorrelated trials.
        # 50 AR(1) nulls (coefficient 0.3, T = 1000), 120 sets, seeds 0-119.
        # By default the IID threshold and IID standard error let the best null
        # through about 10% of the time; with the HAC variance and the trials'
        # own Sharpe spread it should pass no more than the 5% the 0.95 cut-off
        # promises.
        n_obs, n_trials, sets = 1000, 50, 120
        default = corrected = 0
        for seed in range(sets):
            returns = make_return_matrix(n_obs=n_obs, n_candidates=n_trials, ar1=0.3, seed=seed)
            x = returns.returns
            sharpe = x.mean() / x.std(ddof=0) * np.sqrt(252)
            best = sharpe.idxmax()
            default += deflated_sharpe_ratio(float(sharpe[best]), n_trials, n_obs) > 0.95
            corrected += (
                deflated_sharpe_ratio(
                    float(sharpe[best]),
                    n_trials,
                    n_obs,
                    variance=sharpe_variance(x[best], method="hac"),
                    trials_sharpe_variance=float(sharpe.var(ddof=1)),
                )
                > 0.95
            )
        assert corrected / sets <= 0.05
        assert default / sets > corrected / sets


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
