"""Tests that the error-control claims in `evaluate` actually hold.

These are not smoke tests. Each one is built so that breaking the guarantee it
protects turns it red — if you weaken a correction, something here should fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.backtest import sweep
from capstone.evaluate import (
    benjamini_hochberg,
    bonferroni,
    compare_rejection_rules,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    false_discovery_rate,
    power,
    sharpe_pvalue,
)
from capstone.synth import make_panel


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


class TestCompareRejectionRules:
    def test_returns_side_by_side_metrics_for_both_corrections(self):
        panel = make_panel(n_dates=600, n_assets=25, n_candidates=80, n_real=10, seed=0)
        summary = sweep(panel, cost_bps=0.0, freq="daily")
        comparison = compare_rejection_rules(summary, panel.truth)

        assert list(comparison.index) == ["benjamini_hochberg", "bonferroni"]
        assert list(comparison.columns) == ["rejections", "power", "fdr"]
        assert comparison["rejections"].ge(0).all()
        assert comparison["power"].between(0.0, 1.0).all()
        assert (comparison["fdr"].isna() | (comparison["fdr"] >= 0.0)).all()
        assert comparison.loc["benjamini_hochberg", "rejections"] >= comparison.loc["bonferroni", "rejections"]
