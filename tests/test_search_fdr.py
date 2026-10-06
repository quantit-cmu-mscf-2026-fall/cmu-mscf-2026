"""Tests for capstone.search_fdr.

Paper reproductions first (López de Prado & Fabozzi 2026, SSRN 6450418), then
the claims the module makes on data with known truth: the adjusted p-value is
calibrated under the null and keeps power, and the hidden-search experiment of
issue #41 behaves as the paper predicts.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from capstone.evaluate import (
    average_correlation,
    benjamini_hochberg,
    false_discovery_rate,
    power,
    sharpe_test,
)
from capstone.search_fdr import (
    familywise_errors,
    fdr,
    fdr_by_search_intensity,
    fit_max_of_mixture,
    max_of_mixture_cdf,
    search_adjusted_fdr,
    search_adjusted_pvalue,
    search_intensity,
    sharpe_threshold,
    simulate_family_winners,
)
from capstone.synth import make_return_matrix


class TestPaperExamples:
    """Each number here is printed in the paper; the functions must reproduce it."""

    # Section 4.4: N(0, 1) nulls, N(0.3, 1) alternatives, threshold 1.96.
    ALPHA = 1 - norm.cdf(1.96)
    BETA = norm.cdf(1.96 - 0.3)

    def test_case_a_search_and_selection(self):
        # Eqs. 21-23: pi0 = 0.95, K = 10.
        assert self.ALPHA == pytest.approx(0.025, abs=5e-4)
        assert self.BETA == pytest.approx(0.952, abs=5e-4)
        e = familywise_errors(self.ALPHA, self.BETA, 0.95, 10)
        assert e.pi0_k == pytest.approx(0.599, abs=5e-4)
        assert e.alpha_k == pytest.approx(0.224, abs=5e-4)
        assert e.beta_k == pytest.approx(0.753, abs=5e-4)
        assert search_adjusted_fdr(self.ALPHA, self.BETA, 0.95, 10) == pytest.approx(
            0.575, abs=5e-4
        )

    def test_case_b_no_search(self):
        # Eq. 25: pi0 = 0.15, K = 1.
        assert fdr(self.ALPHA, self.BETA, 0.15) == pytest.approx(0.083, abs=5e-4)
        assert search_adjusted_fdr(self.ALPHA, self.BETA, 0.15, 1) == pytest.approx(0.083, abs=5e-4)

    @pytest.mark.parametrize(
        "x, case_a, case_b",
        [
            (2.00, 0.920, 0.919),
            (2.50, 0.275, 0.284),
            (3.00, 0.062, 0.070),
            (3.50, 0.011, 0.014),
            (4.00, 0.002, 0.002),
        ],
    )
    def test_identical_upper_tails_table(self, x, case_a, case_b):
        # Section 4.4.3's table: P[X >= x | X >= c] under both cases. The two
        # worlds report the same tail, yet their FDRs are 0.575 and 0.083.
        def cond_tail(k, pi0):
            tail = lambda v: 1 - max_of_mixture_cdf(v, k, pi0, 0.3)  # noqa: E731
            return tail(x) / tail(1.96)

        assert cond_tail(10, 0.95) == pytest.approx(case_a, abs=6e-4)
        assert cond_tail(1, 0.15) == pytest.approx(case_b, abs=6e-4)

    def test_section_5_conservative_benchmark(self):
        # Section 5.3: pi0 = 0.90, K = 5, alpha = 0.05, beta = 0.90 gives
        # pi0_K ~ 0.59, alpha_K ~ 0.23, beta_K ~ 0.73 and FDR ~ 0.54. The paper
        # rounds beta_K (0.7246) up to 0.73 and computes the FDR from the
        # rounded inputs; unrounded, it is 0.542.
        e = familywise_errors(0.05, 0.90, 0.90, 5)
        assert e.pi0_k == pytest.approx(0.59, abs=0.005)
        assert e.alpha_k == pytest.approx(0.23, abs=0.005)
        assert e.beta_k == pytest.approx(0.73, abs=0.006)
        assert search_adjusted_fdr(0.05, 0.90, 0.90, 5) == pytest.approx(0.54, abs=0.005)

    def test_ar1_threshold(self):
        # Eq. 30 reduces to 1.96 / sqrt(T) without autocorrelation, and positive
        # autocorrelation raises it.
        assert sharpe_threshold(100) == pytest.approx(0.196, abs=5e-4)
        assert sharpe_threshold(100, rho=0.071) > sharpe_threshold(100)


class TestFamilywiseErrors:
    def test_one_variant_is_the_single_trial_experiment(self):
        e = familywise_errors(0.05, 0.4, 0.8, 1)
        assert (e.alpha_k, e.beta_k, e.pi0_k) == pytest.approx((0.05, 0.4, 0.8))

    def test_search_raises_type_i_and_lowers_type_ii_error(self):
        e1, e5 = familywise_errors(0.05, 0.9, 0.9, 1), familywise_errors(0.05, 0.9, 0.9, 5)
        assert e5.alpha_k > e1.alpha_k
        assert e5.beta_k < e1.beta_k

    @pytest.mark.parametrize("pi0", [0.5, 0.9, 0.99])
    @pytest.mark.parametrize("k", [2, 5, 20])
    def test_theorem_2_holding_the_trial_parameters_fixed(self, pi0, k):
        # With the same trial-level parameters, search lowers the family-level
        # FDR (Theorem 2). The danger is not this comparison but that the
        # parameters themselves are misread from the selected cross-section.
        alpha, beta = 0.05, 0.8
        assert search_adjusted_fdr(alpha, beta, pi0, k) < fdr(alpha, beta, pi0)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            familywise_errors(1.5, 0.5, 0.5, 2)
        with pytest.raises(ValueError):
            familywise_errors(0.05, 0.5, 0.5, 0)

    def test_fdr_is_undefined_when_nothing_is_rejected(self):
        assert np.isnan(fdr(0.0, 1.0, 0.5))


class TestSearchAdjustedPvalue:
    def test_one_variant_leaves_the_pvalue_alone(self):
        p = pd.Series([0.001, 0.2, 0.9])
        pd.testing.assert_series_equal(search_adjusted_pvalue(p, 1), p)

    def test_matches_the_formula_and_keeps_small_values_accurate(self):
        p = pd.Series([0.01, 1e-12])
        adjusted = search_adjusted_pvalue(p, 10)
        assert adjusted.iloc[0] == pytest.approx(1 - 0.99**10)
        assert adjusted.iloc[1] == pytest.approx(1e-11, rel=1e-6)

    def test_per_hypothesis_k_is_aligned_by_name(self):
        p = pd.Series({"a": 0.01, "b": 0.01})
        adjusted = search_adjusted_pvalue(p, pd.Series({"b": 5, "a": 1}))
        assert adjusted["a"] == pytest.approx(0.01)
        assert adjusted["b"] == pytest.approx(1 - 0.99**5)

    def test_nan_stays_nan_and_bad_k_raises(self):
        assert np.isnan(search_adjusted_pvalue(pd.Series([np.nan]), 3).iloc[0])
        for bad in (0, np.inf):
            with pytest.raises(ValueError):
                search_adjusted_pvalue(pd.Series([0.1]), bad)

    def test_missing_k_raises_rather_than_dropping_the_hypothesis(self):
        # A NaN here would make BH drop "b" from the family: never selectable,
        # and the family too small.
        p = pd.Series({"a": 0.01, "b": 0.01})
        with pytest.raises(ValueError, match="missing"):
            search_adjusted_pvalue(p, pd.Series({"a": 3}))
        with pytest.raises(ValueError, match="missing"):
            search_adjusted_pvalue(p, pd.Series({"a": 3, "b": np.nan}))

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            search_adjusted_pvalue(pd.Series([1.2]), 3)
        with pytest.raises(TypeError):
            search_adjusted_pvalue([0.1, 0.2], 3)
        with pytest.raises(ValueError):
            search_adjusted_pvalue(pd.Series([0.1]), 3, rho=-0.1)

    def test_is_uniform_under_the_null(self):
        # The best of 20 null variants: raw p piles up near 0, adjusted p is flat.
        d = simulate_family_winners(20_000, 20, 0.0, 1.0, 2520, seed=0)
        adjusted = search_adjusted_pvalue(d["pvalue"], 20)
        assert (d["pvalue"] < 0.05).mean() > 0.6
        for level in (0.01, 0.05, 0.5):
            assert (adjusted < level).mean() == pytest.approx(level, abs=0.01)


class TestCorrelatedVariants:
    """`rho`: the exact adjustment for k equicorrelated variants."""

    def test_endpoints(self):
        p = pd.Series([0.05, 0.001])
        pd.testing.assert_series_equal(
            search_adjusted_pvalue(p, 20, rho=0.0), search_adjusted_pvalue(p, 20)
        )
        pd.testing.assert_series_equal(search_adjusted_pvalue(p, 20, rho=1.0), p)

    def test_falls_as_correlation_rises(self):
        p = pd.Series([0.01])
        values = [search_adjusted_pvalue(p, 20, rho=r).iloc[0] for r in (0, 0.5, 0.9, 0.99)]
        assert values == sorted(values, reverse=True)
        assert values[-1] > 0.01

    def test_matches_monte_carlo(self):
        # P(best of 20 variants with correlation 0.9 reaches p), simulated.
        rng = np.random.default_rng(0)
        k, r, n = 20, 0.9, 200_000
        z = np.sqrt(r) * rng.standard_normal((n, 1)) + np.sqrt(1 - r) * rng.standard_normal((n, k))
        best = z.max(axis=1)
        for p in (0.05, 0.01):
            simulated = (best >= norm.isf(p)).mean()
            exact = search_adjusted_pvalue(pd.Series([p]), k, rho=r).iloc[0]
            assert exact == pytest.approx(simulated, abs=3 * np.sqrt(simulated / n) + 1e-4)

    def test_keeps_small_values_accurate(self):
        adjusted = search_adjusted_pvalue(pd.Series([1e-12]), 20, rho=0.9).iloc[0]
        assert 1e-12 <= adjusted < 2e-11

    @pytest.mark.parametrize("r", [0.5, 0.9, 0.99])
    def test_is_calibrated_where_the_independent_formula_is_not(self, r):
        # Measured over 50 seeds of 1,000 null families, k = 20: the share
        # called real at 0.05 is 0.033 / 0.011 / 0.004 with the independent
        # formula and 0.048-0.049 with rho.
        d = simulate_family_winners(5_000, 20, 0.0, 1.0, 2520, within_rho=r, seed=1)
        assert (search_adjusted_pvalue(d["pvalue"], 20, rho=r) < 0.05).mean() == pytest.approx(
            0.05, abs=0.01
        )
        assert (search_adjusted_pvalue(d["pvalue"], 20) < 0.05).mean() < 0.04

    def test_recovers_the_power_the_independent_formula_loses(self):
        # Near-copies (rho 0.9), k = 20. Measured over 50 seeds: power 0.64
        # with the independent formula, 0.81 with rho, FDR 0.042.
        # Over 10 seeds at three ranges: 0.82 vs 0.64-0.66.
        seeds = range(10)
        r = _hidden_search(20, within_rho=0.9, rho=0.9, seeds=seeds)
        independent = _hidden_search(20, within_rho=0.9, seeds=seeds)
        assert r["adjusted_fdr"] <= 0.05
        assert r["adjusted_power"] > independent["adjusted_power"] + 0.1


def _hidden_search(k, *, share_real=0.2, within_rho=0.0, rho=0.0, seeds=None):
    """Issue #41: report each family's best variant, then run BH with and without
    the search adjustment. Returns mean realised FDR and power of each."""
    rows = []
    for seed in range(20) if seeds is None else seeds:
        d = simulate_family_winners(
            1000, k, share_real, 1.0, 2520, within_rho=within_rho, seed=seed
        )
        naive = benjamini_hochberg(d["pvalue"], 0.05)
        adjusted = benjamini_hochberg(search_adjusted_pvalue(d["pvalue"], k, rho=rho), 0.05)
        rows.append(
            {
                "naive_fdr": false_discovery_rate(naive, d["truth"]),
                "adjusted_fdr": false_discovery_rate(adjusted, d["truth"]),
                "naive_power": power(naive, d["truth"]),
                "adjusted_power": power(adjusted, d["truth"]),
            }
        )
    return pd.DataFrame(rows).mean()


class TestHiddenSearch:
    """The issue #41 experiment, at suite size. 1,000 hypotheses, 20% real
    (annualised Sharpe 1.0, 10 years), each reporting the best of K variants.

    Measured over 50 seeds: naive BH's realised FDR is 0.04 / 0.19 / 0.37 /
    0.62 at K = 1 / 5 / 10 / 20, and the search-adjusted version's is 0.04 at
    every K. BH controls the FDR at pi0 x alpha = 0.8 x 0.05 = 0.04.
    """

    def test_naive_fdr_climbs_with_hidden_search(self):
        fdrs = [_hidden_search(k)["naive_fdr"] for k in (1, 5, 20)]
        assert fdrs[0] < 0.06
        assert fdrs[0] < fdrs[1] < fdrs[2]
        assert fdrs[2] > 0.5

    @pytest.mark.parametrize("k", [1, 5, 20])
    def test_adjusted_fdr_is_controlled_at_every_k(self, k):
        assert _hidden_search(k)["adjusted_fdr"] <= 0.05

    def test_adjustment_keeps_the_power(self):
        # Real families gain from search too, so adjusting costs little.
        r = _hidden_search(5)
        assert r["adjusted_power"] > 0.95

    def test_correlated_variants_make_the_adjustment_conservative(self):
        # Five variants with correlation 0.5 search less than five independent
        # ones, so 1 - (1 - p)^5 over-corrects: FDR falls further below 0.04
        # (measured 0.033-0.038) and power drops (0.99 -> 0.90). This is the
        # case for passing `rho` (TestCorrelatedVariants).
        independent = _hidden_search(5)
        correlated = _hidden_search(5, within_rho=0.5)
        assert correlated["adjusted_fdr"] <= 0.05
        assert correlated["adjusted_power"] < independent["adjusted_power"]

    def test_null_any_discovery_rate(self):
        # No real family anywhere: the chance of any discovery should be alpha.
        # Measured over 200 seeds: naive 0.03 / 0.26 / 0.96 at K = 1 / 5 / 20,
        # adjusted 0.03 / 0.04 / 0.06.
        rates = {}
        for k in (5, 20):
            naive, adjusted = [], []
            for seed in range(100):
                d = simulate_family_winners(1000, k, 0.0, 1.0, 2520, seed=seed)
                naive.append(benjamini_hochberg(d["pvalue"], 0.05).any())
                adjusted.append(
                    benjamini_hochberg(search_adjusted_pvalue(d["pvalue"], k), 0.05).any()
                )
            rates[k] = (np.mean(naive), np.mean(adjusted))
        assert rates[20][0] > 0.8
        for k in (5, 20):
            assert rates[k][1] <= 0.1


def _family_winners_from_returns(n_families, k, n_real_families, seed, **kwargs):
    """Group a `make_return_matrix` matrix into families of k columns, real
    columns together. Returns each family's best one-sided HAC p-value, its
    truth, and the correlation between its variants estimated from returns."""
    m = make_return_matrix(
        n_candidates=n_families * k, n_real=n_real_families * k, seed=seed, **kwargs
    )
    columns = list(m.truth[m.truth].index) + list(m.truth[~m.truth].index)
    family = pd.Series(np.repeat([f"hyp_{i:03d}" for i in range(n_families)], k), index=columns)
    p = pd.Series({c: sharpe_test(m.returns[c]).pvalue_greater for c in columns})
    rho = pd.Series(
        {
            h: max(average_correlation(m.returns[cols.index]), 0.0)
            for h, cols in family.groupby(family)
        }
    )
    return p.groupby(family).min(), m.truth.reindex(columns).groupby(family).any(), rho


class TestOnReturnSeries:
    """The same claims on simulated return series with HAC p-values, including
    correlated, autocorrelated, fat-tailed and volatility-clustered nulls."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rho": 0.3, "ar1": 0.3},
            {"rho": 0.3, "ar1": 0.3, "t_df": 5, "garch": (0.05, 0.9)},
        ],
    )
    def test_null_calibration(self, kwargs):
        # 20 families of 5 variants, 4 years, no real family. Measured over 300
        # seeds, the adjusted any-discovery rate is 0.067 with rho and ar1 and
        # 0.097 with every defect on, against 0.2-0.4 for naive BH; over 40
        # seeds it ranged 0.05-0.175 across five seed ranges. IID nulls are
        # covered exactly by `test_is_uniform_under_the_null`. The excess
        # over 0.05 is the HAC p-value's, not the adjustment's: with fat tails
        # and GARCH, HAC rejects 1.3-1.6x its nominal rate at p < 0.01 per
        # variant, and the best of five lives in that tail.
        # With `rho` estimated from each family's returns, the rate stays in
        # the same band.
        naive, adjusted, with_rho = [], [], []
        for seed in range(40):
            best, _, rho = _family_winners_from_returns(20, 5, 0, seed, n_obs=1000, **kwargs)
            naive.append(benjamini_hochberg(best, 0.05).any())
            adjusted.append(benjamini_hochberg(search_adjusted_pvalue(best, 5), 0.05).any())
            with_rho.append(
                benjamini_hochberg(search_adjusted_pvalue(best, 5, rho=rho), 0.05).any()
            )
        assert np.mean(adjusted) <= 0.2
        assert np.mean(with_rho) <= 0.2
        assert np.mean(adjusted) < np.mean(naive)

    def test_power(self):
        # 2 real families of 20 (annualised Sharpe 1.5, 10 years); measured
        # mean power 0.93 over 30 seeds, 0.85-1.0 over 10 seeds at four ranges.
        powers = []
        for seed in range(10):
            best, truth, _ = _family_winners_from_returns(
                20, 5, 2, seed, n_obs=2520, rho=0.3, ar1=0.3, sharpe_real=1.5
            )
            rejected = benjamini_hochberg(search_adjusted_pvalue(best, 5), 0.05)
            powers.append(power(rejected, truth))
        assert np.mean(powers) >= 0.75


class TestMaxOfMixtureFit:
    @staticmethod
    def _draw(n, k, pi0, delta1, sigma0, sigma1, seed):
        rng = np.random.default_rng(seed)
        real = rng.random((n, k)) > pi0
        x = np.where(real, rng.normal(delta1, sigma1, (n, k)), rng.normal(0, sigma0, (n, k)))
        return x.max(axis=1)

    def test_recovers_the_truth_at_the_true_k_and_misses_it_at_k_one(self):
        # 400 reported best-of-5 statistics from the paper's fitted K = 5 world
        # (pi0 0.95, delta1 0.3, sigma0 0.1, sigma1 0.12), threshold 0.25. The
        # true FDR is 0.13. Fitted at K = 5 it lands near it (0.06-0.22 over
        # seeds 0-5); fitted assuming no search it says 0 on every seed.
        x = self._draw(400, 5, 0.95, 0.3, 0.1, 0.12, seed=0)
        table = fdr_by_search_intensity(x, 0.25, [1, 5])
        assert table.loc[5, "pi0"] == pytest.approx(0.95, abs=0.06)
        assert 0.03 < table.loc[5, "fdr"] < 0.3
        assert table.loc[1, "fdr"] < 0.01
        assert table.loc[5, "loglik"] > table.loc[1, "loglik"]

    def test_cannot_say_nothing_here_but_flags_it(self):
        # All-null best-of-5 statistics, fitted at the true K. Over seeds 0-9
        # the implied FDR ranged 0.20-0.999: seed 9 gives 0.20, which reads as
        # "mostly real". That is why this is a sensitivity table and not a
        # gate. Every one of those null fits sat on a bound, and says so.
        x = np.random.default_rng(9).normal(0, 0.1, (400, 5)).max(axis=1)
        row = fdr_by_search_intensity(x, 0.25, [5]).loc[5]
        assert row["fdr"] < 0.5
        assert row["at_bound"]

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            fit_max_of_mixture([0.1, 0.2], 3)
        with pytest.raises(ValueError):
            fit_max_of_mixture(np.arange(10.0), 0)


class TestSearchIntensity:
    def test_counts_trials_per_hypothesis(self):
        k = search_intensity(["h1", "h2", "h1", None, "h1"])
        assert k.to_dict() == {"h1": 3, "h2": 1}


@pytest.mark.network
def test_reproduces_section_6_table_on_the_papers_data():
    # Table 3, on the 212 Chen-Zimmermann predictors the authors publish with
    # their code. Downloaded, never committed.
    requests = pytest.importorskip("requests")
    url = (
        "https://raw.githubusercontent.com/lopezdeprado/FDR-in-Finance/main/PredictorLSretWide.csv"
    )
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"network unavailable: {exc}")
    returns = pd.read_csv(io.StringIO(response.text)).drop(columns=["date"], errors="ignore")

    sharpe, thresholds = [], []
    for name in returns:
        s = returns[name].dropna()
        rho = float(np.clip(s.autocorr(1), -0.99, 0.99))
        sharpe.append(s.mean() / s.std())
        thresholds.append(sharpe_threshold(len(s), rho))
    assert len(sharpe) == 212
    # The paper prints 0.0778; its own code on this file gives 0.07799.
    assert np.mean(thresholds) == pytest.approx(0.078, abs=3e-4)

    table = fdr_by_search_intensity(sharpe, thresholds, [1, 5])
    assert table.loc[1, "fdr"] == pytest.approx(0.0, abs=1e-3)
    assert table.loc[1, "loglik"] == pytest.approx(179.985, abs=1e-2)
    row = table.loc[5]
    assert row["pi0"] == pytest.approx(0.973, abs=1e-3)
    assert row["delta1"] == pytest.approx(0.316, abs=1e-3)
    assert row["pi0_k"] == pytest.approx(0.872, abs=1e-3)
    assert row["alpha_k"] == pytest.approx(0.724, abs=1e-3)
    assert row["loglik"] == pytest.approx(203.046, abs=1e-2)
    assert row["fdr"] == pytest.approx(0.833, abs=1e-3)
