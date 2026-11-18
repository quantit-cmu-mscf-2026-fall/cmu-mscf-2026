"""Tests for capstone.grading: graded scores that are reported, never decisive.

Every score is computed from the search period, so the holdout is never read
here. `stage4_stat` is the one exception by design: it reads the holdout
statistics the gate hands it, after the look.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from capstone import calibration, funnel, grading
from capstone.evaluate import bh_adjusted, deflated_sharpe_ratio, haircut_sharpe, sharpe_test
from capstone.synth import make_return_matrix

N_FAMILIES = 10
K = 4
N_OBS = 8 * 252


def _search(share_real: float = 0.3, seed: int = 0):
    fs = calibration.make_family_set(
        N_FAMILIES, K, share_real, 1.5, n_search=N_OBS, n_holdout=252, seed=seed
    )
    returns = fs.returns.iloc[: fs.n_search]
    returns = returns.set_axis(pd.bdate_range("2000-01-03", periods=len(returns)))
    return returns, fs.family


def _best_and_p1(returns: pd.DataFrame, family: pd.Series):
    sharpe = returns.mean() / returns.std()
    best = funnel.best_by_family(sharpe, family)
    p1 = pd.Series({f: sharpe_test(returns[b]).pvalue_greater for f, b in best.items()})
    return best, p1


@pytest.fixture(scope="module")
def planted():
    returns, family = _search(0.3, seed=1)
    best, p1 = _best_and_p1(returns, family)
    return returns, best, p1


class TestGradedScores:
    def test_columns_and_index(self, planted):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        assert list(scores.columns) == [
            "p1",
            "by",
            "holm",
            "lfdr",
            "sharpe",
            "psr",
            "min_trl_years",
            "haircut_sharpe",
        ]
        assert scores.index.equals(p1.index)
        pd.testing.assert_series_equal(scores["p1"], p1, check_names=False)

    def test_sharpe_and_psr_are_the_search_period_test(self, planted):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        for fam, b in best.items():
            t = sharpe_test(returns[b])
            assert scores.loc[fam, "sharpe"] == pytest.approx(t.sharpe)
            assert scores.loc[fam, "psr"] == pytest.approx(t.psr)

    def test_adjusted_p_values_are_never_below_the_raw_ones(self, planted):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        assert (scores["by"] >= scores["p1"] - 1e-12).all()
        assert (scores["holm"] >= scores["p1"] - 1e-12).all()

    def test_min_trl_is_in_years(self, planted):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        positive = scores["sharpe"] > 0
        assert positive.any()
        assert np.isfinite(scores.loc[positive, "min_trl_years"]).all()
        assert (scores.loc[positive, "min_trl_years"] > 0).all()
        # Years, not periods: a strong 8-year series needs far fewer than 8 years.
        strongest = scores["sharpe"].idxmax()
        assert scores.loc[strongest, "min_trl_years"] < N_OBS / 252
        assert np.isinf(scores.loc[~positive, "min_trl_years"]).all()

    def test_haircut_is_below_the_sharpe_and_undefined_without_a_positive_one(self, planted):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        positive = scores["sharpe"] > 0
        assert positive.any()
        assert scores.loc[positive, "haircut_sharpe"].notna().all()
        assert (scores.loc[positive, "haircut_sharpe"] <= scores.loc[positive, "sharpe"]).all()
        assert scores.loc[~positive, "haircut_sharpe"].isna().all()

    def test_more_trials_haircut_harder(self, planted):
        returns, best, p1 = planted
        few = grading.graded_scores(returns, best, p1, 20, 0.3, 0)["haircut_sharpe"]
        many = grading.graded_scores(returns, best, p1, 2_000, 0.3, 0)["haircut_sharpe"]
        both = many.dropna().index
        assert (many[both] <= few[both]).all()  # a heavy haircut can floor at zero
        assert (many[both] < few[both]).any()

    def test_one_trial_has_no_haircut(self, planted):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 1, 0.3, 0)
        assert scores["haircut_sharpe"].isna().all()
        assert scores["sharpe"].notna().all()  # the other scores still come back

    def test_seed_fixes_the_haircut(self, planted):
        returns, best, p1 = planted
        a = grading.graded_scores(returns, best, p1, 50, 0.3, 7)
        b = grading.graded_scores(returns, best, p1, 50, 0.3, 7)
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.parametrize("avg_corr", [float("nan"), -0.5, 1.5])
    def test_an_unusable_average_correlation_still_gives_a_haircut(self, planted, avg_corr):
        returns, best, p1 = planted
        scores = grading.graded_scores(returns, best, p1, 50, avg_corr, 0)
        positive = scores["sharpe"] > 0
        assert scores.loc[positive, "haircut_sharpe"].notna().all()

    def test_inputs_are_left_alone(self, planted):
        returns, best, p1 = planted
        before = (returns.copy(), best.copy(), p1.copy())
        grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        pd.testing.assert_frame_equal(returns, before[0])
        pd.testing.assert_series_equal(best, before[1])
        pd.testing.assert_series_equal(p1, before[2])

    def test_grading_leaves_a_screen_unchanged(self, planted):
        """Reported, not read back: screening the same statistics gives the same flags."""
        returns, best, p1 = planted
        stats = pd.DataFrame({"best": best, "p1_rho": p1, "p2": p1})
        th = funnel.Thresholds()
        before = funnel.screen(stats, th)
        grading.graded_scores(returns, best, p1, 50, 0.3, 0)
        pd.testing.assert_frame_equal(before, funnel.screen(stats, th))


class TestHaircut:
    def test_it_is_the_row_minimum_of_the_three_adjusted_sharpes(self):
        sharpe = pd.Series([0.4, 0.8, 1.2, 1.6], index=list("abcd"))
        n_obs = pd.Series(2000, index=sharpe.index)
        frame = haircut_sharpe(
            sharpe, 2000, n_trials=60, frequency="daily", avg_correlation=0.3, seed=5
        )
        adjusted = frame[["sharpe_bonferroni", "sharpe_holm", "sharpe_bhy"]]
        # The case discriminates: the three adjustments differ, row by row.
        assert ((adjusted.max(axis=1) - adjusted.min(axis=1)) > 1e-3).all()
        out = grading._haircut(sharpe, n_obs, 60, 0.3, 5)
        pd.testing.assert_series_equal(
            out, adjusted.min(axis=1).reindex(sharpe.index), check_names=False
        )
        assert (out < adjusted.max(axis=1).reindex(sharpe.index)).all()


class TestFamilyLocalFdr:
    def test_only_with_200_candidates(self):
        rng = np.random.default_rng(0)
        small = pd.Series(rng.uniform(size=199))
        assert grading.family_lfdr(small).isna().all()
        large = pd.Series(np.concatenate([rng.uniform(size=240), np.full(10, 1e-8)]))
        scores = grading.family_lfdr(large)
        assert scores.notna().all() and scores.between(0, 1).all()
        assert scores.iloc[-10:].max() < scores.iloc[:240].median()

    def test_missing_p_values_do_not_count_toward_the_200(self):
        rng = np.random.default_rng(1)
        p = pd.Series(np.concatenate([rng.uniform(size=150), np.full(100, np.nan)]))
        assert grading.family_lfdr(p).isna().all()


def _stage1_batches(n_batches: int, n_real: int, **defects):
    """Yield (p1, passed) for batches of 50 candidates: BH at 0.10, as stage 1 does."""
    for seed in range(n_batches):
        batch = make_return_matrix(1000, 50, n_real, sharpe_real=1.5, seed=seed, **defects).returns
        p1 = pd.Series({c: sharpe_test(batch[c]).pvalue_greater for c in batch.columns})
        yield p1, bh_adjusted(p1) <= 0.10


class TestSearchFdrEstimate:
    def test_nan_when_nothing_passes(self):
        p = pd.Series([0.4, 0.6, 0.9])
        assert math.isnan(grading.search_fdr_estimate(p, pd.Series(False, index=p.index)))

    def test_the_formula_is_storeys_pfdr_at_the_largest_passing_p(self):
        # 2 small, 58 middling, 40 above 0.5: pi0 = 40 / (0.5 x 100) = 0.8 < 1.
        p = pd.Series(
            np.concatenate([[0.004, 0.01], np.linspace(0.1, 0.5, 58), np.linspace(0.51, 1.0, 40)])
        )
        estimate = grading.search_fdr_estimate(p, p <= 0.01)
        t, m, r, pi0 = 0.01, 100, 2, 0.8
        assert estimate == pytest.approx(pi0 * t / ((r / m) * (1 - (1 - t) ** m)))
        # Each piece matters: not pi0 m t / R (0.4), not without pi0 (0.79).
        assert abs(estimate - pi0 * m * t / r) > 0.1
        assert abs(estimate - t / ((r / m) * (1 - (1 - t) ** m))) > 0.1

    def test_capped_at_one_and_a_p_of_zero_is_finite(self):
        p = pd.Series([0.0, 0.0, 0.3, 0.6, 0.9, 0.99])
        assert grading.search_fdr_estimate(p, p <= 0.0) <= 1.0
        rng = np.random.default_rng(0)
        p = pd.Series(rng.uniform(size=50))
        assert grading.search_fdr_estimate(p, p <= 1.0) <= 1.0

    def test_missing_p_values_are_ignored(self):
        p = pd.Series([0.001, 0.2, 0.7, 0.9, np.nan])
        passed = pd.Series([True, False, False, False, False])
        assert grading.search_fdr_estimate(p, passed) == pytest.approx(
            grading.search_fdr_estimate(p.dropna(), passed.iloc[:4])
        )

    @pytest.mark.parametrize(
        "defects", [{}, {"ar1": 0.2}, {"rho": 0.2, "ar1": 0.2}], ids=["iid", "ar1", "rho-ar1"]
    )
    def test_says_nothing_here_on_all_null_batches(self, defects):
        """With no real candidate every pass is false: the estimate must come out high.

        The old estimate, pi0 m t / R at BH's own cutoff, was at most pi0 x 0.10
        by construction (mean 0.05 here).
        """
        estimates = [
            grading.search_fdr_estimate(p1, passed)
            for p1, passed in _stage1_batches(80, 0, **defects)
            if passed.any()
        ]
        assert len(estimates) >= 5
        assert np.mean(estimates) > 0.4

    @pytest.mark.parametrize(
        "defects", [{}, {"ar1": 0.2}, {"rho": 0.2, "ar1": 0.2}], ids=["iid", "ar1", "rho-ar1"]
    )
    def test_says_little_when_real_candidates_pass(self, defects):
        estimates = [
            grading.search_fdr_estimate(p1, passed)
            for p1, passed in _stage1_batches(30, 10, **defects)
            if passed.any()
        ]
        assert len(estimates) >= 25
        assert np.mean(estimates) < 0.3


class TestPbo:
    def test_a_probability_on_a_search_period(self, planted):
        returns, _, _ = planted
        assert 0.0 <= grading.pbo(returns) <= 1.0

    def test_nan_when_the_period_is_too_short(self):
        short = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 4)))
        assert math.isnan(grading.pbo(short))


class TestStage4Stat:
    def _stats(self) -> pd.DataFrame:
        stats = pd.DataFrame(
            {
                "ho_p": [0.001, 0.02, 0.4],
                "ho_sharpe": [2.0, 1.0, 0.2],
                "ho_n_obs": [1000, 1000, 1000],
                "ho_skew": [0.0, 0.0, 0.0],
                "ho_kurtosis": [3.0, 3.0, 3.0],
                "ho_variance": [1.0, 1.0, 1.0],
            },
            index=["a", "b", "c"],
        )
        stats.attrs["n_variants"] = 30
        return stats

    def test_bh_is_the_adjusted_holdout_p_across_the_survivors(self):
        stats = self._stats()
        alive = pd.Series([True, True, False], index=stats.index)
        out = grading.stage4_stat(stats, alive, funnel.Thresholds(final="bh"))
        expected = bh_adjusted(stats.loc[alive, "ho_p"])
        pd.testing.assert_series_equal(out[alive], expected, check_names=False)
        assert math.isnan(out["c"])

    def test_nan_for_everyone_when_nothing_is_alive(self):
        stats = self._stats()
        alive = pd.Series(False, index=stats.index)
        assert grading.stage4_stat(stats, alive, funnel.Thresholds()).isna().all()

    @pytest.mark.parametrize("final_trials", ["survivors", "searched"])
    def test_dsr_charges_the_registered_trial_count(self, final_trials):
        stats = self._stats()
        alive = pd.Series([True, True, False], index=stats.index)
        th = funnel.Thresholds(final="dsr", final_trials=final_trials)
        out = grading.stage4_stat(stats, alive, th)
        n_trials = 30 if final_trials == "searched" else 2
        for h in ("a", "b"):
            r = stats.loc[h]
            expected = deflated_sharpe_ratio(
                r["ho_sharpe"], n_trials, int(r["ho_n_obs"]), 0.0, 3.0, variance=1.0
            )
            assert out[h] == pytest.approx(expected)
        assert math.isnan(out["c"])


class TestExpectedMean:
    def _card(self, **columns) -> pd.DataFrame:
        base = {"haircut_sharpe": [1.0], "sharpe": [2.0], "is_mean": [0.01], "ho_mean": [0.02]}
        base.update({k: [v] for k, v in columns.items()})
        return pd.DataFrame(base, index=["f"])

    def test_the_haircut_scales_the_search_mean(self):
        out = grading.expected_mean(self._card())
        assert out["f"] == pytest.approx(0.005)  # 0.01 x (1 / 2), below both other terms

    def test_the_holdout_mean_binds_when_it_is_smaller(self):
        out = grading.expected_mean(self._card(ho_mean=0.001))
        assert out["f"] == pytest.approx(0.001)

    def test_never_above_the_raw_search_mean(self):
        out = grading.expected_mean(self._card(haircut_sharpe=2.0, ho_mean=0.5))
        assert out["f"] == pytest.approx(0.01)

    def test_a_negative_haircut_clips_to_zero(self):
        out = grading.expected_mean(self._card(haircut_sharpe=-1.0))
        assert out["f"] == 0.0

    def test_missing_terms_are_skipped(self):
        out = grading.expected_mean(self._card(haircut_sharpe=np.nan, ho_mean=np.nan))
        assert out["f"] == pytest.approx(0.01)
        out = grading.expected_mean(self._card(haircut_sharpe=np.nan))
        assert out["f"] == pytest.approx(0.01)
