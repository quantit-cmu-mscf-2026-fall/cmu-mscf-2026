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
from capstone.evaluate import bh_adjusted, deflated_sharpe_ratio, sharpe_test

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


class TestSearchFdrEstimate:
    def test_nan_when_nothing_passes(self):
        p = pd.Series([0.4, 0.6, 0.9])
        assert math.isnan(grading.search_fdr_estimate(p, pd.Series(False, index=p.index)))

    def test_plug_in_formula_and_cap(self):
        rng = np.random.default_rng(0)
        p = pd.Series(np.concatenate([np.full(5, 1e-6), rng.uniform(size=95)]))
        estimate = grading.search_fdr_estimate(p, p <= 0.01)
        assert 0.0 < estimate < 0.5
        assert grading.search_fdr_estimate(p, p <= 1.0) <= 1.0


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
