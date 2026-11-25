"""Tests for capstone.calibration: the test bench for the validation pipeline.

Small family sets (5 years of search, 2 of holdout) keep these fast. The
full calibration lives in scripts/calibrate_validation.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.calibration import (
    Thresholds,
    family_statistics,
    gate_report,
    gates_alone,
    idle_days,
    make_family_set,
    run_baseline,
    run_funnel,
    run_naive_stack,
    score,
)
from capstone.evaluate import average_correlation, sharpe_test

SMALL = {"n_search": 5 * 252, "n_holdout": 2 * 252}


def _stats(share, sharpe, seed, **kwargs):
    fs = make_family_set(20, 5, share, sharpe, seed=seed, **SMALL, **kwargs)
    return fs, family_statistics(fs, stage2_years=2, n_boot=0, seed=seed)


class TestMakeFamilySet:
    def test_shape_truth_and_periods(self):
        fs = make_family_set(20, 5, 0.1, 1.0, seed=0, **SMALL)
        assert fs.returns.shape == (7 * 252, 100)
        assert fs.truth.sum() == 2
        assert fs.family.value_counts().eq(5).all()
        assert len(fs.search) == 5 * 252 and len(fs.holdout) == 2 * 252
        assert fs.search.index[-1] < fs.holdout.index[0]

    def test_variants_carry_the_planted_sharpe_and_correlation(self):
        # Long IID series so the estimates are tight: each real variant's
        # annualised Sharpe should be near sharpe_real, and a family's
        # variants should be correlated near within_rho.
        fs = make_family_set(
            10, 8, 0.5, 1.0, within_rho=0.9, seed=1, n_search=50 * 252, n_holdout=252
        )
        real_cols = fs.family[fs.family.isin(fs.truth[fs.truth].index)].index
        sharpe = fs.returns[real_cols].mean() / fs.returns[real_cols].std() * np.sqrt(252)
        assert sharpe.mean() == pytest.approx(1.0, abs=0.15)
        one = fs.family[fs.family == fs.truth.index[0]].index
        assert average_correlation(fs.returns[one]) == pytest.approx(0.9, abs=0.02)

    def test_rejects_bad_correlation(self):
        with pytest.raises(ValueError):
            make_family_set(5, 2, 0.0, 1.0, within_rho=0.0, **SMALL)


class TestIdleDays:
    def test_share_and_run_length_are_as_asked(self):
        idle = idle_days(50_000, 20, 0.4, 21.0, seed=0)
        assert idle.mean() == pytest.approx(0.4, abs=0.02)
        column = np.r_[False, idle[:, 0], False].astype(int)
        starts, ends = np.flatnonzero(np.diff(column) == 1), np.flatnonzero(np.diff(column) == -1)
        assert (ends - starts).mean() == pytest.approx(21.0, rel=0.1)

    def test_a_family_is_idle_together_and_keeps_its_active_returns(self):
        active = make_family_set(6, 4, 0.5, 1.0, seed=3, **SMALL)
        gappy = make_family_set(6, 4, 0.5, 1.0, seed=3, idle_share=0.3, **SMALL)
        nan = gappy.returns.isna()
        assert nan.to_numpy().mean() == pytest.approx(0.3, abs=0.08)
        for f in gappy.family.unique():
            assert nan[gappy.family.index[gappy.family == f]].nunique(axis=1).eq(1).all()
        pd.testing.assert_frame_equal(gappy.returns, active.returns.where(~nan))

    @pytest.mark.parametrize("share, run", [(-0.1, 21.0), (1.0, 21.0), (0.5, 0.5), (0.9, 1.0)])
    def test_bad_settings_are_refused(self, share, run):
        with pytest.raises(ValueError, match="idle"):
            make_family_set(3, 2, 0.0, 0.0, seed=0, idle_share=share, idle_run=run, **SMALL)

    def test_negative_control_holds_with_idle_days(self):
        accepted = {"baseline": 0, "funnel": 0}
        for seed in range(8):
            _, stats = _stats(
                0.0, 1.0, seed=seed, ar1=0.3, t_df=5, garch=(0.05, 0.9), idle_share=0.5
            )
            truth = stats["reuse"]["truth"]
            accepted["baseline"] += score(run_baseline(stats["reuse"]), truth)["any_false"]
            th = Thresholds(q1=0.2, alpha2=0.2, final="bh", q4=0.2)
            accepted["funnel"] += score(run_funnel(stats["split"], th)["stage4"], truth)[
                "any_false"
            ]
        assert accepted["funnel"] <= 1
        assert accepted["baseline"] <= 3

    def test_fewer_active_days_find_fewer_real_families(self):
        power = {}
        for share in (0.0, 0.6):
            found = []
            for seed in range(6):
                stats = _stats(0.3, 1.5, seed=seed, idle_share=share)[1]["reuse"]
                found.append(score(run_baseline(stats), stats["truth"])["power"])
            power[share] = np.mean(found)
        assert power[0.6] < power[0.0]


class TestHoldoutCoverage:
    def test_a_thinly_covered_survivor_fails_stage_4_and_the_rest_are_unchanged(self):
        _, stats = _stats(0.3, 2.5, seed=0)
        stats = stats["reuse"]
        th = Thresholds(q1=0.2, final="bh", q4=0.5)
        free = run_funnel(stats, th)
        passer = free.index[free["stage4"]][0]
        thin = stats.assign(ho_n_obs=stats["ho_n_obs"].where(stats.index != passer, 10.0))
        guarded = run_funnel(thin, th, min_holdout_obs=100)
        assert not guarded.loc[passer, "stage4"]
        pd.testing.assert_frame_equal(guarded.drop(columns="stage4"), free.drop(columns="stage4"))
        pd.testing.assert_frame_equal(run_funnel(stats, th, min_holdout_obs=100), free)

    def test_a_survivor_at_exactly_the_guard_is_tested(self):
        _, stats = _stats(0.3, 2.5, seed=0)
        stats = stats["reuse"]
        th = Thresholds(q1=0.2, final="bh", q4=0.5)
        free = run_funnel(stats, th)
        passer = free.index[free["stage4"]][0]
        at = stats.assign(ho_n_obs=stats["ho_n_obs"].where(stats.index != passer, 100.0))
        pd.testing.assert_frame_equal(run_funnel(at, th, min_holdout_obs=100), free)


class TestVariantsPerFamily:
    """`k` may list each family's variant count (the gate's power analysis);
    a scalar `k` gives exactly the data it always did."""

    def test_scalar_k_is_unchanged(self):
        fs = make_family_set(3, 2, 0.34, 1.0, seed=5, n_search=300, n_holdout=100, rho=0.1)
        # Computed before `k` could be a list.
        assert np.float64(fs.returns.to_numpy().sum()).hex() == "0x1.36e94c172e08cp+0"
        assert fs.truth.to_dict() == {"cand_0000": False, "cand_0001": False, "cand_0002": True}
        assert list(fs.returns.columns) == [f"cand_000{h}_v{j}" for h in range(3) for j in range(2)]

    def test_equal_counts_listed_equal_the_scalar(self):
        a = make_family_set(4, 3, 0.25, 1.0, seed=1, **SMALL)
        b = make_family_set(4, [3, 3, 3, 3], 0.25, 1.0, seed=1, **SMALL)
        pd.testing.assert_frame_equal(a.returns, b.returns, rtol=0, atol=0)
        pd.testing.assert_series_equal(a.family, b.family)
        pd.testing.assert_series_equal(a.truth, b.truth)

    def test_each_family_gets_its_own_count(self):
        fs = make_family_set(3, (2, 5, 1), 0.0, 0.0, seed=2, **SMALL)
        assert fs.returns.shape[1] == 8
        assert fs.family.value_counts().sort_index().tolist() == [2, 5, 1]
        assert list(fs.returns.columns[:3]) == ["cand_0000_v0", "cand_0000_v1", "cand_0001_v0"]

    @pytest.mark.parametrize("k", [[2, 2], [2, 0, 2], 0])
    def test_bad_counts_are_refused(self, k):
        with pytest.raises(ValueError, match="k"):
            make_family_set(3, k, 0.0, 0.0, seed=0, **SMALL)


class TestFamilyStatistics:
    def test_designs_and_columns(self):
        fs, stats = _stats(0.1, 1.0, seed=0)
        assert set(stats) == {"split", "reuse"}
        for s in stats.values():
            assert {"best", "p1_independent", "p1_rho", "p2", "ho_sharpe", "naive_pass"} <= set(s)
            assert s["p1_bootstrap"].isna().all()  # n_boot=0 skips it
            pd.testing.assert_series_equal(s["truth"], fs.truth.reindex(s.index), check_names=False)

    def test_reuse_confirms_on_the_data_it_screened(self):
        # With no fresh data, stage 2's p-value is the best variant's p-value
        # on the same search period stage 1 used.
        fs, stats = _stats(0.1, 1.0, seed=2)
        s = stats["reuse"]
        h = s.index[0]
        assert s.loc[h, "p2"] == pytest.approx(
            sharpe_test(fs.search[s.loc[h, "best"]]).pvalue_greater
        )

    def test_split_confirms_on_years_stage_1_never_saw(self):
        fs, stats = _stats(0.1, 1.0, seed=3)
        s = stats["split"]
        h = s.index[0]
        last = fs.search.iloc[-2 * 252 :]
        assert s.loc[h, "p2"] == pytest.approx(sharpe_test(last[s.loc[h, "best"]]).pvalue_greater)

    def test_rejects_stage2_longer_than_the_search(self):
        fs = make_family_set(5, 2, 0.0, 1.0, seed=0, **SMALL)
        with pytest.raises(ValueError):
            family_statistics(fs, stage2_years=6, n_boot=0)


class TestPipelines:
    def test_funnel_stages_are_nested(self):
        _, stats = _stats(0.2, 1.5, seed=4)
        stages = run_funnel(stats["split"], Thresholds(q1=0.3, alpha2=0.3, final="bh", q4=0.3))
        for earlier, later in zip(stages.columns, stages.columns[1:], strict=False):
            assert not (stages[later] & ~stages[earlier]).any()

    def test_looser_thresholds_never_accept_fewer(self):
        _, stats = _stats(0.2, 1.0, seed=5)
        s = stats["split"]
        strict = run_funnel(s, Thresholds(q1=0.05, alpha2=0.05, final="bh", q4=0.05))
        loose = run_funnel(s, Thresholds(q1=0.3, alpha2=0.2, final="bh", q4=0.05))
        assert (loose["stage2"] >= strict["stage2"]).all()

    def test_charging_the_searched_count_is_never_looser(self):
        for seed in range(3):
            _, stats = _stats(0.2, 1.5, seed=seed)
            s = stats["split"]
            survivors = run_funnel(s, Thresholds(final="dsr", d4=0.8, final_trials="survivors"))
            searched = run_funnel(s, Thresholds(final="dsr", d4=0.8, final_trials="searched"))
            assert (searched["stage4"] <= survivors["stage4"]).all()

    def test_positive_control_strong_signals_are_found(self):
        # Sharpe 2.5, IID: every approach worth having finds them.
        powers = {"baseline": [], "funnel": []}
        for seed in range(3):
            _, stats = _stats(0.2, 2.5, seed=seed)
            truth = stats["reuse"]["truth"]
            powers["baseline"].append(score(run_baseline(stats["reuse"]), truth)["power"])
            th = Thresholds(q1=0.2, alpha2=0.2, final="bh", q4=0.2)
            powers["funnel"].append(score(run_funnel(stats["split"], th)["stage4"], truth)["power"])
        assert np.mean(powers["baseline"]) >= 0.9
        assert np.mean(powers["funnel"]) >= 0.6

    def test_negative_control_nothing_real_is_accepted_rarely(self):
        # Nothing real, every defect on: the funnel and the baseline should
        # almost never accept anything.
        accepted = {"baseline": 0, "funnel": 0}
        for seed in range(8):
            _, stats = _stats(0.0, 1.0, seed=seed, ar1=0.3, t_df=5, garch=(0.05, 0.9))
            truth = stats["reuse"]["truth"]
            accepted["baseline"] += score(run_baseline(stats["reuse"]), truth)["any_false"]
            th = Thresholds(q1=0.2, alpha2=0.2, final="bh", q4=0.2)
            accepted["funnel"] += score(run_funnel(stats["split"], th)["stage4"], truth)[
                "any_false"
            ]
        assert accepted["funnel"] <= 1
        assert accepted["baseline"] <= 3

    def test_the_naive_stack_loses_real_signals(self):
        # The advisor's point, pinned: at a realistic Sharpe of 1.0, requiring
        # every screening method at once finds fewer real families than the
        # single-gate baseline.
        naive, baseline = [], []
        for seed in range(4):
            _, stats = _stats(0.2, 1.0, seed=seed)
            truth = stats["reuse"]["truth"]
            baseline.append(score(run_baseline(stats["reuse"]), truth)["power"])
            th = Thresholds(final="bh", q4=0.999)  # isolate the stacked screening gates
            naive.append(score(run_naive_stack(stats["reuse"], th), truth)["power"])
        assert np.mean(naive) < np.mean(baseline)

    def test_rejects_unknown_choices(self):
        _, stats = _stats(0.1, 1.0, seed=0)
        with pytest.raises(ValueError):
            run_funnel(stats["split"], Thresholds(final="holm"))
        with pytest.raises(ValueError):
            run_funnel(stats["split"], Thresholds(final_trials="half"))
        with pytest.raises(ValueError):
            run_funnel(stats["split"], Thresholds(), stage1="lfdr")


class TestDefaults:
    def test_thresholds_default_to_the_test_runs_best_set(self):
        # The best set from the synthetic test runs, a starting point (not registered).
        th = Thresholds()
        assert (th.q1, th.alpha2, th.final, th.q4) == (0.10, 1.0, "bh", 0.30)

    def test_a_hypothesis_with_no_data_in_the_window_is_named(self):
        fs = make_family_set(4, 2, 0.0, 1.0, seed=0, **SMALL)
        returns = fs.returns.copy()
        returns.loc[:, fs.family[fs.family == fs.truth.index[0]].index] = np.nan
        holed = type(fs)(returns, fs.family, fs.truth, fs.n_search, fs.n_holdout)
        with pytest.raises(ValueError, match="no variant with data"):
            family_statistics(holed, stage2_years=2, n_boot=0)

    def test_families_with_no_stacked_pass_come_back_empty(self):
        # Nothing real: no variant passes the naive stack, so every family's
        # naive_best is missing. pandas 3 raised here (#68 review).
        _, stats = _stats(0.0, 1.0, seed=1)
        assert not stats["reuse"]["naive_pass"].any()
        assert stats["reuse"]["nv_p"].isna().all()


class TestReports:
    def test_score(self):
        truth = pd.Series([True, False, False, True])
        r = score(pd.Series([True, True, False, False]), truth)
        assert (r["discoveries"], r["true"], r["false"]) == (2, 1, 1)
        assert r["fdr"] == 0.5 and r["power"] == 0.5 and r["any_false"]
        nothing = score(pd.Series([False] * 4), truth)
        assert nothing["fdr"] == 0.0 and np.isnan(nothing["precision"])
        assert np.isnan(score(pd.Series([True]), pd.Series([False]))["power"])

    def test_gate_report_chains_and_flags_stage3(self):
        _, stats = _stats(0.2, 1.5, seed=6)
        stages = run_funnel(stats["split"], Thresholds(q1=0.3, alpha2=0.3, final="bh", q4=0.3))
        report = gate_report(stages, stats["split"]["truth"])
        assert not report.loc["stage3", "measured"]
        assert report.loc["stage3", ["pass_rate_real", "pass_rate_null"]].isna().all()
        for earlier, later in zip(report.index, report.index[1:], strict=False):
            assert report.loc[later, "real_in"] == report.loc[earlier, "real_out"]
            assert report.loc[later, "null_in"] == report.loc[earlier, "null_out"]

    def test_gates_alone(self):
        _, stats = _stats(0.2, 1.5, seed=7)
        alone = gates_alone(stats["split"], Thresholds(final="bh", q4=0.2))
        assert list(alone.index) == ["stage1", "stage2", "stage4"]
        assert alone.loc["stage1", "pass_rate_real"] >= alone.loc["stage1", "pass_rate_null"]
