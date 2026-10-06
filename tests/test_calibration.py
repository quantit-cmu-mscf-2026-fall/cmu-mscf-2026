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
        for earlier, later in zip(report.index, report.index[1:], strict=False):
            assert report.loc[later, "real_in"] == report.loc[earlier, "real_out"]
            assert report.loc[later, "null_in"] == report.loc[earlier, "null_out"]

    def test_gates_alone(self):
        _, stats = _stats(0.2, 1.5, seed=7)
        alone = gates_alone(stats["split"], Thresholds(final="bh", q4=0.2))
        assert list(alone.index) == ["stage1", "stage2", "stage4"]
        assert alone.loc["stage1", "pass_rate_real"] >= alone.loc["stage1", "pass_rate_null"]
