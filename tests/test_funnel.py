"""Tests for capstone.funnel: the funnel's statistics on dated data.

Periods are cut by date, never by row position, and a frame that reaches
past the search period is refused unless the caller names the holdout
explicitly. Stages 1-3 never see a holdout date.
"""

from __future__ import annotations

import json
import typing
import warnings

import numpy as np
import pandas as pd
import pytest

from capstone import calibration, funnel
from capstone.evaluate import sharpe_test
from capstone.funnel import Thresholds, family_statistics, run_funnel, thresholds_dict

SEARCH_END = pd.Timestamp("2004-12-31")
HOLDOUT_START = pd.Timestamp("2005-01-01")


def _dated(n_families=6, k=3, seed=0, drift=None, start="2000-01-03", years=7):
    """Business-day returns, `k` variants per family, optional per-family drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=years * 252)
    columns = [f"h{i}_v{j}" for i in range(n_families) for j in range(k)]
    x = rng.normal(0.0, 0.01, (len(dates), len(columns)))
    returns = pd.DataFrame(x, index=dates, columns=columns)
    family = pd.Series([c.split("_")[0] for c in columns], index=columns, name="hypothesis")
    for h, mu in (drift or {}).items():
        returns.loc[:, family[family == h].index] += mu
    return returns, family


class TestPeriodsByDate:
    def test_dates_after_search_end_need_an_explicit_holdout(self):
        returns, family = _dated()
        with pytest.raises(ValueError, match="holdout"):
            family_statistics(returns, family, search_end=SEARCH_END, n_boot=0)

    def test_search_only_frame_needs_no_holdout(self):
        returns, family = _dated()
        stats = family_statistics(returns.loc[:SEARCH_END], family, search_end=SEARCH_END, n_boot=0)
        assert stats["reuse"]["ho_p"].isna().all()

    def test_holdout_must_follow_the_search(self):
        returns, family = _dated()
        with pytest.raises(ValueError):
            family_statistics(
                returns, family, search_end=SEARCH_END, holdout_start=SEARCH_END, n_boot=0
            )

    def test_requires_a_date_index(self):
        returns, family = _dated()
        with pytest.raises(TypeError):
            family_statistics(
                returns.reset_index(drop=True), family, search_end=SEARCH_END, n_boot=0
            )

    def test_stage1_sees_only_search_dates(self):
        # The holdout's values can't move a single stage-1 or stage-2 number.
        returns, family = _dated(drift={"h0": 0.002})
        a = family_statistics(
            returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0
        )["reuse"]
        shocked = returns.copy()
        shocked.loc[HOLDOUT_START:] += 0.05
        b = family_statistics(
            shocked, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0
        )["reuse"]
        cols = ["best", "p1_independent", "p1_rho", "p2"]
        pd.testing.assert_frame_equal(a[cols], b[cols])
        assert not np.allclose(a["ho_sharpe"], b["ho_sharpe"])

    def test_splits_are_by_date_not_position(self):
        returns, family = _dated(drift={"h1": 0.001})
        stage2_start = pd.Timestamp("2003-01-01")
        stats = family_statistics(
            returns,
            family,
            search_end=SEARCH_END,
            holdout_start=HOLDOUT_START,
            stage2_start=stage2_start,
            n_boot=0,
        )
        s = stats["split"]
        best = s.loc["h1", "best"]
        window2 = returns.loc[stage2_start:SEARCH_END, best]
        assert s.loc["h1", "p2"] == pytest.approx(sharpe_test(window2).pvalue_greater)
        ho = returns.loc[HOLDOUT_START:, best]
        assert s.loc["h1", "ho_sharpe"] == pytest.approx(sharpe_test(ho).sharpe)
        assert s.loc["h1", "ho_n_obs"] == len(ho)

    def test_calibration_runs_the_dated_statistics(self):
        # One code path: calibration's positional periods, mapped to dates.
        fs = calibration.make_family_set(6, 3, 0.2, 1.5, seed=0, n_search=3 * 252, n_holdout=252)
        via_calibration = calibration.family_statistics(fs, stage2_years=1, n_boot=0)
        direct = family_statistics(
            fs.returns,
            fs.family,
            search_end=fs.search.index[-1],
            holdout_start=fs.holdout.index[0],
            stage2_start=fs.search.index[-252],
            n_boot=0,
        )
        for design in ("reuse", "split"):
            pd.testing.assert_frame_equal(
                via_calibration[design].drop(columns="truth"), direct[design]
            )


class TestPandas3Safety:
    def test_best_by_family_all_nan_family_gives_nan(self):
        values = pd.Series({"a1": 0.1, "a2": 0.3, "b1": np.nan, "b2": np.nan})
        family = pd.Series({"a1": "A", "a2": "A", "b1": "B", "b2": "B"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            best = funnel.best_by_family(values, family)
        assert best["A"] == "a2"
        assert pd.isna(best["B"])
        assert list(best.index) == ["A", "B"]

    def test_a_null_set_where_nothing_passes_the_naive_stack(self):
        returns, family = _dated(seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = family_statistics(
                returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0
            )["reuse"]
        assert not s["naive_pass"].any()
        assert s["nv_p"].isna().all()


class TestHoldoutStatistics:
    """Stage 4's input on its own, so the gate can open the holdout last."""

    def test_holdout_statistics_match_family_statistics(self):
        returns, family = _dated(drift={"h0": 0.003})
        stats = family_statistics(
            returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0
        )["reuse"]
        holdout = returns.loc[HOLDOUT_START:]
        ho = funnel.holdout_statistics(holdout, stats["best"])
        expected = stats[[c for c in stats.columns if c.startswith("ho_")]]
        pd.testing.assert_frame_equal(ho, expected)

    def test_holdout_statistics_read_only_the_chosen_columns(self):
        returns, family = _dated()
        holdout = returns.loc[HOLDOUT_START:]
        chosen = pd.Series({"h0": "h0_v1", "h1": np.nan})
        poisoned = holdout.copy()
        poisoned[[c for c in holdout.columns if c != "h0_v1"]] = np.inf
        ho = funnel.holdout_statistics(poisoned, chosen)
        assert np.isfinite(ho.loc["h0"]).all()
        assert ho.loc["h1"].isna().all()


class TestOneFunnel:
    def test_calibration_uses_the_funnel_objects(self):
        assert calibration.Thresholds is funnel.Thresholds
        assert calibration.run_funnel is funnel.run_funnel
        assert calibration._final_gate is funnel._final_gate
        assert calibration.STAGE1_METHODS == funnel.STAGE1_METHODS


class TestThresholds:
    def test_defaults_are_the_test_runs_best_set_on_rho(self):
        th = Thresholds()
        assert (th.q1, th.stage1, th.alpha2, th.final, th.q4) == (0.10, "rho", 1.0, "bh", 0.30)

    def test_modes_are_literals(self):
        hints = typing.get_type_hints(Thresholds)
        assert typing.get_origin(hints["final"]) is typing.Literal
        assert set(typing.get_args(hints["final"])) == {"dsr", "bh"}
        assert set(typing.get_args(hints["stage1"])) == {"independent", "rho", "bootstrap"}
        assert set(typing.get_args(hints["final_trials"])) == {"searched", "survivors"}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"final": "holm"},
            {"stage1": "lfdr"},
            {"final_trials": "half"},
            {"q1": 0.0},
            {"q4": 1.0},
            {"alpha2": 0.0},
        ],
    )
    def test_bad_settings_raise_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            Thresholds(**kwargs)

    def test_thresholds_dict_is_plain_data_that_rebuilds_the_setting(self):
        th = Thresholds(q1=0.05, stage1="bootstrap", final="dsr", d4=0.9)
        d = thresholds_dict(th)
        assert set(d) == set(funnel._REGISTERED_FIELDS)
        assert json.loads(json.dumps(d)) == d
        assert Thresholds(**d) == th


class TestRunningTheFunnel:
    def test_default_funnel_on_rho_p_values(self):
        returns, family = _dated(drift={"h0": 0.003})
        stats = family_statistics(
            returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0
        )["reuse"]
        stages = run_funnel(stats, Thresholds())
        assert stages.loc["h0"].all()
        assert (stages["stage2"] == stages["stage1"]).all()  # no stage-2 gate

    def test_missing_stage1_p_values_raise(self):
        returns, family = _dated()
        stats = family_statistics(
            returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0
        )["reuse"]
        with pytest.raises(ValueError, match="bootstrap"):
            run_funnel(stats, Thresholds(stage1="bootstrap"))

    def test_stage4_without_holdout_statistics_raises(self):
        returns, family = _dated(drift={"h0": 0.003})
        stats = family_statistics(
            returns.loc[:SEARCH_END], family, search_end=SEARCH_END, n_boot=0
        )["reuse"]
        with pytest.raises(ValueError, match="holdout"):
            run_funnel(stats, Thresholds(q1=0.5))

    def test_screen_is_run_funnels_first_three_stages(self):
        returns, family = _dated(drift={"h0": 0.003, "h1": 0.001})
        stats = family_statistics(
            returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=199
        )["reuse"]
        for th in (Thresholds(), Thresholds(stage1="bootstrap", alpha2=0.05)):
            pd.testing.assert_frame_equal(
                funnel.screen(stats, th), run_funnel(stats, th)[["stage1", "stage2", "stage3"]]
            )

    def test_screen_needs_no_holdout(self):
        returns, family = _dated(drift={"h0": 0.003})
        stats = family_statistics(
            returns.loc[:SEARCH_END], family, search_end=SEARCH_END, n_boot=0
        )["reuse"]
        assert funnel.screen(stats, Thresholds()).loc["h0"].all()
