"""Tests for capstone.funnel: the funnel's statistics on dated data.

Periods are cut by date, never by row position, and a frame that reaches
past the search period is refused unless the caller names the holdout
explicitly. Stages 1-3 never see a holdout date.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from capstone import calibration, funnel
from capstone.evaluate import sharpe_test
from capstone.funnel import family_statistics

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

    @pytest.mark.parametrize("n_boot", [0, 199])
    @pytest.mark.parametrize("label", ["absent", "nan"])
    def test_every_variant_needs_a_hypothesis_label(self, n_boot, label):
        # A variant without a label would drop out of its family and k would
        # undercount the search; n_boot=0 skips the bootstrap's own check.
        returns, family = _dated(n_families=4, drift={"h0": 0.003})
        h0_v2 = family.index == "h0_v2"
        family = family[~h0_v2] if label == "absent" else family.where(~h0_v2)
        with pytest.raises(ValueError, match="family is missing for 1 columns"):
            family_statistics(
                returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=n_boot
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

    def test_best_by_family_keeps_a_family_absent_from_values(self):
        # Callers pass filtered values (funnel's naive stack does); every
        # family in `family` still gets a row, in `family`'s order.
        values = pd.Series({"b1": 0.2, "a1": 0.1})
        family = pd.Series({"a1": "A", "b1": "B", "c1": "C"})
        best = funnel.best_by_family(values, family)
        assert list(best.index) == ["A", "B", "C"]
        assert best["A"] == "a1"
        assert best["B"] == "b1"
        assert pd.isna(best["C"])

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


def _score(returns, family, **kw):
    return family_statistics(
        returns, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=0, **kw
    )


class TestVariantsWithoutPeriodData:
    """An unchosen variant with no data in a period must not block scoring."""

    def _unchosen(self, returns, family, stats):
        carried = set(stats["reuse"]["best"].dropna()) | set(stats["split"]["best"].dropna())
        return next(c for c in returns.columns if c not in carried)

    def _base(self):
        returns, family = _dated(drift={"h0": 0.003}, k=3)
        stats = _score(returns, family, stage2_start="2003-01-01")
        return returns, family, stats

    def test_unchosen_variant_without_holdout_does_not_block(self):
        returns, family, base = self._base()
        victim = self._unchosen(returns, family, base)
        blanked = returns.copy()
        blanked.loc[blanked.index >= HOLDOUT_START, victim] = np.nan
        got = _score(blanked, family, stage2_start="2003-01-01")
        # The search period is untouched, so the whole result is unchanged.
        for design in base:
            pd.testing.assert_frame_equal(got[design], base[design])

    def test_chosen_variant_without_holdout_is_refused_by_name(self):
        returns, family, base = self._base()
        chosen = base["reuse"]["best"].iloc[0]
        blanked = returns.copy()
        blanked.loc[blanked.index >= HOLDOUT_START, chosen] = np.nan
        with pytest.raises(ValueError, match=f"{chosen}.*holdout"):
            _score(blanked, family)

    def test_holdout_statistics_refuses_a_chosen_variant_by_name(self):
        returns, family, base = self._base()
        chosen = base["reuse"]["best"]
        holdout = returns.loc[HOLDOUT_START:].copy()
        holdout[chosen.iloc[0]] = np.nan
        with pytest.raises(ValueError, match=chosen.iloc[0]):
            funnel.holdout_statistics(holdout, chosen)

    @pytest.mark.parametrize("n_boot", [0, 199])
    def test_variant_without_search_data_is_refused_by_name(self, n_boot):
        returns, family, base = self._base()
        victim = self._unchosen(returns, family, base)
        blanked = returns.copy()
        blanked.loc[blanked.index <= SEARCH_END, victim] = np.nan
        with pytest.raises(ValueError, match=f"{victim}.*search period"):
            family_statistics(
                blanked, family, search_end=SEARCH_END, holdout_start=HOLDOUT_START, n_boot=n_boot
            )

    def test_period_shortfalls_counts_each_period(self):
        returns, _ = _dated(n_families=1, k=2)
        a = returns.iloc[:50].copy()
        a.iloc[1:, 0] = np.nan
        b = returns.iloc[50:100]
        assert funnel.period_shortfalls({"search": a, "holdout": b}) == {"search": {"h0_v0": 1}}


class TestDateContract:
    """One date contract for the funnel and the intake in front of it."""

    def test_duplicate_dates_are_refused(self):
        returns, family = _dated()
        dup = returns.iloc[list(range(len(returns))) + [10]].sort_index(kind="stable")
        with pytest.raises(ValueError, match="unique"):
            _score(dup, family)

    def test_timezone_aware_dates_are_refused(self):
        returns, family = _dated()
        aware = returns.tz_localize("UTC")
        with pytest.raises(ValueError, match="timezone"):
            _score(aware, family)

    def test_dates_with_a_time_of_day_are_refused(self):
        # 16:00 on search_end would fall in the gap between the periods.
        returns, family = _dated()
        stamped = returns.copy()
        stamped.index = stamped.index + pd.Timedelta(hours=16)
        with pytest.raises(ValueError, match="normalised"):
            _score(stamped, family)

    def test_check_dates_accepts_a_clean_index(self):
        funnel.check_dates(pd.bdate_range("2000-01-03", periods=5))
