"""Tests for capstone.power's simulated worlds: null batches and planted families.

The contract:

- `PowerSpec` normalises its variant counts and refuses a configuration the
  funnel can't run (a family with fewer than 2 variants, too short for the
  bootstrap, too few draws for its resolution, correlations out of order,
  fewer trials than variants);
- the simulation is the funnel's own code: a family's statistics computed
  alone equal its row computed with every family, and planting a Sharpe
  equals the calibration generator's planted family; a planted side's columns
  equal the full batch recomputed with the target planted;
- `within_corr` / `across_corr` are the correlations the generator produces;
- the same seed gives the same world.

Small markets (a few families, a few years) keep these fast.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from capstone import funnel, power
from capstone.calibration import make_family_set
from capstone.power import PowerSpec

SMALL = dict(
    variants=(3, 3, 3, 3),
    within_corr=0.8,
    across_corr=0.1,
    search_periods=4 * 252,
    holdout_periods=2 * 252,
    n_boot=99,  # n_boot + 1 >= 20 x 4 families
)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


class TestPowerSpec:
    def test_frozen_and_normalised(self):
        spec = PowerSpec(
            n_families=3, variants=4, **{k: v for k, v in SMALL.items() if k != "variants"}
        )
        assert spec.variants == (4, 4, 4) and spec.n_families == 3
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.n_boot = 1

    def test_list_of_variants_sets_the_family_count(self):
        spec = PowerSpec(**{**SMALL, "variants": [2, 5, 3], "n_boot": 59})
        assert spec.variants == (2, 5, 3) and spec.n_families == 3

    def test_from_years(self):
        spec = PowerSpec.from_years(
            search_years=31, holdout_years=5, variants=(2, 2), within_corr=0.5, n_boot=39
        )
        assert spec.search_periods == 31 * 252 and spec.holdout_periods == 5 * 252

    @pytest.mark.parametrize(
        "change",
        [
            {"variants": (3, 0)},
            {"n_families": 5},  # disagrees with the 4 listed families
            {"within_corr": 0.05},  # below the across-family correlation
            {"across_corr": -0.1},
            {"within_corr": 1.2},
            {"search_periods": 10},
            {"n_trials": 5},  # fewer trials than the 12 variants scored
        ],
    )
    def test_refuses(self, change):
        with pytest.raises(ValueError):
            PowerSpec(**{**SMALL, **change})

    @pytest.mark.parametrize("change", [{"variants": (2, 5, 1)}, {"variants": 1, "n_families": 4}])
    def test_refuses_a_family_of_one_variant(self, change):
        # The funnel needs at least 2 trials per family; one variant would
        # only fail later, inside null_side.
        with pytest.raises(ValueError, match="at least 2 variants"):
            PowerSpec(**{**SMALL, **change})

    @pytest.mark.parametrize("stage1", ["bootstrap", "rho", "independent"])
    def test_bootstrap_resolution_is_checked_whenever_it_runs(self, stage1):
        # null_side computes the bootstrap whenever n_boot > 0, whatever the
        # stage-1 method, so a spec it can't run is refused here, not there.
        th = funnel.Thresholds(stage1=stage1)
        with pytest.raises(ValueError, match="n_boot"):
            PowerSpec(**{**SMALL, "n_boot": 50, "thresholds": th})  # 51 < 20 x 4

    def test_no_draws_are_needed_without_the_bootstrap(self):
        spec = PowerSpec(**{**SMALL, "n_boot": 0})  # "rho" skips the bootstrap
        assert power.null_side(spec, seed=5).stats["p1_bootstrap"].isna().all()
        with pytest.raises(ValueError, match="n_boot"):
            PowerSpec(**{**SMALL, "n_boot": 0, "thresholds": funnel.Thresholds(stage1="bootstrap")})

    def test_bootstrap_resolution_is_checked_at_the_registered_q1(self):
        # BH at q1 = 0.01 across 4 families needs n_boot + 1 >= 400, not 80.
        boot = funnel.Thresholds(q1=0.01, stage1="bootstrap")
        with pytest.raises(ValueError, match="n_boot"):
            PowerSpec(**{**SMALL, "n_boot": 199, "thresholds": boot})
        spec = PowerSpec(**{**SMALL, "n_boot": 399, "thresholds": boot})
        side = power.null_side(spec, seed=5)
        assert side.kwargs["q"] == 0.01
        target = side.stats.index[0]
        strong = power.planted_side(side, [target], 5.0)
        assert funnel.screen(strong, spec.thresholds).loc[target, "stage1"]


# ---------------------------------------------------------------------------
# The simulation is the funnel's own code
# ---------------------------------------------------------------------------


class TestSameCodePaths:
    def test_a_family_alone_has_its_row_from_the_full_batch(self):
        # Power recomputes only the planted family per Sharpe: valid because
        # each family's statistics (bootstrap included, same seed) depend on
        # its own columns alone.
        fs = make_family_set(4, 3, 0.25, 1.0, seed=4, n_search=3 * 252, n_holdout=252)
        dates = pd.bdate_range("1990-01-01", periods=len(fs.returns))
        r = fs.returns.set_axis(dates)
        kwargs = dict(
            search_end=dates[fs.n_search - 1], holdout_start=dates[fs.n_search], n_boot=99, seed=7
        )
        full = funnel.family_statistics(r, fs.family, **kwargs)["reuse"]
        cols = ["best", "p1_bootstrap", "p2", "ho_p", "ho_sharpe", "ho_n_obs"]
        for fam in full.index:
            own = fs.family.index[fs.family == fam]
            alone = funnel.family_statistics(r[own], fs.family[own], **kwargs)["reuse"]
            assert alone.loc[fam, cols].tolist() == full.loc[fam, cols].tolist()
            # The formula p-value is vectorised across families: equal to the last bits.
            assert alone.loc[fam, "p1_rho"] == pytest.approx(full.loc[fam, "p1_rho"], rel=1e-12)

    def test_planting_recomputes_only_the_target(self):
        side = power.null_side(PowerSpec(**SMALL), seed=5)
        target = side.stats.index[0]
        read = ["best", "p1_rho", "p1_bootstrap", "p2", "ho_p"]
        zero = power.planted_side(side, [target], 0.0)
        pd.testing.assert_frame_equal(zero[read], side.stats[read], rtol=1e-12)
        strong = power.planted_side(side, [target], 5.0)
        pd.testing.assert_frame_equal(
            strong.drop(index=target), side.stats[strong.columns].drop(index=target)
        )
        assert strong.loc[target, "p2"] < side.stats.loc[target, "p2"]
        assert strong.attrs["n_variants"] == side.stats.attrs["n_variants"]

    def test_planted_side_equals_the_full_batch_recomputed(self):
        # Every column planted_side returns is what the funnel gives on the
        # whole planted batch. The naive stack's columns (`naive_pass`,
        # `nv_*`) are left out: their Holm, BY and BH corrections and the
        # DSR's trial count run over the whole batch, not the target's
        # variants. In this world the target alone passes the naive stack;
        # in the full batch it doesn't.
        spec = PowerSpec(
            variants=(3,) * 6,
            within_corr=0.5,
            search_periods=4 * 252,
            holdout_periods=2 * 252,
            n_boot=0,
        )
        side = power.null_side(spec, seed=3)
        target = side.stats.index[0]
        planted = power.planted_side(side, [target], 1.0)
        returns = power.plant(side.fs.returns, side.fs.family, target, 1.0)
        full = funnel.family_statistics(returns, side.fs.family, **side.kwargs)["reuse"]
        assert not any(c == "naive_pass" or c.startswith("nv_") for c in planted.columns)
        pd.testing.assert_frame_equal(planted, full[planted.columns], rtol=1e-12)

    def test_planting_matches_the_calibration_generator(self):
        planted = make_family_set(5, 3, 0.2, 0.8, within_rho=0.7, rho=0.1, seed=2)
        null = make_family_set(5, 3, 0.0, 0.0, within_rho=0.7, rho=0.1, seed=2)
        target = planted.truth.index[planted.truth][0]
        shifted = power.plant(null.returns, null.family, target, 0.8)
        np.testing.assert_allclose(shifted.to_numpy(), planted.returns.to_numpy(), atol=1e-15)

    def test_correlations_map_to_the_generator(self):
        # within_corr / across_corr are the measured pairwise correlations of
        # variants in one family / in different families.
        spec = PowerSpec(
            **{**SMALL, "within_corr": 0.76, "across_corr": 0.13, "search_periods": 58 * 252}
        )
        fs = power.null_world(spec, seed=0)
        corr = fs.returns.corr().to_numpy()
        fam = fs.family.to_numpy()
        same = (fam[:, None] == fam[None, :]) & ~np.eye(len(fam), dtype=bool)
        assert corr[same].mean() == pytest.approx(0.76, abs=0.02)
        assert corr[fam[:, None] != fam[None, :]].mean() == pytest.approx(0.13, abs=0.03)
        assert not fs.truth.any()


class TestDeterminism:
    def test_same_seed_same_world(self):
        spec = PowerSpec(**SMALL)
        a, b = power.null_side(spec, seed=9), power.null_side(spec, seed=9)
        pd.testing.assert_frame_equal(a.fs.returns, b.fs.returns, rtol=0, atol=0)
        pd.testing.assert_frame_equal(a.stats, b.stats, rtol=0, atol=0)
        target = a.stats.index[2]
        pd.testing.assert_frame_equal(
            power.planted_side(a, [target], 1.0), power.planted_side(b, [target], 1.0)
        )

    def test_another_seed_another_world(self):
        spec = PowerSpec(**SMALL)
        a, b = power.null_side(spec, seed=9), power.null_side(spec, seed=10)
        assert not a.fs.returns.equals(b.fs.returns)


# ---------------------------------------------------------------------------
# Synthetic defects: autocorrelation, fat tails, volatility clustering
# ---------------------------------------------------------------------------

LONG = {**SMALL, "search_periods": 20_000, "holdout_periods": 1_000, "n_boot": 99}


def _lag1(x: np.ndarray) -> float:
    """Mean lag-1 autocorrelation of the columns of `x`."""
    x = x - x.mean(axis=0)
    return float(((x[1:] * x[:-1]).sum(0) / (x * x).sum(0)).mean())


def _excess_kurtosis(x: np.ndarray) -> float:
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    return float(((z**4).mean(0) - 3).mean())


class TestDefects:
    def test_no_defects_is_the_gaussian_world_bit_for_bit(self):
        spec = PowerSpec(**SMALL)
        assert spec.defects == dict(rho=0.1, ar1=0.0, t_df=None, garch=None)
        fs = power.null_world(spec, seed=3)
        ref = make_family_set(
            4, (3,) * 4, 0.0, 0.0, within_rho=spec.within_rho, rho=0.1,
            n_search=spec.search_periods, n_holdout=spec.holdout_periods, seed=3,
        )  # fmt: skip
        np.testing.assert_array_equal(fs.returns.to_numpy(), ref.returns.to_numpy())

    def test_ar1_worlds_are_autocorrelated(self):
        iid = power.null_world(PowerSpec(**LONG), seed=1).returns.to_numpy()
        ar = power.null_world(PowerSpec(**LONG, ar1=0.4), seed=1).returns.to_numpy()
        assert abs(_lag1(iid)) < 0.03
        assert _lag1(ar) == pytest.approx(0.4, abs=0.04)

    def test_t_worlds_have_fat_tails(self):
        iid = power.null_world(PowerSpec(**LONG), seed=1).returns.to_numpy()
        fat = power.null_world(PowerSpec(**LONG, t_df=4.0), seed=1).returns.to_numpy()
        assert abs(_excess_kurtosis(iid)) < 0.3
        assert _excess_kurtosis(fat) > 1.0

    def test_garch_worlds_cluster_volatility(self):
        iid = power.null_world(PowerSpec(**LONG), seed=1).returns.to_numpy()
        spec = PowerSpec(**LONG, garch=(0.1, 0.85))
        g = power.null_world(spec, seed=1).returns.to_numpy()
        assert abs(_lag1(iid**2)) < 0.03
        assert _lag1(g**2) > 0.1
        assert abs(_lag1(g)) < 0.03  # returns themselves stay uncorrelated

    @pytest.mark.parametrize(
        "defects",
        [
            dict(ar1=0.5),
            dict(t_df=5.0),
            dict(garch=(0.1, 0.8)),
            dict(ar1=0.3, t_df=6, garch=(0.05, 0.9)),
        ],
    )
    def test_defect_worlds_stay_null_with_unit_volatility(self, defects):
        spec = PowerSpec(**LONG, **defects)
        fs = power.null_world(spec, seed=2)
        assert not fs.truth.any()
        r = fs.returns.to_numpy()
        assert r.std(axis=0).mean() == pytest.approx(power.VOL, rel=0.08)
        assert abs(r.mean(axis=0)).max() < 4 * power.VOL / np.sqrt(len(r)) * 1.5

    @pytest.mark.parametrize(
        "defects", [dict(ar1=0.5), dict(t_df=5.0), dict(garch=(0.1, 0.8))]
    )  # fmt: skip
    def test_planting_matches_the_generator_under_defects(self, defects):
        kwargs = dict(within_rho=0.7, rho=0.1, seed=2, **defects)
        planted = make_family_set(5, 3, 0.2, 0.8, **kwargs)
        null = make_family_set(5, 3, 0.0, 0.0, **kwargs)
        target = planted.truth.index[planted.truth][0]
        shifted = power.plant(null.returns, null.family, target, 0.8)
        np.testing.assert_allclose(shifted.to_numpy(), planted.returns.to_numpy(), atol=1e-15)

    def test_planted_side_equals_the_full_batch_recomputed_under_defects(self):
        spec = PowerSpec(**SMALL, ar1=0.4, t_df=5.0, garch=(0.1, 0.8))
        side = power.null_side(spec, seed=3)
        target = side.stats.index[0]
        planted = power.planted_side(side, [target], 1.0)
        returns = power.plant(side.fs.returns, side.fs.family, target, 1.0)
        full = funnel.family_statistics(returns, side.fs.family, **side.kwargs)["reuse"]
        pd.testing.assert_frame_equal(planted, full[planted.columns], rtol=1e-12)

    def test_an_ar1_null_is_not_flagged_as_power(self):
        # The point of an AR(1) null: with no plant, stage 1's bootstrap should
        # not reject families at a rate beyond its level across seeds.
        spec = PowerSpec(**{**SMALL, "n_boot": 199}, ar1=0.3)
        hits = [
            (power.null_side(spec, seed=s).stats["p2"] < 0.05).mean() for s in range(6)
        ]  # fmt: skip
        assert np.mean(hits) < 0.3

    def test_same_seed_same_defect_world(self):
        spec = PowerSpec(**SMALL, ar1=0.4, t_df=5.0, garch=(0.1, 0.8))
        a, b = power.null_side(spec, seed=9), power.null_side(spec, seed=9)
        pd.testing.assert_frame_equal(a.fs.returns, b.fs.returns, rtol=0, atol=0)
        pd.testing.assert_frame_equal(a.stats, b.stats, rtol=0, atol=0)
        assert not a.fs.returns.equals(power.null_world(PowerSpec(**SMALL), seed=9).returns)

    @pytest.mark.parametrize(
        "change",
        [
            dict(ar1=1.0), dict(ar1=-1.0), dict(t_df=2.0), dict(t_df=float("nan")),
            dict(garch=(0.5, 0.5)), dict(garch=(-0.1, 0.5)), dict(garch=(0.1, 0.2, 0.3)),
        ],
    )  # fmt: skip
    def test_refuses_defects_the_generator_cannot_make(self, change):
        with pytest.raises(ValueError):
            PowerSpec(**SMALL, **change)

    def test_garch_is_stored_as_a_tuple(self):
        assert PowerSpec(**SMALL, garch=[0.1, 0.8]).garch == (0.1, 0.8)


class TestPlantTargets:
    def test_a_duplicate_target_is_refused_not_planted_twice(self):
        side = power.null_side(PowerSpec(**SMALL), seed=5)
        target = side.stats.index[0]
        with pytest.raises(ValueError, match="duplicate"):
            power.planted_side(side, [target, target], 1.0)

    def test_an_unknown_target_is_named_not_a_holdout_error(self):
        side = power.null_side(PowerSpec(**SMALL), seed=5)
        with pytest.raises(ValueError, match="unknown plant targets.*nope"):
            power.planted_side(side, ["nope"], 1.0)
        with pytest.raises(ValueError, match="unknown plant targets"):
            power.planted_side(side, [side.stats.index[0], "nope"], 1.0)

    def test_plant_refuses_an_unknown_target(self):
        fs = power.null_world(PowerSpec(**SMALL), seed=5)
        with pytest.raises(ValueError, match="unknown plant target"):
            power.plant(fs.returns, fs.family, "nope", 1.0)

    def test_several_distinct_targets_still_plant(self):
        side = power.null_side(PowerSpec(**SMALL), seed=5)
        two = list(side.stats.index[:2])
        stats = power.planted_side(side, two, 3.0)
        assert (stats.loc[two, "p2"] < side.stats.loc[two, "p2"]).all()
