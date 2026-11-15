"""Tests for capstone.power's simulated worlds: null batches and planted families.

The contract:

- `PowerSpec` normalises its variant counts and refuses a configuration the
  funnel can't run (too short for the bootstrap, too few draws for its
  resolution, correlations out of order, fewer trials than variants);
- the simulation is the funnel's own code: a family's statistics computed
  alone equal its row computed with every family, and planting a Sharpe
  equals the calibration generator's planted family;
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
        spec = PowerSpec(**{**SMALL, "variants": [2, 5, 1], "n_boot": 59})
        assert spec.variants == (2, 5, 1) and spec.n_families == 3

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

    def test_bootstrap_resolution_is_checked_only_for_the_bootstrap(self):
        boot = funnel.Thresholds(stage1="bootstrap")
        with pytest.raises(ValueError, match="n_boot"):
            PowerSpec(**{**SMALL, "n_boot": 50, "thresholds": boot})  # 51 < 20 x 4
        assert PowerSpec(**{**SMALL, "n_boot": 50}).n_boot == 50  # "rho" reads no bootstrap


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
        pd.testing.assert_frame_equal(strong.drop(index=target), side.stats.drop(index=target))
        assert strong.loc[target, "p2"] < side.stats.loc[target, "p2"]
        assert strong.attrs["n_variants"] == side.stats.attrs["n_variants"]

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
