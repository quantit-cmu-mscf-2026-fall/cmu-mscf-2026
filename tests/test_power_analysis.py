"""Tests for capstone.power's analysis: stage power and minimum detectable Sharpe.

The contract:

- at Sharpe 0 each gated stage passes at its nominal level (stage 1: any
  family, BH's familywise rate under the complete null, q1; stage 4: q4 for a
  lone null survivor), on iid, cross-correlated (`rho`), autocorrelated (`ar1`)
  and fat-tailed GARCH worlds, so the procedure can say "nothing here";
- power rises with the true Sharpe, per stage and end to end, and a planted
  strength changes it under fixed seeds;
- with a single survivor the holdout's minimum detectable Sharpe at 80% power
  is the one-sided closed form (z_{1-q4} + z_{0.8}) / sqrt(years);
- the same seed gives the same report;
- the batched Sharpe grid is the funnel's own code: each Sharpe equals
  `planted_side` at that Sharpe.

Small markets (a few families, a few years) keep these fast.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats

from capstone import funnel, power
from capstone.funnel import Thresholds
from capstone.power import PowerBudget, PowerReport, PowerSpec, power_analysis

SMALL = dict(
    variants=(3, 3, 3, 3),
    within_corr=0.8,
    across_corr=0.1,
    search_periods=4 * 252,
    holdout_periods=2 * 252,
    n_boot=99,  # n_boot + 1 >= 20 x 4 families
)
# The null-calibration worlds: the defaults, then each defect the generator has.
WORLDS = {
    "iid": {},
    "rho": {"across_corr": 0.4},
    "ar1": {"ar1": 0.4},
    "fat_tails_garch": {"t_df": 4.0, "garch": (0.1, 0.85)},
}


def stage(report: PowerReport, name: str) -> power.StagePower:
    return {s.stage: s for s in report.stages}[name]


class TestBudget:
    def test_defaults_and_frozen(self):
        budget = PowerBudget(sharpe_grid=[0, 1])
        assert budget.sharpe_grid == (0.0, 1.0) and budget.n_sims == 200
        with pytest.raises(dataclasses.FrozenInstanceError):
            budget.seed = 1

    @pytest.mark.parametrize(
        "bad",
        [
            {"n_sims": 0},
            {"sharpe_grid": ()},
            {"sharpe_grid": (0.5, 0.2)},
            {"sharpe_grid": (0.0, 0.0)},
            {"sharpe_grid": (-0.1, 0.5)},
            {"sharpe_grid": (0.0, float("nan"))},
        ],
    )
    def test_refuses(self, bad):
        with pytest.raises(ValueError):
            PowerBudget(**bad)


class TestCrossing:
    grid = (0.0, 1.0, 2.0, 3.0)

    def test_interpolates_between_grid_points(self):
        curve = np.array([0.0, 0.4, 0.8, 1.0])
        assert power._crossing(self.grid, curve, 0.5) == pytest.approx(1.25)
        assert power._crossing(self.grid, curve, 0.8) == pytest.approx(2.0)

    def test_uses_the_running_maximum(self):
        # Monte Carlo noise: a dip after the first crossing doesn't move it.
        curve = np.array([0.0, 0.6, 0.4, 0.9])
        assert power._crossing(self.grid, curve, 0.5) == pytest.approx(5 / 6)
        assert power._crossing(self.grid, curve, 0.8) == pytest.approx(2 + 0.2 / 0.3)

    def test_none_when_never_reached_and_first_point_when_already_there(self):
        assert power._crossing(self.grid, np.array([0.0, 0.1, 0.2, 0.3]), 0.5) is None
        assert power._crossing(self.grid, np.array([0.7, 0.8, 0.9, 1.0]), 0.5) == 0.0


class TestSameCodePaths:
    @pytest.mark.parametrize("defects", [{}, {"ar1": 0.4}], ids=["iid", "ar1"])
    def test_batched_sharpe_grid_equals_one_sharpe_at_a_time(self, defects, monkeypatch):
        # `planted_sides` (what `power_analysis` runs) scores every grid Sharpe in a
        # few funnel calls; each must equal `planted_side` at that Sharpe on every
        # column the stages read. A tiny chunk bound forces several calls.
        spec = PowerSpec(**{**SMALL, **defects})
        side = power.null_side(spec, seed=21)
        targets, grid = list(side.stats.index[[1, 3]]), (0.0, 0.7, 1.5, 3.0)
        monkeypatch.setattr(power, "_CHUNK_ELEMENTS", 3 * spec.search_periods * 6)
        batched = power.planted_sides(side, targets, grid)
        assert len(batched) == len(grid)
        for got, sharpe in zip(batched, grid, strict=True):
            want = power.planted_side(side, targets, sharpe)
            assert list(got.columns) == list(want.columns)
            assert got["best"].tolist() == want["best"].tolist()
            numeric = [c for c in want if c != "best"]
            np.testing.assert_allclose(
                got[numeric].to_numpy(float), want[numeric].to_numpy(float), rtol=1e-12
            )
            assert got.attrs == want.attrs

    def test_batched_grid_refuses_what_planted_side_refuses(self):
        side = power.null_side(PowerSpec(**SMALL), seed=1)
        first = side.stats.index[0]
        with pytest.raises(ValueError, match="duplicate plant targets"):
            power.planted_sides(side, [first, first], (0.0, 1.0))
        with pytest.raises(ValueError, match="unknown plant targets"):
            power.planted_sides(side, ["no such family"], (0.0, 1.0))

    def test_end_to_end_acceptance_is_the_funnels(self):
        # `accepted` is `run_funnel`'s stage 4, for the target and the null families alike.
        spec = PowerSpec(**SMALL)
        side = power.null_side(spec, seed=3)
        planted = power.planted_side(side, [side.stats.index[0]], 3.0)
        want = funnel.run_funnel(planted, spec.thresholds)["stage4"]
        assert power.accepted(planted, spec.thresholds).equals(want)
        assert want.any()  # a Sharpe-3 family is accepted: the check is not vacuous


@pytest.fixture(scope="module", params=sorted(WORLDS))
def null_report(request) -> PowerReport:
    spec = PowerSpec(**{**SMALL, **WORLDS[request.param]})
    return power_analysis(spec, PowerBudget(n_sims=150, sharpe_grid=(0.0,), seed=11))


class TestNullLevels:
    def test_nominal_levels_are_the_thresholds(self, null_report):
        th = Thresholds()
        assert [s.stage for s in null_report.stages] == ["stage1", "stage2", "stage3", "stage4"]
        nominal = {s.stage: s.nominal for s in null_report.stages}
        assert nominal == {"stage1": th.q1, "stage2": th.alpha2, "stage3": 1.0, "stage4": th.q4}

    def test_stage1_any_family_passes_at_about_q1(self, null_report):
        # BH's familywise rate under the complete null is q1 = 0.10 (standard error
        # 0.025 over 150 worlds); the bounds catch a stage that never or always passes.
        s = stage(null_report, "stage1")
        assert s.gated and 0.03 <= s.null_rate <= 0.20

    def test_stage4_passes_a_lone_null_survivor_at_q4(self, null_report):
        s = stage(null_report, "stage4")
        assert s.gated and s.null_rate == pytest.approx(0.30, abs=0.08)

    def test_end_to_end_null_accepts_rarely(self, null_report):
        # Stage 1 passes in ~q1 of null worlds and stage 4 at ~q4: about 0.03.
        assert null_report.end_to_end_null <= 0.09

    def test_stages_that_are_no_gate_pass_everything(self, null_report):
        for name in ("stage2", "stage3"):
            s = stage(null_report, name)
            assert not s.gated and s.null_rate == 1.0 and s.power == (1.0,)
            assert s.mds50 is None and s.mds80 is None


class TestPower:
    GRID = (0.0, 0.5, 1.0, 1.5, 2.5)

    @pytest.fixture(scope="class", params=["iid", "rho", "ar1"])
    @classmethod
    def report(cls, request) -> PowerReport:
        spec = PowerSpec(**{**SMALL, **WORLDS[request.param]})
        return power_analysis(spec, PowerBudget(n_sims=40, sharpe_grid=cls.GRID, seed=5))

    def test_power_rises_with_sharpe(self, report):
        assert report.sharpe_grid == self.GRID
        curves = [s.power for s in report.stages if s.gated] + [report.end_to_end]
        for curve in curves:
            assert len(curve) == 5
            assert (np.diff(curve) >= -0.15).all(), curve  # Monte Carlo noise
            assert curve[-1] > curve[0] + 0.5, curve
        assert report.end_to_end[-1] >= 0.8
        assert report.end_to_end[0] <= 0.2  # no strength, little acceptance

    def test_minimum_detectable_sharpe_is_on_the_curve(self, report):
        for s in [*report.stages, report.end_to_end_stage]:
            if not s.gated:
                assert s.mds50 is None and s.mds80 is None
                continue
            assert s.mds50 is not None and s.mds80 is not None
            assert 0.0 <= s.mds50 <= s.mds80 <= 2.5
            for level, mds in ((0.5, s.mds50), (0.8, s.mds80)):
                at_mds = np.interp(mds, report.sharpe_grid, np.maximum.accumulate(s.power))
                assert at_mds == pytest.approx(level, abs=1e-9) or mds == 0.0

    def test_the_summary_lists_every_stage_and_end_to_end(self, report):
        mds = report.min_detectable_sharpe()
        assert [m.stage for m in mds] == ["stage1", "stage2", "stage3", "stage4", "end_to_end"]
        by_stage = {m.stage: m for m in mds}
        assert by_stage["end_to_end"].power_80 == report.end_to_end_mds80
        assert not by_stage["stage2"].gated and by_stage["stage2"].power_50 is None

    def test_the_holdout_test_matches_the_closed_form(self):
        # One family (two identical variants: the funnel needs two to measure a
        # family's correlation): the only stage-4 survivor, tested alone, one-sided
        # at q4 over 5 years. 250 worlds: each point of the curve has a standard
        # error of at most 0.032, the 80% crossing about 0.045.
        spec = PowerSpec(
            variants=(2,), within_corr=1.0, search_periods=252, holdout_periods=5 * 252, n_boot=19
        )
        grid = (0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8)
        s4 = stage(
            power_analysis(spec, PowerBudget(n_sims=250, sharpe_grid=grid, seed=3)), "stage4"
        )
        z, q4 = stats.norm.ppf, Thresholds().q4
        closed = stats.norm.cdf(np.array(grid) * np.sqrt(5) - z(1 - q4))
        np.testing.assert_allclose(s4.power, closed, atol=0.08)
        assert s4.null_rate == pytest.approx(q4, abs=0.08)
        assert s4.mds80 == pytest.approx((z(1 - q4) + z(0.8)) / np.sqrt(5), abs=0.1)
        assert s4.mds50 == pytest.approx(z(1 - q4) / np.sqrt(5), abs=0.1)

    def test_dsr_stage4_is_measured_on_its_own_level(self):
        # Another registered gate: nominal is 1 - d4, and a Sharpe 2.5 target is found.
        th = Thresholds(final="dsr", d4=0.95)
        spec = PowerSpec(**SMALL, thresholds=th)
        report = power_analysis(spec, PowerBudget(n_sims=12, sharpe_grid=(0.0, 2.5), seed=2))
        s4 = stage(report, "stage4")
        assert s4.nominal == pytest.approx(0.05) and s4.power[-1] > s4.power[0]

    def test_bootstrap_stage1_is_measured_at_the_bootstrap_level(self):
        # The grid is batched under the bootstrap's own resolution check (q = 0.05).
        spec = PowerSpec(**SMALL, thresholds=Thresholds(stage1="bootstrap"))
        report = power_analysis(spec, PowerBudget(n_sims=8, sharpe_grid=(0.0, 1.0, 2.5), seed=6))
        s1 = stage(report, "stage1")
        assert s1.gated and s1.power[-1] > s1.power[0]
        assert report.end_to_end[-1] > report.end_to_end[0]


class TestDeterminism:
    BUDGET = PowerBudget(n_sims=4, sharpe_grid=(0.0, 1.0), seed=9)

    def test_same_seed_same_report(self):
        spec = PowerSpec(**SMALL)
        assert power_analysis(spec, self.BUDGET) == power_analysis(spec, self.BUDGET)

    def test_another_seed_another_report(self):
        spec = PowerSpec(**SMALL)
        other = dataclasses.replace(self.BUDGET, seed=10)
        assert power_analysis(spec, self.BUDGET).stages != power_analysis(spec, other).stages

    def test_report_records_its_inputs(self):
        spec = PowerSpec(**SMALL, ar1=0.3)
        report = power_analysis(spec, self.BUDGET)
        assert report.spec == spec and report.budget == self.BUDGET

    def test_the_spec_defects_reach_the_worlds(self):
        # What `power_analysis` simulates on is `null_side(spec, seed)`: its batch
        # carries the spec's autocorrelation, and the plain spec's does not.
        def lag1(spec):
            x = power.null_side(spec, seed=4).fs.returns.to_numpy()
            return np.mean([np.corrcoef(c[:-1], c[1:])[0, 1] for c in x.T])

        assert lag1(PowerSpec(**SMALL)) == pytest.approx(0.0, abs=0.05)
        assert lag1(PowerSpec(**SMALL, ar1=0.5)) == pytest.approx(0.5, abs=0.05)
