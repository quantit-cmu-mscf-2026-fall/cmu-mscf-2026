"""Ratio-based thresholds: the simulation, the choice, and the check before a look.

Small batches (8 families of 3 variants, 3 years of search) and a few worlds keep this
fast; the production defaults are documented in `capstone.ratio`. The simulated worlds
are `power`'s and the accepted families are the funnel's own stage 4: no new statistics.
Every test that logs points `CAPSTONE_LEDGER_DIR` at a temporary directory.
"""

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from capstone import funnel, power, ratio, runlog
from capstone.funnel import Thresholds

FDR = Thresholds()
SHAPE = dict(
    variants=(3,) * 8,
    within_corr=0.8,
    across_corr=0.1,
    search_periods=3 * 252,
    holdout_periods=252,
    n_boot=159,  # n_boot + 1 >= 20 x 8 families
)
ALT = 2.0  # detectable about half the time on 3 years at the FDR thresholds
N_SIMS = 12
LAMBDAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1e4)


def small_spec(**overrides) -> ratio.RatioSpec:
    kwargs = dict(SHAPE, miss_cost=1.0, pi1=0.0, alt_sharpe=ALT, n_sims=N_SIMS, seed=0)
    return ratio.RatioSpec(**kwargs | overrides)


@pytest.fixture(scope="module")
def table() -> ratio.RatioTable:
    return ratio.simulate(small_spec())


def test_lambda_one_pi1_zero_is_at_least_as_strict_as_fdr(table):
    th = ratio.calibrate(small_spec(miss_cost=1.0, pi1=0.0))
    assert th.mode == "ratio" and th.ratio == small_spec(miss_cost=1.0, pi1=0.0)
    assert th.q1 <= FDR.q1 and th.q4 <= FDR.q4
    # pi1 = 0: no real family to miss, so the strictest point is chosen.
    assert table.choose(1.0, 0.0) == 0


@pytest.mark.parametrize("pi1", [0.02, 0.05, 0.25])
def test_thresholds_loosen_monotonically_with_lambda(table, pi1):
    chosen = [table.grid[table.choose(lam, pi1)] for lam in LAMBDAS]
    q1, q4 = np.array(chosen).T
    assert (np.diff(q1) >= 0).all() and (np.diff(q4) >= 0).all(), chosen
    false, missed, _ = zip(
        *(table.expected(lam, pi1, at=table.choose(lam, pi1)) for lam in LAMBDAS),
        strict=True,
    )
    assert (np.diff(false) >= 0).all() and (np.diff(missed) <= 0).all()


def test_monotone_even_when_raw_rates_are_noisy():
    # Raw Monte Carlo rates need not be monotone along the chain; the choice reads their
    # envelopes (a looser point passes at least what a stricter one does), so the chosen
    # index still never decreases.
    rng = np.random.default_rng(3)
    grid = ratio.DEFAULT_GRID
    for _ in range(200):
        t = ratio.RatioTable(
            grid=grid,
            false_passes=tuple(tuple(rng.uniform(0, 8 - r, len(grid))) for r in range(9)),
            misses=tuple(tuple(rng.uniform(0, r, len(grid))) for r in range(9)),
            n_families=8,
            n_sims=10,
        )
        for pi1 in (0.0, 0.02, 0.3):
            picks = [t.choose(lam, pi1) for lam in LAMBDAS]
            assert picks == sorted(picks)


def test_the_cost_is_false_plus_lambda_times_missed(table):
    for lam, pi1 in [(1.0, 0.02), (10.0, 0.05)]:
        false, missed, cost = table.expected(lam, pi1)
        np.testing.assert_allclose(cost, false + lam * missed)
        k = table.choose(lam, pi1)
        assert cost[k] == cost.min() and k == int(np.argmin(cost))
        assert false[k] <= 8 * (1 - pi1) and missed[k] <= 8 * pi1


def test_deterministic_for_a_seed_and_independent_of_lambda_and_prior(table):
    assert ratio.simulate(small_spec(miss_cost=7.0, pi1=0.3)) == table  # only weights
    assert ratio.simulate(small_spec(seed=1)) != table


def test_one_real_family_misses_what_the_funnel_does_not_accept(table):
    # With one real family the misses at a grid point are the share of worlds whose
    # planted target the funnel's own stage 4 did not accept: same worlds, same seed.
    spec = small_spec()
    k = table.grid.index((FDR.q1, FDR.q4))
    th = dataclasses.replace(FDR, q1=table.grid[k][0], q4=table.grid[k][1])
    pspec = spec.power_spec(FDR)
    seeds = np.random.default_rng(spec.seed).integers(0, 2**31 - 1, size=spec.n_sims)
    missed = 0
    for i, seed in enumerate(seeds):
        side = power.null_side(pspec, int(seed))
        families = side.stats.index
        target = np.roll(families, -(i % len(families)))[0]
        stats = power.planted_side(side, [target], spec.alt_sharpe)
        missed += not funnel.run_funnel(stats, th)["stage4"][target]
    assert table.misses[1][k] == pytest.approx(missed / spec.n_sims, abs=1e-12)
    assert table.misses[0] == (0.0,) * len(table.grid)  # nothing real, nothing missed


def test_verify_does_nothing_in_fdr_mode(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("FDR mode must not calibrate")

    monkeypatch.setattr(ratio, "simulate", boom)
    monkeypatch.setattr(power, "null_side", boom)
    ratio.verify(FDR, variants=(1,), search_periods=1, holdout_periods=1, n_boot=1)


def test_a_calibrated_batch_verifies_and_a_wrong_choice_is_refused():
    spec = small_spec(miss_cost=5.0, pi1=0.25)
    th = ratio.calibrate(spec)
    shape = dict(
        variants=spec.variants[::-1],  # order does not matter
        search_periods=spec.search_periods,
        holdout_periods=spec.holdout_periods,
        n_boot=spec.n_boot,
    )
    ratio.verify(th, **shape)
    k = spec.grid.index((th.q1, th.q4))
    q1, q4 = spec.grid[k - 1] if k else spec.grid[k + 1]
    wrong = dataclasses.replace(th, q1=q1, q4=q4)
    with pytest.raises(ratio.RatioMismatch, match="recomputed"):
        ratio.verify(wrong, **shape)


@pytest.mark.parametrize(
    "change",
    [
        {"n_boot": 399},
        {"search_periods": 300 * 21},
        {"variants": (5,) * 7 + (6,)},
        {"holdout_periods": 9},
    ],
)
def test_a_batch_of_another_shape_is_refused_before_anything_is_simulated(monkeypatch, change):
    th = Thresholds(mode="ratio", ratio=small_spec(), q1=0.10, q4=0.30)

    def boom(*args, **kwargs):
        raise AssertionError("a shape mismatch is refused before simulating")

    monkeypatch.setattr(ratio, "simulate", boom)
    shape = dict(
        variants=SHAPE["variants"],
        search_periods=SHAPE["search_periods"],
        holdout_periods=SHAPE["holdout_periods"],
        n_boot=SHAPE["n_boot"],
    )
    with pytest.raises(ratio.RatioMismatch, match="calibrated for another batch"):
        ratio.verify(th, **shape | change)


# --- the script ---------------------------------------------------------------------


def test_the_script_logs_before_it_prints_and_stays_synthetic(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts" / "ratio_thresholds.py"
    loaded = importlib.util.spec_from_file_location("ratio_thresholds", path)
    script = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(script)
    params = {
        "search_years": 3,
        "holdout_years": 1,
        "variants": [3] * 8,
        "within_corr": 0.8,
        "across_corr": 0.1,
        "alt_sharpe": ALT,
        "n_sims": 3,
        "seed": 0,
        "miss_costs": [1, 10],
        "pi1s": [0.05],
    }
    (tmp_path / "params.json").write_text(json.dumps(params))
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setattr("sys.argv", ["ratio_thresholds", "--params", str(tmp_path / "params.json")])
    script.main()
    (entry,) = runlog.read_entries()
    assert entry["name"] == "validation-ratio" and entry["tags"] == ["synthetic", "ratio"]
    assert entry["params"]["data_version"] == "synthetic" and entry["seed"] == 0
    assert len(entry["metrics"]["choices"]) == 2
    assert "grid point" in capsys.readouterr().out
