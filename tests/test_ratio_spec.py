"""The ratio mode's registered spec, and that the default (flat FDR) mode is untouched.

`Thresholds.mode` ("fdr" | "ratio") and `Thresholds.ratio` enter `thresholds_dict`,
and so a registered config's hash, only when they differ from their defaults: every
FDR-mode setting keeps the content it had before the fields existed. Ratio thresholds
are opt-in; nothing here chooses a `miss_cost` or a prior for the team.
"""

import dataclasses
import hashlib
import json

import pytest

from capstone import ratio_spec
from capstone.funnel import Thresholds, thresholds_dict
from capstone.ratio_spec import DEFAULT_GRID, RatioSpec

SHAPE = dict(
    variants=(3,) * 8,
    within_corr=0.8,
    across_corr=0.1,
    search_periods=3 * 252,
    holdout_periods=252,
    n_boot=159,  # n_boot + 1 >= 20 x 8 families
)


def spec(**changes) -> RatioSpec:
    return RatioSpec(**dict(SHAPE, miss_cost=2.0, pi1=0.1) | changes)


def ratio_thresholds(**changes) -> Thresholds:
    return Thresholds(mode="ratio", ratio=spec(), q1=0.10, q4=0.30, **changes)


def content_hash(th: Thresholds) -> str:
    return hashlib.sha256(json.dumps(thresholds_dict(th), sort_keys=True).encode()).hexdigest()


# --- the default is the flat FDR target and does not change ---------------------------


def test_the_default_is_the_fdr_mode_and_ratio_is_opt_in():
    default = Thresholds()
    assert (default.mode, default.ratio) == ("fdr", None)
    assert Thresholds(mode="fdr", ratio=None) == default
    assert (default.q1, default.q4) in DEFAULT_GRID  # the registered point is a candidate


def test_default_content_is_exactly_the_registered_fields():
    original = {"q1", "stage1", "alpha2", "final", "d4", "q4", "final_trials"}
    assert set(thresholds_dict(Thresholds())) == original
    assert set(thresholds_dict(Thresholds(q4=0.2, final="dsr"))) == original
    # Pinned: this content, and so every registration's hash, must not move.
    assert thresholds_dict(Thresholds()) == {
        "q1": 0.10,
        "stage1": "rho",
        "alpha2": 1.0,
        "final": "bh",
        "d4": 0.95,
        "q4": 0.30,
        "final_trials": "survivors",
    }


def test_ratio_mode_enters_the_registered_content_and_reads_back():
    th = ratio_thresholds()
    registered = thresholds_dict(th)
    assert registered["mode"] == "ratio" and registered["ratio"]["miss_cost"] == 2.0
    assert Thresholds(**json.loads(json.dumps(registered))) == th  # a JSON round trip
    assert content_hash(th) != content_hash(Thresholds())
    # Every ratio field moves the hash: lambda, prior, grid, seed, alternative Sharpe.
    for change in (
        {"miss_cost": 3.0},
        {"pi1": 0.2},
        {"seed": 1},
        {"alt_sharpe": 1.0},
        {"n_sims": 100},
        {"grid": DEFAULT_GRID[:-1]},
    ):
        other = dataclasses.replace(th, ratio=dataclasses.replace(th.ratio, **change))
        assert content_hash(other) != content_hash(th), change


# --- the spec's own checks -------------------------------------------------------------


def test_defaults_are_documented():
    assert ratio_spec.DEFAULT_ALT_SHARPE == 0.75
    assert ratio_spec.DEFAULT_N_SIMS == 400
    s = spec()
    assert s.grid == DEFAULT_GRID and s.seed == 0


def test_the_grid_is_a_chain_from_strict_to_loose():
    pairs = zip(DEFAULT_GRID, DEFAULT_GRID[1:], strict=False)
    assert all(a[0] <= b[0] and a[1] <= b[1] and a != b for a, b in pairs)
    with pytest.raises(ValueError, match="chain"):
        spec(grid=((0.05, 0.30), (0.10, 0.20)))
    with pytest.raises(ValueError, match="chain"):
        spec(grid=((0.05, 0.30), (0.05, 0.30)))
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        spec(grid=((0.0, 0.3),))
    with pytest.raises(ValueError, match="at least one"):
        spec(grid=())


@pytest.mark.parametrize(
    "bad",
    [
        {"miss_cost": 0.0},
        {"miss_cost": -1.0},
        {"pi1": 1.0},
        {"pi1": -0.1},
        {"alt_sharpe": 0.0},
        {"n_sims": 0},
        {"n_boot": 99},  # below 20 x 8 families - 1
        {"search_periods": 50},  # below the bootstrap's 10 blocks
        {"within_corr": 0.05},  # below across_corr
    ],
)
def test_bad_values_are_refused(bad):
    with pytest.raises(ValueError):
        spec(**bad)


def test_lists_round_trip_to_tuples():
    s = spec()
    data = dataclasses.asdict(s)
    data["grid"] = [list(pair) for pair in data["grid"]]
    data["variants"] = list(data["variants"])
    assert RatioSpec(**data) == s


def test_the_spec_makes_the_power_batch_it_was_calibrated_for():
    power_spec = spec().power_spec(None)
    assert power_spec.variants == (3,) * 8 and power_spec.n_boot == 159
    assert (power_spec.search_periods, power_spec.holdout_periods) == (756, 252)


# --- how a Thresholds checks it ---------------------------------------------------------


def test_ratio_mode_needs_a_spec_and_a_spec_needs_ratio_mode():
    with pytest.raises(ValueError, match="needs ratio"):
        Thresholds(mode="ratio")
    with pytest.raises(ValueError, match="only read in mode"):
        Thresholds(ratio=spec())
    with pytest.raises(ValueError, match="mode must be one of"):
        Thresholds(mode="cost")


def test_the_chosen_pair_must_be_on_the_registered_grid():
    with pytest.raises(ValueError, match="not on the ratio grid"):
        Thresholds(mode="ratio", ratio=spec(), q1=0.11, q4=0.30)
    Thresholds(mode="ratio", ratio=spec(), q1=0.05, q4=0.10)  # a grid point


def test_ratio_mode_calibrates_q4_so_stage_4_must_be_bh():
    with pytest.raises(ValueError, match='final="bh"'):
        Thresholds(mode="ratio", ratio=spec(), q1=0.10, q4=0.30, final="dsr")


def test_a_dict_spec_is_read_back_into_a_spec():
    data = dataclasses.asdict(spec())
    th = Thresholds(mode="ratio", ratio=data, q1=0.10, q4=0.30)
    assert isinstance(th.ratio, RatioSpec) and th.ratio == spec()
