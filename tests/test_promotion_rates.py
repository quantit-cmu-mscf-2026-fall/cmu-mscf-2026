"""`scripts/promotion_rates.py`: the incubate outcome is off unless asked for (D-4).

Synthetic worlds only, shrunk to a few families and years so the gate runs in seconds; the
ledger is a temporary directory. No real data is read.
"""

import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone import runlog
from capstone.funnel import Thresholds

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "promotion_rates.py"
spec = importlib.util.spec_from_file_location("promotion_rates", SCRIPT)
rates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rates)

SMALL = {"n_families": 4, "k": 2, "n_boot": 199, "sharpe": 1.0}
TINY = SMALL | {
    "share_real": 0.5,
    "planted_worlds": 2,
    "null_worlds": 2,
    "first_seed": 5,
    "q4_incubate": None,
    "alpha": 0.05,
    "horizons": [20, 40],
    "block": 5,
    "confirm_horizons": [20, 40],
    "live_draws": 3,
    "power": {"sharpes": [1.0], "years": [1], "paths": 20, "chunk": 10, "seed": 1},
}


@pytest.fixture(autouse=True)
def small_worlds(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    shrink = functools.partial(rates.make_family_set, n_search=6 * 252, n_holdout=2 * 252)
    monkeypatch.setattr(rates, "make_family_set", shrink)


def log_trials(p):
    for i in range(p["n_families"] * p["k"]):
        runlog.log_run(rates.TRIAL_NAME, params={"variant": i}, seed=i, tags=["synthetic"])


def test_the_shipped_parameters_leave_the_incubate_outcome_off():
    params = json.loads(rates.PARAMS.read_text())
    assert params["q4_incubate"] is None


def test_without_a_level_no_family_is_labelled_incubate():
    log_trials(SMALL)
    card = rates.world(SMALL, 5, 0.5, Thresholds())
    assert not card["incubate"].any() and card["incubate"].dtype == bool
    assert card["accepted"].sum() >= 1  # the planted families still reach acceptance
    assert set(card["real"]) == {True, False}
    assert card["search"].map(len).eq(6 * 252).all()


def test_an_incubate_family_is_a_stage_3_survivor_that_was_not_accepted():
    log_trials(SMALL)
    wide = Thresholds(promote="incubate", q4_incubate=0.999)
    card = rates.world(SMALL | {"sharpe": 0.8}, 6, 0.5, wide)  # seed 6: one near-miss
    assert not (card["incubate"] & card["accepted"]).any()
    assert card["incubate"].sum() == 1 and card["accepted"].sum() == 0
    for family in card.index[card["incubate"]]:
        assert pd.notna(card.loc[family, "expected_mean"])  # graded for the live check


def rows(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "search": [pd.Series(rng.normal(0, 0.01, 400)) for _ in range(n)],
            "scale": [0.01] * n,
            "expected_mean": [0.0005] * n,
            "is_mean": [0.0005] * n,
            "seed": range(n),
        }
    )


def test_monitors_report_nan_for_no_families_and_a_rate_otherwise():
    empty = rows(0)
    assert np.isnan(rates.monitored(empty, [20], 0.05, 5, "expected_mean")["20"])
    nothing = rates.confirming(empty, TINY, lambda r: 0.0)
    assert set(nothing) == {"20", "40"} and all(np.isnan(v) for v in nothing["20"].values())
    held = rows(3)
    retire = rates.monitored(held, [20, 40], 0.05, 5, "expected_mean")
    assert all(0.0 <= v <= 1.0 for v in retire.values())
    split = rates.confirming(held, TINY, lambda r: 0.0)["40"]
    assert sum(split.values()) == pytest.approx(1.0)


def test_confirm_power_grows_with_the_edge():
    power = {"seed": 3, "paths": 40, "chunk": 20, "years": [1, 2], "sharpes": [0.0, 3.0]}
    out = rates.confirm_power(power, 0.05)
    null, strong = out["0.0"]["promote_within_years"], out["3.0"]["promote_within_years"]
    assert strong["2"] > null["2"] and strong["2"] >= strong["1"]
    assert out["0.0"]["years_to_power"]["80%"] is None


def test_main_logs_one_entry_with_the_rates_before_printing(tmp_path, monkeypatch, capsys):
    params = tmp_path / "params.json"
    params.write_text(json.dumps(TINY), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["promotion_rates.py", "--params", str(params)])
    rates.main()
    (entry,) = [e for e in runlog.read_entries() if e["name"] == "promotion-rates"]
    shown = json.loads(capsys.readouterr().out)
    assert entry["tags"] == ["synthetic", "promotion"]
    assert entry["params"]["planted_worlds"] == 2 and entry["params"]["q4_incubate"] is None
    assert entry["metrics"]["planted_real_incubate"] == 0 == shown["planted_real_incubate"]
    assert entry["metrics"]["null_incubate_rate"] == 0.0
    trials = [e for e in runlog.read_entries() if e["name"] == rates.TRIAL_NAME]
    assert len(trials) == TINY["n_families"] * TINY["k"]


@pytest.mark.parametrize(
    ("extra", "promote", "level"),
    [([], "accept", None), (["--q4-incubate", "0.7"], "incubate", 0.7)],
)
def test_only_the_flag_registers_the_incubate_outcome(
    extra, promote, level, tmp_path, monkeypatch, capsys
):
    params = tmp_path / "params.json"
    params.write_text(json.dumps(TINY | {"planted_worlds": 1, "null_worlds": 1}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["promotion_rates.py", "--params", str(params), *extra])
    seen, world = [], rates.world
    monkeypatch.setattr(rates, "world", lambda *a: seen.append(a[3]) or world(*a))
    rates.main()
    assert {(t.promote, t.q4_incubate) for t in seen} == {(promote, level)}
    (entry,) = [e for e in runlog.read_entries() if e["name"] == "promotion-rates"]
    assert entry["params"]["q4_incubate"] == level
    capsys.readouterr()
