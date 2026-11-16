"""Tests for scripts/measure_generator.py and scripts/calibrate_generator.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from capstone.calibration import make_family_set

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure_generator = _load("measure_generator")
calibrate_generator = _load("calibrate_generator")


def test_measure_recovers_a_known_structure():
    # Variants correlate w + (1 - w) x rho within a family and rho across.
    fs = make_family_set(
        15, 10, 0.0, 0.0, within_rho=0.6, rho=0.1, ar1=0.2, n_search=2520, n_holdout=2, seed=1
    )
    returns = fs.returns.set_axis([f"{c}-v" for c in fs.returns.columns], axis=1)
    family = pd.Series(fs.family.to_numpy(), index=returns.columns)
    m = measure_generator.measure(returns, family)
    assert m["n_families"] == 15 and m["variants_per_family"] == [10] * 15
    assert m["within_family_corr"] == pytest.approx(0.64, abs=0.04)
    assert m["across_family_corr"] == pytest.approx(0.10, abs=0.03)
    assert m["ar1"]["median"] == pytest.approx(0.2, abs=0.03)


PARAMS = {
    "variants": [2, 3, 4],
    "within_corr": 0.5,
    "across_corr": 0.0,
    "ar1": 0.0,
    "t_df": None,
    "garch": None,
    "search_years": 2,
    "holdout_years": 1,
    "n_boot": 99,
    "n_real": [0, 1],
    "sharpe_real": [3.0],
    "sweep": {"q1": [0.1], "q4": [0.5]},
    "baseline_q": [0.1],
}


def test_one_seed_counts_true_and_false_discoveries():
    rows = pd.DataFrame(calibrate_generator.one_seed((PARAMS, 0)))
    assert set(rows["rule"]) == {"funnel q1=0.1 q4=0.5", "baseline q=0.1"}
    assert (rows["true"] + rows["false"] == rows["discoveries"]).all()
    assert (rows.loc[rows["n_real"] == 0, "true"] == 0).all()
    assert (rows["true"] <= rows["n_real"]).all()


def test_summarise_marks_a_rule_unsafe_in_its_worst_setting():
    # Rule B is clean except one setting with an FDR of 1/11: unsafe at 0.10 - 0.02.
    rows = pd.DataFrame(
        [
            {"rule": r, "n_real": n, "sharpe": s, "discoveries": d, "false": f, "true": t}
            for r, n, s, d, f, t in [
                ("A", 0, 0.0, 0, 0, 0),
                ("A", 1, 1.0, 1, 0, 1),
                ("B", 0, 0.0, 0, 0, 0),
                ("B", 1, 1.0, 1, 0, 1),
                ("B", 2, 1.0, 11, 1, 10),
            ]
        ]
    )
    by_rule, _ = calibrate_generator.summarise(rows, target=0.10, margin=0.02)
    assert bool(by_rule.loc["A", "safe"]) and not bool(by_rule.loc["B", "safe"])
    assert by_rule.loc["B", "worst_fdr"] == pytest.approx(1 / 11)
