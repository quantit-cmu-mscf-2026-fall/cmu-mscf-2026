"""The placebo re-score: a panel built from permuted rows, and the guards on the script.
Synthetic CRSP-shaped rows only: no data, no model, no network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone.factors import evaluation as ev
from capstone.factors.tree import parse
from capstone.pipeline import placebo

sys.path.insert(0, str(Path(__file__).parent))
from test_factor_evaluation import fake_rows  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "placebo_rescore", Path(__file__).parents[1] / "research" / "placebo_rescore.py"
)
rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore)


@pytest.fixture
def placebo_ledger(tmp_path, monkeypatch):
    main = tmp_path / "experiments"
    monkeypatch.setattr(placebo, "main_ledger_dir", lambda: main)
    monkeypatch.setattr(placebo, "_main_ledger_dirs", lambda: [main])
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(main / "placebo"))
    return main


def test_build_panel_is_unchanged_by_the_split():
    rows = fake_rows(n_stocks=8, n_days=30)
    real = ev.build_panel(rows)
    again = ev.panel_from_rows(ev.derived_rows(rows))
    pd.testing.assert_frame_equal(real.fields["returns"], again.fields["returns"])
    pd.testing.assert_frame_equal(real.half_spread, again.half_spread)


def test_placebo_panel_keeps_each_dates_returns_but_not_stock_identity(placebo_ledger):
    rows = fake_rows(n_stocks=20, n_days=40)
    real = ev.build_panel(rows, "v1")
    null = ev.build_placebo_panel(rows, 3, "v1")
    assert null.data_version == placebo.placebo_version("cross_section_permutation", 3, "v1")
    r, p = real.fields["returns"], null.fields["returns"]
    assert r.shape == p.shape
    np.testing.assert_allclose(np.sort(r.to_numpy(), axis=1), np.sort(p.to_numpy(), axis=1))
    assert not r.equals(p)
    # sector codes stay with the target id
    pd.testing.assert_frame_equal(real.sic, null.sic)


def test_planted_signal_is_lost_on_the_placebo(placebo_ledger):
    rows = fake_rows(n_stocks=40, n_days=500, planted=0.004)
    tree = parse("volume")
    assert ev.factor_returns(tree, ev.build_panel(rows)).sharpe("gross") > 4.0
    for seed in (0, 1, 2):
        null = ev.build_placebo_panel(rows, seed)
        # gross: the net is a cost loss (every stock is replaced daily), not a signal
        assert abs(ev.factor_returns(tree, null).sharpe("gross")) < 2.0


def test_placebo_panel_refuses_a_main_or_unset_ledger(placebo_ledger, monkeypatch):
    rows = fake_rows(n_stocks=5, n_days=10)
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(placebo_ledger))
    with pytest.raises(placebo.PlaceboRefused):
        ev.build_placebo_panel(rows, 0)
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR")
    with pytest.raises(placebo.PlaceboRefused):
        ev.build_placebo_panel(rows, 0)


def test_output_dir_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(placebo, "main_ledger_dir", lambda: tmp_path / "experiments")
    monkeypatch.setattr(rescore, "REAL_OUT", tmp_path / "experiments" / "factor_evaluation")
    for bad in (
        tmp_path / "experiments" / "factor_evaluation",
        tmp_path / "experiments" / "factor_evaluation" / "crsp-v1",
    ):
        with pytest.raises(placebo.PlaceboRefused, match="factor_evaluation"):
            rescore.check_out(bad)
    ok = tmp_path / "experiments" / "placebo" / "crsp-v1" / "seed0"
    assert rescore.check_out(ok) == ok


def test_factors_of_takes_unique_factors():
    summary = pd.DataFrame(
        {
            "factor_id": ["a", "a", "b"],
            "expression": ["volume", "volume", "open"],
            "hypothesis_id": ["h1", "h1", "h2"],
        }
    )
    trees, families = rescore.factors_of(summary)
    assert list(trees) == ["a", "b"] and families == {"a": "h1", "b": "h2"}
