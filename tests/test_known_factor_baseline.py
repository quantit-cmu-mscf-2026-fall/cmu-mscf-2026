"""The known factor: 12-1 momentum, its synthetic market and its dated periods.

Synthetic data only. The module builds inputs and never imports the gate.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone.factors.evaluation import DISCOVERY_END, build_panel, factor_returns
from capstone.factors.tree import parse, unparse

SCRIPT = Path(__file__).resolve().parent.parent / "research" / "known_factor_baseline.py"
_spec = importlib.util.spec_from_file_location("known_factor_baseline", SCRIPT)
kfb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kfb  # dataclasses look their module up
_spec.loader.exec_module(kfb)

PARAMS = kfb.load_params()


def market_rows(*, premium=0.001, n_days=500, n_stocks=30, seed=7):
    return kfb.momentum_market(n_stocks=n_stocks, n_days=n_days, premium=premium, seed=seed)


@pytest.fixture(scope="module")
def panel():
    return build_panel(market_rows(), "synthetic-momentum")


def test_the_expression_is_twelve_minus_one_month_momentum():
    tree = parse(PARAMS["expression"])
    assert unparse(tree) == "shift(ts_sum(returns, 231), 21)"
    assert PARAMS["generator"] == "known-factor-baseline/momentum-12-1"
    assert kfb.LOOKBACK - kfb.SKIP == 231  # t-251 .. t-21


# ---------------------------------------------------------------------------
# terciles


def test_terciles_are_plus_one_for_the_top_third_and_minus_one_for_the_bottom():
    values = np.array([[3.0, 1.0, 2.0, 6.0, 5.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(
        kfb.terciles(values), [[0, -1, -1, 1, 1, 0], [-1, -1, 0, 0, 1, 1]]
    )


def test_a_row_with_a_missing_value_holds_nothing():
    values = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
    out = kfb.terciles(values)
    np.testing.assert_array_equal(out[0], [0, 0, 0])
    np.testing.assert_array_equal(out[1], [-1, 0, 1])


# ---------------------------------------------------------------------------
# The synthetic market and the factors built from it


def test_the_market_is_shaped_like_the_shared_daily_file_and_deterministic():
    a, b = market_rows(n_days=60, n_stocks=6), market_rows(n_days=60, n_stocks=6)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 60 * 6 and a["permno"].nunique() == 6
    assert (
        a["date"].is_monotonic_increasing
        or a.groupby("permno")["date"].is_monotonic_increasing.all()
    )
    assert a["in_universe"].all()


def test_the_momentum_factors_have_the_ff5_plus_mom_columns_on_the_panels_dates(panel):
    frame = kfb.momentum_factors(panel, seed=7)
    assert list(frame.columns) == list(kfb.FACTOR_COLUMNS)
    assert frame.index.name == "date" and frame.index.is_monotonic_increasing
    assert frame.index.isin(panel.universe.index).all()
    assert not frame.isna().any().any()
    # Mom needs 252 days of returns before its first signal, and a day to earn it.
    assert frame.index[0] > panel.universe.index[kfb.LOOKBACK - 1]
    np.testing.assert_array_equal(
        frame["Mkt-RF"].to_numpy(), panel.fields["mkt_return"].loc[frame.index].to_numpy()
    )


def test_the_markets_momentum_premium_is_what_its_mom_factor_earns(panel):
    mom = kfb.momentum_factors(panel, seed=7)["Mom"]
    flat = build_panel(market_rows(premium=0.0), "no-premium")
    flat_mom = kfb.momentum_factors(flat, seed=7)["Mom"]
    assert mom.mean() > 0.0005  # about twice the daily premium on each side
    assert abs(flat_mom.mean()) < mom.mean() / 2
    # The known factor, traded by evaluation, earns before costs on this market.
    result = factor_returns(parse(PARAMS["expression"]), panel, 5)
    assert result.gross.mean() > 0


def test_the_momentum_factors_are_seeded(panel):
    a, b = kfb.momentum_factors(panel, seed=1), kfb.momentum_factors(panel, seed=1)
    pd.testing.assert_frame_equal(a, b)
    other = kfb.momentum_factors(panel, seed=2)
    pd.testing.assert_series_equal(a["Mom"], other["Mom"])  # Mom is not random
    assert not a["SMB"].equals(other["SMB"])  # the placeholders are


def test_mom_on_a_day_is_the_return_of_yesterdays_terciles(panel):
    # Reference loop: sort on the signal known at the previous close, earn today's return.
    from capstone.factors.interpret import evaluate

    signal = evaluate(parse(PARAMS["expression"]), panel.fields).where(panel.universe)
    returns = panel.fields["returns"]
    mom = kfb.momentum_factors(panel, seed=7)["Mom"]
    dates = list(panel.universe.index)
    for day in mom.index[::40]:
        i = dates.index(day)
        yesterday = signal.iloc[i - 1]
        third = len(yesterday) // 3
        order = yesterday.sort_values(kind="stable").index
        top, bottom = order[-third:], order[:third]
        expected = returns.iloc[i][top].mean() - returns.iloc[i][bottom].mean()
        assert mom.loc[day] == pytest.approx(expected, rel=1e-9, abs=1e-12), day


def test_mom_on_a_day_uses_no_later_returns(panel):
    # Truncate the market at day 400: Mom and the market factor on every shared
    # day are unchanged, so nothing after a day feeds its value.
    rows = market_rows()
    cut = rows["date"].drop_duplicates().sort_values().iloc[399]
    short = build_panel(rows[rows["date"] <= cut], "short")
    full = kfb.momentum_factors(panel, seed=7)
    part = kfb.momentum_factors(short, seed=7)
    shared = part.index
    assert len(shared) > 100 and shared.max() <= cut
    for column in ("Mom", "Mkt-RF"):
        np.testing.assert_allclose(
            part[column].to_numpy(), full.loc[shared, column].to_numpy(), rtol=0, atol=1e-15
        )


# ---------------------------------------------------------------------------
# Periods


def test_the_synthetic_search_period_is_the_panel_and_the_holdout_comes_after_it(panel):
    periods = kfb.synthetic_periods(panel)
    last = panel.universe.index[-1]
    assert pd.Timestamp(periods["search_end"]) == last
    assert pd.Timestamp(periods["holdout_start"]) == last + pd.offsets.BDay(1)
    assert pd.Timestamp(periods["holdout_end"]) > pd.Timestamp(periods["holdout_start"])
    assert not (panel.universe.index > pd.Timestamp(periods["search_end"])).any()


def test_the_crsp_periods_are_the_discovery_period_and_never_the_holdout():
    periods = kfb.crsp_periods()
    assert periods["search_end"] == DISCOVERY_END == "2020-12-31"
    assert periods["holdout_start"] > periods["search_end"]
    assert periods["holdout_end"] > periods["holdout_start"]


def test_crsp_periods_refuse_a_search_end_past_the_discovery_period():
    late = {"crsp": {**PARAMS["crsp"], "search_end": "2021-06-30"}}
    with pytest.raises(ValueError, match="discovery period"):
        kfb.crsp_periods(late)


def test_the_module_builds_its_inputs_without_the_validation_gate():
    code = (
        "import importlib.util, sys; "
        f"s = importlib.util.spec_from_file_location('kfb', r'{SCRIPT}'); "
        "m = importlib.util.module_from_spec(s); sys.modules['kfb'] = m; s.loader.exec_module(m); "
        "print(*sorted(n for n in sys.modules if n.startswith('capstone')), sep=chr(10))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=SCRIPT.parent.parent
    )
    assert out.returncode == 0, out.stderr
    loaded = set(out.stdout.split())
    banned = ("capstone.funnel", "capstone.intake", "capstone.validation", "capstone.calibration")
    assert "capstone.factors.evaluation" in loaded  # the module did load
    assert not [m for m in loaded if m.startswith(banned)]
