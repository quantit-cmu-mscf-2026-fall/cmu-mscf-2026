"""The no-trade band in evaluation: `TradingSpec.band`, traded by the one backtest.

The rule itself (`backtest.band_book`) is tested in test_backtest. Here: a spec
with band 0 is the backtest it was before bands, byte for byte; the band is
validated, part of the spec's dict and variant id, ledgered and summarised.
Synthetic CRSP-shaped rows only.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone.backtest import backtest_frame
from capstone.factors import evaluation as ev
from capstone.factors.evaluation import (
    TradingSpec,
    build_panel,
    evaluate_factors,
    factor_returns,
    factor_signal,
    grid,
    trading_signal,
)
from capstone.factors.store import variant_id
from capstone.factors.tree import factor_id, parse

EXPRESSION = "rank(ts_mean(volume, 5)) - rank(delta(close, 3))"
PINNED = [(1, "none"), (5, "none"), (5, "sector")]


def digest(result: ev.FactorReturns) -> str:
    frame = pd.concat([result.net, result.gross, result.cost, result.turnover], axis=1)
    return hashlib.sha256(np.ascontiguousarray(frame.to_numpy()).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def small_panel():
    return build_panel(fake_rows(n_stocks=12, n_days=120, seed=3))


@pytest.mark.parametrize(("hold", "neutral"), PINNED)
def test_band_zero_is_the_backtest_without_a_band_byte_for_byte(small_panel, hold, neutral):
    tree = parse(EXPRESSION)
    traded = trading_signal(factor_signal(tree, small_panel), small_panel, neutral)
    frame = backtest_frame(  # the call as it was written before bands existed
        traded,
        small_panel.fields["returns"],
        cost_bps=small_panel.half_spread * 1e4,
        demean=neutral == "none",
        gross=1.0,
        hold=hold,
    )
    expected = hashlib.sha256(np.ascontiguousarray(frame.to_numpy()).tobytes()).hexdigest()[:16]
    for spec in (TradingSpec(hold, neutral), TradingSpec(hold, neutral, band=0.0)):
        assert digest(factor_returns(tree, small_panel, spec)) == expected


def test_band_zero_keeps_the_spec_dict_and_the_variant_id():
    fid = factor_id(parse(EXPRESSION))
    assert TradingSpec(5, "sector", 0.0).as_dict() == {
        "hold": 5,
        "neutral": "sector",
        "weighting": "signal",
    }
    assert variant_id(fid, TradingSpec(5, "sector", band=0.0).as_dict()) == (
        "ec5e374dcb7b1b55-86608223"
    )
    assert variant_id(fid, TradingSpec().as_dict()) == "ec5e374dcb7b1b55-49e33e0d"
    banded = TradingSpec(5, "sector", band=0.002)
    assert banded.as_dict() == {
        "hold": 5,
        "neutral": "sector",
        "weighting": "signal",
        "band": 0.002,
    }
    assert variant_id(fid, banded.as_dict()) != variant_id(fid, TradingSpec(5, "sector").as_dict())


def test_grid_adds_bands_innermost_and_defaults_to_none():
    assert grid([1, 5], ["none"]) == [TradingSpec(1), TradingSpec(5)]
    assert grid([1], ["none", "sector"], bands=[0.0, 0.01]) == [
        TradingSpec(1, "none", 0.0),
        TradingSpec(1, "none", 0.01),
        TradingSpec(1, "sector", 0.0),
        TradingSpec(1, "sector", 0.01),
    ]


@pytest.mark.parametrize("band", [-0.001, float("nan"), float("inf"), 1.0, "0.1"])
def test_an_invalid_band_is_refused(band):
    with pytest.raises(ValueError, match="band"):
        TradingSpec(band=band)


def test_backtest_frame_with_band_zero_equals_the_default(small_panel):
    signal = factor_signal(parse(EXPRESSION), small_panel)
    returns = small_panel.fields["returns"]
    for hold in (1, 3):
        kwargs = dict(cost_bps=small_panel.half_spread * 1e4, demean=True, gross=1.0, hold=hold)
        pd.testing.assert_frame_equal(
            backtest_frame(signal, returns, band=0.0, **kwargs),
            backtest_frame(signal, returns, **kwargs),
            check_exact=True,
        )


# ---------------------------------------------------------------------------
# On a synthetic market

BANDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)


@pytest.fixture(scope="module")
def market():
    return build_panel(fake_rows(n_stocks=30, n_days=500, planted=0.004, seed=11))


@pytest.mark.parametrize(
    ("expression", "hold"),
    [("volume", 1), ("volume", 5), ("ts_mean(volume, 10)", 1), ("ts_mean(volume, 10)", 5)],
)
def test_turnover_and_cost_fall_as_the_band_widens(market, expression, hold, capsys):
    # "volume" is the planted signal, new every day; its 10-day mean is persistent.
    tree = parse(expression)
    rows = []
    for band in BANDS:
        r = factor_returns(tree, market, TradingSpec(hold, band=band))
        rows.append((band, r.turnover.mean(), r.cost.mean(), r.sharpe("gross"), r.sharpe("net")))
    table = pd.DataFrame(rows, columns=["band", "turnover", "cost", "sharpe_gross", "sharpe_net"])
    table["cost_saving"] = 1 - table["cost"] / table["cost"].iloc[0]
    with capsys.disabled():
        print(f"\nno-trade band: {expression}, hold={hold} (30 stocks, 500 days, planted volume)")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    assert (np.diff(table["turnover"]) <= 1e-12).all()
    assert (np.diff(table["cost"]) <= 1e-12).all()
    assert table["turnover"].iloc[-1] < 0.85 * table["turnover"].iloc[0]


def test_the_band_is_ledgered_and_summarised(small_panel, tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    specs = [TradingSpec(1), TradingSpec(1, band=0.01)]
    matrix, summary = evaluate_factors({"f": parse(EXPRESSION)}, small_panel, specs=specs)
    assert matrix.shape[1] == 2
    entries = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [e["params"]["spec"].get("band") for e in entries] == [None, 0.01]
    banded = summary.loc[variant_id("f", specs[1].as_dict())]
    assert banded["band"] == 0.01
    assert ev.spec_from_summary(banded) == specs[1]
    assert ev.spec_from_summary(summary.loc[variant_id("f", specs[0].as_dict())]) == specs[0]
