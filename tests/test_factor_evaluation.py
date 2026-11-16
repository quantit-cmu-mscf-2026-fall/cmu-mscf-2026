"""Evaluating factor trees on the CRSP panel (QUANTIT-81), on synthetic CRSP-shaped rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd
from capstone.backtest import backtest_components, backtest_frame
from capstone.factors import evaluation as ev
from capstone.factors.evaluation import (
    HoldoutError,
    Panel,
    build_panel,
    evaluate_factors,
    factor_returns,
    load_panel,
)
from capstone.factors.tree import parse


def fake_rows(n_stocks=30, n_days=260, *, spread=0.002, seed=0, start="2015-01-02", planted=0.0):
    """Daily rows shaped like the shared file. With `planted`, a stock's next-day
    return rises with its volume today."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    volume = np.exp(rng.normal(13, 0.5, (n_days, n_stocks)))
    z = (volume - volume.mean(axis=1, keepdims=True)) / volume.std(axis=1, keepdims=True)
    ret = rng.normal(0, 0.02, (n_days, n_stocks))
    ret[1:] += planted * z[:-1]
    price = 50 * np.cumprod(1 + ret, axis=0)
    rows = []
    for j in range(n_stocks):
        for t, date in enumerate(dates):
            p = price[t, j]
            rows.append(
                {
                    "date": date,
                    "permno": 10000 + j,
                    "in_universe": True,
                    "dlyret": ret[t, j],
                    "dlyretmissflg": None,
                    "dlydelflg": "N",
                    "delactiontype": None,
                    "dlyopen": p,
                    "dlyhigh": p * 1.01,
                    "dlylow": p * 0.99,
                    "dlyprc": p,
                    "dlyvol": volume[t, j],
                    "shrout": 1000.0,
                    "dlycap": p * 1000.0,
                    "dlycumfacpr": 1.0,
                    "dlycumfacshr": 1.0,
                    "dlybid": p * (1 - spread / 2),
                    "dlyask": p * (1 + spread / 2),
                }
            )
    return pd.DataFrame(rows)


def test_a_split_moves_no_adjusted_price_or_share_count():
    rows = fake_rows(n_stocks=3, n_days=40)
    split = (rows.permno == 10000) & (rows.date < rows.date.unique()[20])
    # Before a 2-for-1 split: raw prices twice as high, half the shares; CRSP's
    # cumulative factors are 2 on those days.
    rows.loc[split, ["dlyopen", "dlyhigh", "dlylow", "dlyprc", "dlybid", "dlyask"]] *= 2
    rows.loc[split, "shrout"] /= 2
    rows.loc[split, ["dlycumfacpr", "dlycumfacshr"]] = 2.0
    panel = build_panel(rows)
    close, shares = panel.fields["close"][10000], panel.fields["shares"][10000]
    implied = close.pct_change().iloc[1:].to_numpy()
    np.testing.assert_allclose(implied, panel.fields["returns"][10000].iloc[1:], rtol=1e-12)
    assert shares.nunique() == 1


def test_stocks_outside_the_universe_are_never_held():
    rows = fake_rows(n_stocks=10, n_days=60)
    rows.loc[rows.permno == 10003, "in_universe"] = False
    panel = build_panel(rows)
    result = factor_returns(parse("volume"), panel)
    weights = ev.factor_weights(parse("volume"), panel)
    assert (weights[10003].fillna(0) == 0).all()
    assert (weights.drop(columns=10003).fillna(0) != 0).any().all()
    assert result.net.iloc[1:].notna().all()


def test_costs_match_the_backtest_when_every_spread_is_the_same():
    rows = fake_rows(spread=0.002)
    panel = build_panel(rows)
    result = factor_returns(parse("ts_mean(returns, 5)"), panel)
    signal = ev.evaluate(parse("ts_mean(returns, 5)"), panel.fields)
    net, turnover = backtest_components(
        signal, panel.fields["returns"], cost_bps=10.0, demean=True, gross=1.0
    )  # half of a 20 bps spread is 10 bps per unit traded
    # Same returns and turnover on every day, warm-up NaNs included (#55).
    held = result.net.notna()
    np.testing.assert_allclose(result.net, net, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.turnover[held], turnover[held], rtol=1e-9)
    assert net[~held].isna().all() and (~held).sum() == 5


def test_a_planted_factor_earns_and_its_opposite_loses():
    panel = build_panel(fake_rows(n_days=500, planted=0.004))
    assert factor_returns(parse("volume"), panel).sharpe() > 1.0
    assert factor_returns(parse("-volume"), panel).sharpe() < -1.0


def test_the_market_return_is_cap_weighted_on_the_previous_day():
    panel = build_panel(fake_rows(n_stocks=5, n_days=30))
    cap, returns = panel.fields["cap"], panel.fields["returns"]
    t = returns.index[10]
    w = cap.shift(1).loc[t]
    assert panel.fields["mkt_return"].loc[t] == pytest.approx((w * returns.loc[t]).sum() / w.sum())


def test_the_market_return_leaves_out_stocks_outside_that_days_universe():
    rows = fake_rows(n_stocks=3, n_days=10)
    dates = rows.date.unique()
    rows.loc[(rows.permno == 10000) & (rows.date < dates[5]), "in_universe"] = False
    rows.loc[(rows.permno == 10001) & (rows.date >= dates[7]), "in_universe"] = False
    panel = build_panel(rows)
    cap, returns, universe = panel.fields["cap"], panel.fields["returns"], panel.universe
    mkt = panel.fields["mkt_return"]
    for t in dates[1:]:
        # That day's universe, each stock weighted by its previous day's cap
        # (tree.py's definition): 10000 counts from the day it joins, 10001
        # stops counting the day it leaves.
        members = universe.loc[t]
        w = cap.shift(1).loc[t][members]
        assert mkt.loc[t] == pytest.approx((w * returns.loc[t][members]).sum() / w.sum())


def test_the_holdout_is_refused():
    with pytest.raises(HoldoutError):
        build_panel(fake_rows(n_days=30, start="2020-12-15"))  # runs into 2021
    with pytest.raises(HoldoutError):
        load_panel(end="2021-06-30")


def test_load_panel_never_asks_for_holdout_dates(monkeypatch):
    asked = {}

    def fake_load(name, *, columns, start, end):
        asked.update(name=name, start=start, end=end)
        return fake_rows(n_stocks=3, n_days=20)

    monkeypatch.setattr(sd, "load", fake_load)
    monkeypatch.setattr(sd, "data_version", lambda: "test-version")
    panel = load_panel()
    assert asked == {"name": ev.DAILY_FILE, "start": ev.DISCOVERY_START, "end": ev.DISCOVERY_END}
    assert panel.data_version == "test-version"


def test_every_factor_and_holding_period_is_a_logged_trial(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = build_panel(fake_rows(n_stocks=10, n_days=80), data_version="v-test")
    trees = {"f1": parse("volume"), "f2": parse("ts_mean(returns, 5)"), "f3": parse("open")}
    original = ev.evaluate

    def failing(tree, fields):
        if tree == trees["f3"]:
            raise ValueError("cannot compute")
        return original(tree, fields)

    monkeypatch.setattr(ev, "evaluate", failing)
    matrix, summary = evaluate_factors(trees, panel, holds=(1, 5))
    entries = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    tried = [(e["params"]["factor_id"], e["params"]["hold"]) for e in entries]
    assert tried == [("f1", 1), ("f1", 5), ("f2", 1), ("f2", 5), ("f3", 1), ("f3", 5)]
    assert all(e["params"]["data_version"] == "v-test" for e in entries)
    assert entries[4]["metrics"] == {"error": "cannot compute"}  # a failure is a trial too
    assert list(matrix.columns) == ["f1@1", "f1@5", "f2@1", "f2@5"]
    assert list(summary.index) == list(matrix.columns)
    assert matrix.index[0] == panel.fields["returns"].index[1]  # the untraded first day is dropped
    assert isinstance(panel, Panel)


def test_holding_one_day_is_daily_rebalancing():
    panel = build_panel(fake_rows())
    signal = ev.factor_signal(parse("ts_mean(returns, 5)"), panel)
    daily = ev.signal_returns(signal, panel, 1)
    np.testing.assert_allclose(daily.net, factor_returns(parse("ts_mean(returns, 5)"), panel).net)


def test_a_longer_hold_trades_only_on_rebalance_days_and_less():
    panel = build_panel(fake_rows(n_days=200))
    weekly = factor_returns(parse("ts_mean(returns, 5)"), panel, 5)
    daily = factor_returns(parse("ts_mean(returns, 5)"), panel, 1)
    traded = weekly.turnover.fillna(0) > 0
    # Trades land the day after a rebalance day (the position is first held then).
    rebalance_next = np.arange(len(panel.fields["returns"])) % 5 == 1
    assert not (traded & ~rebalance_next).any()
    assert weekly.turnover.sum() < daily.turnover.sum()
    assert weekly.cost.sum() < daily.cost.sum()


def test_a_hold_below_one_day_is_refused():
    panel = build_panel(fake_rows(n_stocks=5, n_days=20))
    with pytest.raises(ValueError):
        factor_returns(parse("volume"), panel, 0)


def test_a_lookback_warming_up_is_not_counted_as_flat_days():
    # Before ts_mean(returns, 63) has a value there is no position: those days
    # are NaN, not 0.0, so they don't pad the series and pull its Sharpe to 0.
    panel = build_panel(fake_rows(n_stocks=20, n_days=200))
    for hold in (1, 5, 21):
        result = factor_returns(parse("ts_mean(returns, 63)"), panel, hold)
        first_signal = ev.factor_weights(parse("ts_mean(returns, 63)"), panel).notna().any(axis=1)
        start = int(np.argmax(first_signal.to_numpy()))
        assert start > 50
        assert result.net.iloc[: start + 1].isna().all(), hold
        assert result.net.iloc[start + 1 :].notna().any(), hold
        assert result.turnover.iloc[: start + 1].isna().all(), hold


def test_a_real_signal_with_zero_net_weight_still_earns_zero():
    # #25's rule: idleness comes from a missing signal, not from zero weights.
    panel = build_panel(fake_rows(n_stocks=10, n_days=30))
    signal = ev.factor_signal(parse("volume"), panel)
    signal.iloc[10:15] = 1.0  # a signal that is there but the same for every stock
    result = ev.signal_returns(signal, panel)
    assert (result.gross.iloc[12:16] == 0.0).all()


def per_stock_spread_rows(**kwargs):
    """Rows where each stock quotes its own spread, 5 to 80 bps."""
    rows = fake_rows(**kwargs)
    width = 0.0005 + 0.0015 * (rows.permno - 10000) % 0.0076
    mid = rows.dlyprc
    rows["dlybid"], rows["dlyask"] = mid * (1 - width / 2), mid * (1 + width / 2)
    return rows


@pytest.mark.parametrize("hold", [1, 5])
def test_every_column_is_the_shared_backtest_with_each_stocks_own_spread(hold):
    panel = build_panel(per_stock_spread_rows(n_stocks=20, n_days=120))
    tree = parse("ts_mean(returns, 5)")
    frame = backtest_frame(
        ev.factor_signal(tree, panel),
        panel.fields["returns"],
        cost_bps=panel.half_spread * 1e4,
        demean=True,
        gross=1.0,
        hold=hold,
    )
    result = factor_returns(tree, panel, hold)
    for name, series in (("net", result.net), ("gross", result.gross), ("cost", result.cost)):
        np.testing.assert_allclose(series, frame[name], rtol=1e-12, atol=1e-15, equal_nan=True)
    np.testing.assert_allclose(result.turnover, frame["turnover"], equal_nan=True)
    # The spreads differ by stock, so the cost is not turnover times one rate.
    assert result.cost.sum() != pytest.approx(
        result.turnover.sum() * panel.half_spread.stack().mean()
    )


def test_a_trade_pays_each_stocks_half_spread_and_a_missing_one_pays_the_days_median():
    panel = build_panel(per_stock_spread_rows(n_stocks=12, n_days=40))
    tree = parse("volume")
    weights = ev.factor_weights(tree, panel).fillna(0.0)
    t = 20  # trade at the close of day t, held (and charged) from day t+1
    gap = panel.half_spread.columns[:3]
    panel.half_spread.iloc[t, panel.half_spread.columns.get_indexer(gap)] = np.nan
    spread = panel.half_spread.iloc[t]
    spread = spread.fillna(spread.median())
    expected = ((weights.iloc[t] - weights.iloc[t - 1]).abs() * spread).sum()
    assert factor_returns(tree, panel).cost.iloc[t + 1] == pytest.approx(expected, rel=1e-12)


def test_a_trading_day_with_no_cost_rate_is_refused_not_traded_for_free():
    panel = build_panel(fake_rows(n_stocks=10, n_days=40))
    panel.half_spread.iloc[15] = np.nan  # no stock has a spread that day
    with pytest.raises(ValueError, match="no rate"):
        factor_returns(parse("volume"), panel)


def test_a_refused_trial_is_logged_and_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = build_panel(fake_rows(n_stocks=10, n_days=40), data_version="v-test")
    panel.half_spread.iloc[15] = np.nan
    with pytest.raises(ValueError, match="no rate"):
        evaluate_factors({"f1": parse("volume")}, panel, holds=(1, 5))
    entries = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [(e["params"]["factor_id"], e["params"]["hold"]) for e in entries] == [("f1", 1)]
    assert "no rate" in entries[0]["metrics"]["error"]


def test_a_day_when_no_stock_has_a_return_still_pays_its_cost():
    # An intended difference from the module's old loop, which read the day's
    # gross as NaN and dropped the net with the cost: the backtest's gross is 0.0.
    panel = build_panel(fake_rows(n_stocks=10, n_days=40))
    panel.fields["returns"].iloc[20] = np.nan
    result = factor_returns(parse("volume"), panel)
    assert result.gross.iloc[20] == 0.0
    assert result.cost.iloc[20] > 0
    assert result.net.iloc[20] == pytest.approx(-result.cost.iloc[20])
