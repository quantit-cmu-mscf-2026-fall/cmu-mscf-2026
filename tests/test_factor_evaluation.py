"""Evaluating factor trees on the CRSP panel (QUANTIT-81), on synthetic CRSP-shaped rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd
from capstone.backtest import _backtest_components
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
    signal_held = ev.to_weights(ev.evaluate(parse("volume"), panel.fields).where(panel.universe))
    assert (signal_held[10003].fillna(0) == 0).all()
    assert result.net.iloc[1:].notna().all()


def test_costs_match_the_backtest_when_every_spread_is_the_same():
    rows = fake_rows(spread=0.002)
    panel = build_panel(rows)
    result = factor_returns(parse("ts_mean(returns, 5)"), panel)
    signal = ev.evaluate(parse("ts_mean(returns, 5)"), panel.fields)
    net, turnover = _backtest_components(
        signal, panel.fields["returns"], cost_bps=10.0, demean=True, gross=1.0
    )  # half of a 20 bps spread is 10 bps per unit traded
    # Same returns and turnover on every day a position is held. During the
    # 5-day warm-up this module reports NaN; the backtest here still reports
    # 0.0, which #55 changes to NaN too.
    held = result.net.notna()
    np.testing.assert_allclose(result.net[held], net[held], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.turnover[held], turnover[held], rtol=1e-9)
    assert (net.iloc[1:][~held.iloc[1:]] == 0.0).all() and (~held).sum() == 5


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
    weights = ev.factor_weights(parse("ts_mean(returns, 5)"), panel)
    daily = ev.returns_from_weights(weights, panel, 1)
    np.testing.assert_allclose(daily.net, factor_returns(parse("ts_mean(returns, 5)"), panel).net)


def test_a_longer_hold_trades_only_on_rebalance_days_and_less():
    panel = build_panel(fake_rows(n_days=200))
    weights = ev.factor_weights(parse("ts_mean(returns, 5)"), panel)
    weekly = ev.returns_from_weights(weights, panel, 5)
    daily = ev.returns_from_weights(weights, panel, 1)
    traded = weekly.turnover.fillna(0) > 0
    # Trades land the day after a rebalance day (the position is first held then).
    rebalance_next = np.arange(len(weights)) % 5 == 1
    assert not (traded & ~rebalance_next).any()
    assert weekly.turnover.sum() < daily.turnover.sum()
    assert weekly.cost.sum() < daily.cost.sum()


def test_a_hold_below_one_day_is_refused():
    panel = build_panel(fake_rows(n_stocks=5, n_days=20))
    with pytest.raises(ValueError):
        ev.returns_from_weights(ev.factor_weights(parse("volume"), panel), panel, 0)


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
    weights = ev.factor_weights(parse("volume"), panel)
    weights.iloc[10:15] = 0.0  # a signal that is there but flat
    result = ev.returns_from_weights(weights, panel)
    assert (result.gross.iloc[12:16] == 0.0).all()
