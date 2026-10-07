"""Evaluating factor trees on the CRSP panel (QUANTIT-81), on synthetic CRSP-shaped rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd
from capstone.backtest import backtest_components
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
                    "siccd": (2834, 3674, 6020)[j % 3],
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
    net, turnover = backtest_components(
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


def test_every_variant_is_a_logged_trial_and_a_column(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = build_panel(fake_rows(n_stocks=12, n_days=80), data_version="v-test")
    trees = {"f1": parse("volume"), "f2": parse("ts_mean(returns, 5)"), "f3": parse("open")}
    original = ev.evaluate

    def failing(tree, fields):
        if tree == trees["f3"]:
            raise ValueError("cannot compute")
        return original(tree, fields)

    monkeypatch.setattr(ev, "evaluate", failing)
    specs = ev.grid(holds=[1, 5], neutral=["none", "sector"])
    matrix, summary = evaluate_factors(trees, panel, specs=specs)
    entries = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(entries) == 3 * 4  # every (factor, spec), the failing factor included
    assert entries[-1]["metrics"] == {"error": "cannot compute"}
    assert all(e["params"]["data_version"] == "v-test" for e in entries)
    assert len(matrix.columns) == 2 * 4 and list(summary.index) == list(matrix.columns)
    assert summary.loc[matrix.columns[0], "factor_id"] == "f1"
    assert set(zip(summary.hold, summary.neutral, strict=True)) == {
        (1, "none"),
        (5, "none"),
        (1, "sector"),
        (5, "sector"),
    }
    assert (summary.sharpe_net_full_spread <= summary.sharpe_net + 1e-12).all()
    assert matrix.index[0] == panel.fields["returns"].index[1]  # the untraded first day is dropped
    assert isinstance(panel, Panel)


def test_variants_are_recorded_in_the_store(tmp_path, monkeypatch):
    from capstone.factors import store

    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "p", "Paper")
    hid = store.add_hypothesis(con, store.Hypothesis("p", "o", "k", "j", "s", "f"))
    fid, _ = store.add_factor(con, parse("volume"), hid)
    panel = build_panel(fake_rows(n_stocks=9, n_days=40))
    matrix, _ = evaluate_factors(
        {fid: parse("volume")}, panel, specs=ev.grid([1, 21], ["none"]), con=con
    )
    stored = store.variants(con, fid)
    assert [v["id"] for v in stored] == list(matrix.columns)
    assert [v["spec"]["hold"] for v in stored] == [1, 21]


def test_sector_neutral_weights_net_to_zero_within_every_sector():
    panel = build_panel(fake_rows(n_stocks=30, n_days=40))
    weights = ev.factor_weights(parse("volume"), panel, neutral="sector")
    sectors = ev.sector_groups(panel)
    long = weights.stack().reindex(sectors.index)
    by_sector = long.groupby([long.index.get_level_values(0), sectors.values]).sum()
    assert by_sector.abs().max() < 1e-12
    np.testing.assert_allclose(weights.abs().sum(axis=1).iloc[5:], 1.0)


def test_a_thin_sic_group_falls_back_to_its_division():
    rows = fake_rows(n_stocks=12, n_days=10)
    rows.loc[rows.permno == 10000, "siccd"] = 2010  # alone in group 20: manufacturing
    sectors = ev.sector_groups(build_panel(rows))
    day = sectors.index.get_level_values(0)[0]
    assert sectors.loc[(day, 10000)] == "manufacturing"
    assert sectors.loc[(day, 10003)] == "manufacturing"  # group 28 has 4 stocks: too thin
    assert sectors.loc[(day, 10002)] == "finance"


def test_a_bad_spec_is_refused():
    with pytest.raises(ValueError):
        ev.TradingSpec(hold=0)
    with pytest.raises(ValueError):
        ev.TradingSpec(neutral="industry")


def test_the_full_spread_view_doubles_the_cost():
    result = factor_returns(parse("volume"), build_panel(fake_rows(n_days=40)))
    np.testing.assert_allclose(result.net_full_spread, result.gross - 2 * result.cost)


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


def test_the_research_script_end_to_end(tmp_path, monkeypatch):
    """The script on a fake panel: backs up the store, records variants, writes outputs."""
    import importlib.util
    import sys
    from pathlib import Path

    from capstone.factors import store

    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    db = tmp_path / "factors.db"
    con = store.connect(db)
    store.add_paper(con, "p", "Paper")
    hid = store.add_hypothesis(con, store.Hypothesis("p", "o", "k", "j", "s", "f"))
    for e in ["volume", "ts_mean(returns, 5)", "-ts_std(returns, 21)", "rank(cap)"]:
        store.add_factor(con, parse(e), hid)
    con.close()

    path = Path(__file__).resolve().parent.parent / "research" / "evaluate_factors.py"
    spec = importlib.util.spec_from_file_location("evaluate_factors_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    panel = build_panel(fake_rows(n_stocks=12, n_days=60), data_version="v-fake")
    monkeypatch.setattr(script, "load_panel", lambda: panel)
    monkeypatch.setattr(script, "OUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["x", "--store", str(db), "--record-variants"])
    script.main()

    assert list(tmp_path.glob("factors.db.bak-*"))  # backed up before writing
    con = store.connect(db)
    n_specs = len(
        script.grid(**{k: v for k, v in script.tomllib.loads(script.PARAMS.read_text()).items()})
    )
    assert len(store.variants(con)) == 4 * n_specs
    con.close()
    out = tmp_path / "out" / "v-fake"
    assert (out / "matrix.parquet").exists() and (out / "summary.csv").exists()
    ledger = (tmp_path / "ledger" / "runs.jsonl").read_text().splitlines()
    assert len(ledger) == 4 * n_specs


def test_a_correction_excludes_entries_without_deleting_them(tmp_path, monkeypatch):
    from capstone.runlog import _read_entries, log_run

    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    first = [log_run(ev.LEDGER_NAME, params={"factor_id": f}) for f in ("a", "b")]
    log_run(ev.LEDGER_NAME, params={"factor_id": "a", "variant_id": "a-1"})
    log_run("other-experiment", params={"factor_id": "a"})
    assert ev.trial_count() == 3
    log_run(
        ev.CORRECTION_NAME,
        params={
            "ledger_name": ev.LEDGER_NAME,
            "excludes": [
                {"ts_utc": e["ts_utc"], "factor_id": e["params"]["factor_id"]} for e in first
            ],
            "reason": "duplicates",
        },
    )
    assert ev.trial_count() == 1
    assert len(_read_entries()) == 5  # nothing deleted


# ---------------------------------------------------------------------------
# The screen statistic


def test_the_screen_pvalue_is_one_sided():
    panel = build_panel(fake_rows(n_days=500, planted=0.004))
    assert ev.screen_pvalue(factor_returns(parse("volume"), panel).net) < 0.01
    assert ev.screen_pvalue(factor_returns(parse("-volume"), panel).net) > 0.99


@pytest.mark.parametrize("ar1", [0.0, 0.3])
def test_the_screen_pvalue_holds_its_size_on_signal_free_returns(ar1):
    # 400 zero-mean series of 1,000 days: about 5% should fall below 0.05, with
    # autocorrelated returns too, since the variance is HAC.
    rng = np.random.default_rng(7)
    shocks = rng.standard_t(5, (1000, 400)) * 0.01
    returns = np.empty_like(shocks)
    returns[0] = shocks[0]
    for t in range(1, len(shocks)):
        returns[t] = ar1 * returns[t - 1] + shocks[t]
    p = np.array([ev.screen_pvalue(pd.Series(returns[:, j])) for j in range(400)])
    assert 0.02 <= np.mean(p < 0.05) <= 0.09
    assert 0.4 <= np.median(p) <= 0.6


def test_the_screen_pvalue_is_nan_without_variance():
    assert np.isnan(ev.screen_pvalue(pd.Series([0.0] * 50)))
    assert np.isnan(ev.screen_pvalue(pd.Series([np.nan, 0.01])))


def test_each_variant_logs_its_screen_pvalue_and_names_its_family(tmp_path, monkeypatch):
    from capstone.factors import store

    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "p", "Paper")
    h1 = store.add_hypothesis(con, store.Hypothesis("p", "volume", "k", "j", "s", "f"))
    h2 = store.add_hypothesis(con, store.Hypothesis("p", "reversal", "k", "j", "s", "f"))
    f1, _ = store.add_factor(con, parse("volume"), h1)
    f2, _ = store.add_factor(con, parse("ts_mean(returns, 5)"), h2)
    store.add_factor(con, parse("volume"), h2)  # proposed again: a duplicate, stays in h1
    families = {fid: ev.hypothesis_of(con, fid) for fid in (f1, f2)}
    assert families == {f1: h1, f2: h2}

    panel = build_panel(fake_rows(n_stocks=12, n_days=80))
    trees = {f1: parse("volume"), f2: parse("ts_mean(returns, 5)")}
    matrix, summary = evaluate_factors(trees, panel, families=families)
    entries = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    for entry, vid in zip(entries, matrix.columns, strict=True):
        p = entry["metrics"]["pvalue_screen"]
        assert p == pytest.approx(ev.screen_pvalue(matrix[vid]))
        assert summary.loc[vid, "pvalue_screen"] == pytest.approx(p)
    assert list(summary["hypothesis_id"]) == [h1, h2]


# ---------------------------------------------------------------------------
# Evaluation as the deepen move's scorer (QUANTIT-114)


def test_the_scorer_gives_the_screen_z_and_the_variants_net_returns(tmp_path, monkeypatch):
    from scipy import stats

    from capstone.factors.scoring import EvaluationScorer

    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = build_panel(fake_rows(n_days=500, planted=0.004), data_version="v-test")
    spec = ev.TradingSpec(hold=5)
    scorer = EvaluationScorer(panel, spec)
    good, bad = scorer(parse("volume")), scorer(parse("-volume"))
    net = factor_returns(parse("volume"), panel, spec).net
    assert good.quality == pytest.approx(stats.norm.isf(ev.screen_pvalue(net)))
    assert good.quality > 2.3 and bad.quality < -2.3  # the sign is kept
    np.testing.assert_array_equal(good.values, net.to_numpy())
    assert good.params["data_version"] == "v-test" and good.params["spec"]["hold"] == 5
    assert good.metrics["pvalue_screen"] == pytest.approx(ev.screen_pvalue(net))
    assert not (tmp_path / "runs.jsonl").exists()  # the scorer never logs; deepen does


def test_the_scorer_returns_none_for_a_factor_it_cannot_compute(monkeypatch):
    from capstone.factors.scoring import EvaluationScorer

    def failing(tree, fields):
        raise ValueError("cannot compute")

    monkeypatch.setattr(ev, "evaluate", failing)
    scorer = EvaluationScorer(build_panel(fake_rows(n_stocks=9, n_days=40)), ev.TradingSpec())
    assert scorer(parse("volume")) is None


def test_crsp_settings_put_the_memory_on_the_z_scale(tmp_path):
    from capstone.factors.llm import load_config
    from capstone.factors.scoring import crsp_settings

    config, spec = crsp_settings(load_config())
    assert spec == ev.TradingSpec(hold=21, neutral="none")
    assert config.memory_signed_quality and config.memory_quality_edges == (0.0, 1.0, 1.645)
    assert config.memory_min_quality == 0.0 and config.memory_high_quality == 1.645
    assert load_config().memory_quality_edges == (0.05, 0.10, 0.20)  # the default is untouched
    bad = tmp_path / "bad.toml"
    bad.write_text("[memory]\nmemory_min_qualty = 1.0\n")
    with pytest.raises(ValueError, match="memory_min_qualty"):
        crsp_settings(load_config(), bad)


def test_a_seed_scored_by_evaluation_is_a_logged_trial(tmp_path, monkeypatch):
    from capstone.factors import store
    from capstone.factors.deepen import Deepener
    from capstone.factors.llm import load_config
    from capstone.factors.scoring import EvaluationScorer, crsp_settings

    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "p", "Paper")
    hid = store.add_hypothesis(con, store.Hypothesis("p", "o", "k", "j", "s", "f"))
    fid, _ = store.add_factor(con, parse("volume"), hid)
    config, spec = crsp_settings(load_config())
    panel = build_panel(fake_rows(n_days=300, planted=0.004), data_version="v-test")
    d = Deepener(con, config, EvaluationScorer(panel, spec), None, run_id="r", seed=0)
    assert d.add_parent(fid)
    (entry,) = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert entry["name"] == "deepen" and entry["params"]["role"] == "seed"
    assert entry["params"]["data_version"] == "v-test"
    assert entry["params"]["variant_id"] == store.variant_id(fid, spec.as_dict())
    assert entry["metrics"]["quality"] == pytest.approx(d.pool.quality[fid])


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


def test_the_warm_up_is_left_out_under_sector_neutrality_too():
    panel = build_panel(fake_rows(n_stocks=30, n_days=150))
    tree = parse("ts_mean(returns, 63)")
    plain = factor_returns(tree, panel, ev.TradingSpec(hold=5))
    sector = factor_returns(tree, panel, ev.TradingSpec(hold=5, neutral="sector"))
    assert plain.net.isna().sum() == sector.net.isna().sum() > 60
