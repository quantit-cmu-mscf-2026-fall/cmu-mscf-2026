import pandas as pd
import pytest

from capstone.execution import daily_planning
from capstone.execution.planning import (
    UnqualifiedCost,
    execution_risk,
    portfolio_risk,
    share_plan,
)
from capstone.factors.evaluation import Panel
from capstone.pipeline import replay as module
from capstone.pipeline.market_costs import HistoricalCostSchedule
from capstone.pipeline.replay import ReplayCosts, replay
from capstone.strategy.reference import BenchmarkSpec


def test_sizing_rounds_target_not_trade_and_exits_small_positions():
    plan = share_plan(pd.Series({1: 0.4, 2: 0}), {1: 3, 2: 1}, {1: 30, 2: 5}, 1000)
    assert plan["targets"] == {1: 13, 2: 0}
    assert [o["shares"] for o in plan["orders"]] == [10, -1]
    assert (
        "rounded_single_name_limit"
        in portfolio_risk({1: 1}, {1: 101}, 1000, side="long_only")["blockers"]
    )


def Spec(side="long_short"):
    return BenchmarkSpec("equal_weight_universe_proxy", "synthetic", "2000-01-27", side)


def replay_panel(dates):
    close = pd.DataFrame(1.0, index=dates, columns=[1])
    return Panel({"close": close}, close > 0, close * 0, data_version="synthetic", calendar=dates)


def test_delayed_fills_share_drift_dividends_and_splits(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    dates = pd.DatetimeIndex(
        ["2000-01-28", "2000-01-31", "2000-02-01", "2000-02-02", "2000-02-03", "2000-02-04"]
    )

    prices = pd.DataFrame({1: [10.0, 10.0, 5.0, 2.5, 2.5, 2.5]}, index=dates)
    splits = prices * 0 + 1
    splits.loc["2000-02-02", 1] = 2
    dividends = prices * 0
    dividends.loc["2000-02-03", 1] = 1
    monkeypatch.setattr(module, "target_weights", lambda *a: pd.Series({1: 1.0}))
    result = replay(
        Spec(),
        replay_panel(dates),
        prices,
        prices,
        start=dates[0],
        end=dates[-2],
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        splits=splits,
        dividends=dividends,
        min_notional=0,
    )
    assert len(result["orders"]) == 1
    assert result["orders"][0]["date"] == "2000-02-01"
    assert result["orders"][0]["shares"] == 100  # fixed at previous close, not resized at the gap
    assert result["positions"][1] == 200
    assert result["history"].loc["2000-02-02", "equity"] == 1000
    assert result["cash"] == 700
    assert result["history"].loc["2000-02-03", "equity"] == 1200


def test_missing_fill_price_and_holdout_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="holdout"):
        replay(
            None,
            None,
            None,
            None,
            start="2021-01-01",
            end="2021-01-02",
            capital=1000,
            costs=ReplayCosts(0, 0, 0, 0, 0),
        )


def test_current_market_recheck_includes_remaining_quantities_and_costs():
    plan = {
        "projected_shares": {"1": 10},
        "capital": 1000,
        "profile": "long_only",
        "execution_cost_bps": 11,
        "limits": {"max_gross": 1, "max_weight": 1, "max_order_notional": 1000},
    }
    account = {"positions": {1: 9}, "cash": 100}
    blocked = execution_risk(plan, account, {1: 100})
    assert blocked["blockers"] == ["current_cash_deficit"]
    allowed = execution_risk(plan, account, {1: 98})
    assert not allowed["blockers"]
    assert allowed["projected_cash_at_current_marks"] == pytest.approx(100 - 98 * 1.0011)
    filled = execution_risk(plan, {"positions": {1: 10}, "cash": 10}, {1: 98})
    assert filled["projected_cash_at_current_marks"] == 10


def test_split_expires_pending_shares_and_replans_from_the_completed_session(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    dates = pd.to_datetime(["2000-01-28", "2000-01-31", "2000-02-01", "2000-02-02", "2000-02-03"])
    panel = replay_panel(dates)
    prices = pd.DataFrame({1: [10.0, 10.0, 5.0, 5.0, 5.0]}, index=dates)
    splits = prices * 0 + 1
    splits.loc["2000-02-01", 1] = 2
    monkeypatch.setattr(module, "target_weights", lambda *a: pd.Series({1: 1.0}))
    result = replay(
        Spec(side="long_only"),
        panel,
        prices,
        prices,
        start=dates[0],
        end=dates[-2],
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        splits=splits,
        min_notional=0,
    )
    assert result["orders"][0]["status"] == "expired_corporate_action"
    assert result["orders"][1]["date"] == "2000-02-02"
    assert result["orders"][1]["shares"] == 200 and result["cash"] == 0


def test_terminal_cash_settlement_removes_shares_without_a_fictitious_zero_mark(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    dates = pd.to_datetime(["2000-01-28", "2000-01-31", "2000-02-01", "2000-02-02", "2000-02-03"])
    panel = replay_panel(dates)
    prices = pd.DataFrame({1: [10.0, 10.0, 10.0, 0.0, 0.0]}, index=dates)
    settlement = prices * float("nan")
    settlement.loc["2000-02-02", 1] = 5
    monkeypatch.setattr(module, "target_weights", lambda *a: pd.Series({1: 1.0}))
    result = replay(
        Spec(),
        panel,
        prices,
        prices,
        start=dates[0],
        end=dates[-2],
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        cash_settlements=settlement,
    )
    assert result["positions"] == {} and result["cash"] == 500
    assert result["history"].iloc[-1]["equity"] == 500
    assert result["corporate_actions"][0]["shares"] == 100


def test_initial_investment_uses_only_the_prior_completed_session(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    dates = pd.to_datetime(["2000-01-27", "2000-01-28", "2000-01-31", "2000-02-01"])
    panel = replay_panel(dates)
    prices = pd.DataFrame({1: 10.0}, index=dates)
    observed = []

    def targets(spec, panel, day):
        observed.append(day)
        return pd.Series({1: 1.0})

    monkeypatch.setattr(module, "target_weights", targets)
    monkeypatch.setattr(daily_planning, "target_weights", targets)
    result = replay(
        Spec(),
        panel,
        prices,
        prices,
        start=dates[1],
        end=dates[-2],
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
    )
    assert observed[0] == dates[0]
    assert result["orders"][0]["decision"] == "2000-01-27"
    assert result["orders"][0]["date"] == "2000-01-28"


def test_unheld_zero_weights_need_no_price_and_small_reversals_still_close():
    sized = share_plan(pd.Series({1: 0.5, 2: 0.0}), {2: 0}, {1: 100}, 1000, min_notional=0)
    assert sized["targets"] == {1: 5}
    reversal = share_plan(pd.Series({1: 0.005}), {1: -1}, {1: 5}, 1000)
    assert reversal["orders"][0]["shares"] == 1
    assert reversal["projected"] == {1: 0} and reversal["targets"] == {1: 1}
    assert reversal["omitted"][0]["reason"] == "minimum_notional_opening_leg"


NAN = float("nan")


def test_held_name_without_price_is_carried():
    # fixed-share-next-open-v3: a held name with no sizing price keeps its shares at its
    # last valid mark, gets no order, and is listed in the plan's `carried` field.
    carry = {"marks": {2: 7.0}, "fallback_bps": None}
    plan = share_plan(
        pd.Series({1: 0.5, 2: 0.5}),
        {2: 10},
        {1: 10.0, 2: NAN},
        1000,
        min_notional=0,
        side="long_only",
        carry=carry,
    )
    assert plan["carried"] == [
        {"permno": 2, "shares": 10, "mark": 7.0, "reason": "held_without_sizing_price"}
    ]
    assert plan["targets"][2] == plan["projected"][2] == 10
    assert [o["permno"] for o in plan["orders"]] == [1]
    assert plan["orders"][0]["shares"] == 50 and plan["sizing_capital"] == 1000
    # A held name with neither a price nor a last valid mark is still a blocker.
    with pytest.raises(ValueError, match="finite and positive"):
        share_plan(pd.Series({1: 1.0}), {2: 10}, {1: 10.0, 2: NAN}, 1000, carry={"marks": {}})


def test_held_name_without_quote_exits_at_capped_fallback_cost():
    schedule = HistoricalCostSchedule(
        pd.DataFrame({1: [2.0]}, index=pd.to_datetime(["2000-01-03"])),
        ReplayCosts(1, 0, 5, 0, 0),
        multiplier=2,
    )
    # The scenario's capped half-spread is sd.MAX_SPREAD / 2 = 250 bps, scaled like any cost.
    assert schedule.fallback_components() == {"commission": 2.0, "spread": 500.0, "slippage": 10.0}
    assert schedule.fallback_bps() == 512.0
    carry = {"marks": {}, "fallback_bps": schedule.fallback_bps()}
    rates = schedule.rates("2000-01-03")
    plan = share_plan(
        pd.Series({1: 0.5, 2: 0.5}),
        {2: 10},
        {1: 10.0, 2: 10.0},
        1000,
        min_notional=0,
        side="long_only",
        fee_bps=rates,
        carry=carry,
    )
    assert plan["targets"][2] == plan["projected"][2] == 0
    exit_order = next(o for o in plan["orders"] if o["permno"] == 2)
    assert exit_order["shares"] == -10 and exit_order["fallback_cost"] == 512.0
    assert "fallback_cost" not in next(o for o in plan["orders"] if o["permno"] == 1)
    # Names that are not held never reach the fallback: formation excludes them (U4).
    with pytest.raises(UnqualifiedCost):
        share_plan(pd.Series({3: 1.0}), {}, {3: 10.0}, 1000, fee_bps=rates, carry=carry)


def test_v2_contract_still_raises():
    with pytest.raises(ValueError, match="sizing prices must be finite and positive"):
        share_plan(pd.Series({1: 1.0}), {2: 10}, {1: 10.0, 2: NAN}, 1000, min_notional=0)
    with pytest.raises(UnqualifiedCost, match="instrument 2"):
        share_plan(pd.Series({1: 1.0}), {2: 10}, {1: 10.0, 2: 10.0}, 1000, fee_bps={1: 1.0})
