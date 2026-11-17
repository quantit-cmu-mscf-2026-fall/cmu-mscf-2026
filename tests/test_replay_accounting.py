"""Replay accounting steps: conserved cash and shares, held carry, fallback exits, open gaps.

Synthetic prices and orders only; no strategy engine is involved.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.replay_accounting import (
    ReplayCosts,
    carried_close,
    carried_open,
    fallback_exits,
    fill_orders,
    held_carry,
    legacy_actions,
    mark_session,
    open_gap_run,
    open_gaps,
    order_rates,
)
from capstone.pipeline.market_costs import HistoricalCostSchedule

DAYS = pd.bdate_range("2020-01-02", periods=8)
COSTS = ReplayCosts(1.0, 2.0, 1.0, 100.0, 50.0)
ZERO = ReplayCosts(0.0, 0.0, 0.0, 0.0, 0.0)


def parts():
    return {"commission": 0.0, "spread": 0.0, "slippage": 0.0}


def schedule(spread=5.0):
    table = pd.DataFrame(spread, index=DAYS, columns=[101, 102])
    return HistoricalCostSchedule(table, COSTS)


def rates_of(bps):
    return lambda decision: {101: bps, 102: bps}


# -------------------------------------------------------------------- costs


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_cost_assumptions_must_be_finite_and_nonnegative(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ReplayCosts(bad, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ReplayCosts(0.0, 0.0, 0.0, bad, 0.0)


# --------------------------------------------------------------- held carry


def test_carry_state_exists_only_for_contracts_that_declare_it():
    assert held_carry({"version": "fixed-share-next-open-v2"}, None) is None
    declared = {"held_without_price": "carry_at_last_valid_mark_v1"}
    assert held_carry(declared, None) == {"marks": {}, "fallback_bps": None}
    carry = held_carry(declared | {"held_open_gap_exit_sessions": 5}, schedule())
    assert carry["open_gap_sessions"] == 5
    assert carry["fallback_bps"] == schedule().fallback_bps() > 0


def test_a_held_name_without_an_open_or_order_is_valued_at_its_last_mark():
    carry = {"marks": {101: 10.0, 102: 20.0}}
    row = pd.Series({101: np.nan, 102: 21.0})
    assert carried_open(row, {101: 5, 102: 3}, carry, None)[101] == 10.0
    # A name with an order today is not carried: the order itself is handled by the caller.
    ordered = {"orders": [{"permno": 101, "shares": -5}]}
    assert math.isnan(carried_open(row, {101: 5, 102: 3}, carry, ordered)[101])
    assert math.isnan(carried_open(row, {101: 5}, None, None)[101])


def test_planned_rates_keep_a_fallback_orders_own_cost():
    pending = {"decision": "d", "orders": [{"permno": 101, "fallback_cost": 7.0}]}
    assert order_rates(pending, rates_of(3.0)) == {101: 7.0, 102: 3.0}
    assert order_rates({"decision": "d", "orders": []}, rates_of(3.0)) == {101: 3.0, 102: 3.0}


def test_closes_refresh_marks_and_a_missing_close_keeps_the_last_valid_mark():
    carry = {"marks": {}}
    history = []
    positions = {101: 5, 102: 3}
    raw_open = pd.Series({101: 10.0, 102: 20.0})
    close = pd.Series({101: 11.0, 102: np.nan})
    # 102 has no close today: it keeps its open as the last valid mark.
    result = carried_close(DAYS[0], positions, raw_open, close, carry, history)
    assert result[102] == 20.0
    assert history == [
        {"date": "2020-01-02", "status": "carried_at_last_valid_mark", "permnos": [102]}
    ]
    assert carry["marks"] == {101: 11.0, 102: 20.0}
    # With no carry state the close row is untouched.
    assert carried_close(DAYS[0], positions, raw_open, close, None, history) is close


# ------------------------------------------------------------ corporate actions


def actions_frame(value, name):
    return pd.DataFrame({101: 0.0, 102: 0.0} | {name: value}, index=DAYS).astype(float)


def test_a_split_pays_cash_for_the_fractional_share_and_conserves_value():
    positions, cash, history = {101: 5}, 100.0, []
    splits = pd.DataFrame({101: [1.0, 1.5]}, index=DAYS[:2])
    result = legacy_actions(
        DAYS[1],
        positions,
        cash,
        pd.Series({101: 10.0}),
        splits,
        None,
        None,
        {"affected": set(), "retired": set()},
        0.0,
        history,
        set(),
    )
    cash, distribution, split_names, settled, changed = result
    # 5 shares x 1.5 = 7.5: 7 shares and half a share's open value in cash.
    assert positions == {101: 7}
    assert cash == pytest.approx(100.0 + 0.5 * 10.0)
    assert split_names == {101} and changed and not settled and distribution == 0.0


def test_dividends_and_cash_settlements_move_cash_and_retire_positions():
    positions, history, terminated = {101: 4, 102: 2}, [], set()
    dividends = pd.DataFrame({101: [0.5, 0.5], 102: [0.0, 0.0]}, index=DAYS[:2])
    settlements = pd.DataFrame({101: [np.nan, np.nan], 102: [np.nan, 3.0]}, index=DAYS[:2])
    cash, distribution, _, settled, changed = legacy_actions(
        DAYS[1],
        positions,
        0.0,
        pd.Series({101: 10.0, 102: 10.0}),
        None,
        dividends,
        settlements,
        {"affected": set(), "retired": set()},
        0.0,
        history,
        terminated,
    )
    assert cash == pytest.approx(4 * 0.5 + 2 * 3.0)
    assert distribution == pytest.approx(4 * 0.5)
    assert positions == {101: 4} and settled == {102} and terminated == {102} and changed
    assert history == [{"date": "2020-01-03", "permno": 102, "shares": 2, "cash_per_share": 3.0}]


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_an_invalid_split_or_settlement_is_refused(bad):
    empty = {"affected": set(), "retired": set()}
    splits = pd.DataFrame({101: [bad]}, index=DAYS[:1])
    with pytest.raises(ValueError, match="invalid corporate action"):
        legacy_actions(
            DAYS[0], {101: 1}, 0.0, pd.Series({101: 1.0}), splits, None, None, empty, 0.0, [], set()
        )
    settlements = pd.DataFrame({101: [bad if bad < 0 else -2.0]}, index=DAYS[:1])
    with pytest.raises(ValueError, match="nonnegative"):
        legacy_actions(
            DAYS[0],
            {101: 1},
            0.0,
            pd.Series({101: 1.0}),
            None,
            None,
            settlements,
            empty,
            0.0,
            [],
            set(),
        )


# ------------------------------------------------------------------- fills


def test_fills_move_cash_and_shares_by_exactly_the_order_and_its_cost():
    positions = {102: 3}
    pending = {
        "decision": "2020-01-02",
        "orders": [{"permno": 101, "shares": 10}, {"permno": 102, "shares": -3}],
    }
    components = {"commission": 0.0, "spread": 0.0, "slippage": 0.0}
    cash, cost, turnover, orders = fill_orders(
        pending,
        positions,
        1000.0,
        pd.Series({101: 10.0, 102: 20.0}),
        COSTS,
        None,
        rates_of(4.0),
        DAYS[1],
        components,
        event_protocol=False,
    )
    fees = (100.0 + 60.0) * 4e-4
    assert positions == {101: 10, 102: 0}
    assert cash == pytest.approx(1000.0 - 100.0 + 60.0 - fees)
    assert cost == pytest.approx(fees) and turnover == pytest.approx(160.0)
    assert components == {
        "commission": pytest.approx(160.0 * 1e-4),
        "spread": pytest.approx(160.0 * 2e-4),
        "slippage": pytest.approx(160.0 * 1e-4),
    }
    assert [o["shares"] for o in orders] == [10, -3] and orders[0]["fill"] == 10.0


def test_a_fallback_fill_uses_the_capped_rate_and_event_fills_carry_their_intents():
    positions = {101: 5}
    pending = {
        "decision": "2020-01-03",
        "intent_id": "i1",
        "source_intent": {"source_intent_id": "s1"},
        "superseded_source_intent_id": "s0",
        "orders": [{"permno": 101, "shares": -5, "fallback_cost": 50.0}],
    }
    components = {"commission": 0.0, "spread": 0.0, "slippage": 0.0}
    cost_schedule = schedule()
    _, cost, _, orders = fill_orders(
        pending,
        positions,
        0.0,
        pd.Series({101: 10.0}),
        COSTS,
        cost_schedule,
        cost_schedule.rates,
        DAYS[2],
        components,
        event_protocol=True,
    )
    assert cost == pytest.approx(50.0 * 50.0 * 1e-4)
    assert orders[0]["fallback_cost"] == 50.0
    assert (orders[0]["intent_id"], orders[0]["source_intent_id"]) == ("i1", "s1")
    assert orders[0]["superseded_source_intent_id"] == "s0"
    assert components["spread"] == pytest.approx(
        50.0 * cost_schedule.fallback_components()["spread"] / 1e4
    )


@pytest.mark.parametrize("price", [np.nan, 0.0, -3.0])
def test_a_missing_execution_price_is_refused(price):
    pending = {"decision": "d", "orders": [{"permno": 101, "shares": 1}]}
    with pytest.raises(ValueError, match="execution price"):
        fill_orders(
            pending, {}, 0.0, pd.Series({101: price}), COSTS, None, rates_of(1.0), DAYS[0],
            {"commission": 0.0, "spread": 0.0, "slippage": 0.0}, event_protocol=False,
        )  # fmt: skip


# ------------------------------------------------------------------- marking


def mark(positions, cash, close, **change):
    arguments = {
        "day": DAYS[1],
        "positions": positions,
        "cash": cash,
        "row_close": pd.Series(close),
        "previous": 1000.0,
        "cost": 0.0,
        "turnover": 0.0,
        "components": {"commission": 0.0, "spread": 0.0, "slippage": 0.0},
        "costs": ZERO,
        "cost_schedule": None,
        "actions": SimpleNamespace(value=0.0),
        "distribution_pnl": 0.0,
        "side": "long_short",
        "event_protocol": False,
    } | change
    return mark_session(**arguments)


def test_equity_is_cash_plus_marked_shares_and_return_is_against_the_prior_equity():
    cash, equity, row = mark({101: 10, 102: -5}, 900.0, {101: 12.0, 102: 20.0})
    assert equity == pytest.approx(900.0 + 120.0 - 100.0)
    assert row["net"] == pytest.approx(equity / 1000.0 - 1)
    assert row["positions"] == {101: 10, 102: -5}
    assert row["gross_exposure"] == pytest.approx(220.0 / equity)
    assert row["net_exposure"] == pytest.approx(20.0 / equity)
    assert row["market_pnl"] == pytest.approx(equity - 1000.0)


def test_shares_and_cash_are_conserved_across_a_fill_and_a_mark():
    positions = {}
    pending = {"decision": "d", "orders": [{"permno": 101, "shares": 10}]}
    components = {"commission": 0.0, "spread": 0.0, "slippage": 0.0}
    cash, cost, turnover, _ = fill_orders(
        pending, positions, 1000.0, pd.Series({101: 10.0}), COSTS, None, rates_of(4.0),
        DAYS[1], components, event_protocol=False,
    )  # fmt: skip
    # Prices do not move: the only change in equity is the fee that was paid.
    _, equity, row = mark(positions, cash, {101: 10.0}, cost=cost, turnover=turnover)
    assert equity == pytest.approx(1000.0 - cost)
    assert row["cost"] == pytest.approx(cost) and row["turnover"] == pytest.approx(0.1)
    assert row["market_pnl"] == pytest.approx(0.0, abs=1e-9)


def test_short_borrow_and_negative_cash_financing_are_charged_daily():
    costs = ReplayCosts(0.0, 0.0, 0.0, 252.0, 252.0)
    # Short 200 of stock with the proceeds in cash: borrow only, no financing.
    cash, _, row = mark({101: -10}, 1200.0, {101: 20.0}, costs=costs)
    assert row["borrow_cost"] == pytest.approx(200.0 * 252 / 252e4)
    assert row["financing_cost"] == 0.0
    assert cash == pytest.approx(1200.0 - row["borrow_cost"])
    # A levered long: 500 of negative cash is financed, nothing is borrowed.
    cash, equity, row = mark({101: 100}, -500.0, {101: 20.0}, costs=costs)
    assert row["borrow_cost"] == 0.0
    assert row["financing_cost"] == pytest.approx(500.0 * 252 / 252e4)
    assert equity == pytest.approx(1500.0 - row["financing_cost"])


def test_a_long_only_account_cannot_borrow_and_a_dead_account_is_refused():
    with pytest.raises(ValueError, match="cannot borrow"):
        mark({101: 100}, -50.0, {101: 20.0}, side="long_only")
    with pytest.raises(ValueError, match="exhausted"):
        mark({101: 1}, -100.0, {101: 20.0})


def test_a_held_name_without_a_closing_price_is_never_marked_at_zero():
    with pytest.raises(ValueError, match="no silent zero marks"):
        mark({101: 1}, 0.0, {101: np.nan})
    with pytest.raises(ExecutionContractBlocked, match="held_valuation_missing"):
        mark({101: 1}, 0.0, {101: np.nan}, event_protocol=True)
    with pytest.raises(ValueError, match="no silent zero marks"):
        mark({101: 1}, 0.0, {102: 5.0})


def test_pending_receivables_count_towards_equity():
    _, equity, row = mark({}, 500.0, {}, actions=SimpleNamespace(value=40.0))
    assert equity == pytest.approx(540.0) and row["receivables"] == 40.0


# ----------------------------------------------------------- gaps and fallbacks


def opens_frame(prices):
    return pd.DataFrame(prices, index=DAYS[: len(next(iter(prices.values())))])


def test_an_open_gap_run_counts_consecutive_sessions_without_a_valid_open():
    opens = opens_frame({101: [10.0, np.nan, np.nan, 0.0, np.nan], 102: [1, 1, 1, 1, 1]})
    assert open_gap_run(opens, DAYS[4], 101) == 4
    assert open_gap_run(opens, DAYS[1], 101) == 1
    assert open_gap_run(opens, DAYS[4], 102) == 0
    assert open_gap_run(opens, DAYS[4], 999) == 1
    # A later valid open never changes an earlier answer.
    later = opens.copy()
    later.loc[DAYS[4], 101] = 12.0
    assert open_gap_run(later, DAYS[3], 101) == open_gap_run(opens, DAYS[3], 101) == 3


def fallback_session(cost_schedule=None):
    return (
        COSTS,
        cost_schedule,
        lambda decision: 3.0,
        {"commission": 0.0, "spread": 0.0, "slippage": 0.0},
    )


def test_a_held_name_missing_its_open_for_the_registered_sessions_exits_at_its_last_close():
    opens = opens_frame({101: [10.0, np.nan, np.nan, np.nan], 102: [5.0] * 4})
    closes = opens_frame({101: [10.0, 11.0, 12.0, 13.0], 102: [5.0] * 4})
    day = DAYS[3]
    positions = {101: 4, 102: 2}
    carry = {"marks": {}, "open_gap_sessions": 3}
    cash, cost, turnover, orders = fallback_exits(
        {"cash": 0.0}, opens, closes, day, positions, carry, fallback_session()
    )
    # Exit at the last close strictly before the session (12.0), never the same-day close.
    assert positions == {102: 2}
    assert cash == pytest.approx(4 * 12.0 - cost)
    assert turnover == pytest.approx(48.0) and cost == pytest.approx(48.0 * 3.0 / 1e4)
    exit_row = orders[0]["fallback_exit"]
    assert exit_row["reason"] == "held_without_open_for_registered_sessions"
    assert exit_row["price"] == 12.0 and exit_row["last_price_date"] == "2020-01-06"
    assert exit_row["cost_rule"] == "exit_at_capped_half_spread_v1"
    # One session short of the registered gap does not exit.
    shorter = {101: 4, 102: 2}
    fallback_exits(
        {"cash": 0.0},
        opens,
        closes,
        day,
        shorter,
        {"marks": {}, "open_gap_sessions": 4},
        fallback_session(),
    )
    assert shorter == {101: 4, 102: 2}


def test_unqualifiable_held_events_exit_by_fallback_and_without_a_price_are_blocked():
    opens = opens_frame({101: [10.0, 10.0, 10.0]})
    closes = opens_frame({101: [10.0, 10.5, 10.5]})
    event = {
        "cash": 2.0,
        "fallback_exits": [
            {"event_id": "e1", "reason": "unqualifiable", "permno": 101, "shares": 3}
        ],
    }
    cost_schedule = schedule()
    positions = {101: 3}
    cash, cost, _, orders = fallback_exits(
        event, opens, closes, DAYS[2], positions, None, fallback_session(cost_schedule)
    )
    assert positions == {} and cash == pytest.approx(2.0 + 3 * 10.5 - cost)
    assert cost == pytest.approx(3 * 10.5 * cost_schedule.fallback_bps() / 1e4)
    assert orders[0]["fallback_exit"]["event_id"] == "e1"
    # No close before the session: nothing observed to exit at.
    with pytest.raises(ExecutionContractBlocked, match="without_observed_price"):
        fallback_exits(event, opens, closes, DAYS[0], {101: 3}, None, fallback_session())
    assert fallback_exits(
        {"cash": 7.0}, opens, closes, DAYS[1], {101: 3}, None, fallback_session()
    ) == (
        7.0,
        0.0,
        0.0,
        [],
    )


def test_a_name_without_an_open_drops_only_its_own_order():
    opens = opens_frame({101: [10.0, np.nan], 102: [20.0, 20.0], 103: [5.0, np.nan]})
    pending = {
        "decision": "2020-01-02",
        "orders": [
            {"permno": 101, "shares": -2},
            {"permno": 102, "shares": 3},
            {"permno": 103, "shares": 4},
        ],
    }
    positions = {101: 2}
    carry = {"marks": {101: 10.0}}
    kept, missing, row_open, history = open_gaps(
        pending, [101, 103], positions, opens, DAYS[1], carry
    )
    assert [o["permno"] for o in kept["orders"]] == [102]
    # The held name is valued at its carried mark, so it no longer blocks; the unheld entry
    # was dropped and nothing blocks the rest of the book.
    assert missing == [] and row_open[101] == 10.0
    assert history[0]["status"] == "dropped_missing_open"
    assert {r["permno"]: r["reason"] for r in history[0]["orders"]} == {
        101: "held_without_open_carried",
        103: "entry_without_open_dropped",
    }


def test_a_held_name_with_no_order_and_no_mark_still_blocks():
    opens = opens_frame({101: [np.nan], 102: [20.0]})
    pending = {"decision": "d", "orders": [{"permno": 102, "shares": 1}]}
    _, missing, _, history = open_gaps(pending, [101], {101: 2}, opens, DAYS[0], {"marks": {}})
    assert missing == [101] and history == []
