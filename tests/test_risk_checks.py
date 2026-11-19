"""Portfolio and execution risk checks on a whole-share plan (`capstone.execution.planning`).

Pure arithmetic on synthetic positions and marks; no data, no broker.
"""

from __future__ import annotations

import pytest

from capstone.execution.planning import execution_risk, portfolio_risk


def test_the_default_limits_are_the_testbed_values_and_a_big_name_is_blocked():
    # One name at 10.1% of capital breaks the 10% default.
    risk = portfolio_risk({1: 1}, {1: 101}, 1000, side="long_only")
    assert risk["blockers"] == ["rounded_single_name_limit"]
    assert risk["gross"] == pytest.approx(0.101) and risk["largest_weight"] == risk["gross"]
    assert portfolio_risk({1: 1}, {1: 100}, 1000, side="long_only")["blockers"] == []


def test_each_limit_blocks_by_name():
    prices = {1: 10.0, 2: 10.0}
    limits = {"max_gross": 0.5, "max_weight": 0.4}
    gross = portfolio_risk({1: 30, 2: 30}, prices, 1000, side="long_short", **limits)
    assert gross["blockers"] == ["rounded_gross_limit"]  # 0.6 in all; each name 0.3
    name = portfolio_risk({1: 50}, prices, 1000, side="long_short", max_gross=1.0, max_weight=0.4)
    assert name["blockers"] == ["rounded_single_name_limit"]
    both = portfolio_risk({1: 60, 2: 60}, prices, 1000, side="long_short", **limits)
    assert both["blockers"] == ["rounded_gross_limit", "rounded_single_name_limit"]
    short = portfolio_risk({1: 10, 2: -10}, prices, 1000, side="long_only")
    assert short["blockers"] == ["long_only_short_position"]
    assert short["net"] == 0
    big = portfolio_risk({}, prices, 1000, side="long_short", orders=[{"permno": 1, "shares": 6}])
    assert big["blockers"] == []
    over = portfolio_risk(
        {},
        prices,
        1000,
        side="long_short",
        max_order_notional=50,
        orders=[{"permno": 1, "shares": 6}],
    )
    assert over["blockers"] == ["order_notional_limit"]


def test_a_limit_at_the_boundary_passes():
    risk = portfolio_risk({1: 10}, {1: 10.0}, 1000, side="long_only", max_gross=0.1, max_weight=0.1)
    assert risk["blockers"] == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capital": 0}, "capital"),
        ({"capital": float("inf")}, "capital"),
        ({"capital": 100, "max_weight": 0.5, "max_gross": 0.4}, "limits"),
        ({"capital": 100, "max_weight": 0.0}, "limits"),
        ({"capital": 100, "side": "either"}, "side"),
        ({"capital": 100, "max_order_notional": float("nan")}, "limits"),
        ({"capital": 100, "max_order_notional": 0}, "limits"),
    ],
)
def test_invalid_risk_inputs_are_refused(kwargs, match):
    kwargs = {"side": "long_only", **kwargs}
    capital = kwargs.pop("capital")
    with pytest.raises(ValueError, match=match):
        portfolio_risk({1: 1}, {1: 10.0}, capital, **kwargs)


@pytest.mark.parametrize("mark", [0.0, -5.0, float("nan")])
def test_a_position_needs_a_positive_finite_mark(mark):
    with pytest.raises(ValueError, match="marks"):
        portfolio_risk({1: 1}, {1: mark}, 1000, side="long_only")


def plan(**overrides):
    values = {
        "projected_shares": {"1": 10},
        "capital": 1000,
        "profile": "long_only",
        "execution_cost_bps": 11,
        "limits": {"max_gross": 1, "max_weight": 1, "max_order_notional": 1000},
    }
    values.update(overrides)
    return values


def test_the_current_market_recheck_includes_the_remaining_quantities_and_costs():
    account = {"positions": {1: 9}, "cash": 100}
    blocked = execution_risk(plan(), account, {1: 100})
    assert blocked["blockers"] == ["current_cash_deficit"]
    allowed = execution_risk(plan(), account, {1: 98})
    assert not allowed["blockers"]
    assert allowed["projected_cash_at_current_marks"] == pytest.approx(100 - 98 * 1.0011)
    filled = execution_risk(plan(), {"positions": {1: 10}, "cash": 10}, {1: 98})
    assert filled["projected_cash_at_current_marks"] == 10


def test_the_recheck_prices_each_instrument_at_its_own_cost():
    rates = {"1": 10, "2": 100}
    both = plan(projected_shares={"1": 5, "2": 5}, execution_cost_bps=rates)
    risk = execution_risk(both, {"positions": {}, "cash": 1000}, {1: 20, 2: 20})
    assert risk["projected_cash_at_current_marks"] == pytest.approx(
        1000 - 5 * 20 * 1.001 - 5 * 20 * 1.01
    )


def test_the_recheck_never_resizes_and_closes_what_the_plan_drops():
    risk = execution_risk(
        plan(projected_shares={"1": 0}), {"positions": {1: 4}, "cash": 0}, {1: 50}
    )
    # The 4 held shares are sold at the current mark, less the cost: cash comes in.
    assert risk["projected_cash_at_current_marks"] == pytest.approx(4 * 50 * (1 - 0.0011))
    assert risk["blockers"] == []


def test_the_recheck_refuses_fractional_shares_and_unusable_marks():
    with pytest.raises(ValueError, match="whole-share"):
        execution_risk(plan(projected_shares={"1": 1.5}), {"positions": {}, "cash": 10}, {1: 5})
    with pytest.raises(ValueError, match="marks"):
        execution_risk(plan(), {"positions": {}, "cash": 10}, {1: 0.0})
    with pytest.raises(ValueError, match="cash"):
        execution_risk(plan(), {"positions": {}, "cash": float("nan")}, {1: 5})
