"""Whole-share sizing with per-instrument costs (`capstone.execution.planning`).

Pure arithmetic on synthetic weights, prices and holdings; no data, no broker.
"""

from __future__ import annotations

import pandas as pd
import pytest

from capstone.execution.planning import (
    UnqualifiedCost,
    carry_marks,
    cost_rate,
    positive_price,
    share_plan,
)


def test_sizing_rounds_the_target_not_the_trade_and_exits_small_positions():
    plan = share_plan(pd.Series({1: 0.4, 2: 0}), {1: 3, 2: 1}, {1: 30, 2: 5}, 1000)
    assert plan["targets"] == {1: 13, 2: 0}
    assert [o["shares"] for o in plan["orders"]] == [10, -1]
    assert plan["projected"] == {1: 13, 2: 0}


def test_halves_round_away_from_zero_and_long_only_floors():
    weights = pd.Series({1: 0.25, 2: -0.25})
    plan = share_plan(weights, {}, {1: 10.0, 2: 10.0}, 20, min_notional=0)
    assert plan["targets"] == {1: 1, 2: -1}
    only = share_plan(pd.Series({1: 0.25}), {}, {1: 10.0}, 20, min_notional=0, side="long_only")
    assert only["targets"] == {1: 0} and only["orders"] == []


def test_a_scalar_cost_and_the_same_cost_per_instrument_size_the_same():
    weights = pd.Series({1: 0.5, 2: 0.5})
    scalar = share_plan(weights, {}, {1: 10, 2: 20}, 1000, fee_bps=10)
    mapped = share_plan(weights, {}, {1: 10, 2: 20}, 1000, fee_bps={"1": 10, "2": 10})
    assert scalar == mapped
    assert scalar["estimated_fees"] == pytest.approx(
        sum(abs(o["shares"]) * {1: 10, 2: 20}[o["permno"]] for o in scalar["orders"]) * 10e-4
    )


def test_a_dearer_instrument_leaves_fewer_shares_to_buy():
    weights = pd.Series({1: 0.5, 2: 0.5})
    cheap = share_plan(weights, {}, {1: 10, 2: 10}, 1000, fee_bps={1: 0, 2: 0})
    dear = share_plan(weights, {}, {1: 10, 2: 10}, 1000, fee_bps={1: 0, 2: 500})
    assert dear["sizing_capital"] < cheap["sizing_capital"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -1.0])
def test_an_unqualified_cost_is_refused_naming_the_instrument(bad):
    with pytest.raises(UnqualifiedCost, match="instrument 7$"):
        cost_rate({7: bad}, 7)
    with pytest.raises(UnqualifiedCost, match="instrument 8$"):
        cost_rate({7: 1.0}, 8)
    assert cost_rate({7: 25}, 7) == 0.0025 and cost_rate(25, 7) == 0.0025
    assert cost_rate({"7": 25}, 7) == 0.0025
    with pytest.raises(UnqualifiedCost):
        share_plan(pd.Series({7: 1.0}), {}, {7: 10.0}, 100, fee_bps={7: None})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capital": 0}, "capital"),
        ({"capital": float("nan")}, "capital"),
        ({"capital": 100, "min_notional": -1}, "capital"),
        ({"capital": 100, "side": "short_only"}, "profile"),
    ],
)
def test_invalid_sizing_inputs_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        share_plan(pd.Series({1: 0.5}), {}, {1: 10.0}, **kwargs)


def test_weights_and_holdings_must_be_well_formed():
    prices = {1: 10.0}
    with pytest.raises(ValueError, match="unique"):
        share_plan(pd.Series([0.5, 0.5], index=[1, 1]), {}, prices, 100)
    with pytest.raises(ValueError, match="finite"):
        share_plan(pd.Series({1: float("nan")}), {}, prices, 100)
    with pytest.raises(ValueError, match="long-only"):
        share_plan(pd.Series({1: -0.5}), {}, prices, 100, side="long_only")
    with pytest.raises(ValueError, match="whole shares"):
        share_plan(pd.Series({1: 0.5}), {1: 1.5}, prices, 100)
    with pytest.raises(ValueError, match="sizing prices"):
        share_plan(pd.Series({1: 0.5}), {}, {1: 0.0}, 100)


def test_a_small_opening_leg_is_omitted_but_a_close_never_is():
    weights = pd.Series({1: 0.01, 2: 0.0})
    plan = share_plan(weights, {2: 1}, {1: 10.0, 2: 10.0}, 1000, min_notional=100)
    assert [o["permno"] for o in plan["orders"]] == [2]
    assert plan["omitted"] == [{"permno": 1, "shares": 1, "reason": "minimum_notional"}]
    flip = share_plan(pd.Series({1: -0.01}), {1: 1}, {1: 10.0}, 1000, min_notional=100)
    assert flip["omitted"][0]["reason"] == "minimum_notional_opening_leg"
    assert flip["projected"] == {1: 0}


def test_positive_price_and_carry_marks():
    assert positive_price(1.0) and not any(positive_price(p) for p in (None, 0, -1, float("nan")))
    row = pd.Series({1: 10.0, 2: float("nan"), 3: float("nan")})
    filled, carried = carry_marks(row, {1: 5, 2: 3, 3: 0}, {2: 7.0, 3: 9.0})
    assert carried == [2] and filled[2] == 7.0 and pd.isna(filled[3])
    assert pd.isna(row[2])
    same, none = carry_marks(row, {2: 3}, None)
    assert same is row and none == []
    _, skipped = carry_marks(row, {2: 3}, {2: 7.0}, skip=(2,))
    assert skipped == []


def test_carry_keeps_a_held_name_without_a_price_and_exits_one_without_a_quote_at_the_fallback():
    carry = {"marks": {2: 20.0, 3: 5.0}, "fallback_bps": 50}
    plan = share_plan(
        pd.Series({1: 1.0, 2: 0.0, 3: 0.0}),
        {2: 4, 3: 10},
        {1: 10.0, 3: 5.0},
        1000,
        fee_bps={1: 10, 2: 10},
        carry=carry,
        min_notional=0,
    )
    assert plan["carried"] == [
        {"permno": 2, "shares": 4, "mark": 20.0, "reason": "held_without_sizing_price"}
    ]
    assert plan["projected"][2] == 4 and plan["targets"][3] == 0
    exit_order = next(o for o in plan["orders"] if o["permno"] == 3)
    assert exit_order["shares"] == -10 and exit_order["fallback_cost"] == 50.0
    with pytest.raises(UnqualifiedCost, match="instrument 3"):
        share_plan(pd.Series({1: 1.0}), {3: 10}, {1: 10.0, 3: 5.0}, 1000, fee_bps={1: 10, 2: 10})


def test_a_held_name_with_no_price_and_no_mark_is_refused():
    carry = {"marks": {}, "fallback_bps": 50}
    with pytest.raises(ValueError, match="sizing prices"):
        share_plan(pd.Series({1: 0.0}), {1: 2}, {}, 1000, fee_bps=10, carry=carry)


def test_formation_eligibility_is_carried_into_the_plan_by_copy():
    weights = pd.Series({1: 0.5})
    weights.attrs["formation_eligibility"] = {"1": "eligible"}
    plan = share_plan(weights, {}, {1: 10.0}, 100)
    assert plan["formation_eligibility"] == {"1": "eligible"}
    plan["formation_eligibility"]["1"] = "changed"
    assert weights.attrs["formation_eligibility"] == {"1": "eligible"}


def test_a_string_keyed_current_book_is_read_like_an_int_keyed_one():
    args = (pd.Series({2: 0.5}), {1: 10.0, 2: 10.0}, 1000)
    as_int = share_plan(args[0], {1: 10}, *args[1:], min_notional=0)
    as_str = share_plan(args[0], {"1": 10}, *args[1:], min_notional=0)
    assert as_str["orders"] == as_int["orders"]
    assert (1, -10) in [(o["permno"], o["shares"]) for o in as_str["orders"]]
    assert as_str["projected"] == as_int["projected"]


def test_current_keys_that_collide_after_normalising_are_refused():
    with pytest.raises(ValueError, match="collide"):
        share_plan(pd.Series({1: 0.5}), {1: 10, "1": 5}, {1: 10.0}, 1000, min_notional=0)


def test_a_fallback_exit_ignores_the_target_weight():
    carry = {"marks": {3: 5.0}, "fallback_bps": 50}
    plan = share_plan(
        pd.Series({1: 0.5, 3: 0.5}),
        {3: 10},
        {1: 10.0, 3: 5.0},
        1000,
        fee_bps={1: 10},
        carry=carry,
        min_notional=0,
    )
    exit_order = next(o for o in plan["orders"] if o["permno"] == 3)
    assert exit_order["shares"] == -10 and exit_order["fallback_cost"] == 50.0


def test_a_reduction_below_the_minimum_notional_is_still_sent():
    plan = share_plan(pd.Series({1: 0.09}), {1: 10}, {1: 10.0}, 1000, min_notional=100)
    assert plan["targets"] == {1: 9}
    assert [o["shares"] for o in plan["orders"]] == [-1] and plan["omitted"] == []


def test_a_none_price_on_a_held_name_is_carried_not_a_type_error():
    carry = {"marks": {1: 5.0}, "fallback_bps": 50}
    plan = share_plan(pd.Series({1: 0.0}), {1: 2}, {1: None}, 1000, fee_bps=10, carry=carry)
    assert plan["carried"][0]["permno"] == 1 and plan["projected"] == {1: 2}


def test_string_keyed_weights_read_as_the_same_targets_as_integer_keys():
    args = ({1: 10}, {1: 10.0, 2: 10.0}, 1000)
    want = share_plan(pd.Series({1: 0.5, 2: -0.5}), *args, min_notional=0)
    got = share_plan(pd.Series({"1": 0.5, "2": -0.5}), *args, min_notional=0)
    assert got["orders"] == want["orders"] and got["targets"] == want["targets"]
    assert got["targets"] == {1: 50, 2: -50}
    held = share_plan(pd.Series({"1": 0.5}), {1: 10}, {1: 10.0}, 1000, min_notional=0)
    assert [(o["permno"], o["shares"]) for o in held["orders"]] == [(1, 40)]
    empty = share_plan(pd.Series({"1": 0.5, "2": -0.5}), {}, {1: 10.0, 2: 10.0}, 1000)
    assert empty["targets"] == {1: 50, 2: -50}


def test_weights_keys_that_collide_as_integers_are_refused():
    with pytest.raises(ValueError, match="weights collide"):
        share_plan(pd.Series([0.5, 0.5], index=[1, "1"]), {1: 10}, {1: 10.0}, 1000)


@pytest.mark.parametrize("mark", [float("nan"), 0.0, -3.0, float("inf")])
def test_a_carried_mark_must_be_positive_and_finite(mark):
    carry = {"marks": {1: mark}, "fallback_bps": 50}
    with pytest.raises(ValueError, match="sizing prices"):
        share_plan(
            pd.Series({2: 0.5}),
            {1: 5, 2: 50},
            {1: float("nan"), 2: 10.0},
            1000,
            min_notional=0,
            carry=carry,
        )


def test_a_nan_minimum_notional_is_refused():
    with pytest.raises(ValueError, match="minimum notional"):
        share_plan(pd.Series({1: 0.5}), {}, {1: 10.0}, 1000, min_notional=float("nan"))
