"""Orders = target - current, deterministic."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from capstone.strategy.orders import Orders, orders


def book(weights: dict[int, float]) -> pd.Series:
    out = pd.Series(weights, dtype="float64")
    out.index.name = "permno"
    return out


def test_a_trade_is_target_minus_current_with_a_missing_name_held_at_zero():
    target = book({3: 0.25, 1: -0.5, 7: 0.25})
    current = book({1: -0.25, 5: 0.5})
    result = orders(target, current, min_trade=0.0)
    expected = book({1: -0.25, 3: 0.25, 5: -0.5, 7: 0.25}).rename("trade")
    pd.testing.assert_series_equal(result.trades, expected, check_exact=True)


def test_trades_below_the_minimum_are_dropped_and_one_at_it_is_kept():
    # Binary fractions, so each difference is exact.
    target = book({1: 0.5, 2: 0.5, 3: 0.5})
    current = book({1: 0.375, 2: 0.4375, 3: 0.0})
    result = orders(target, current, min_trade=0.125)
    pd.testing.assert_series_equal(result.trades, book({1: 0.125, 3: 0.5}).rename("trade"))
    small = orders(book({1: 0.5}), book({1: 0.25}), min_trade=0.25)
    assert list(small.trades.index) == [1]
    tiny = orders(book({1: 0.5}), book({1: 0.25}), min_trade=0.2500001)
    assert tiny.trades.empty


def test_no_trade_is_emitted_for_a_name_already_on_target():
    result = orders(book({1: 0.5, 2: -0.5}), book({1: 0.5, 2: -0.25}), min_trade=0.0)
    assert list(result.trades.index) == [2]


def test_trades_are_signed_sorted_by_permno_and_sum_to_the_turnover():
    rng = np.random.default_rng(0)
    names = rng.permutation(np.arange(10000, 10200))
    target = book(dict(zip(names[:150].tolist(), rng.normal(0, 0.01, 150), strict=True)))
    current = book(dict(zip(names[50:].tolist(), rng.normal(0, 0.01, 150), strict=True)))
    result = orders(target, current, min_trade=0.002)
    trades = result.trades
    assert trades.index.is_monotonic_increasing and trades.index.is_unique
    assert trades.index.name == "permno"
    assert (trades.abs() >= 0.002).all()
    assert (trades < 0).any() and (trades > 0).any()
    assert result.turnover == trades.abs().sum()
    full = target.reindex(trades.index, fill_value=0.0) - current.reindex(
        trades.index, fill_value=0.0
    )
    np.testing.assert_array_equal(trades.to_numpy(), full.to_numpy())


def test_the_order_of_the_inputs_does_not_change_the_orders():
    rng = np.random.default_rng(1)
    target = book(dict(enumerate(rng.normal(0, 0.01, 40))))
    current = book(dict(enumerate(rng.normal(0, 0.01, 40), start=20)))
    a = orders(target, current, min_trade=0.001)
    b = orders(
        target.sample(frac=1, random_state=3),
        current.sample(frac=1, random_state=4),
        min_trade=0.001,
    )
    pd.testing.assert_series_equal(a.trades, b.trades, check_exact=True)
    assert a.turnover == b.turnover


def test_orders_are_frozen():
    result = orders(book({1: 0.5}), book({}), min_trade=0.0)
    assert isinstance(result, Orders)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.turnover = 0.0


@pytest.mark.parametrize(
    ("target", "current", "min_trade"),
    [
        (book({1: float("nan")}), book({}), 0.0),
        (book({1: 0.5}), book({2: float("inf")}), 0.0),
        (book({1: 0.5}), book({}), -0.1),
        (pd.Series([0.1, 0.2], index=pd.Index([1, 1], name="permno")), book({}), 0.0),
    ],
)
def test_bad_inputs_are_refused(target, current, min_trade):
    with pytest.raises(ValueError):
        orders(target, current, min_trade=min_trade)
