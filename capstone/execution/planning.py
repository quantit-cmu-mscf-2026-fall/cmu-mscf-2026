"""Pure fixed-share sizing, shared by historical replay and frozen order plans.

Nothing here reads data or talks to a broker: whole-share quantities come from
weights, prices, current holdings and per-instrument costs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

import pandas as pd


def _whole(x: float) -> int:
    """Nearest whole number, halves away from zero."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


class UnqualifiedCost(ValueError):
    """A required instrument cost is unavailable; this is an input blocker."""


def cost_rate(bps, name):
    """Scalar compatibility or a frozen per-instrument quote, expressed as a fraction."""
    value = bps.get(name, bps.get(str(name))) if isinstance(bps, Mapping) else bps
    if value is None or not math.isfinite(float(value)) or float(value) < 0:
        raise UnqualifiedCost(
            f"qualified finite nonnegative instrument costs are required: instrument {name}"
        )
    return float(value) / 1e4


def positive_price(price):
    return price is not None and math.isfinite(float(price)) and float(price) > 0


def carry_marks(row, positions, marks, *, skip=()):
    """Held names without a valid observation take their last valid mark (v3/v4 carry).

    Returns the filled row and the carried names; `marks=None` leaves the row untouched.
    """
    if marks is None:
        return row, []
    row, carried = row.copy(), []
    for name, quantity in sorted(positions.items()):
        if quantity and name not in skip and not positive_price(row.get(name)) and name in marks:
            row[name] = marks[name]
            carried.append(name)
    return row, carried


def share_plan(
    weights,
    current,
    prices,
    capital,
    *,
    min_notional=100.0,
    side="long_short",
    fee_bps=0.0,
    carry=None,
):
    """Fixed-share sizing. `carry` (fixed-share-next-open-v3, program-events-v4) is
    `{"marks": last valid prices, "fallback_bps": scenario capped-spread rate}`: a held name
    without a sizing price keeps its shares (listed in `carried`); a held name with a price
    but no qualified quote exits at `fallback_bps` (recorded as the order's `fallback_cost`).
    """
    if not math.isfinite(capital) or capital <= 0 or min_notional < 0:
        raise ValueError("positive finite capital and nonnegative minimum notional are required")
    if not isinstance(weights, pd.Series) or not weights.index.is_unique:
        raise ValueError("weights require unique instrument identifiers")
    if not all(math.isfinite(float(w)) for w in weights):
        raise ValueError("weights must be finite")
    names = sorted(
        {int(n) for n, w in weights.items() if w} | {int(n) for n, q in current.items() if q}
    )
    if side not in {"long_only", "long_short"}:
        raise ValueError("valid profile and finite nonnegative costs are required")
    if side == "long_only" and (weights < 0).any():
        raise ValueError("long-only target weights cannot be negative")
    rates, fallback, marks = {}, set(), dict(prices)
    for n in names:
        try:
            rates[n] = cost_rate(fee_bps, n)
        except UnqualifiedCost:
            if carry is None or not current.get(n) or carry.get("fallback_bps") is None:
                raise
            rates[n] = cost_rate(carry["fallback_bps"], n)
            fallback.add(n)
        if carry is not None and current.get(n) and not positive_price(prices.get(n)):
            if n not in carry["marks"]:
                raise ValueError("sizing prices must be finite and positive")
            marks[n] = carry["marks"][n]
    rate = max(rates.values(), default=cost_rate(0 if isinstance(fee_bps, Mapping) else fee_bps, 0))
    old_gross = sum(abs(q) * float(marks[int(n)]) for n, q in current.items() if q)
    old_cost = (
        sum(abs(q) * float(marks[int(n)]) * rates[int(n)] for n, q in current.items() if q)
        if isinstance(fee_bps, Mapping)
        else rate * old_gross
    )
    budget = max(0.0, (capital - old_cost) / (1 + rate))
    targets, projected, orders, omitted, carried = {}, {}, [], [], []
    for name in names:
        price = float(prices[name]) if carry is None else float(prices.get(name, math.nan))
        held = current.get(name, 0)
        if isinstance(held, bool) or int(held) != held:
            raise ValueError("current positions must be whole shares")
        if carry is not None and held and not positive_price(price):
            targets[name] = projected[name] = int(held)
            carried.append(
                {
                    "permno": name,
                    "shares": int(held),
                    "mark": float(marks[name]),
                    "reason": "held_without_sizing_price",
                }
            )
            continue
        if not math.isfinite(price) or price <= 0:
            raise ValueError("sizing prices must be finite and positive")
        weight = 0.0 if name in fallback else round(float(weights.get(name, 0)), 12)
        target = (
            math.floor(weight * budget / price)
            if side == "long_only"
            else _whole(weight * budget / price)
        )
        quantity = target - int(held)
        targets[name] = target
        # Closing a position is never suppressed by a minimum-size convenience rule.
        if held * target < 0 and abs(target) * price < min_notional:
            omitted.append(
                {"permno": name, "shares": target, "reason": "minimum_notional_opening_leg"}
            )
            quantity = -int(held)
        elif (
            quantity
            and abs(quantity) * price < min_notional
            and target != 0
            and abs(target) >= abs(held)
        ):
            omitted.append({"permno": name, "shares": quantity, "reason": "minimum_notional"})
            quantity = 0
        projected[name] = int(held) + quantity
        if quantity:
            orders.append({"permno": name, "shares": quantity, "reference_price": price})
            if name in fallback:
                orders[-1]["fallback_cost"] = float(carry["fallback_bps"])
    fees = sum(abs(o["shares"]) * prices[o["permno"]] * rates[o["permno"]] for o in orders)
    result = {
        "targets": targets,
        "projected": projected,
        "orders": orders,
        "omitted": omitted,
        "sizing_capital": budget,
        "estimated_fees": fees,
    }
    if carry is not None:
        result["carried"] = carried
    if eligibility := weights.attrs.get("formation_eligibility"):
        result["formation_eligibility"] = deepcopy(eligibility)
    return result
