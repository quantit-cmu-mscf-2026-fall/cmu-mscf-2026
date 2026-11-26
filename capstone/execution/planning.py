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
    """Held names without a valid observation take their last valid mark.

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


def _int_keys(mapping, label):
    """Re-key by integer permno; refuse keys that collide after conversion."""
    out = {int(n): q for n, q in mapping.items()}
    if len(out) != len(mapping):
        raise ValueError(f"{label} collide after normalising keys to integer permnos")
    return out


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
    """Fixed-share sizing. `carry` is
    `{"marks": last valid prices, "fallback_bps": scenario capped-spread rate}`: a held name
    without a sizing price keeps its shares (listed in `carried`); a held name with a price
    but no qualified quote exits at `fallback_bps` (recorded as the order's `fallback_cost`).
    """
    if not math.isfinite(capital) or capital <= 0 or not min_notional >= 0:
        raise ValueError("positive finite capital and nonnegative minimum notional are required")
    current = _int_keys(current, "current positions")
    prices = {n: (math.nan if p is None else p) for n, p in prices.items()}
    if not isinstance(weights, pd.Series) or not weights.index.is_unique:
        raise ValueError("weights require unique instrument identifiers")
    keys = [int(n) for n in weights.index]
    if len(set(keys)) != len(keys):
        raise ValueError("weights collide after normalising keys to integer permnos")
    weights = weights.copy()
    weights.index = keys
    if not all(math.isfinite(float(w)) for w in weights):
        raise ValueError("weights must be finite")
    names = sorted({n for n, w in weights.items() if w} | {n for n, q in current.items() if q})
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
            if n not in carry["marks"] or not positive_price(carry["marks"][n]):
                raise ValueError("sizing prices must be finite and positive")
            marks[n] = carry["marks"][n]
    rate = max(rates.values(), default=cost_rate(0 if isinstance(fee_bps, Mapping) else fee_bps, 0))
    old_gross = sum(abs(q) * float(marks[n]) for n, q in current.items() if q)
    old_cost = (
        sum(abs(q) * float(marks[n]) * rates[n] for n, q in current.items() if q)
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


def portfolio_risk(
    positions,
    prices,
    capital,
    *,
    side,
    max_gross=1.0,
    max_weight=0.1,
    max_order_notional=50_000,
    orders=(),
):
    """Check whole-share `positions` (permno -> shares) at `prices` against limits.

    Returns the book's gross, net and largest weight (fractions of `capital`)
    and the names of the limits it breaks (`blockers`): gross above
    `max_gross`, a name above `max_weight`, a short under `long_only`, an order
    above `max_order_notional`.

    The defaults (gross 1.0, one name 0.1, one order 50,000) are the testbed's
    values, documented here and not a policy: a caller with a real mandate
    passes its own limits, and adopting these is the team's decision.
    """
    if capital <= 0 or not math.isfinite(capital):
        raise ValueError("capital must be finite and positive")
    if not 0 < max_weight <= max_gross or not math.isfinite(max_gross):
        raise ValueError("invalid portfolio limits")
    if not math.isfinite(max_order_notional) or max_order_notional <= 0:
        raise ValueError("invalid portfolio limits")
    if any(isinstance(q, bool) or int(q) != q for q in positions.values()):
        raise ValueError("risk checks require whole-share positions")
    priced = [n for n, q in positions.items() if q] + [o["permno"] for o in orders]
    if any(not math.isfinite(float(prices[n])) or prices[n] <= 0 for n in priced):
        raise ValueError("risk checks require positive finite marks")
    values = {n: int(q) * float(prices[n]) / capital for n, q in positions.items() if q}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("position values must be finite")
    blockers = []
    gross = sum(abs(v) for v in values.values())
    if gross > max_gross + 1e-10:
        blockers.append("rounded_gross_limit")
    if any(abs(v) > max_weight + 1e-10 for v in values.values()):
        blockers.append("rounded_single_name_limit")
    if side == "long_only" and any(v < 0 for v in values.values()):
        blockers.append("long_only_short_position")
    if side not in {"long_only", "long_short"}:
        raise ValueError("unknown portfolio side")
    if any(abs(o["shares"]) * float(prices[o["permno"]]) > max_order_notional for o in orders):
        blockers.append("order_notional_limit")
    return {
        "gross": gross,
        "net": sum(values.values()),
        "largest_weight": max((abs(v) for v in values.values()), default=0),
        "blockers": blockers,
    }


def execution_risk(plan, account, marks):
    """Recheck remaining fixed quantities at current marks; never resize an instruction."""
    marks = {n: float(p) for n, p in _int_keys(marks, "marks").items()}
    if any(
        isinstance(q, bool) or int(q) != q
        for q in [*plan["projected_shares"].values(), *account["positions"].values()]
    ):
        raise ValueError("execution risk requires whole-share positions")
    target = _int_keys(plan["projected_shares"], "projected shares")
    held = _int_keys(account["positions"], "account positions")
    target = {n: int(q) for n, q in target.items()}
    held = {n: int(q) for n, q in held.items()}
    required = {n for n, q in target.items() if q} | {n for n, q in held.items() if q}
    if any(not math.isfinite(marks[n]) or marks[n] <= 0 for n in required):
        raise ValueError("execution risk requires positive finite marks")
    orders = [
        {"permno": n, "shares": target.get(n, 0) - held.get(n, 0)}
        for n in sorted(set(target) | set(held))
        if target.get(n, 0) != held.get(n, 0)
    ]
    result = portfolio_risk(
        target, marks, plan["capital"], side=plan["profile"], orders=orders, **plan["limits"]
    )
    cash = float(account["cash"]) - sum(
        (o["shares"] + abs(o["shares"]) * cost_rate(plan["execution_cost_bps"], o["permno"]))
        * marks[o["permno"]]
        for o in orders
    )
    if not math.isfinite(cash):
        raise ValueError("current execution cash projection is invalid")
    if plan["profile"] == "long_only" and cash < -1e-8:
        result["blockers"].append("current_cash_deficit")
    result["projected_cash_at_current_marks"] = cash
    return result
