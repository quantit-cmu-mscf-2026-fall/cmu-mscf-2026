"""Accounting steps extracted from the one shared replay loop; no strategy engine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.planning import carry_marks, cost_rate, positive_price


@dataclass(frozen=True)
class ReplayCosts:
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    short_borrow_bps_annual: float
    financing_bps_annual: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("all cost assumptions must be finite and nonnegative")


def held_carry(execution, cost_schedule):
    """Carry state for contracts declaring `held_without_price` (v3/v4), else None."""
    if "held_without_price" not in execution:
        return None
    carry = {
        "marks": {},
        "fallback_bps": None if cost_schedule is None else cost_schedule.fallback_bps(),
    }
    if "held_open_gap_exit_sessions" in execution:  # fallback versions
        carry["open_gap_sessions"] = execution["held_open_gap_exit_sessions"]
    return carry


def carried_open(row_open, positions, carry, pending):
    """Held names without an open and without an order are valued at their last mark."""
    skip = {o["permno"] for o in (pending or {}).get("orders", ())}
    return carry_marks(row_open, positions, carry and carry["marks"], skip=skip)[0]


def order_rates(pending, rates):
    """Planned rates; a held fallback exit keeps the `fallback_cost` it was planned at."""
    fallback = {o["permno"]: o["fallback_cost"] for o in pending["orders"] if "fallback_cost" in o}
    return {**rates(pending["decision"]), **fallback} if fallback else rates(pending["decision"])


def carried_close(day, positions, raw_open, row_close, carry, order_history):
    """Refresh last valid marks from today's opens; held names without a close keep them."""
    if carry is None:
        return row_close
    marks = carry["marks"]
    marks.update(
        {
            n: float(raw_open[n])
            for n, q in positions.items()
            if q and positive_price(raw_open.get(n))
        }
    )
    row_close, carried = carry_marks(row_close, positions, marks)
    if carried:
        order_history.append(
            {"date": str(day.date()), "status": "carried_at_last_valid_mark", "permnos": carried}
        )
    marks.clear()
    marks.update(
        {
            n: float(row_close[n])
            for n, q in positions.items()
            if q and positive_price(row_close.get(n))
        }
    )
    return row_close


def legacy_actions(
    day,
    positions,
    cash,
    row_open,
    splits,
    dividends,
    cash_settlements,
    event_result,
    distribution_pnl,
    action_history,
    terminated,
):
    split_names = event_result["affected"]
    settlements = {} if cash_settlements is None else cash_settlements.loc[day].dropna().to_dict()
    if any(not math.isfinite(v) or v < 0 for v in settlements.values()):
        raise ValueError("cash settlements must be finite and nonnegative")
    settled = set(settlements) | event_result["retired"]
    terminated.update(settled)
    changed = False
    for name, quantity in list(positions.items()):
        ratio = 1.0 if splits is None else float(splits.loc[day, name])
        dividend = 0.0 if dividends is None else float(dividends.loc[day, name])
        if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(dividend):
            raise ValueError("invalid corporate action")
        if ratio != 1:
            changed = True
            split_names.add(name)
            exact = quantity * ratio
            whole = math.trunc(exact)
            cash += (exact - whole) * float(row_open[name])
            positions[name] = whole
        cash += positions[name] * dividend
        changed |= dividend != 0
        distribution_pnl += positions[name] * dividend
        if name in settlements:
            changed = True
            cash += positions[name] * settlements[name]
            action_history.append(
                {
                    "date": str(day.date()),
                    "permno": name,
                    "shares": positions[name],
                    "cash_per_share": float(settlements[name]),
                }
            )
            positions.pop(name)
            settled.add(name)
    return cash, distribution_pnl, split_names, settled, changed


def fill_orders(
    pending,
    positions,
    cash,
    row_open,
    costs,
    cost_schedule,
    rates,
    day,
    components,
    *,
    event_protocol,
):
    cost = turnover = 0.0
    orders = []
    for order in pending["orders"]:
        name, quantity = order["permno"], order["shares"]
        price = float(row_open[name])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("missing or invalid next-session execution price")
        notional = abs(quantity) * price
        fallback = "fallback_cost" in order
        fee = notional * cost_rate(
            order["fallback_cost"] if fallback else rates(pending["decision"]), name
        )
        parts = (
            {
                "commission": costs.commission_bps,
                "spread": costs.half_spread_bps,
                "slippage": costs.slippage_bps,
            }
            if cost_schedule is None
            else cost_schedule.fallback_components()
            if fallback
            else cost_schedule.components(pending["decision"])[name]
        )
        for key, bps in parts.items():
            components[key] += notional * bps / 1e4
        cash -= quantity * price + fee
        positions[name] = positions.get(name, 0) + quantity
        cost += fee
        turnover += notional
        orders.append(
            {
                "date": str(day.date()),
                "decision": pending["decision"],
                "permno": name,
                "shares": quantity,
                "fill": price,
                "cost": fee,
            }
        )
        if fallback:
            orders[-1]["fallback_cost"] = order["fallback_cost"]
        if event_protocol:
            orders[-1].update(
                intent_id=pending["intent_id"],
                source_intent_id=pending["source_intent"]["source_intent_id"],
            )
            if "superseded_source_intent_id" in pending:
                orders[-1]["superseded_source_intent_id"] = pending["superseded_source_intent_id"]
    return cash, cost, turnover, orders


def mark_session(
    day,
    positions,
    cash,
    row_close,
    previous,
    cost,
    turnover,
    components,
    costs,
    cost_schedule,
    actions,
    distribution_pnl,
    side,
    *,
    event_protocol,
):
    for name, quantity in positions.items():
        if quantity and (
            name not in row_close
            or not math.isfinite(float(row_close[name]))
            or row_close[name] <= 0
        ):
            if event_protocol:
                raise ExecutionContractBlocked("held_valuation_missing")
            raise ValueError(
                "held positions require valid raw closing prices; no silent zero marks"
            )
    short = sum(-q * float(row_close[n]) for n, q in positions.items() if q < 0)
    borrow, financing = (
        (costs.short_borrow_bps_annual, costs.financing_bps_annual)
        if cost_schedule is None
        else cost_schedule.carry_rates()
    )
    components["borrow"] = short * borrow / 252e4
    components["financing"] = max(-cash, 0) * financing / 252e4
    carry = components["borrow"] + components["financing"]
    cash -= carry
    if side == "long_only" and cash < -1e-8:
        raise ValueError("long-only replay cannot borrow cash")
    cost += carry
    equity = cash + actions.value + sum(q * float(row_close[n]) for n, q in positions.items() if q)
    if equity <= 0 or not math.isfinite(equity):
        raise ValueError("replay account exhausted its equity")
    row = {
        "date": day,
        "equity": equity,
        "cash": cash,
        "net": equity / previous - 1,
        "cost": cost,
        "turnover": turnover / previous,
        "gross_exposure": sum(abs(q) * float(row_close[n]) for n, q in positions.items() if q)
        / equity,
        "net_exposure": sum(q * float(row_close[n]) for n, q in positions.items() if q) / equity,
        "positions": {n: q for n, q in sorted(positions.items()) if q},
        "market_values": {n: q * float(row_close[n]) for n, q in sorted(positions.items()) if q},
        "receivables": actions.value,
        "distribution_pnl": distribution_pnl,
        "market_pnl": equity - previous + cost - distribution_pnl,
        "gross_same_holdings": equity / previous - 1 + cost / previous,
        **{f"{key}_cost": value for key, value in components.items()},
    }
    return cash, equity, row


def open_gap_run(opens, day, name):
    """Consecutive sessions through `day` (inclusive) without a valid open; never later."""
    seen = opens.loc[:day, name].to_numpy(dtype=float) if name in opens else np.array([np.nan])
    good = np.flatnonzero(np.isfinite(seen) & (seen > 0))
    return len(seen) - (good[-1] + 1 if len(good) else 0)


def fallback_exits(event_result, opens, closes, day, positions, carry, session):
    """Book fallback exits before any fill: `ActionBook` held events that stay
    unqualifiable, then held names whose open has been missing `open_gap_sessions`
    consecutive sessions through `day` (a stale or delisted price series included). Each
    exits at its last valid close strictly before `day` (later closes are never read) at
    the capped half-spread cost (`sd.MAX_SPREAD / 2`, scenario multiplier). `session` is
    (costs, cost_schedule, rates, components). Returns (cash, cost, turnover, orders)."""
    cash, exits = event_result["cash"], list(event_result.get("fallback_exits", ()))
    limit = (carry or {}).get("open_gap_sessions")
    for name, held in sorted(positions.items()) if limit else ():
        if held and not positive_price(opens.loc[day].get(name)):
            run = open_gap_run(opens, day, name)
            if run >= limit and name not in {row["permno"] for row in exits}:
                reason = "held_without_open_for_registered_sessions"
                exits.append({"event_id": None, "reason": reason, "permno": name, "shares": held})
    if not exits:
        return cash, 0.0, 0.0, []
    costs, cost_schedule, rates, components = session
    bps = rates(day) if cost_schedule is None else cost_schedule.fallback_bps()
    prices, orders, rows = {}, [], []
    for row in exits:
        n = row["permno"]
        seen = closes.loc[closes.index < day, n] if n in closes else closes.iloc[:0, 0]
        seen = seen[np.isfinite(seen.astype(float)) & (seen > 0)]
        if seen.empty:
            raise ExecutionContractBlocked("fallback_exit_without_observed_price", str(row))
        prices[n] = float(seen.iloc[-1])
        orders.append({"permno": n, "shares": -row["shares"], "fallback_cost": float(bps)})
        rows.append(
            row | {"session": str(day.date()), "last_price_date": str(seen.index[-1].date())}
        )
    pending = {"orders": orders, "decision": str(day.date())}
    cash, cost, turnover, filled = fill_orders(
        pending,
        positions,
        cash,
        pd.Series(prices),
        costs,
        cost_schedule,
        rates,
        day,
        components,
        event_protocol=False,
    )
    for order, row in zip(filled, rows, strict=True):
        del order["fallback_cost"]
        order["fallback_exit"] = row | {
            "price": order["fill"],
            "cost": order["cost"],
            "cost_rule": "exit_at_capped_half_spread_v1",
        }
        positions.pop(row["permno"])
    return cash, cost, turnover, filled


def open_gaps(pending, missing, positions, opens, day, carry):
    """Open gaps: a name without a valid open no longer expires the whole book.
    Only its own order is dropped (recorded) and every other order executes: an unheld
    entry is not entered; a held name keeps its shares at the carried mark (it exits by
    `fallback_exits` once the gap reaches `open_gap_sessions`). A held name without any
    order or mark still blocks as before. Returns (pending, missing, row_open, records)."""
    kept = [o for o in pending["orders"] if o["permno"] not in missing]
    dropped = [
        {
            "permno": o["permno"],
            "shares": o["shares"],
            "reason": "held_without_open_carried"
            if positions.get(o["permno"])
            else "entry_without_open_dropped",
        }
        for o in pending["orders"]
        if o["permno"] in missing
    ]
    pending = {**pending, "orders": kept}
    row_open = carried_open(opens.loc[day], positions, carry, pending)
    names = set(positions) | {o["permno"] for o in kept}
    missing = [n for n in missing if n in names and not positive_price(row_open.get(n))]
    status = {"date": str(day.date()), "status": "dropped_missing_open", "orders": dropped}
    return pending, missing, row_open, [status] if dropped else []
