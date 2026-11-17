"""Existing fractional total-return diagnostic, with explicit assumed cost sensitivity.

This is not the fixed-share execution engine. Reinvested distributions, fractional
units and following-session-close fills cannot establish broker cash accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(
    returns,
    universe,
    targets,
    terminal,
    *,
    side,
    multiplier=0,
    missing=None,
    flags=None,
):
    if side not in {"long_only", "long_short"} or multiplier not in {0, 1, 2}:
        raise ValueError("unknown diagnostic profile or cost scenario")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("diagnostic dates must be unique and increasing")
    if returns.columns.has_duplicates or returns.index.max() > pd.Timestamp("2020-12-31"):
        raise ValueError("duplicate instruments or sealed holdout observation")
    returns = returns.sort_index(axis=1)
    columns = returns.columns
    r = returns.to_numpy(dtype=float)
    active = universe.reindex_like(returns).fillna(False).to_numpy(dtype=bool, copy=True)
    active &= ~terminal.reindex_like(returns).fillna(False).to_numpy(dtype=bool)
    marked = (
        missing.reindex_like(returns).fillna(False).to_numpy(dtype=bool)
        if missing is not None
        else np.zeros_like(active)
    )
    held = np.zeros(len(columns))
    cash = previous = 100000.0
    pending = None
    rows, used = [], []
    rate = multiplier * 6 / 10000
    for i, day in enumerate(returns.index):
        mask = np.abs(held) > 1e-10
        if not np.isfinite(r[i, mask]).all() or (r[i, mask] < -1).any():
            raise ValueError(f"unresolved/invalid held return {day.date()}")
        for j in np.flatnonzero(mask & marked[i]):
            used.append(
                {
                    "date": str(day.date()),
                    "permno": int(columns[j]),
                    "return": float(r[i, j]),
                    "flag": str(flags.at[day, columns[j]]) if flags is not None else "unknown",
                }
            )
        held[mask] *= 1 + r[i, mask]
        pnl = float(held.sum()) + cash - previous
        borrow = float(-held[held < 0].sum()) * multiplier * 200 / 252e4
        financing = max(-cash, 0) * multiplier * 500 / 252e4
        cash -= borrow + financing
        exits = mask & ~active[i]
        exit_notional = float(np.abs(held[exits]).sum())
        cash += float(held[exits].sum())
        held[exits] = 0
        exit_cost = exit_notional * rate
        cash -= exit_cost
        before_trade = cash + float(held.sum())
        if not np.isfinite(before_trade) or before_trade <= 0:
            raise ValueError(f"nonpositive diagnostic equity {day.date()}")
        traded, trade_cost = 0.0, 0.0
        if pending is not None:
            if pending.index.has_duplicates or not pending.index.isin(columns).all():
                raise ValueError("duplicate or unmapped diagnostic target")
            weights = pending.reindex(columns, fill_value=0).to_numpy(dtype=float).copy()
            weights[~active[i]] = 0
            if not np.isfinite(weights).all() or abs(weights).sum() > 1 + 1e-10:
                raise ValueError("invalid diagnostic target exposure")
            if side == "long_only" and (weights < -1e-12).any():
                raise ValueError("long-only diagnostic cannot short")
            # Fees change investable equity; solve without borrowing a fee shortfall.
            for _ in range(30):
                desired = (before_trade - trade_cost) * weights
                traded = float(abs(desired - held).sum())
                updated = traded * rate
                if abs(updated - trade_cost) < 1e-9:
                    trade_cost = updated
                    break
                trade_cost = updated
            else:
                raise ValueError("diagnostic sizing did not reconcile")
            held = (before_trade - trade_cost) * weights
            cash = before_trade - trade_cost - float(held.sum())
        equity = cash + float(held.sum())
        cost = borrow + financing + exit_cost + trade_cost
        if equity <= 0 or abs(previous + pnl - cost - equity) > 1e-6:
            raise ValueError("diagnostic P&L failed reconciliation")
        if side == "long_only" and (cash < -1e-7 or held.min() < -1e-7):
            raise ValueError("long-only diagnostic cash or holdings became negative")
        rows.append(
            {
                "date": day,
                "return": equity / previous - 1,
                "equity": equity,
                "cash": cash,
                "gross": float(abs(held).sum()) / equity,
                "net": float(held.sum()) / equity,
                "turnover": (exit_notional + traded) / before_trade,
                "total_return_pnl": pnl,
                "transaction_cost": exit_cost + trade_cost,
                "borrow_cost": borrow,
                "financing_cost": financing,
                "cost": cost,
                "names": int((abs(held) > 1e-10).sum()),
            }
        )
        previous = equity
        pending = targets.get(day)
    return pd.DataFrame(rows).set_index("date"), used
