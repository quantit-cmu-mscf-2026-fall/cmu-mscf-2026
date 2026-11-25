"""Orders: the trades that move the current book to the target, deterministically.

A trade is `target - current` per permno, a name missing from either book
held at zero. Trades smaller than `min_trade` in absolute weight are not sent
(the book stays that close to target). The trades are signed weights, sorted
by permno, and their absolute sum is the turnover; the same books give the
same bytes whatever order their names arrive in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class Orders:
    trades: pd.Series
    turnover: float


def _check(book: pd.Series, name: str) -> None:
    if not book.index.is_unique:
        raise ValueError(f"{name} names a permno more than once")
    if not np.isfinite(book.to_numpy(dtype="float64")).all():
        raise ValueError(f"{name} holds a non-finite weight")


def orders(target: pd.Series, current: pd.Series, *, min_trade: float) -> Orders:
    """Trades taking `current` to `target`, dropping any with |trade| < `min_trade`."""
    if not min_trade >= 0:
        raise ValueError(f"min_trade must be >= 0, got {min_trade!r}")
    _check(target, "target")
    _check(current, "current")
    names = target.index.union(current.index).sort_values()
    trade = target.astype("float64").reindex(names, fill_value=0.0) - current.astype(
        "float64"
    ).reindex(names, fill_value=0.0)
    trade = trade[(trade != 0) & (trade.abs() >= min_trade)].rename("trade")
    trade.index.name = "permno"
    return Orders(trade, float(trade.abs().sum()))
