"""Point-in-time level fields: on each date, only the splits known on that date.

CRSP's cumulative adjustment factors in the shared file (`dlycumfacpr` for
prices, `dlycumfacshr` for shares) are anchored to the file's last date. A
price divided by its factor is a level that already reflects every split the
stock makes later, through the end of the file (holdout years included): a low
"adjusted close" marks a stock that will split, which is look-ahead. Ratios over
time are unaffected, since the anchor cancels; cross-sectional levels are not.
Rebasing each stock to its first loaded row removes the leak but ranks price
relative to that row, which depends on where the loaded window starts.

The rule here: **a factor's value on date t is its tree evaluated on the panel
re-anchored at t.** Every level on an earlier date s <= t is restated in date-t
units by the ratio of cumulative factors f(t) / f(s), which holds only the
splits between s and t, all known at t. On date t itself every level is its
nominal value. Time-series windows are consistent (no split moves a ratio), and
nothing from after t enters. Any other anchor for the file changes nothing
(anchor invariance).

So the panel's level fields (`LEVEL_FIELDS`) hold the nominal values of each
date, and the panel carries the factors (`SplitFactors`). `evaluate_pit` is the
tree's value by that rule, one block at a time: a block is a run of dates on
which no stock's factor changes, evaluated on the panel re-anchored at the block.
A tree that reads no level field, or a panel without `split_factors` (synthetic
panels), is evaluated as is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone.factors import interpret
from capstone.factors.tree import (
    REGIME_FUNCS,
    BinOp,
    Call,
    Const,
    Field_,
    Node,
    Unary,
    features_used,
    uses_regime,
)

PRICE_FIELDS = ("open", "high", "low", "close")
SHARE_FIELDS = ("volume", "shares")
LEVEL_FIELDS = (*PRICE_FIELDS, *SHARE_FIELDS)


@dataclass(frozen=True)
class SplitFactors:
    """CRSP's cumulative factors per (date, permno): `price` (`dlycumfacpr`),
    `shares` (`dlycumfacshr`); NaN where missing or not positive. Their anchor
    is arbitrary: only ratios between two dates are ever used."""

    price: pd.DataFrame
    shares: pd.DataFrame


def consistent_fields(
    fields: Mapping[str, pd.DataFrame | pd.Series], factors: SplitFactors
) -> dict[str, pd.DataFrame | pd.Series]:
    """`fields` with the levels on one consistent basis across splits (prices
    over the price factor, share counts times the share factor), whose anchor is
    arbitrary. Right for ratios over time, never for a level on its own."""
    out = dict(fields)
    for name in PRICE_FIELDS:
        if name in out:
            out[name] = out[name] / factors.price
    for name in SHARE_FIELDS:
        if name in out:
            out[name] = out[name] * factors.shares
    return out


def anchors(factors: SplitFactors) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each date's own factors (the last known, carried over non-trading days)."""
    return factors.price.ffill(), factors.shares.ffill()


def anchored_at(
    fields: Mapping[str, pd.DataFrame | pd.Series], factors: SplitFactors, date
) -> dict[str, pd.DataFrame | pd.Series]:
    """The fields through `date`, restated in that date's units: the panel a
    reader with data through `date` sees (its levels on `date` are nominal)."""
    base = consistent_fields(fields, factors)
    price, shares = anchors(factors)
    p, s = price.loc[date], shares.loc[date]
    out = {name: frame.loc[:date] for name, frame in base.items()}
    for name in PRICE_FIELDS:
        if name in out:
            out[name] = out[name] * p
    for name in SHARE_FIELDS:
        if name in out:
            out[name] = out[name] / s
    return out


def _lookback(node: Node) -> int | None:
    """Rows of history the tree's value on a date needs (None: all of it)."""
    if isinstance(node, Field_ | Const):
        return 0
    if isinstance(node, Unary):
        return _lookback(node.operand)
    if isinstance(node, BinOp):
        a, b = _lookback(node.left), _lookback(node.right)
        return None if a is None or b is None else max(a, b)
    if isinstance(node, Call):
        if node.func in REGIME_FUNCS or node.func == "ema":
            return None
        a = _lookback(node.arg)
        b = _lookback(node.arg2) if node.arg2 is not None else 0
        if a is None or b is None:
            return None
        return max(a, b) + (int(node.window) if node.window is not None else 0)
    raise TypeError(f"not a Node: {node!r}")


def _block_starts(price: pd.DataFrame, shares: pd.DataFrame) -> np.ndarray:
    """Row positions where some stock's factor changes (row 0 included)."""
    changed = np.zeros(len(price), dtype=bool)
    changed[0] = True
    for frame in (price, shares):
        a = frame.to_numpy(dtype=float)
        same = (a[1:] == a[:-1]) | (np.isnan(a[1:]) & np.isnan(a[:-1]))
        changed[1:] |= ~same.all(axis=1)
    return np.flatnonzero(changed)


def _evaluate_blocks(
    node: Node, base, price, shares, *, strict_domains=False, cross_sectional_universe=None
) -> pd.DataFrame:
    """The exact rule, one block of constant factors at a time. `base` holds the
    levels on one consistent basis (`consistent_fields`)."""
    starts = _block_starts(price, shares)
    ends = np.append(starts[1:], len(price))
    need = _lookback(node)
    used = features_used(node) | ({"mkt_return"} if uses_regime(node) else set())
    base = {k: v for k, v in base.items() if k in used}
    template = interpret._template(base)
    out = np.full(template.shape, np.nan)
    arrays = {
        k: (v.to_numpy(dtype=float) if isinstance(v, pd.DataFrame) else v) for k, v in base.items()
    }
    p_all, s_all = price.to_numpy(dtype=float), shares.to_numpy(dtype=float)
    for r0, r1 in zip(starts, ends, strict=True):
        lo = 0 if need is None else max(0, r0 - need)
        index = template.index[lo:r1]
        fields = {}
        for name, value in base.items():
            if isinstance(value, pd.Series):
                fields[name] = value.iloc[lo:r1]
                continue
            a = arrays[name][lo:r1]
            if name in PRICE_FIELDS:
                a = a * p_all[r0]
            elif name in SHARE_FIELDS:
                a = a / s_all[r0]
            fields[name] = pd.DataFrame(a, index=index, columns=template.columns)
        universe = (
            None if cross_sectional_universe is None else cross_sectional_universe.iloc[lo:r1]
        )
        value = interpret.evaluate(
            node, fields, strict_domains=strict_domains, cross_sectional_universe=universe
        )
        out[r0:r1] = value.to_numpy(dtype=float)[r0 - lo :]
    return pd.DataFrame(out, index=template.index, columns=template.columns)


def evaluate_pit(
    node: Node,
    fields: Mapping[str, pd.DataFrame | pd.Series],
    factors: SplitFactors | None,
    *,
    strict_domains: bool = False,
    cross_sectional_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The tree's value on each date t, on the panel re-anchored at t (see the
    module docstring). `fields` hold nominal levels; `factors` None: a panel
    without split information, evaluated as is. The other arguments are
    `interpret.evaluate`'s."""
    mode = {"strict_domains": strict_domains, "cross_sectional_universe": cross_sectional_universe}
    if factors is None or not (features_used(node) & set(LEVEL_FIELDS)):
        return interpret.evaluate(node, fields, **mode)
    price, shares = anchors(factors)
    return _evaluate_blocks(node, consistent_fields(fields, factors), price, shares, **mode)
