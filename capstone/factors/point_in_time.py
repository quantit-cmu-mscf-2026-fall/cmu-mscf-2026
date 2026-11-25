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

So the panel's level fields (`PRICE_FIELDS`, `SHARE_FIELDS`) hold the nominal
values of each date, and the panel carries the factors (`SplitFactors`).
`evaluate_pit` computes the rule exactly, without re-evaluating the tree once
per date, by tracking how each subtree depends on a per-stock rescaling of the
anchor:

- `hom(p, s)`: the value scales by k_p ** p * k_s ** s when every price of a
  stock is multiplied by k_p and every share count by k_s (a price is
  hom(1, 0), a share count hom(0, 1), returns, cap, market and accounting
  fields hom(0, 0): anchor-free);
- `log(p, s)`: the value shifts by p log k_p + s log k_s (a log of a level);
- `dated`: neither; the frame's row t already holds the date-t value.

The time-series operators commute with a constant per-stock rescaling (a
window of k x is k times the window of x, a rank or correlation over time
ignores it), so a subtree that is hom or log is computed once, on levels on any
one consistent basis, and restated at the end (or where a cross-sectional
operator needs real levels) by the date's own factor: row t times f(t) ** p.
A cross-sectional operator (`rank`, `scale`, `winsorize`) on an anchor-dependent
input, a sum of levels with different degrees and similar mixtures produce a
`dated` frame. A time-series operator on a `dated` input has no shortcut: then
the whole tree is evaluated block by block, one block per run of dates on which
no stock's factor changes, each on the panel re-anchored at that block (exact,
and slow: about a thousand blocks on the 1998-2020 file).

A tree that reads no level field, or a panel without `split_factors` (synthetic
panels), is evaluated as is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from capstone.factors import interpret
from capstone.factors.tree import (
    BINARY_PARAMETER_FUNCS,
    BINARY_WINDOW_FUNCS,
    FUNCS,
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
CROSS_SECTIONAL = frozenset({"rank", "scale", "winsorize"})
# Time-series operators by how they treat a constant per-stock rescaling x -> k x
# (k > 0) and shift x -> x + c of their input.
_KEEP_SCALE = frozenset(
    {"shift", "delta", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum", "decay_linear", "ema"}
)
_DROP_SCALE = frozenset({"ts_rank", "ts_argmax", "ts_argmin", "zscore"})
_KEEP_SHIFT = frozenset({"shift", "ts_mean", "ts_min", "ts_max", "decay_linear", "ema"})
_DROP_SHIFT = frozenset({"delta", "ts_std", "ts_rank", "ts_argmax", "ts_argmin", "zscore"})


@dataclass(frozen=True)
class SplitFactors:
    """CRSP's cumulative factors per (date, permno): `price` (`dlycumfacpr`),
    `shares` (`dlycumfacshr`); NaN where missing or not positive. Their anchor
    is arbitrary: only ratios between two dates are ever used."""

    price: pd.DataFrame
    shares: pd.DataFrame


# ---------------------------------------------------------------------------
# Kinds


@dataclass(frozen=True)
class _Kind:
    tag: str  # "hom", "log" or "dated"
    p: Fraction = Fraction(0)
    s: Fraction = Fraction(0)

    @property
    def free(self) -> bool:
        """Anchor-free: the same value under any anchor."""
        return self.tag == "hom" and self.p == 0 and self.s == 0


FREE = _Kind("hom")
DATED = _Kind("dated")


def _hom(p, s) -> _Kind:
    return _Kind("hom", Fraction(p), Fraction(s))


def _log(p, s) -> _Kind:
    p, s = Fraction(p), Fraction(s)
    return FREE if p == 0 and s == 0 else _Kind("log", p, s)


def _field_kind(name: str) -> _Kind:
    if name in PRICE_FIELDS:
        return _hom(1, 0)
    if name in SHARE_FIELDS:
        return _hom(0, 1)
    return FREE


class _NeedsBlocks(Exception):
    """A time-series operator reads an anchor-dependent cross-sectional value."""


def _const(node: Node) -> float | None:
    if isinstance(node, Const):
        return float(node.value)
    if isinstance(node, Unary) and isinstance(node.operand, Const):
        return -float(node.operand.value)
    return None


def _binop_kind(op: str, a: _Kind, b: _Kind, left: Node, right: Node) -> _Kind | None:
    """The kind of `left op right`, or None if it is anchor-dependent and mixed."""
    if a.tag == "dated" or b.tag == "dated":
        return None
    if op in "+-":
        sign = 1 if op == "+" else -1
        if a.tag == "hom" and b.tag == "hom" and (a.p, a.s) == (b.p, b.s):
            return a
        if a.tag == "log" and b.tag == "log":
            return _log(a.p + sign * b.p, a.s + sign * b.s)
        if a.free and b.tag == "log":
            return _log(sign * b.p, sign * b.s)
        if b.free and a.tag == "log":
            return a
        return None
    if a.tag == "hom" and b.tag == "hom":
        sign = 1 if op == "*" else -1
        return _hom(a.p + sign * b.p, a.s + sign * b.s)
    c_left, c_right = _const(left), _const(right)
    if op == "*" and a.tag == "log" and c_right is not None:
        return _log(a.p * Fraction(c_right), a.s * Fraction(c_right))
    if op == "*" and b.tag == "log" and c_left is not None:
        return _log(b.p * Fraction(c_left), b.s * Fraction(c_left))
    if op == "/" and a.tag == "log" and c_right not in (None, 0.0):
        return _log(a.p / Fraction(c_right), a.s / Fraction(c_right))
    return None


def _windowed_kind(func: str, window: int, a: _Kind) -> _Kind:
    if a.tag == "dated":
        raise _NeedsBlocks(func)
    if a.tag == "hom":
        if func in _KEEP_SCALE:
            return a
        if func in _DROP_SCALE:
            return FREE
        if func == "ts_product":
            return _hom(a.p * window, a.s * window)
    else:
        if func in _KEEP_SHIFT:
            return a
        if func in _DROP_SHIFT:
            return FREE
        if func == "ts_sum":
            return _log(a.p * window, a.s * window)
    raise _NeedsBlocks(func)


def _pair_kind(func: str, a: _Kind, b: _Kind) -> _Kind:
    if a.tag == "dated" or b.tag == "dated":
        raise _NeedsBlocks(func)
    if func == "ts_corr":
        return FREE
    # A covariance ignores a shift; a log input counts as anchor-free.
    pa = (a.p, a.s) if a.tag == "hom" else (0, 0)
    pb = (b.p, b.s) if b.tag == "hom" else (0, 0)
    return _hom(pa[0] + pb[0], pa[1] + pb[1])


def _plain_kind(func: str, a: _Kind) -> _Kind | None:
    """Kind of a pointwise or cross-sectional call; None: materialize the input."""
    if a.tag == "dated":
        return DATED
    if func in CROSS_SECTIONAL:
        return FREE if a.free else None
    if func == "log":
        return _log(a.p, a.s) if a.tag == "hom" else None
    if func == "abs":
        return a if a.tag == "hom" else None
    if func == "sign":
        return FREE if a.tag == "hom" else None
    if func in REGIME_FUNCS:  # x times a market-only (anchor-free) multiplier
        return a if a.tag == "hom" else None
    return None


def anchor_kind(node: Node) -> str:
    """ "free" if the tree's value does not depend on the anchor of the factors
    (any ratio or time-series rank of levels), "levels" if it does but has an
    exact shortcut, "blocks" if it needs the block-by-block evaluation."""
    try:
        kind = _Rewriter(None).kind_only(node)
    except _NeedsBlocks:
        return "blocks"
    return "free" if kind.free else "levels"


# ---------------------------------------------------------------------------
# Fields


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


# ---------------------------------------------------------------------------
# Evaluation


def _raw_value(node: Node, fields, **kwargs) -> pd.DataFrame:
    """The subtree's value as the interpreter computes it inside a tree (no
    clean-up of infinities in between, as in one `interpret.evaluate` call).
    `kwargs` (`strict_domains`, `cross_sectional_universe`) reach the interpreter."""
    template = interpret._template(fields)
    out = interpret._eval(node, fields, template, **kwargs)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(float(out), index=template.index, columns=template.columns)
    return out


class _Rewriter:
    """Rewrites a tree into one over consistent fields and lifted, dated frames."""

    def __init__(self, fields, price=None, shares=None, nominal=None, kwargs=None):
        self.fields = None if fields is None else dict(fields)
        self.price, self.shares, self.nominal = price, shares, nominal
        self.kwargs = kwargs or {}
        self.n = 0

    def kind_only(self, node: Node) -> _Kind:
        return self.walk(node)[1]

    def lift(self, frame: pd.DataFrame) -> Field_:
        name = f"__pit_{self.n}"
        self.n += 1
        self.fields[name] = frame
        return Field_(name)

    def materialize(self, node: Node, kind: _Kind) -> Node:
        """`node` as a dated frame: row t restated at date t's own anchor."""
        if kind.tag == "dated" or kind.free or self.fields is None:
            return node
        if isinstance(node, Field_) and node.name in LEVEL_FIELDS:
            return self.lift(self.nominal[node.name])  # exactly the nominal level
        base = _raw_value(node, self.fields, **self.kwargs)
        price_p, share_p = float(kind.p), float(kind.s)
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind.tag == "hom":
                if price_p:
                    base = base * self.price**price_p
                if share_p:
                    base = base * self.shares ** (-share_p)
            else:
                if price_p:
                    base = base + price_p * np.log(self.price)
                if share_p:
                    base = base - share_p * np.log(self.shares)
        return self.lift(base)

    def walk(self, node: Node) -> tuple[Node, _Kind]:
        if isinstance(node, Field_):
            return node, _field_kind(node.name)
        if isinstance(node, Const):
            return node, FREE
        if isinstance(node, Unary):
            inner, kind = self.walk(node.operand)
            if kind.tag == "log":
                kind = _log(-kind.p, -kind.s)
            return Unary(node.op, inner), kind
        if isinstance(node, BinOp):
            left, a = self.walk(node.left)
            right, b = self.walk(node.right)
            kind = _binop_kind(node.op, a, b, left, right)
            if kind is None:
                left, right, kind = self.materialize(left, a), self.materialize(right, b), DATED
            return BinOp(node.op, left, right), kind
        if isinstance(node, Call):
            return self.call(node)
        raise TypeError(f"not a Node: {node!r}")

    def call(self, node: Call) -> tuple[Node, _Kind]:
        arg, a = self.walk(node.arg)
        if node.func in BINARY_WINDOW_FUNCS:
            arg2, b = self.walk(node.arg2)
            return Call(node.func, arg, node.window, arg2), _pair_kind(node.func, a, b)
        if node.func in BINARY_PARAMETER_FUNCS:
            arg2, b = self.walk(node.arg2)
            exponent = _const(node.arg2)
            if a.tag == "hom" and exponent is not None:
                e = Fraction(exponent)
                return Call(node.func, arg, node.window, arg2), _hom(a.p * e, a.s * e)
            if a.tag != "dated" and a.free and b.free:
                return Call(node.func, arg, node.window, arg2), FREE
            arg, arg2 = self.materialize(arg, a), self.materialize(arg2, b)
            return Call(node.func, arg, node.window, arg2), DATED
        if FUNCS.get(node.func):
            return Call(node.func, arg, node.window, None), _windowed_kind(
                node.func, node.window, a
            )
        kind = _plain_kind(node.func, a)
        if kind is None:
            arg, kind = self.materialize(arg, a), DATED
        return Call(node.func, arg, node.window, None), kind


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
        inner = max(a, b)
        if node.window is None:
            return inner
        return inner + int(node.window)
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
    """The exact rule, one block of constant factors at a time."""
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
    without split information, evaluated as is. `cross_sectional_universe` (dates
    x permno, bool) limits every cross-sectional operator to that date's members,
    so a stock that joins later never enters an earlier date's rank or scale."""
    kwargs = {"strict_domains": strict_domains}
    if cross_sectional_universe is not None:
        kwargs["cross_sectional_universe"] = cross_sectional_universe
    if factors is None or not (features_used(node) & set(LEVEL_FIELDS)):
        return interpret.evaluate(node, fields, **kwargs)
    base = consistent_fields(fields, factors)
    price, shares = anchors(factors)
    rewriter = _Rewriter(base, price, shares, fields, kwargs)
    try:
        rewritten, kind = rewriter.walk(node)
    except _NeedsBlocks:
        return _evaluate_blocks(node, base, price, shares, **kwargs)
    rewritten = rewriter.materialize(rewritten, kind)
    return interpret.evaluate(rewritten, rewriter.fields, **kwargs)
