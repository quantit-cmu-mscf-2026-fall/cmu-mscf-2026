"""The operator library alpha expressions are built from.

A starter subset of the paper's operator families (time-series, cross-sectional,
group-wise, element-wise). Every operator is causal by construction: time-series
operators use trailing windows or positive shifts only, cross-sectional and
group-wise operators use only the current row. `tests/alpha_gpt/test_ag_operators.py`
checks that for every entry in `OPERATORS`, so adding an operator that peeks at
the future fails the suite rather than a backtest.

Operators receive already-evaluated arguments: DataFrames (date x asset) for
series, numbers for literal values, ints for windows/lags, and an asset -> label
Series for groups. `OpSpec.__call__` turns +/-inf into NaN on the way out, so no
infinity ever reaches metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

#: series: a field or nested call. value: series or numeric literal.
#: window: int literal >= 2. lag: int literal >= 1. group: a group name.
ArgKind = Literal["series", "value", "window", "lag", "group"]

_EPS = 1e-12

Operand = pd.DataFrame | float | int


@dataclass(frozen=True)
class OpSpec:
    """One operator: its name, argument kinds, implementation and prompt doc."""

    name: str
    args: tuple[ArgKind, ...]
    fn: Callable[..., pd.DataFrame]
    doc: str
    commutative: bool = False

    @property
    def signature(self) -> str:
        names = {"series": "x", "value": "x", "window": "d", "lag": "n", "group": "group"}
        labels = [names[kind] for kind in self.args]
        if labels.count("x") == 2:
            labels[labels.index("x", 1)] = "y"
        return f"{self.name}({', '.join(labels)})"

    def __call__(self, *args) -> pd.DataFrame:
        result = self.fn(*args)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"{self.name} did not produce a DataFrame (no series argument?)")
        return result.replace([np.inf, -np.inf], np.nan)


# --- time-series ---------------------------------------------------------------


def _rolling(x: pd.DataFrame, d: int):
    return x.rolling(d, min_periods=d)


def _ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    std = _rolling(x, d).std()
    return (x - _rolling(x, d).mean()) / std.where(std > _EPS)


def _ts_decayed_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    values = x.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) >= d:
        # windows[t, i] holds rows t .. t+d-1 of column i, oldest first.
        windows = sliding_window_view(values, d, axis=0)
        out[d - 1 :] = windows @ weights
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return _rolling(x, d).corr(y)


# --- cross-sectional / group-wise -----------------------------------------------


def _row_std(x: pd.DataFrame) -> pd.Series:
    std = x.std(axis=1)
    return std.where(std > _EPS)


def _zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(_row_std(x), axis=0)


def _winsorize(x: pd.DataFrame) -> pd.DataFrame:
    mean, std = x.mean(axis=1), x.std(axis=1)
    return x.clip(lower=mean - 3 * std, upper=mean + 3 * std, axis=0)


def _grouped_demean(x: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    labels = group.reindex(x.columns)
    means = x.T.groupby(labels).transform("mean").T
    return x - means


# --- element-wise ----------------------------------------------------------------


def _div(x: Operand, y: Operand) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        denominator = y.where(y.abs() > _EPS)
    else:
        denominator = y if abs(y) > _EPS else np.nan
    return x / denominator


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


def _sign(x: pd.DataFrame) -> pd.DataFrame:
    # A named function, not np.sign itself: ufuncs have no inspectable signature.
    return np.sign(x)


def _compare(x: Operand, y: Operand, op: Callable) -> pd.DataFrame:
    result = op(x, y).astype(float)
    for side in (x, y):
        if isinstance(side, pd.DataFrame):
            result = result.where(side.notna())
    return result


_SPECS = [
    # time-series
    OpSpec("shift", ("series", "lag"), lambda x, n: x.shift(n), "x as of n days ago"),
    OpSpec("ts_delta", ("series", "window"), lambda x, d: x - x.shift(d), "x minus x d days ago"),
    OpSpec(
        "ts_mean", ("series", "window"), lambda x, d: _rolling(x, d).mean(), "mean of last d days"
    ),
    OpSpec("ts_std", ("series", "window"), lambda x, d: _rolling(x, d).std(), "std of last d days"),
    OpSpec("ts_min", ("series", "window"), lambda x, d: _rolling(x, d).min(), "min of last d days"),
    OpSpec("ts_max", ("series", "window"), lambda x, d: _rolling(x, d).max(), "max of last d days"),
    OpSpec(
        "ts_rank",
        ("series", "window"),
        lambda x, d: _rolling(x, d).rank(pct=True),
        "percentile (0,1] of today's x within its last d days",
    ),
    OpSpec(
        "ts_zscore_scale",
        ("series", "window"),
        _ts_zscore,
        "(x - mean of last d days) / std of last d days",
    ),
    OpSpec(
        "ts_decayed_linear",
        ("series", "window"),
        _ts_decayed_linear,
        "linearly decaying weighted mean of last d days (today weighted most)",
    ),
    OpSpec(
        "ts_corr",
        ("series", "series", "window"),
        _ts_corr,
        "correlation of x and y over the last d days, per asset",
    ),
    # cross-sectional
    OpSpec(
        "normed_rank",
        ("series",),
        lambda x: x.rank(axis=1, pct=True),
        "percentile rank (0,1] across assets on each day",
    ),
    OpSpec("zscore_scale", ("series",), _zscore, "z-score across assets on each day"),
    OpSpec(
        "winsorize_scale",
        ("series",),
        _winsorize,
        "clip to mean +/- 3 std across assets on each day",
    ),
    # group-wise
    OpSpec(
        "grouped_demean",
        ("series", "group"),
        _grouped_demean,
        "x minus its group's mean across assets on each day",
    ),
    # element-wise
    OpSpec("add", ("value", "value"), lambda x, y: x + y, "x + y", commutative=True),
    OpSpec("minus", ("value", "value"), lambda x, y: x - y, "x - y"),
    OpSpec("cwise_mul", ("value", "value"), lambda x, y: x * y, "x * y", commutative=True),
    OpSpec("div", ("value", "value"), _div, "x / y (missing where |y| is ~0)"),
    OpSpec("neg", ("series",), lambda x: -x, "-x"),
    OpSpec("abs", ("series",), lambda x: x.abs(), "|x|"),
    OpSpec("log", ("series",), _log, "natural log (missing where x <= 0)"),
    OpSpec("sign", ("series",), _sign, "sign of x: -1, 0 or 1"),
    OpSpec("relu", ("series",), lambda x: x.clip(lower=0), "max(x, 0)"),
    OpSpec(
        "greater",
        ("value", "value"),
        lambda x, y: _compare(x, y, lambda a, b: a > b),
        "1 where x > y else 0",
    ),
    OpSpec(
        "less",
        ("value", "value"),
        lambda x, y: _compare(x, y, lambda a, b: a < b),
        "1 where x < y else 0",
    ),
]

OPERATORS: dict[str, OpSpec] = {spec.name: spec for spec in _SPECS}
