"""Compute a factor tree on a panel: one value per stock per day.

`evaluate(node, panel)` walks a parsed tree (`capstone.factors.tree`) over a
panel of daily fields, each a dates x stocks DataFrame on one shared index and
columns. Market-wide fields (`MARKET_FIELDS`) may instead be a Series by date;
they are broadcast to every stock.

Operators follow Kakushadze (2016), "101 Formulaic Alphas", Appendix A, where
it defines them:

- `shift(x, d)`, `delta(x, d)`: x d days ago; x minus that.
- `ts_mean`, `ts_std`, `ts_min`, `ts_max`, `ts_sum`, `ts_product` over the past
  d days, today included. A window that isn't full yet is NaN, so no value
  rests on less history than its formula says.
- `ts_rank(x, d)`: today's value's rank among the past d days, as a fraction
  in (0, 1], ties averaged.
- `ts_argmax`, `ts_argmin`: on which of the past d days the max (min) fell,
  counting 1 for the oldest and d for today. Alpha101 says "which day"; this
  is the usual reading, and ties take the earliest day.
- `decay_linear(x, d)`: weighted mean over the past d days, weights d, d-1,
  ..., 1 from today back, scaled to sum to 1.
- `ts_corr`, `ts_cov(x, y, d)`: rolling correlation and covariance per stock.
- `rank(x)`: cross-sectional rank per day, as a fraction in (0, 1], ties
  averaged. `scale(x)`: rescaled per day so the absolute values sum to 1.
- `signed_power(x, a)`: sign(x) * |x| ** a.

Three are not in Alpha101 and are defined here:

- `ema(x, d)`: exponential moving average with span d (smoothing 2 / (d + 1)),
  NaN until d days have been seen.
- `zscore(x, d)`: (x - ts_mean(x, d)) / ts_std(x, d), per stock over time.
- `winsorize(x)`: each day, clipped to the cross-sectional mean plus or minus
  three standard deviations.

Every operator is causal: a value on day t uses no data after t (a test
checks every one). Division by zero, log of a non-positive number and any
other undefined result is NaN, never inf.

`evaluate(..., strict_domains=True)` is the stricter mode new programs use: an operator
never turns an undefined domain into a value. See `evaluate` for it and for the dated
`cross_sectional_universe`.

The panel's prices and shares must already be split-adjusted: a ratio of raw
closes across a split is not a return. Building the panel from the shared
data, including `mkt_return`, is the evaluator's job (QUANTIT-81).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from capstone.factors.tree import (
    BINARY_PARAMETER_FUNCS,
    BINARY_WINDOW_FUNCS,
    FUNCS,
    MARKET_FIELDS,
    BinOp,
    Call,
    Const,
    Field_,
    Node,
    Unary,
)

Panel = Mapping[str, pd.DataFrame | pd.Series]
WINSOR_SIGMAS = 3.0


class EvaluationError(ValueError):
    """The panel cannot evaluate this tree (a field is missing or misshapen)."""


def evaluate(
    node: Node,
    panel: Panel,
    *,
    strict_domains: bool = False,
    cross_sectional_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The factor's value for every stock on every day, as a dates x stocks frame.

    `strict_domains` additionally refuses to let an operator invent a value from an
    undefined domain: a constant argument is broadcast explicitly, scale normalises by
    the largest magnitude first, and zscore, winsorize and ts_corr are undefined
    wherever their moments are. `cross_sectional_universe` is a dated boolean frame on
    the panel's axes; with it, rank, scale and winsorize see only each date's members
    (and are undefined on a date whose member has no value). It needs strict mode.
    """
    template = _template(panel)
    if cross_sectional_universe is not None:
        mask = cross_sectional_universe
        if (
            not strict_domains
            or not mask.index.equals(template.index)
            or not mask.columns.equals(template.columns)
        ):
            raise EvaluationError(
                "dated cross-sectional universe requires strict mode and matching axes"
            )
        if mask.isna().any().any() or not mask.isin([True, False]).all().all():
            raise EvaluationError("dated cross-sectional membership must be explicitly known")
    out = _eval(
        node,
        panel,
        template,
        strict_domains=strict_domains,
        cross_sectional_universe=cross_sectional_universe,
    )
    if not isinstance(out, pd.DataFrame):  # a tree of constants only; parse forbids it
        out = pd.DataFrame(float(out), index=template.index, columns=template.columns)
    return out.replace([np.inf, -np.inf], np.nan)


def _template(panel: Panel) -> pd.DataFrame:
    frames = [v for v in panel.values() if isinstance(v, pd.DataFrame)]
    if not frames:
        raise EvaluationError("the panel has no per-stock field")
    template = frames[0]
    for frame in frames[1:]:
        if not (frame.index.equals(template.index) and frame.columns.equals(template.columns)):
            raise EvaluationError("panel fields must share one index and one set of columns")
    return template


def _field(name: str, panel: Panel, template: pd.DataFrame) -> pd.DataFrame:
    if name not in panel:
        raise EvaluationError(f"the panel has no field {name!r}")
    value = panel[name]
    if isinstance(value, pd.Series):
        if name not in MARKET_FIELDS:
            raise EvaluationError(f"{name!r} is a per-stock field; pass a dates x stocks frame")
        value = value.reindex(template.index)
        return pd.DataFrame(
            np.repeat(value.to_numpy(dtype=float)[:, None], template.shape[1], axis=1),
            index=template.index,
            columns=template.columns,
        )
    return value.astype(float)


def _eval(node: Node, panel: Panel, template: pd.DataFrame, **mode):
    # A parent must never rank, sign or scale an undefined child's infinity.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = _eval_node(node, panel, template, **mode)
    if isinstance(out, pd.DataFrame):
        return out.replace([np.inf, -np.inf], np.nan)
    return out if np.isfinite(out) else np.float64(np.nan)


def _eval_node(
    node: Node,
    panel: Panel,
    template: pd.DataFrame,
    *,
    strict_domains: bool = False,
    cross_sectional_universe: pd.DataFrame | None = None,
):
    mode = {
        "strict_domains": strict_domains,
        "cross_sectional_universe": cross_sectional_universe,
    }
    if isinstance(node, Field_):
        return _field(node.name, panel, template)
    if isinstance(node, Const):
        return np.float64(node.value)
    if isinstance(node, Unary):
        return -_eval(node.operand, panel, template, **mode)
    if isinstance(node, BinOp):
        left = _eval(node.left, panel, template, **mode)
        right = _eval(node.right, panel, template, **mode)
        with np.errstate(divide="ignore", invalid="ignore"):
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            return left / right
    if isinstance(node, Call):
        x = _eval(node.arg, panel, template, **mode)
        if strict_domains and np.isscalar(x) and node.func not in BINARY_PARAMETER_FUNCS:
            x = pd.DataFrame(x, index=template.index, columns=template.columns)
        if node.func in BINARY_WINDOW_FUNCS:
            y = _eval(node.arg2, panel, template, **mode)
            if strict_domains and np.isscalar(y):
                y = pd.DataFrame(y, index=template.index, columns=template.columns)
            return _pair(node.func, x, y, node.window, strict_domains=strict_domains)
        if node.func in BINARY_PARAMETER_FUNCS:
            exponent = _eval(node.arg2, panel, template, **mode)
            return np.sign(x) * np.abs(x) ** exponent
        if FUNCS.get(node.func):
            return _windowed(node.func, x, node.window, strict_domains=strict_domains)
        return _plain(node.func, x, **mode)
    raise TypeError(f"not a Node: {node!r}")


def _windowed(func: str, x: pd.DataFrame, d: int, *, strict_domains=False) -> pd.DataFrame:
    if func == "shift":
        return x.shift(d)
    if func == "delta":
        return x - x.shift(d)
    if func == "ema":
        return x.ewm(span=d, adjust=False, min_periods=d).mean()
    roll = x.rolling(d, min_periods=d)
    if func == "ts_mean":
        return roll.mean()
    if func == "ts_std":
        return roll.std()
    if func == "ts_min":
        return roll.min()
    if func == "ts_max":
        return roll.max()
    if func == "ts_sum":
        return roll.sum()
    if func == "ts_product":
        return roll.apply(np.prod, raw=True)
    if func == "ts_rank":
        return roll.rank(pct=True)
    if func == "ts_argmax":
        return roll.apply(lambda w: np.argmax(w) + 1.0, raw=True)
    if func == "ts_argmin":
        return roll.apply(lambda w: np.argmin(w) + 1.0, raw=True)
    if func == "decay_linear":
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()
        return roll.apply(lambda w: float(np.dot(w, weights)), raw=True)
    if func == "zscore":
        with np.errstate(divide="ignore", invalid="ignore"):
            mean, std = roll.mean(), roll.std()
            result = (x - mean) / std
            return result.where(np.isfinite(mean) & np.isfinite(std)) if strict_domains else result
    raise EvaluationError(f"no implementation for {func}")


def _plain(
    func: str, x: pd.DataFrame, *, strict_domains=False, cross_sectional_universe=None
) -> pd.DataFrame:
    if func in {"rank", "scale", "winsorize"} and cross_sectional_universe is not None:
        mask = cross_sectional_universe.astype(bool)
        x = x.where(mask)
        complete = (x.notna() | ~mask).all(axis=1)
        x = x.where(complete, axis=0)
    if func == "rank":
        return x.rank(axis=1, pct=True)
    if func == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(x.where(x > 0))
    if func == "abs":
        return x.abs()
    if func == "sign":
        return np.sign(x)
    if func == "scale":
        if strict_domains:
            largest = x.abs().max(axis=1)
            scaled = x.div(largest.where(largest > 0), axis=0)
            total = scaled.abs().sum(axis=1)
            return scaled.div(total.where(total > 0), axis=0)
        total = x.abs().sum(axis=1)
        return x.div(total.where(total > 0), axis=0)
    if func == "winsorize":
        mean, std = x.mean(axis=1), x.std(axis=1)
        result = x.clip(mean - WINSOR_SIGMAS * std, mean + WINSOR_SIGMAS * std, axis=0)
        return (
            result.where(np.isfinite(mean) & np.isfinite(std), axis=0) if strict_domains else result
        )
    raise EvaluationError(f"no implementation for {func}")


def _pair(
    func: str, x: pd.DataFrame, y: pd.DataFrame, d: int, *, strict_domains=False
) -> pd.DataFrame:
    roll = x.rolling(d, min_periods=d)
    if strict_domains and func == "ts_corr":
        cov, var_x = roll.cov(y), roll.var()
        var_y = y.rolling(d, min_periods=d).var()
        valid = (
            np.isfinite(cov) & np.isfinite(var_x) & np.isfinite(var_y) & (var_x > 0) & (var_y > 0)
        )
        result = cov / np.sqrt(var_x) / np.sqrt(var_y)
        return result.where(valid & (result.abs() <= 1 + 1e-10))
    return roll.corr(y) if func == "ts_corr" else roll.cov(y)
