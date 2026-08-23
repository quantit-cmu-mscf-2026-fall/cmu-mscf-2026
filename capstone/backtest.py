"""Cross-sectional backtest: weights, returns, and summary statistics.

The convention throughout this kit (see `synth.py`): a signal observed at date
`t` targets the return realised at `t+1`. `run_backtest` enforces this with an
explicit `.shift(1)` on positions — remove it and every backtest becomes
look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from capstone.synth import SyntheticPanel

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


def to_weights(signal: pd.DataFrame, *, demean: bool = True, gross: float = 1.0) -> pd.DataFrame:
    """Convert a raw signal panel into cross-sectional portfolio weights.

    Args:
        signal: dates x assets raw signal values.
        demean: if True, subtract each row's cross-sectional mean first,
            producing a dollar-neutral book.
        gross: target sum of |weight| per date.

    Returns:
        DataFrame of the same shape/index/columns as `signal`. A row that is
        all-zero or all-NaN (post-demean) is returned as all-zero rather than
        divided by zero.
    """
    values = signal.sub(signal.mean(axis=1), axis=0) if demean else signal.copy()
    values = values.fillna(0.0)

    abs_sum = values.abs().sum(axis=1)
    scale = pd.Series(0.0, index=abs_sum.index)
    nonzero = abs_sum > 0
    scale.loc[nonzero] = gross / abs_sum.loc[nonzero]

    return values.mul(scale, axis=0)


def _backtest_components(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: float,
    demean: bool,
    gross: float,
) -> tuple[pd.Series, pd.Series]:
    """Shared core of `run_backtest` and `sweep`.

    Aligns signal/returns on their common index and columns, shifts positions
    one period so `signal.loc[t]` earns `returns.loc[t+1]`, and returns both
    the net strategy return series and the per-period turnover series (the
    latter is not part of `run_backtest`'s public return value, but `sweep`
    needs it to populate `BacktestSummary.turnover`).
    """
    dates = signal.index.intersection(returns.index)
    assets = signal.columns.intersection(returns.columns)
    signal = signal.loc[dates, assets]
    returns = returns.loc[dates, assets]

    weights = to_weights(signal, demean=demean, gross=gross)
    positions = weights.shift(1)

    turnover = positions.diff().abs().sum(axis=1)
    gross_returns = (positions * returns).sum(axis=1)
    costs = cost_bps / 10000.0 * turnover
    net_returns = gross_returns - costs
    return net_returns, turnover


def run_backtest(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
    demean: bool = True,
    gross: float = 1.0,
) -> pd.Series:
    """Run a cross-sectional backtest from a signal panel.

    Args:
        signal: dates x assets raw signal values.
        returns: dates x assets simple returns, same convention as `signal`.
        cost_bps: proportional transaction cost per unit of turnover, in
            basis points, charged as `cost_bps / 10000 * turnover`.
        demean: passed through to `to_weights`.
        gross: passed through to `to_weights`.

    Returns:
        Per-period strategy return series, net of costs. Positions are
        `to_weights(signal).shift(1)`, so the first observation is always
        NaN (no prior signal to trade on).
    """
    net_returns, _turnover = _backtest_components(
        signal, returns, cost_bps=cost_bps, demean=demean, gross=gross
    )
    return net_returns


@dataclass
class BacktestSummary:
    """Summary statistics for a backtested return series.

    Attributes:
        n_obs: number of non-NaN periods used.
        mean: annualised mean return.
        vol: annualised return volatility.
        sharpe: annualised mean / annualised vol; NaN if vol is 0.
        max_drawdown: peak-to-trough drawdown on cumulative returns, <= 0.
        hit_rate: share of strictly positive periods among non-NaN periods.
        turnover: mean per-period turnover, if supplied by the caller.
    """

    n_obs: int
    mean: float
    vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float

    def __repr__(self) -> str:
        return (
            f"BacktestSummary(n_obs={self.n_obs}, sharpe={self.sharpe:.2f}, "
            f"mean={self.mean:.2%}, vol={self.vol:.2%}, "
            f"max_drawdown={self.max_drawdown:.2%}, hit_rate={self.hit_rate:.2%}, "
            f"turnover={self.turnover:.3f})"
        )

    def as_dict(self) -> dict:
        """Return the summary as a plain dict, suitable for a DataFrame row."""
        return {
            "n_obs": self.n_obs,
            "mean": self.mean,
            "vol": self.vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
        }


def summarize(
    strategy_returns: pd.Series,
    *,
    freq: str = "daily",
    turnover: float = float("nan"),
) -> BacktestSummary:
    """Summarize a per-period strategy return series.

    Args:
        strategy_returns: per-period simple returns, as produced by
            `run_backtest`.
        freq: one of `PERIODS_PER_YEAR`, used to annualise mean and vol.
        turnover: mean turnover to record on the result; not computed here
            because this function only sees returns, not positions.

    Returns:
        A `BacktestSummary`.

    Raises:
        ValueError: if fewer than 60 non-NaN observations are present. A
        24-day window annualised at 252 is not wrong so much as meaningless
        — short windows must not be annualised.
    """
    if freq not in PERIODS_PER_YEAR:
        raise KeyError(f"unknown freq {freq!r}; options: {sorted(PERIODS_PER_YEAR)}")

    clean = strategy_returns.dropna()
    n_obs = len(clean)
    if n_obs < 60:
        raise ValueError(
            f"only {n_obs} observations; short windows must not be annualised "
            "(need at least 60) — the annualised numbers would be meaningless, "
            "not just imprecise"
        )

    periods = PERIODS_PER_YEAR[freq]
    mean = float(clean.mean() * periods)
    vol = float(clean.std() * np.sqrt(periods))
    sharpe = mean / vol if vol != 0 else float("nan")

    cumulative = (1.0 + clean).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = min(float(drawdown.min()), 0.0)

    hit_rate = float((clean > 0).mean())

    return BacktestSummary(
        n_obs=n_obs,
        mean=mean,
        vol=vol,
        sharpe=float(sharpe),
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        turnover=turnover,
    )


def sweep(panel: "SyntheticPanel", *, cost_bps: float = 0.0, freq: str = "daily") -> pd.DataFrame:
    """Backtest every candidate in a `SyntheticPanel` and summarize each.

    Args:
        panel: a `capstone.synth.SyntheticPanel`.
        cost_bps: transaction cost passed to the underlying backtest.
        freq: annualisation frequency passed to `summarize`.

    Returns:
        DataFrame indexed by candidate name, with the columns of
        `BacktestSummary.as_dict()` plus a bool `is_real` column taken from
        `panel.truth`.
    """
    from capstone.synth import candidate_frames  # lazy import: avoids a circular import

    rows = {}
    for name, signal, is_real in candidate_frames(panel):
        net_returns, turnover = _backtest_components(
            signal, panel.returns, cost_bps=cost_bps, demean=True, gross=1.0
        )
        summary = summarize(net_returns, freq=freq, turnover=float(turnover.mean()))
        row = summary.as_dict()
        row["is_real"] = is_real
        rows[name] = row

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "candidate"
    return result
