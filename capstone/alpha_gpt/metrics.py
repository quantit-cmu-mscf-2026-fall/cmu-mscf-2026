"""Deterministic alpha metrics. The LLM never computes these; it only reads them.

Convention (same as `capstone.backtest`): the alpha at date t is information
available at t's close, and it is scored against the return realised on t+1.
Metrics for a split are reported by signal date, so a split's numbers depend on
signals inside the split and on returns up to one day past its end — never
further. The embargo in `splits.py` keeps that one day out of the next split.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from capstone.backtest import PERIODS_PER_YEAR, backtest_components, summarize
from capstone.evaluate import sharpe_pvalue


def rank_ic_series(
    signal: pd.DataFrame, returns: pd.DataFrame, *, min_assets: int = 20
) -> pd.Series:
    """Daily Spearman correlation across assets of signal[t] with returns[t+1].

    Pairs are masked jointly before ranking, so an asset with a missing signal
    or a missing next-day return drops out of both sides. Dates with fewer than
    `min_assets` usable pairs, or with a constant signal, are NaN.
    """
    target = returns.shift(-1).reindex(index=signal.index, columns=signal.columns)
    usable = signal.notna() & target.notna()
    signal_ranks = signal.where(usable).rank(axis=1)
    target_ranks = target.where(usable).rank(axis=1)

    s = signal_ranks.sub(signal_ranks.mean(axis=1), axis=0)
    r = target_ranks.sub(target_ranks.mean(axis=1), axis=0)
    numerator = (s * r).sum(axis=1)
    denominator = np.sqrt((s**2).sum(axis=1) * (r**2).sum(axis=1))
    ic = numerator / denominator.where(denominator > 0)
    ic[usable.sum(axis=1) < min_assets] = np.nan
    return ic


@dataclass(frozen=True)
class SplitMetrics:
    """One alpha's metrics on one split.

    Attributes:
        split: "train", "valid" or "test".
        n_days: dates with a usable IC.
        ic_mean, ic_std: mean and std of the daily rank IC.
        icir: ic_mean / ic_std (daily, not annualised).
        ic_tstat: icir * sqrt(n_days). Treats daily ICs as independent.
        ic_pvalue: two-sided normal p-value of ic_tstat.
        ls_sharpe, ls_ann_return: annualised dollar-neutral long-short Sharpe
            and mean return from `capstone.backtest`, net of costs.
        ls_sharpe_pvalue: `evaluate.sharpe_pvalue` of ls_sharpe.
        turnover: mean daily turnover of the long-short book.
        coverage: share of (date, asset) cells where the alpha is defined.
    """

    split: str
    n_days: int
    ic_mean: float
    ic_std: float
    icir: float
    ic_tstat: float
    ic_pvalue: float
    ls_sharpe: float
    ls_ann_return: float
    ls_sharpe_pvalue: float
    turnover: float
    coverage: float

    def as_flat(self, prefix: str) -> dict[str, float | int | None]:
        """Numeric fields as `{prefix}_{name}`, NaN as None (ledger-safe JSON)."""
        out: dict[str, float | int | None] = {}
        for name, value in asdict(self).items():
            if name == "split":
                continue
            is_nan = isinstance(value, float) and math.isnan(value)
            out[f"{prefix}_{name}"] = None if is_nan else value
        return out


def split_metrics(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    split: str,
    cost_bps: float = 0.0,
    min_assets: int = 20,
) -> SplitMetrics:
    """Rank-IC and long-short metrics of `signal` on the signal dates `dates`."""
    nan = float("nan")

    ic = rank_ic_series(signal, returns, min_assets=min_assets).reindex(dates).dropna()
    n_days = len(ic)
    ic_mean = float(ic.mean()) if n_days else nan
    ic_std = float(ic.std()) if n_days >= 2 else nan
    if n_days >= 2 and ic_std > 0:
        icir = ic_mean / ic_std
        ic_tstat = icir * math.sqrt(n_days)
        ic_pvalue = float(2.0 * stats.norm.sf(abs(ic_tstat)))
    else:
        icir = ic_tstat = ic_pvalue = nan

    # backtest_components dates P&L by the return date: net[t+1] = w_t . r_{t+1}.
    # Shift back one day so the long-short series is indexed by signal date too.
    net, turnover = backtest_components(signal, returns, cost_bps=cost_bps, demean=True, gross=1.0)
    long_short = net.shift(-1).reindex(dates)
    try:
        summary = summarize(long_short, freq="daily")
        ls_sharpe, ls_ann_return = summary.sharpe, summary.mean
        ls_sharpe_pvalue = sharpe_pvalue(ls_sharpe, summary.n_obs, PERIODS_PER_YEAR["daily"])
    except ValueError:
        # Fewer than 60 observations: summarize refuses to annualise, and so do we.
        ls_sharpe = ls_ann_return = ls_sharpe_pvalue = nan

    coverage = float(signal.reindex(index=dates).notna().to_numpy().mean()) if len(dates) else nan

    return SplitMetrics(
        split=split,
        n_days=n_days,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=float(icir),
        ic_tstat=float(ic_tstat),
        ic_pvalue=float(ic_pvalue),
        ls_sharpe=float(ls_sharpe),
        ls_ann_return=float(ls_ann_return),
        ls_sharpe_pvalue=float(ls_sharpe_pvalue),
        turnover=float(turnover.shift(-1).reindex(dates).mean()),
        coverage=coverage,
    )
