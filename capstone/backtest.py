"""Cross-sectional backtest: weights, returns, and summary statistics.

The convention throughout this kit (see `synth.py`): a signal observed at date
`t` targets the return realised at `t+1`. `run_backtest` enforces this with an
explicit `.shift(1)` on positions — remove it and every backtest becomes
look-ahead.
"""

from __future__ import annotations

import math
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
        gross: target sum of |weight| per date. With `demean=True`, a nonzero row is dollar-neutral,
            so `gross=1.0` corresponds to 0.5 long and 0.5 short exposure.

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


def _check_band(band: float) -> None:
    if not (math.isfinite(band) and band >= 0):
        raise ValueError(f"band must be a finite number >= 0, got {band!r}")


def band_book(
    weights: pd.DataFrame,
    missing: pd.DataFrame,
    *,
    band: float,
    gross: float,
    hold: int = 1,
) -> pd.DataFrame:
    """The book held after each date's close under a no-trade band.

    `weights` is the target book on each date (`to_weights`), `missing` marks
    the names with no signal that date. Starting flat, on every `hold`-th
    date counted from the first (a rebalance):

    - a name trades to its target only when |target - held| > `band`, or when
      it has no signal (it is closed, whatever its weight); otherwise it keeps
      its held weight;
    - the book is then rescaled to `gross` (the kept weights move by that
      factor too, and pay for it);
    - a target with no position at all (no signal anywhere) is taken as is:
      the book goes flat.

    Between rebalances the book is held. `band` is in units of weight at
    `gross`. The book is not exactly dollar-neutral when weights are kept:
    each kept name is off its target by at most `band` before the rescale.
    """
    _check_band(band)
    if hold < 1:
        raise ValueError("hold must be at least 1 period")
    target = weights.to_numpy(dtype="float64")
    closed = missing.reindex_like(weights).fillna(True).to_numpy(dtype=bool)
    out = np.empty_like(target)
    held = np.zeros(target.shape[1])
    for t in range(len(target)):
        if t % hold == 0:
            row = target[t]
            if np.abs(row).sum() > 0:
                move = (np.abs(row - held) > band) | closed[t]
                held = np.where(move, row, held)
                total = np.abs(held).sum()
                held = held * (gross / total) if total > 0 else held
            else:
                held = row.copy()
        out[t] = held
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


def backtest_frame(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: float | pd.DataFrame,
    demean: bool,
    gross: float,
    hold: int = 1,
    band: float = 0.0,
) -> pd.DataFrame:
    """Per-period net and gross returns, cost and turnover for one signal.

    Aligns signal/returns on their common index and columns, shifts positions
    one period so `signal.loc[t]` earns `returns.loc[t+1]`.

    `cost_bps` is the cost of trading one unit of weight, in basis points:
    one rate for every asset and date, or a dates x assets table (a stock's
    half quoted spread that day, say). A trade made at date t's close pays
    date t's rate. An asset with no rate that day pays that day's median. A
    trade on a date where no asset has a rate raises `ValueError` rather than
    trading for free.

    `hold` rebalances every `hold`-th period, counted from the first date,
    and holds the book in between; `hold=1` rebalances every period. Between
    rebalances the weights are held fixed, so the trades that would keep them
    fixed as prices drift are not charged.

    `band` > 0 is a no-trade band on each rebalance (`band_book`: a name
    trades only when its target is more than `band` from its held weight).
    `band=0` is the backtest without one, computed exactly as before.

    Returns a frame with columns `net`, `gross`, `cost` and `turnover`.
    """
    if hold < 1:
        raise ValueError("hold must be at least 1 period")
    _check_band(band)
    dates = signal.index.intersection(returns.index)
    assets = signal.columns.intersection(returns.columns)
    signal = signal.loc[dates, assets]
    returns = returns.loc[dates, assets]

    weights = to_weights(signal, demean=demean, gross=gross)
    no_signal = signal.isna().all(axis=1)
    if hold > 1:
        rebalance = pd.Series(np.arange(len(dates)) % hold == 0, index=dates)
        no_signal = no_signal.astype(float).where(rebalance).ffill().astype(bool)
    if band > 0:
        weights = band_book(weights, signal.isna(), band=band, gross=gross, hold=hold)
    elif hold > 1:
        weights = weights.where(rebalance, axis=0).ffill()
    positions = weights.shift(1)

    # Diff against a flat book, not against the NaN first row: otherwise the
    # turnover of building the first position is NaN and `.sum` counts it as 0.
    trades = positions.fillna(0.0).diff().abs()
    turnover = trades.sum(axis=1)
    gross_returns = (positions * returns).sum(axis=1)
    if isinstance(cost_bps, pd.DataFrame):
        rate = cost_bps.reindex(index=dates, columns=assets) / 10000.0
        rate = rate.where(rate.notna(), rate.median(axis=1), axis=0)
        # positions[t] - positions[t-1] was traded at date t-1's close.
        charged = rate.shift(1)
        # A date with no rate for any asset has no median either. `.sum` would
        # read its trades as free, overstating net returns without a word, so
        # refuse instead: a missing date, a table that ends early or an index
        # of a different type (strings, not Timestamps) all land here.
        unpriced = (trades > 0) & charged.isna()
        if unpriced.to_numpy().any():
            when = unpriced.any(axis=1)
            first = when.idxmax()
            raise ValueError(
                f"cost_bps has no rate for trades on {int(when.sum())} date(s), the first "
                f"made at the close before {first}: no asset has a rate that day, so there "
                "is no median to fall back on. Fill the table (e.g. carry the last rate "
                "forward) or check that its index matches the returns' dates."
            )
        costs = (trades * charged).sum(axis=1)
    else:
        costs = cost_bps / 10000.0 * turnover
    net_returns = gross_returns - costs
    # A date whose position comes from no signal at all, and that trades
    # nothing, is not a period with a return: NaN, rather than a zero that
    # `summarize` would count as a flat day (#25). That covers the first date
    # and any warm-up or gap where the signal row is entirely NaN. Idleness is
    # decided from the signal, not the weights: a real signal with zero net
    # weight (e.g. a timing overlay gone flat) is a genuine 0.0 day. The shift
    # matches `positions`: the date-t position comes from the date t-1 signal.
    # A date that only closes the book is kept, since its cost is real.
    idle = no_signal.shift(1, fill_value=True) & (turnover == 0)
    frame = pd.DataFrame(
        {"net": net_returns, "gross": gross_returns, "cost": costs, "turnover": turnover}
    )
    frame.loc[idle] = np.nan
    return frame


def backtest_components(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: float | pd.DataFrame,
    demean: bool,
    gross: float,
    hold: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Net returns and turnover for one signal: the core of `run_backtest`, `sweep`
    and `candidate_returns` (`backtest_frame`'s `net` and `turnover`)."""
    frame = backtest_frame(
        signal, returns, cost_bps=cost_bps, demean=demean, gross=gross, hold=hold
    )
    return frame["net"], frame["turnover"]


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
        NaN (no prior signal to trade on). So is every date whose prior
        signal row is entirely NaN and that trades nothing, such as a
        lookback's warm-up; a date that only closes the book carries its
        cost. A real signal with zero net weight is a 0.0 day, not NaN.
    """
    net_returns, _turnover = backtest_components(
        signal, returns, cost_bps=cost_bps, demean=demean, gross=gross
    )
    return net_returns


def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Annualised Sharpe ratio of a per-period return series.

    The mean over the sample standard deviation (ddof=1) of the non-NaN
    periods, times sqrt(periods_per_year). NaN when there is no variance or
    fewer than two periods. `summarize` uses it; anything else that reports a
    backtest's Sharpe ratio should too, so the number has one definition.
    """
    clean = returns.dropna()
    sd = float(clean.std(ddof=1))
    if not sd > 0:
        return float("nan")
    # Annualise the mean and the sd separately, as `summarize` reports them.
    return float(clean.mean() * periods_per_year / (sd * np.sqrt(periods_per_year)))


@dataclass
class BacktestSummary:
    """Summary statistics for a backtested return series.

    Attributes:
        n_obs: number of non-NaN periods used.
        mean: annualised mean return.
        vol: annualised return volatility.
        sharpe: `sharpe_ratio`, the annualised mean over the annualised vol
            (sample sd, ddof=1); NaN if vol is 0.
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

    cumulative = (1.0 + clean).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = min(float(drawdown.min()), 0.0)

    hit_rate = float((clean > 0).mean())

    return BacktestSummary(
        n_obs=n_obs,
        mean=mean,
        vol=vol,
        sharpe=sharpe_ratio(clean, periods),
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        turnover=turnover,
    )


def sweep(panel: SyntheticPanel, *, cost_bps: float = 0.0, freq: str = "daily") -> pd.DataFrame:
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
        net_returns, turnover = backtest_components(
            signal, panel.returns, cost_bps=cost_bps, demean=True, gross=1.0
        )
        summary = summarize(net_returns, freq=freq, turnover=float(turnover.mean()))
        row = summary.as_dict()
        row["is_real"] = is_real
        rows[name] = row

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "candidate"
    return result


def candidate_returns(panel: SyntheticPanel, *, cost_bps: float = 0.0) -> pd.DataFrame:
    """Per-period net returns of every candidate in a `SyntheticPanel`, side by side.

    `sweep` reduces each candidate to one summary row; the validation methods
    need the series themselves (bootstrap, CSCV, robust Sharpe inference all
    resample or split time). Each column is exactly what `run_backtest` returns
    for that candidate, with the same defaults as `sweep`.

    Args:
        panel: a `capstone.synth.SyntheticPanel`.
        cost_bps: transaction cost passed to the underlying backtest.

    Returns:
        dates x candidates DataFrame, columns in `panel.truth` order. The first
        date is dropped: no candidate holds a position on it, so it is not a
        period anyone traded, and keeping it would add a zero-return observation
        to every column.
    """
    from capstone.synth import candidate_frames  # lazy import: avoids a circular import

    columns = {}
    for name, signal, _is_real in candidate_frames(panel):
        net_returns, _turnover = backtest_components(
            signal, panel.returns, cost_bps=cost_bps, demean=True, gross=1.0
        )
        columns[name] = net_returns

    result = pd.DataFrame(columns).iloc[1:]
    result.columns.name = "candidate"
    return result
