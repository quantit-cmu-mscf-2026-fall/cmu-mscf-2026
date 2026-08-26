from __future__ import annotations

import pandas as pd

from capstone import pit


def momentum_signal(
    prices: pd.DataFrame,
    *,
    lookback: int = 20,
    skip: int = 1,
    t: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Return a point-in-time cross-sectional momentum signal.

    The signal is the trailing mean return over the prior `lookback` periods,
    excluding the most recent `skip` periods to avoid immediate-reversal bias. The
    returned frame keeps the full date index and column set, matching the signal
    convention used by `capstone.backtest.run_backtest`.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")

    if t is not None:
        prices = pit.available_at(
            t,
            prices,
            pd.DataFrame(columns=["filed"]),
            pd.DataFrame(index=prices.index),
        )[0]

    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    returns = prices.pct_change().fillna(0.0)
    signal = returns.rolling(window=lookback, min_periods=lookback).mean()
    if skip:
        signal = signal.shift(skip)
    return signal
