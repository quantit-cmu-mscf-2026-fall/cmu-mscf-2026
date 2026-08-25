from __future__ import annotations

import pandas as pd


def available_at(
    t: pd.Timestamp | str,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return data available as of date `t`.

    `prices` and `factors` are sliced to dates <= t. `fundamentals` is filtered to
    rows whose `filed` date is on or before `t`, which matches the public-data
    convention that a firm releases numbers only after the quarter-end date.
    """
    ts = pd.Timestamp(t)

    prices_at = prices.loc[prices.index <= ts].copy()
    factors_at = factors.loc[factors.index <= ts].copy()
    fundamentals_at = fundamentals.loc[fundamentals["filed"] <= ts].copy()

    return prices_at, fundamentals_at, factors_at
