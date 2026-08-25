from __future__ import annotations

import pandas as pd

from capstone.backtest import summarize
from capstone.evaluate import deflated_sharpe_ratio


def strategy_accepts(
    strategy_returns: pd.Series,
    n_trials: int,
    *,
    freq: str = "daily",
    threshold: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> str:
    """Return whether a strategy is strong enough to pass a null-based gate.

    A strategy passes only if its deflated Sharpe ratio clears the acceptance
    threshold for the given trial count. Otherwise the procedure reports the
    conservative result: "nothing".
    """
    summary = summarize(strategy_returns, freq=freq)
    score = deflated_sharpe_ratio(
        summary.sharpe,
        n_trials=n_trials,
        n_obs=summary.n_obs,
        skew=skew,
        kurtosis=kurtosis,
        periods_per_year=252 if freq == "daily" else 52 if freq == "weekly" else 12,
    )
    return "pass" if pd.notna(score) and score > threshold else "nothing"
