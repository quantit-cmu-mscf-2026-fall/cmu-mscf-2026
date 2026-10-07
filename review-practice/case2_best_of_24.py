"""Case 2 — "The best of 24".

The pull request under review is titled *20-day momentum: Sharpe 1.87 on the
sample panel*. It adds one file that loops over lookbacks 5, 10, 15 ... 120, runs
a backtest for each, and prints the best one. There is no call to `log_run`. The
description reports the 1.87 and nothing else.

Write your two lines before running this file.

Run:  python review-practice/case2_best_of_24.py
"""

from __future__ import annotations

import numpy as np

from capstone.backtest import run_backtest
from capstone.evaluate import expected_max_sharpe
from capstone.synth import make_panel

PERIODS_PER_YEAR = 252
LOOKBACKS = list(range(5, 125, 5))  # 5, 10, 15 ... 120 -> 24 trials


def annualised_sharpe(returns) -> float:
    r = returns.dropna()
    if r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR))


def sweep(returns) -> dict[int, float]:
    """Exactly what the pull request does: try every lookback, keep the scores."""
    scores = {}
    for lb in LOOKBACKS:
        # No .shift(1) here on purpose: run_backtest already lags the positions,
        # so shifting again would be a double lag — a bug of its own.
        signal = returns.rolling(lb).mean()
        scores[lb] = annualised_sharpe(run_backtest(signal, returns))
    return scores


def main() -> None:
    # Again a pure-noise panel: there is nothing to find, by construction.
    panel = make_panel(n_dates=2520, n_assets=100, n_candidates=1, n_real=0, seed=11)
    returns = panel.returns

    scores = sweep(returns)
    best_lb = max(scores, key=lambda k: scores[k])
    best = scores[best_lb]
    median = float(np.median(list(scores.values())))

    n_obs = len(returns)
    bar = expected_max_sharpe(n_trials=len(LOOKBACKS), n_obs=n_obs)

    print(__doc__.splitlines()[0])
    print()
    print(f"  trials run:                                {len(LOOKBACKS)}")
    print(f"  best lookback on PURE NOISE:               {best_lb} days")
    print(f"  its Sharpe:                                {best:6.2f}")
    print(f"  median Sharpe across the same 24 trials:   {median:6.2f}")
    print(f"  expected_max_sharpe({len(LOOKBACKS)} trials, {n_obs} obs):     {bar:6.2f}")
    print()
    print("  THE CHECK: this panel contains no signal at all, and the sweep still")
    print("  returns a 'best' lookback with a positive score. argmax always returns")
    print("  something. The number that decides whether a result exists is not the")
    print("  maximum — it is the maximum compared against what the null produces.")
    print()
    print("  Second question for the review: the trial count that feeds that")
    print("  comparison comes from the ledger. If nothing called log_run, what is")
    print("  the trial count six weeks from now?")


if __name__ == "__main__":
    main()
