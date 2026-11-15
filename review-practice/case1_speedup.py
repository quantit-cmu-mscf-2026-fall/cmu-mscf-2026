"""Case 1 — "The speed-up".

The pull request under review is titled *Speed up the momentum backtest*, and the
description says:

    "Same result, avoids the overhead of the full backtest path. Sharpe went from
     0.9 to 2.4, so the old path was also dropping data somewhere."

The diff replaces the call to `run_backtest` with a hand-rolled two-liner:

    -    rets = run_backtest(signal, returns, cost_bps=5.0)
    +    w = to_weights(signal)
    +    rets = (w * returns).sum(axis=1)

Write your two lines before running this file.

Run:  python review-practice/case1_speedup.py
"""

from __future__ import annotations

import numpy as np

from capstone.backtest import run_backtest, to_weights
from capstone.synth import make_panel

PERIODS_PER_YEAR = 252


def annualised_sharpe(returns) -> float:
    """Annualised Sharpe of a return series, NaNs dropped."""
    r = returns.dropna()
    if r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR))


def guarded(signal, returns):
    """What the code did before the diff: positions come from YESTERDAY's signal."""
    return run_backtest(signal, returns)


def refactored(signal, returns):
    """What the diff does: same-day weights multiplied by same-day returns."""
    weights = to_weights(signal)
    return (weights * returns).sum(axis=1)


def main() -> None:
    # A pure-noise world: returns are independent draws, so NOTHING here can
    # predict anything. n_real=0 is the point — any edge that shows up is an
    # artefact of the code, not of the data.
    panel = make_panel(n_dates=2520, n_assets=100, n_candidates=1, n_real=0, seed=7)
    returns = panel.returns

    # The signal is 20-day momentum built FROM the returns, which is how a real
    # momentum signal is built — and why the timing guard matters. The value at
    # date t contains the return at date t.
    signal = returns.rolling(20).mean()

    before = annualised_sharpe(guarded(signal, returns))
    after = annualised_sharpe(refactored(signal, returns))

    print(__doc__.splitlines()[0])
    print()
    print(
        f"  data:        {len(returns)} dates x {returns.shape[1]} assets, "
        f"{panel.n_true} genuinely predictive signals"
    )
    print(f"  guarded      (run_backtest, positions = weights.shift(1)) : Sharpe {before:6.2f}")
    print(f"  refactored   (same-day weights x same-day returns)        : Sharpe {after:6.2f}")
    print()
    print("  THE CHECK: the returns above are independent noise, so the honest")
    print("  answer is ~0. One of these two numbers is not ~0.")
    print()
    print("  (The pull request claimed 0.9 -> 2.4. Here the same defect shows up")
    print("   far larger, because this panel is pure noise and the leak is the only")
    print("   thing left in the number. Size varies; the direction never does.)")
    print()
    print("  Ask yourself which line of the diff moved it, and what that means for")
    print("  the sentence 'so the old path was also dropping data somewhere'.")


if __name__ == "__main__":
    main()
