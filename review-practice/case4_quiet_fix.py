"""Case 4 — "The quiet fix".

The pull request under review is titled *Handle missing prices*. The diff:

    -    returns = prices.pct_change()
    +    try:
    +        returns = prices.pct_change().fillna(0.0)
    +    except Exception:
    +        returns = pd.DataFrame(index=prices.index, columns=prices.columns)

The description says: "Some tickers have gaps, this stops the backtest from
crashing."

Write your two lines before running this file.

Run:  python review-practice/case4_quiet_fix.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(4)


def make_prices_with_gaps(n_dates=500, n_assets=20, gap_rate=0.03):
    """A price panel where some observations are genuinely missing."""
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    steps = RNG.standard_normal((n_dates, n_assets)) * 0.01
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=assets)
    mask = RNG.random(prices.shape) < gap_rate
    return prices.mask(mask)


def before(prices):
    """What the code did before the diff."""
    # fill_method=None so a gap stays a gap. The pandas default pads the price
    # forward first, which is itself a quiet fill — exactly the class of thing
    # this case is about.
    return prices.pct_change(fill_method=None)


def after(prices):
    """What the diff does."""
    try:
        return prices.pct_change(fill_method=None).fillna(0.0)
    except Exception:
        return pd.DataFrame(index=prices.index, columns=prices.columns)


def main() -> None:
    prices = make_prices_with_gaps()
    n_cells = prices.size

    r_before = before(prices)
    r_after = after(prices)

    missing_before = int(r_before.isna().sum().sum())
    zeros_after = int((r_after == 0.0).sum().sum())

    print(__doc__.splitlines()[0])
    print()
    shape = f"{prices.shape[0]} dates x {prices.shape[1]} assets"
    print(f"  panel:                                  {shape}  ({n_cells} cells)")
    print(f"  genuinely missing returns, before:      {missing_before}")
    print(f"  cells now holding exactly 0.0, after:   {zeros_after}")
    print(f"  share of the panel that is invented:    {zeros_after / n_cells:.1%}")
    print()
    print("  THE CHECK: count what the change created. fillna(0.0) does not handle")
    print("  a missing price. It asserts the asset was flat that day, which is a")
    print("  tradeable observation the backtest will happily act on.")
    print()
    print("  The second defect has no output at all: `except Exception` turns any")
    print("  real error into an empty frame that propagates silently. If you cannot")
    print("  say in one sentence which exception you are catching and why, the")
    print("  handler is hiding a bug rather than handling one.")


if __name__ == "__main__":
    main()
