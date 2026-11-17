"""Point-in-time market regime series from `mkt_return`.

Two regime definitions, both computed from the panel's market return only, with no
parameter tuned on our data (the cutoffs are the published ones):

1. **Volatility regime.** `rv126` at date t is the sample standard deviation (ddof 1)
   of the 126 market returns t-126 .. t-1, annualised by sqrt(252): six months of
   trailing realized volatility (a common choice of window, not tuned here). `high_vol` at t is 1
   when `rv126[t]` is above the median of every earlier value `rv126[s]`, s <= t-1
   (an expanding median from the first available value), else 0.
2. **Bear and panic states** of Daniel and Moskowitz (2016, "Momentum Crashes",
   J. Financial Economics 122(2), sections 3-4): `bear` at t is 1 when the cumulative
   (compounded) market return over the past 504 trading days (24 months), t-504 ..
   t-1, is negative, else 0. Their panic state interacts the bear indicator with
   market variance: `panic_raw[t] = bear[t] * rv126[t] ** 2`, and `panic` is
   `panic_raw[t]` divided by the mean of every earlier value `panic_raw[s]`,
   s <= t-1 (an expanding mean), so it is unit-free and comparable across decades.

**Point in time.** Every value at t uses market returns dated t-1 or earlier only;
the return of day t itself is never read. Appending, changing or dropping any data
after t leaves every value at or before t unchanged (tests: one-day shift, future
perturbation, truncation).

**Warm-up.** A value is NaN until its inputs exist, never 0: `rv126` needs 126
earlier returns, `bear` 504, `high_vol` one earlier `rv126`, and `panic` an earlier
`panic_raw` whose expanding mean is positive (before the first bear day the mean is
0 and `panic` is NaN). A NaN market return inside a window makes that window's
value NaN; expanding statistics skip NaN values.

**Exact arithmetic, same answer everywhere.** The windows are summed with
`math.fsum` (exactly rounded), products are taken in date order, and the median is
of a sorted list, so the values depend only on the inputs: not on the pandas or
numpy version, the platform, or any worker count (no rolling-window kernels).

**Holdout.** A series holding a date after `DISCOVERY_END` is refused with
`HoldoutError`, through the same date check `evaluation.build_panel` applies to the
search panel; `panel_regimes` also calls `evaluation.require_discovery`.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
import pandas as pd

from capstone.factors.evaluation import Panel, _check_dates, require_discovery

RV_WINDOW = 126  # trading days of trailing realized volatility
BEAR_WINDOW = 504  # trading days (24 months) of trailing market return, Daniel-Moskowitz
ANNUALISATION = 252  # trading days per year
COLUMNS = ("rv126", "high_vol", "bear", "panic")


def _rv(returns: np.ndarray, window: int = RV_WINDOW) -> np.ndarray:
    """Annualised sample sd of returns[t-window .. t-1] at each t (NaN in warm-up)."""
    out = np.full(len(returns), np.nan)
    scale = math.sqrt(ANNUALISATION)
    for t in range(window, len(returns)):
        w = returns[t - window : t]
        if np.isnan(w).any():
            continue
        mean = math.fsum(w.tolist()) / window
        var = math.fsum(((w - mean) ** 2).tolist()) / (window - 1)
        out[t] = math.sqrt(var) * scale
    return out


def _bear(returns: np.ndarray, window: int = BEAR_WINDOW) -> np.ndarray:
    """1.0 when the compounded return over returns[t-window .. t-1] is negative, else 0.0."""
    out = np.full(len(returns), np.nan)
    for t in range(window, len(returns)):
        w = returns[t - window : t]
        if np.isnan(w).any():
            continue
        growth = 1.0
        for r in w.tolist():  # date order, so the product is the same everywhere
            growth *= 1.0 + r
        out[t] = 1.0 if growth < 1.0 else 0.0
    return out


def _above_prior_median(values: np.ndarray) -> np.ndarray:
    """1.0 when values[t] exceeds the median of the non-NaN values[s], s < t, else 0.0."""
    out = np.full(len(values), np.nan)
    prior: list[float] = []
    for t, v in enumerate(values.tolist()):
        if prior and not math.isnan(v):
            n = len(prior)
            mid = n // 2
            median = prior[mid] if n % 2 else (prior[mid - 1] + prior[mid]) / 2
            out[t] = 1.0 if v > median else 0.0
        if not math.isnan(v):
            bisect.insort(prior, v)
    return out


def _over_prior_mean(values: np.ndarray) -> np.ndarray:
    """values[t] over the mean of the non-NaN values[s], s < t; NaN unless that mean > 0."""
    out = np.full(len(values), np.nan)
    prior: list[float] = []
    for t, v in enumerate(values.tolist()):
        if prior and not math.isnan(v):
            mean = math.fsum(prior) / len(prior)
            if mean > 0:
                out[t] = v / mean
        if not math.isnan(v):
            prior.append(v)
    return out


def regime_series(mkt_return: pd.Series) -> pd.DataFrame:
    """The regime columns `COLUMNS` on `mkt_return`'s dates (see the module docstring).

    `mkt_return` is indexed by trading date in ascending order, one value per date.
    Raises `HoldoutError` for a date after `DISCOVERY_END`, before computing anything.
    """
    index = pd.DatetimeIndex(mkt_return.index)
    _check_dates(index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("mkt_return must be indexed by unique dates in ascending order")
    returns = np.asarray(mkt_return.to_numpy(dtype=float, na_value=np.nan), dtype=float)
    rv = _rv(returns)
    bear = _bear(returns)
    panic_raw = bear * rv**2
    return pd.DataFrame(
        {
            "rv126": rv,
            "high_vol": _above_prior_median(rv),
            "bear": bear,
            "panic": _over_prior_mean(panic_raw),
        },
        index=index,
        columns=list(COLUMNS),
    )


def panel_regimes(panel: Panel) -> pd.DataFrame:
    """`regime_series` of the panel's `mkt_return`; a panel past the discovery period is
    refused by `evaluation.require_discovery` first."""
    require_discovery(panel)
    return regime_series(panel.fields["mkt_return"])
