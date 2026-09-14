"""Synthetic OHLCV panels with an optional planted pattern.

The baseline has to prove two things before it is pointed at CRSP: that it can
find a pattern that is really there, and that it says "nothing here" when
nothing is. This generator produces both panels from the same machinery, so
the only difference between them is the planted effect.

Return model, per asset i and day t::

    r_it = beta_i * m_t + gamma_i * s_sector(i),t + e_it

With `pattern="reversal"` the idiosyncratic part carries short-term reversal:

    e_it = idio_vol_i * (-strength * z_i,t-1 + sqrt(1 - strength^2) * u_it)

where z_i,t-1 is the cross-sectional z-score of yesterday's total return and u
is standard normal noise. The daily rank IC of `neg(returns)` against the next
day's return is then roughly `strength` (a little less, because market and
sector moves dilute the cross-section). With `pattern="none"` strength is 0 and
no field predicts future returns.

Prices, open/high/low, volume and vwap are built around those returns so the
panel looks like OHLCV data, but they carry no information beyond it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from capstone.alpha_gpt.panel import OHLCVPanel

TRADING_DAYS = 252
PATTERNS = ("none", "reversal")


def make_ohlcv_panel(
    n_dates: int = 1260,
    n_assets: int = 100,
    n_sectors: int = 10,
    *,
    pattern: Literal["none", "reversal"] = "none",
    strength: float = 0.03,
    seed: int = 0,
) -> OHLCVPanel:
    """Generate an `OHLCVPanel` (fields open/high/low/close/volume/vwap/returns).

    Args:
        n_dates: business days (1260 ~ five years).
        n_assets: cross-section width.
        n_sectors: sectors, exposed as the "sector" group.
        pattern: "none" for a pure null panel, "reversal" to plant short-term
            reversal of strength `strength`.
        strength: planted correlation in [0, 1); ignored when pattern="none".
        seed: RNG seed; same seed, same panel.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    if not 0.0 <= strength < 1.0:
        raise ValueError("strength must be in [0, 1)")
    if n_dates < 2 or n_assets < 2 or n_sectors < 1:
        raise ValueError("need n_dates >= 2, n_assets >= 2, n_sectors >= 1")
    effect = strength if pattern == "reversal" else 0.0

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_dates)
    assets = [f"S{i:03d}" for i in range(n_assets)]
    sector_of = rng.integers(0, n_sectors, size=n_assets)

    market = rng.normal(0.0003, 0.16 / np.sqrt(TRADING_DAYS), size=n_dates)
    sector_factor = rng.normal(0.0, 0.10 / np.sqrt(TRADING_DAYS), size=(n_dates, n_sectors))
    beta = rng.uniform(0.7, 1.3, size=n_assets)
    gamma = rng.uniform(0.3, 0.9, size=n_assets)
    idio_vol = rng.uniform(0.20, 0.45, size=n_assets) / np.sqrt(TRADING_DAYS)
    noise = rng.standard_normal((n_dates, n_assets))

    returns = np.empty((n_dates, n_assets))
    for t in range(n_dates):
        if t == 0 or effect == 0.0:
            shock = noise[t]
        else:
            prev = returns[t - 1]
            z = (prev - prev.mean()) / prev.std()
            shock = -effect * z + np.sqrt(1.0 - effect**2) * noise[t]
        returns[t] = beta * market[t] + gamma * sector_factor[t, sector_of] + idio_vol * shock

    base = rng.uniform(20.0, 200.0, size=n_assets)
    close = base * np.cumprod(1.0 + returns, axis=0)
    prev_close = np.vstack([base, close[:-1]])

    gap = rng.normal(0.0, 0.3, size=(n_dates, n_assets)) * idio_vol
    open_ = prev_close * np.exp(gap)
    wick_up = np.abs(rng.normal(0.0, 0.5, size=(n_dates, n_assets))) * idio_vol
    wick_down = np.abs(rng.normal(0.0, 0.5, size=(n_dates, n_assets))) * idio_vol
    high = np.maximum(open_, close) * np.exp(wick_up)
    low = np.minimum(open_, close) * np.exp(-wick_down)

    # Log volume: AR(1) around a per-asset level, lifted on large absolute moves.
    level = rng.uniform(12.0, 16.0, size=n_assets)
    log_volume = np.empty((n_dates, n_assets))
    log_volume[0] = level
    vol_noise = rng.normal(0.0, 0.25, size=(n_dates, n_assets))
    for t in range(1, n_dates):
        log_volume[t] = (
            level
            + 0.8 * (log_volume[t - 1] - level)
            + vol_noise[t]
            + 0.5 * np.abs(returns[t]) / idio_vol
        )
    volume = np.exp(log_volume)

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=dates, columns=assets)

    close_frame = frame(close)
    fields = {
        "open": frame(open_),
        "high": frame(high),
        "low": frame(low),
        "close": close_frame,
        "volume": frame(volume),
        "vwap": frame((high + low + close) / 3.0),
        # From prices, so the first day is NaN exactly as it would be on real data.
        "returns": close_frame.pct_change(),
    }
    groups = {"sector": pd.Series([f"sector_{k}" for k in sector_of], index=assets)}
    descriptor = {
        "source": "synthetic_ohlcv",
        "pattern": pattern,
        "strength": effect,
        "seed": seed,
        "n_dates": n_dates,
        "n_assets": n_assets,
        "n_sectors": n_sectors,
    }
    return OHLCVPanel(fields=fields, groups=groups, descriptor=descriptor)
