"""A six-month sample panel, so day one does not depend on a WRDS account.

Getting WRDS credentials takes as long as it takes, and nobody should be blocked
from building the pipeline meanwhile. This module hands you a panel with the same
shape as `capstone.wrds_loader.to_price_panel()` output — date index, ticker
columns — so swapping in real CRSP data later changes one line and nothing else.

The prices here are SYNTHETIC. Real market prices cannot ship in a public
repository: free feeds grant access but not redistribution rights, and the
underlying exchange data is separately licensed no matter what the aggregator's
terms say (docs/data_sources.md). The ticker symbols are real because symbols are
public facts; the numbers attached to them are not, and must never be presented
as though they were.

Being synthetic is also useful. The factor structure is known by construction,
so you can check that your pipeline recovers something you already know the
answer to before pointing it at data where you do not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

# Real symbols, synthetic prices. Spread across sectors so cross-sectional work
# has something to bite on.
SAMPLE_SECTORS: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "AMD": "Technology",
    "TXN": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "BLK": "Financials",
    "AXP": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "JNJ": "Health Care",
    "UNH": "Health Care",
    "LLY": "Health Care",
    "PFE": "Health Care",
    "ABBV": "Health Care",
    "TMO": "Health Care",
    "PG": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",
    "WMT": "Consumer",
    "COST": "Consumer",
    "MCD": "Consumer",
    "NKE": "Consumer",
    "CAT": "Industrials",
    "HON": "Industrials",
    "UNP": "Industrials",
    "GE": "Industrials",
    "BA": "Industrials",
    "NEE": "Utilities",
    "DUK": "Utilities",
}

SAMPLE_TICKERS: list[str] = list(SAMPLE_SECTORS)

FUNDAMENTAL_TAGS = ("Revenues", "NetIncomeLoss", "StockholdersEquity", "Assets")

TRADING_DAYS = 252


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def generate_sample_prices(
    *,
    start: str = "2025-01-02",
    periods: int = 126,
    seed: int = 20260825,
) -> pd.DataFrame:
    """Six months (126 business days) of daily prices for the sample universe.

    The generating model is a two-factor return process, chosen because it is the
    simplest thing that makes cross-sectional work meaningful::

        r_it = beta_i * f_market_t + gamma_i * f_sector(i),t + eps_it

    with betas in [0.7, 1.4], sector loadings in [0.3, 0.9], market volatility
    around 16% annualised and idiosyncratic volatility around 28%. Prices are the
    compounded path from a per-ticker base level.

    Because a market factor and sector factors are present, names co-move and
    same-sector names co-move more — so a long-short cross-sectional signal
    behaves the way one does on real data, and a naive equal-weight backtest
    inherits real market exposure. Deterministic given `seed`: same seed, same
    panel, on every machine.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    tickers = SAMPLE_TICKERS
    sectors = sorted(set(SAMPLE_SECTORS.values()))

    market_vol = 0.16 / np.sqrt(TRADING_DAYS)
    sector_vol = 0.10 / np.sqrt(TRADING_DAYS)
    idio_vol = 0.28 / np.sqrt(TRADING_DAYS)

    market = rng.normal(0.0004, market_vol, size=periods)
    sector_paths = {s: rng.normal(0.0, sector_vol, size=periods) for s in sectors}

    betas = rng.uniform(0.7, 1.4, size=len(tickers))
    gammas = rng.uniform(0.3, 0.9, size=len(tickers))
    bases = rng.uniform(20.0, 600.0, size=len(tickers))

    columns = {}
    for i, ticker in enumerate(tickers):
        sector = SAMPLE_SECTORS[ticker]
        returns = (
            betas[i] * market
            + gammas[i] * sector_paths[sector]
            + rng.normal(0.0, idio_vol, size=periods)
        )
        columns[ticker] = bases[i] * np.cumprod(1.0 + returns)

    return pd.DataFrame(columns, index=dates)


def generate_sample_fundamentals(
    prices: pd.DataFrame | None = None,
    *,
    seed: int = 20260825,
) -> pd.DataFrame:
    """Quarterly fundamentals for the sample universe, with a realistic filing lag.

    Returns a tidy frame: [ticker, period_end, filed, tag, value].

    The point of this function is the gap between `period_end` and `filed`.
    Companies report 30-75 days after the quarter closes, so on the day a quarter
    ends, its numbers do not exist yet for anyone. Joining a signal on
    `period_end` therefore trades on information nobody had — which produces a
    beautiful backtest and a false one. Join on `filed`.

    The values are scaled off the price level so that ratios (earnings yield,
    book-to-price) land in plausible ranges rather than nonsense.
    """
    if prices is None:
        prices = generate_sample_prices(seed=seed)

    rng = np.random.default_rng(seed + 1)
    window_start, window_end = prices.index[0], prices.index[-1]

    # Two quarter-ends inside the price window.
    quarter_ends = pd.date_range(window_start, window_end, freq="QE")
    if len(quarter_ends) < 2:
        quarter_ends = pd.DatetimeIndex(
            [window_start + pd.Timedelta(days=45), window_start + pd.Timedelta(days=135)]
        )

    rows = []
    for ticker in SAMPLE_TICKERS:
        level = float(prices[ticker].iloc[0])
        shares = rng.uniform(0.5, 8.0) * 1e9
        market_cap = level * shares
        for period_end in quarter_ends[:2]:
            lag_days = int(rng.integers(30, 76))
            filed = period_end + pd.Timedelta(days=lag_days)
            revenue = market_cap * rng.uniform(0.05, 0.30)
            values = {
                "Revenues": revenue,
                "NetIncomeLoss": revenue * rng.uniform(0.02, 0.25),
                "StockholdersEquity": market_cap * rng.uniform(0.15, 0.80),
                "Assets": market_cap * rng.uniform(0.5, 3.0),
            }
            for tag in FUNDAMENTAL_TAGS:
                rows.append(
                    {
                        "ticker": ticker,
                        "period_end": period_end,
                        "filed": filed,
                        "tag": tag,
                        "value": values[tag],
                    }
                )

    return pd.DataFrame(rows).sort_values(["ticker", "period_end", "tag"]).reset_index(drop=True)


def load_sample_prices() -> pd.DataFrame:
    """Cached `generate_sample_prices`. The cache is a speed-up, not a source of truth."""
    path = _cache_path("sample_prices")
    if path.exists():
        return pd.read_parquet(path)
    frame = generate_sample_prices()
    frame.to_parquet(path)
    return frame


def load_sample_fundamentals() -> pd.DataFrame:
    """Cached `generate_sample_fundamentals`, aligned to the cached price panel."""
    path = _cache_path("sample_fundamentals")
    if path.exists():
        return pd.read_parquet(path)
    frame = generate_sample_fundamentals(load_sample_prices())
    frame.to_parquet(path)
    return frame
