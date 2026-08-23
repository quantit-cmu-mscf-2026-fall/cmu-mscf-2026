"""Load real data and confirm access works.

Run this first: `python examples/01_hello_data.py`. It works on a cold cache
(needs network, fetches from Ken French's site) and on a warm one (reads only
from `data_cache/`).
"""

from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
import requests

from capstone.data import CACHE_DIR, FRENCH_BASE, load_french, load_industry_returns


def _describe(name: str, frame: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"  shape: {frame.shape}")
    print(f"  dates: {frame.index.min().date()} .. {frame.index.max().date()}")
    columns = list(frame.columns)
    shown = columns if len(columns) <= 8 else columns[:8] + [f"... (+{len(columns) - 8} more)"]
    print(f"  columns ({len(columns)}): {shown}")
    print("  head(3):")
    print(frame.head(3).to_string())


def main() -> None:
    print(f"cache directory: {CACHE_DIR}")

    host = urlparse(FRENCH_BASE).netloc
    try:
        factors = load_french("factors_daily")
        industries = load_industry_returns(12)
    except requests.RequestException:
        print(
            f"\nFailed to fetch data from {host}. Check network/VPN access to "
            f"that host, or pre-populate {CACHE_DIR} from a machine that has it "
            "(the cache is just parquet files named french_<dataset>.parquet)."
        )
        raise

    _describe("factors_daily", factors)
    _describe("industry12_daily", industries)

    print("\nOK")


if __name__ == "__main__":
    main()
