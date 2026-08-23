"""Loaders for public asset-pricing data, with an on-disk cache.

Sources are public and need no credentials. WRDS/CRSP is richer and available to
you through CMU, but is usually not republishable — see the README before making
any CRSP-derived artifact public.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Ken French ships a few CSVs whose layout differs; these are the ones the
# parser below is known to handle.
FRENCH_DATASETS = {
    "factors_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "factors_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "factors5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "industry49_daily": "49_Industry_Portfolios_daily_CSV.zip",
    "industry12_daily": "12_Industry_Portfolios_daily_CSV.zip",
}


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def _download_french_csv(filename: str) -> str:
    """Fetch a Ken French zip and return the CSV text inside it."""
    url = f"{FRENCH_BASE}/{filename}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    member = archive.namelist()[0]
    # The files carry non-UTF-8 bytes in the header notes.
    return archive.read(member).decode("latin-1")


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first data block of a Ken French CSV.

    The files open with several lines of prose, then a header row whose first
    field is empty, then dated rows. Many files contain a SECOND block further
    down (annual data, or equal-weighted variants) separated by a blank line —
    reading the whole file naively silently concatenates them. We stop at the
    first blank line after data begins.
    """
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith(","):
            start = i
            break
    if start is None:
        raise ValueError("no header row found; the file layout may have changed")

    block = [lines[start]]
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if not stripped[0].isdigit():
            break
        block.append(line)

    if len(block) < 2:
        raise ValueError("header found but no data rows followed it")

    frame = pd.read_csv(io.StringIO("\n".join(block)), index_col=0)
    frame.index = frame.index.astype(str).str.strip()

    lengths = frame.index.str.len().unique()
    if set(lengths) <= {8}:
        frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    elif set(lengths) <= {6}:
        # Monthly files date to the month; anchor at month end.
        frame.index = pd.to_datetime(frame.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        raise ValueError(f"unexpected date widths in index: {sorted(lengths)}")

    frame.index.name = "date"
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")

    # French publishes percent; the rest of this kit works in decimals.
    frame = frame / 100.0

    # -99.99 and -999 are the library's missing-value codes. After the division
    # above they are -0.9999 / -9.99, which are plausible-looking returns — so
    # they must be dropped here rather than downstream.
    return frame.mask(frame <= -0.99)


def load_french(dataset: str = "factors_daily", use_cache: bool = True) -> pd.DataFrame:
    """Load a Ken French dataset as decimal returns indexed by date.

    >>> factors = load_french("factors_daily")
    >>> sorted(factors.columns)
    ['HML', 'Mkt-RF', 'RF', 'SMB']
    """
    if dataset not in FRENCH_DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; options: {sorted(FRENCH_DATASETS)}")

    cache = _cache_path(f"french_{dataset}")
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    frame = _parse_french_csv(_download_french_csv(FRENCH_DATASETS[dataset]))
    frame.to_parquet(cache)
    return frame


def load_industry_returns(n: int = 49, use_cache: bool = True) -> pd.DataFrame:
    """Daily value-weighted industry portfolio returns — a ready-made panel.

    A convenient cross-section to develop against: real returns, real
    correlation structure, no credentials, and small enough to iterate on.
    """
    if n not in (12, 49):
        raise ValueError("n must be 12 or 49")
    return load_french(f"industry{n}_daily", use_cache=use_cache)


def clear_cache() -> int:
    """Delete cached parquet files. Returns the number removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.parquet"))
    for path in files:
        path.unlink()
    return len(files)
