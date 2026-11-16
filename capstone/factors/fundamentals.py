"""Annual fundamentals for the factor panel (QUANTIT-118).

The grammar's accounting fields (`tree.FUNDAMENTAL_FIELDS`: book, earnings,
sales, assets) as dates x permno frames, each value dated by when it could
have been known:

- **Rows** (`FUNDAMENTAL_COLUMNS`): one per permno and fiscal year end
  (`datadate`), with a column per field. `from_compustat` makes them from the
  shared `comp_funda` and `ccm_link` files; `load_fundamentals` reads those
  through `capstone.shared_data` (the only way in). Tests pass their own rows.
- **Linking**: Compustat's gvkey becomes a permno through `sd.primary_links`
  (LU/LC, primary security); a fiscal year counts only while its link is live.
- **Availability** (`docs/shared_data.md`): `sd.available_from` with
  report_col=None, since annual data has no report date: the first trading
  day strictly after the fiscal year end + 90 days. A value is never visible
  before that day.
- **As of** (`as_of_fields`): on each date, each permno's latest available
  fiscal year, held until the next one is available. A year reported with a
  missing value is missing until the next year, not filled from the year
  before ("don't fill missing fundamentals"). An older fiscal year that
  becomes available after a newer one never replaces it. A year is held
  only until the following year would be available: an absent year or an
  ended link leaves the value missing, not carried forward indefinitely.

Known limitation (as the shared data's): Compustat values may include later
restatements rather than the figures first reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.factors.tree import FUNDAMENTAL_FIELDS

FUNDAMENTAL_COLUMNS = ("permno", "datadate", *FUNDAMENTAL_FIELDS)
FUNDA_FILE = "comp_funda"
LINK_FILE = "ccm_link"
FUNDA_COLUMNS = [
    "gvkey",
    "datadate",
    "seq",
    "ceq",
    "pstk",
    "pstkrv",
    "pstkl",
    "txditc",
    "at",
    "lt",
    "ib",
    "sale",
]
LINK_COLUMNS = ["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]
# Annual data: a fiscal year's value is held until the next year would be available.
MAX_HOLD_YEARS = 1


def book_equity(funda: pd.DataFrame) -> pd.Series:
    """Fama-French book equity: stockholders' equity, less preferred stock, plus
    deferred taxes and investment tax credit.

    Equity is `seq`, else `ceq` + `pstk`, else `at` - `lt`; preferred is the
    redemption value `pstkrv`, else liquidating `pstkl`, else par `pstk`,
    else 0; `txditc` is 0 when missing (Davis, Fama and French 2000).
    """
    equity = funda["seq"].fillna(funda["ceq"] + funda["pstk"].fillna(0.0))
    equity = equity.fillna(funda["at"] - funda["lt"])
    preferred = funda["pstkrv"].fillna(funda["pstkl"]).fillna(funda["pstk"]).fillna(0.0)
    return (equity - preferred + funda["txditc"].fillna(0.0)).rename("book")


def from_compustat(funda: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """`FUNDAMENTAL_COLUMNS` rows from `comp_funda` and `ccm_link` rows.

    Linked with `sd.primary_links` (LU/LC, primary security), keeping a fiscal
    year only when its `datadate` falls inside the link's [linkdt, linkenddt].
    One row per (permno, datadate): a duplicate keeps the last.
    """
    links = sd.primary_links(link)[["gvkey", "lpermno", "linkdt", "linkenddt"]]
    rows = funda.assign(datadate=pd.to_datetime(funda["datadate"])).merge(links, on="gvkey")
    rows = rows[(rows["datadate"] >= rows["linkdt"]) & (rows["datadate"] <= rows["linkenddt"])]
    out = pd.DataFrame(
        {
            "permno": rows["lpermno"].astype("int64"),
            "datadate": rows["datadate"],
            "book": book_equity(rows),
            "earnings": rows["ib"].astype("float64"),
            "sales": rows["sale"].astype("float64"),
            "assets": rows["at"].astype("float64"),
        }
    )
    out = out.sort_values(["permno", "datadate"], kind="stable")
    return out.drop_duplicates(["permno", "datadate"], keep="last").reset_index(drop=True)


def load_fundamentals(end: str) -> pd.DataFrame:
    """Every linked fiscal year ending on or before `end`, through the shared data.

    Fiscal years ending by `end` can only become visible after it (90 days
    later), so the panel's own calendar decides what is used; nothing dated
    after `end` is kept in the rows. `sd.load` filters only the daily file's
    `date`, so these two small files are read whole and cut here.
    """
    funda = sd.load(FUNDA_FILE, columns=FUNDA_COLUMNS)
    funda = funda[pd.to_datetime(funda["datadate"]) <= pd.Timestamp(end)]
    links = sd.load(LINK_FILE, columns=LINK_COLUMNS)
    # Links starting after `end` cannot cover a fiscal year ending by `end`.
    links = links[pd.to_datetime(links["linkdt"]) <= pd.Timestamp(end)]
    return from_compustat(funda, links)


def _checked(fund: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FUNDAMENTAL_COLUMNS if c not in fund.columns]
    if missing:
        raise ValueError(f"the fundamentals frame has no column(s) {missing}")
    out = fund.assign(datadate=pd.to_datetime(fund["datadate"]))
    if np.isinf(out[list(FUNDAMENTAL_FIELDS)].to_numpy(dtype="float64")).any():
        raise ValueError("fundamental values must be finite (NaN for missing)")
    return out


def with_availability(fund: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """`fund` with an `available` column: `sd.available_from` on `dates`, annual rule."""
    fund = _checked(fund)
    available = sd.available_from(fund, dates, report_col=None, period_col="datadate")
    return fund.assign(available=available)


def as_of_fields(
    fund: pd.DataFrame, dates: pd.DatetimeIndex, permnos: pd.Index
) -> dict[str, pd.DataFrame]:
    """Each field as dates x permnos: the latest fiscal year available on each date.

    `fund` has `FUNDAMENTAL_COLUMNS` and `available` (a trading date of
    `dates`, or NaT: never visible). Before a permno's first available year,
    and for a permno without rows, the value is NaN. A year expires on the day
    the following fiscal year would be available (its fiscal year end plus
    `MAX_HOLD_YEARS`, then the annual 90-day rule); with no newer year by
    then, the value is NaN rather than carried on.
    """
    dates = pd.DatetimeIndex(dates)
    rows = _checked(fund)
    rows = rows[rows["available"].notna() & rows["permno"].isin(permnos)]
    rows = rows.sort_values(["permno", "available", "datadate"], kind="stable")
    # An older fiscal year arriving after a newer one is not news.
    newest = rows.groupby("permno")["datadate"].cummax()
    rows = rows[rows["datadate"] >= newest]
    # Several years visible from one day (all before the panel, say): the latest.
    rows = rows.drop_duplicates(["permno", "available"], keep="last").reset_index(drop=True)
    if rows.empty:
        return {
            name: pd.DataFrame(np.nan, index=dates, columns=permnos) for name in FUNDAMENTAL_FIELDS
        }
    unknown = pd.DatetimeIndex(rows["available"]).difference(dates)
    if len(unknown):
        raise ValueError(f"availability dates outside the panel's calendar, e.g. {unknown[0]}")
    # Which row holds on each (date, permno): its position, carried forward.
    position = rows.assign(position=np.arange(len(rows), dtype="float64")).pivot(
        index="available", columns="permno", values="position"
    )
    position = position.reindex(index=dates, columns=permnos).ffill()
    index = position.fillna(len(rows)).to_numpy().astype("int64")
    # A year is held only until the next one would be available: from that day on,
    # with no newer year (none filed, or the link ended), the value is missing.
    next_year = rows.assign(datadate=rows["datadate"] + pd.DateOffset(years=MAX_HOLD_YEARS))
    expires = sd.available_from(next_year, dates, report_col=None, period_col="datadate")
    held_until = np.append(expires.to_numpy(dtype="datetime64[ns]"), np.datetime64("NaT", "ns"))
    stale = held_until[index] <= dates.to_numpy(dtype="datetime64[ns]")[:, None]  # NaT: False
    out = {}
    for name in FUNDAMENTAL_FIELDS:
        values = np.append(rows[name].to_numpy(dtype="float64"), np.nan)[index]
        values[stale] = np.nan
        out[name] = pd.DataFrame(values, index=dates, columns=permnos)
    return out


def fundamental_fields(
    fund: pd.DataFrame, dates: pd.DatetimeIndex, permnos: pd.Index
) -> dict[str, pd.DataFrame]:
    """The panel's accounting fields from `FUNDAMENTAL_COLUMNS` rows."""
    return as_of_fields(with_availability(fund, dates), dates, permnos)
