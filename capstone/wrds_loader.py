"""CRSP daily prices via WRDS — loaded with YOUR credentials, not ours.

CMU subscribes to WRDS, so CRSP is already yours: it is the academic standard
for US equity prices, and unlike every free alternative it includes delisted
companies and delisting returns. That single property is what separates a
backtest from a measurement of your own hindsight (see docs/data_sources.md on
survivorship bias).

We ship the loader; the data flows through your own WRDS account. We never hold
or redistribute CRSP data — the agreement with CMU allows CMU-licensed academic
data used by you, not vendor data handed out by us. CRSP-derived artifacts also
generally may not be published, so check before making any output public.

First-time setup is in docs/wrds_howto.md. In short::

    pip install wrds
    python -c "import wrds; wrds.Connection(wrds_username='YOUR_ID')"
    # answer 'y' when it offers to create ~/.pgpass, then you never type the
    # password again

If you have no WRDS account yet, everything here has a drop-in stand-in:
`capstone.sample_data.load_sample_prices()` returns the same frame shape from a
synthetic six-month panel, so you can build and test the whole pipeline first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

# CRSP share codes 10 and 11 = ordinary common shares of US-incorporated firms.
# Without this filter you silently pull in ADRs, REITs, closed-end funds and
# trackers, and your "US equity" cross-section is not one.
COMMON_SHARE_CODES = (10, 11)

# Exchange codes 1/2/3 = NYSE / AMEX / NASDAQ.
MAJOR_EXCHANGES = (1, 2, 3)


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def connect(username: str | None = None):
    """Open a WRDS connection, with an actionable error if the package is absent.

    Kept as a thin wrapper so the import of `wrds` stays lazy: the rest of this
    kit must import cleanly for someone who has no WRDS account at all.
    """
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the 'wrds' package is not installed. Run `pip install wrds`, or use "
            "capstone.sample_data.load_sample_prices() to work without WRDS."
        ) from exc

    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def crsp_daily(
    conn,
    start: str,
    end: str,
    *,
    permnos: list[int] | None = None,
    share_codes: tuple[int, ...] = COMMON_SHARE_CODES,
    exchanges: tuple[int, ...] = MAJOR_EXCHANGES,
) -> pd.DataFrame:
    """Daily CRSP prices and returns for a date range, as a tidy frame.

    Returns columns [date, permno, ticker, prc, ret, vol, shrout, cfacpr].

    Two CRSP conventions bite everyone once:

    - `prc` is NEGATIVE when the day had no trade and CRSP stored the bid/ask
      midpoint instead. Take `abs(prc)` for a price level, but know that those
      days are quotes, not transactions.
    - `ret` already includes dividends and is adjusted for splits. Do NOT rebuild
      returns from `prc` unless you also apply `cfacpr` yourself — mixing the two
      is a common source of phantom jumps on split dates.

    The share-code and exchange filters are applied through the names file
    (`crsp.dsenames`), which is time-varying: a company that changed listing is
    correctly included only for the period it qualified.
    """
    where = [
        "d.date between %(start)s and %(end)s",
        f"n.shrcd in {tuple(share_codes)}",
        f"n.exchcd in {tuple(exchanges)}",
        "d.date between n.namedt and coalesce(n.nameendt, current_date)",
    ]
    params: dict[str, object] = {"start": start, "end": end}
    if permnos:
        where.append("d.permno in %(permnos)s")
        params["permnos"] = tuple(permnos)

    query = f"""
        select d.date, d.permno, n.ticker, d.prc, d.ret, d.vol,
               d.shrout, d.cfacpr
        from crsp.dsf as d
        join crsp.dsenames as n on d.permno = n.permno
        where {" and ".join(where)}
    """
    frame = conn.raw_sql(query, params=params, date_cols=["date"])
    return frame.sort_values(["date", "permno"]).reset_index(drop=True)


def to_price_panel(daily: pd.DataFrame, field: str = "prc") -> pd.DataFrame:
    """Pivot the tidy CRSP frame into date x ticker, the shape the kit expects.

    Prices are made positive (see `crsp_daily` on negative quotes). Tickers are
    reused by CRSP over time, so a panel keyed on ticker can collide; permno is
    the stable identifier. Pass field='permno' framing if you hit that.
    """
    values = daily[field].abs() if field == "prc" else daily[field]
    panel = daily.assign(_v=values).pivot_table(
        index="date", columns="ticker", values="_v", aggfunc="last"
    )
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def sp500_members(conn, start: str, end: str) -> pd.DataFrame:
    """S&P 500 membership intervals from CRSP, i.e. the survivorship-bias fix.

    `crsp.dsp500list` gives one row per (permno, start, ending) membership spell,
    so a name that left the index in 2011 is present for the period it was
    actually in it — which is the whole point. Building a universe from today's
    constituent list instead is the single most effective way to make a backtest
    look excellent for no reason.
    """
    query = """
        select permno, start as mbr_start, ending as mbr_end
        from crsp.dsp500list
        where ending >= %(start)s and start <= %(end)s
    """
    return conn.raw_sql(
        query,
        params={"start": start, "end": end},
        date_cols=["mbr_start", "mbr_end"],
    )


def cached_crsp_daily(conn, start: str, end: str, *, name: str, **kwargs) -> pd.DataFrame:
    """`crsp_daily` with an on-disk parquet cache, keyed by a name you choose.

    WRDS queries are slow and rate-limited by the server's patience, not yours.
    Cache aggressively while iterating; delete the file to refresh. The cache
    lives in data_cache/, which is gitignored — CRSP data must not enter the
    public repository.
    """
    path = _cache_path(f"crsp_{name}")
    if path.exists():
        return pd.read_parquet(path)
    frame = crsp_daily(conn, start, end, **kwargs)
    frame.to_parquet(path)
    return frame
