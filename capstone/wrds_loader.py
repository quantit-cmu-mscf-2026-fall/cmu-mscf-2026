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

import os
import sys
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"

# CRSP share codes 10 and 11 = ordinary common shares of US-incorporated firms.
# Without this filter you silently pull in ADRs, REITs, closed-end funds and
# trackers, and your "US equity" cross-section is not one.
COMMON_SHARE_CODES = (10, 11)

# Exchange codes 1/2/3 = NYSE / AMEX / NASDAQ.
MAJOR_EXCHANGES = (1, 2, 3)

# Date-bounded S&P 500 membership join, as documented on the WRDS "S&P 500 Data"
# page. Bounding on the spell dates is what makes it point-in-time: a name that
# left the index in 2014 contributes rows only through 2014. Selecting the same
# permnos with an IN-list would pull their full history and quietly readmit them
# to the universe for every year after they left.
_SP500_JOIN = """join crsp.dsp500list as s
          on {alias}.permno = s.permno
         and {date_col} >= s.start
         and {date_col} <= coalesce(s.ending, current_date)"""

# --- CIZ (the _v2 family) ---------------------------------------------------
#
# CRSP's newer CIZ layout replaces the legacy SIZ tables, and it matters for a
# blunt reason: on CMU's subscription legacy stops at 2024-12-31 while CIZ runs
# to 2025-12-31, so anything built on `crsp.dsf` is quietly a year stale.
#
# CIZ drops the numeric shrcd/exchcd codes for named classification columns
# carried inline on the daily file, which removes the `dsenames` join entirely.
# These values were verified against legacy rather than taken from a mapping
# table: on 2020-06-30, every security with shrcd in (10, 11) and exchcd in
# (1, 2, 3) carried exactly this combination.
#
# All four parts are load-bearing. On that same day, relaxing `issuertype` would
# admit 2,434 funds and 184 REITs, and relaxing `sharetype` a further 393 ADRs.
CIZ_COMMON_SHARE = {
    "sharetype": "NS",  # no special status (excludes ADRs, units, trackers)
    "securitytype": "EQTY",  # equity, not fund or debt
    "issuertype": "CORP",  # ordinary corporation (excludes REIT, ACOR)
    "usincflg": "Y",  # US-incorporated
}

# Primary exchange letters, the CIZ spelling of NYSE / AMEX / NASDAQ.
CIZ_MAJOR_EXCHANGES = ("N", "A", "Q")

# Legacy name -> CIZ name. Aliasing in the SELECT means callers of the tidy
# frame cannot tell which table family produced it.
CIZ_COLUMN_ALIASES = {
    "date": "dlycaldt",
    "openprc": "dlyopen",
    "askhi": "dlyhigh",
    "bidlo": "dlylow",
    "prc": "dlyprc",
    "ret": "dlyret",
    "vol": "dlyvol",
    "cfacpr": "dlycumfacpr",
    "cfacshr": "dlycumfacshr",
}

# Same shape, against the CIZ membership table.
_SP500_JOIN_CIZ = """join crsp.dsp500list_v2 as s
          on v.permno = s.permno
         and v.dlycaldt >= s.mbrstartdt
         and v.dlycaldt <= coalesce(s.mbrenddt, current_date)"""


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def pgpass_path() -> Path:
    """Where the PostgreSQL driver keeps saved passwords on this machine."""
    if os.environ.get("PGPASSFILE"):
        return Path(os.environ["PGPASSFILE"])
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def _pgpass_fields(line: str) -> list[str]:
    """Split a pgpass line on ':', honouring the format's '\\:' and '\\\\' escapes."""
    fields, current, chars = [], "", iter(line)
    for ch in chars:
        if ch == "\\":
            current += next(chars, "")
        elif ch == ":":
            fields.append(current)
            current = ""
        else:
            current += ch
    return [*fields, current]


def wrds_username(explicit: str | None = None) -> str | None:
    """The WRDS username: `explicit`, else $WRDS_USERNAME, else the pgpass file.

    The pgpass file's fourth field is the username, so scripts never need it
    on the command line, where it would land in shell history and session logs.
    """
    if explicit:
        return explicit
    if os.environ.get("WRDS_USERNAME"):
        return os.environ["WRDS_USERNAME"]
    path = pgpass_path()
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        fields = _pgpass_fields(line)
        if len(fields) >= 4 and not line.startswith("#") and fields[0] in (WRDS_HOST, "*"):
            return fields[3] or None
    return None


def connect(username: str | None = None):
    """Open a WRDS connection, with an actionable error if the package is absent.

    The username comes from `wrds_username()` when not given. Kept as a thin
    wrapper so the import of `wrds` stays lazy: the rest of this kit must
    import cleanly for someone who has no WRDS account at all.
    """
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the 'wrds' package is not installed. Run `pip install wrds`, or use "
            "capstone.sample_data.load_sample_prices() to work without WRDS."
        ) from exc

    username = wrds_username(username)
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def crsp_daily(
    conn,
    start: str,
    end: str,
    *,
    permnos: list[int] | None = None,
    sp500_only: bool = False,
    share_codes: tuple[int, ...] = COMMON_SHARE_CODES,
    exchanges: tuple[int, ...] = MAJOR_EXCHANGES,
) -> pd.DataFrame:
    """Daily CRSP prices and returns for a date range, as a tidy frame.

    Prefer `crsp_daily_ciz`: these legacy tables stop a year earlier.

    Returns columns [date, permno, ticker, prc, ret, vol, shrout, cfacpr].

    `sp500_only` restricts the pull to index members, joined on the membership
    spell so each name appears only for the dates it actually belonged. Prefer
    it for index research: on a measured 2022-23 pull the unrestricted query
    returns 4,742 securities where ~500 are wanted, roughly 10x the rows. The
    wide pull stays the default because a cache of it can be re-filtered to any
    universe definition without going back to WRDS.

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

    membership_join = _SP500_JOIN.format(alias="d", date_col="d.date") if sp500_only else ""
    query = f"""
        select d.date, d.permno, n.ticker, d.prc, d.ret, d.vol,
               d.shrout, d.cfacpr
        from crsp.dsf as d
        join crsp.dsenames as n on d.permno = n.permno
        {membership_join}
        where {" and ".join(where)}
    """
    frame = conn.raw_sql(query, params=params, date_cols=["date"])
    return frame.sort_values(["date", "permno"]).reset_index(drop=True)


def crsp_daily_ciz(
    conn,
    start: str,
    end: str,
    *,
    permnos: list[int] | None = None,
    sp500_only: bool = False,
) -> pd.DataFrame:
    """`crsp_daily` against the CIZ (`_v2`) tables, with legacy column names.

    Prefer this over `crsp_daily`: on CMU's subscription the legacy tables stop
    at 2024-12-31 and these run to 2025-12-31, so a legacy pull is silently a
    year behind.

    CIZ names are aliased back to `prc`, `ret`, `cfacpr` and friends in the
    SELECT, so callers cannot tell which family produced the frame. It also
    carries open/high/low and the delisting return, which the legacy query above
    does not.

    Two differences from legacy, both verified against the data rather than
    assumed, because both corrupt silently if guessed wrong:

    - Classification moved inline onto the daily file, so there is no `dsenames`
      join. See `CIZ_COMMON_SHARE`.
    - The delisting return sits on its own row. CIZ adds a row 1-7 days after
      `delistingdt` with `dlydelflg == 'Y'`, `tradingstatusflg == 'D'` and
      `dlyret` equal to `delret` (662 of 662 such rows in the 1990-2025 S&P 500
      pull). That row carries `sharetype`/`securitytype` 'N/A' and
      `primaryexch` 'X', so the classification filter below drops it. The
      delisting return is therefore merged back from `stkdelists` as `dlret`,
      on `delistingdt`, the last trading day, as in the legacy convention:
      compound it with that day's `ret`, (1 + ret) * (1 + dlret) - 1.
    """
    aliased = ", ".join(f"v.{source} as {legacy}" for legacy, source in CIZ_COLUMN_ALIASES.items())
    classification = " and ".join(f"v.{col} = '{value}'" for col, value in CIZ_COMMON_SHARE.items())

    where = [
        "v.dlycaldt between %(start)s and %(end)s",
        classification,
        f"v.primaryexch in {CIZ_MAJOR_EXCHANGES}",
    ]
    params: dict[str, object] = {"start": start, "end": end}
    if permnos:
        where.append("v.permno in %(permnos)s")
        params["permnos"] = tuple(permnos)

    query = f"""
        select {aliased}, v.permno, v.ticker, v.shrout
        from crsp.dsf_v2 as v
        {_SP500_JOIN_CIZ if sp500_only else ""}
        where {" and ".join(where)}
    """
    frame = conn.raw_sql(query, params=params, date_cols=["date"])

    # stkdelists carries no classification columns, so rather than reproduce the
    # filter through another join we keep delisting rows for permnos the price
    # pull already admitted. The merge is outer so a delisting return survives
    # even if its last trading day fell outside the price rows pulled.
    delist_query = """
        select d.delistingdt as date, d.permno, d.delret as dlret
        from crsp.stkdelists as d
        where d.delistingdt between %(start)s and %(end)s
          and d.delret is not null
    """
    delist = conn.raw_sql(delist_query, params={"start": start, "end": end}, date_cols=["date"])
    delist = delist[delist["permno"].isin(set(frame["permno"]))]
    delist = delist.drop_duplicates(subset=["date", "permno"])

    frame = frame.merge(delist, on=["date", "permno"], how="outer", validate="one_to_one")
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
    return _membership_frame(conn, query, start, end)


def sp500_members_ciz(conn, start: str, end: str) -> pd.DataFrame:
    """`sp500_members` against `crsp.dsp500list_v2`, same columns out.

    A permno can hold SEVERAL spells — 11 do over 2010-2025 — because names
    leave the index and later rejoin. Any membership check must test against
    every spell, not against a single end date.
    """
    query = """
        select permno, mbrstartdt as mbr_start, mbrenddt as mbr_end
        from crsp.dsp500list_v2
        where mbrenddt >= %(start)s and mbrstartdt <= %(end)s
    """
    return _membership_frame(conn, query, start, end)


def _membership_frame(conn, query: str, start: str, end: str) -> pd.DataFrame:
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

    Both the universe scope AND the table family are part of the cache key.
    Without that, a scoped and a wide pull under one `name` would silently serve
    each other's rows, and so would a legacy and a CIZ pull — which differ by a
    year of history. A cache that returns the wrong data is worse than none.

    BOTH families carry an explicit suffix, including the default. Leaving CIZ
    unsuffixed would put it on the bare `crsp_<name>` path that every pull
    written before this change already occupies — so the new default would
    silently serve year-stale legacy rows as if they were current. Caches
    predating this change are simply never hit again, which is the right
    outcome: they are legacy, and legacy is a year behind.
    """
    ciz = kwargs.pop("ciz", True)
    scope = "_sp500" if kwargs.get("sp500_only") else ""
    family = "_ciz" if ciz else "_siz"
    path = _cache_path(f"crsp_{name}{scope}{family}")
    if path.exists():
        return pd.read_parquet(path)
    puller = crsp_daily_ciz if ciz else crsp_daily
    frame = puller(conn, start, end, **kwargs)
    frame.to_parquet(path)
    return frame
