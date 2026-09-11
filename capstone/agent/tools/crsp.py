"""CRSP return retrieval, as a tool the agent can call.

A thin adapter over `capstone.wrds_loader`. The credentials, the connection and
the SQL all live there and none of them are duplicated here; what this module
adds is a narrow contract — a fixed column set, stable dtypes, and an empty
frame instead of a surprise when nothing matches. A tool called without a human
watching has to behave the same way on every path, including the boring ones.
"""

from __future__ import annotations

import pandas as pd

from capstone.wrds_loader import connect, crsp_daily

#: Columns every call returns, in this order, rows or no rows.
RETURN_COLUMNS = ("date", "permno", "ret")

_DTYPES = {"date": "datetime64[ns]", "permno": "int64", "ret": "float64"}


def _empty_frame() -> pd.DataFrame:
    """The zero-row result, carrying the same columns and dtypes as a full one.

    Callers that concat, group or merge this output should not have to special-
    case "nothing matched", so the empty result is the normal result minus rows.
    """
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in _DTYPES.items()})


def _to_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Narrow `crsp_daily`'s tidy frame to the columns and dtypes we promise."""
    if frame.empty:
        return _empty_frame()

    out = frame.loc[:, list(RETURN_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"])
    # WRDS hands numeric columns back as float; permno is an identifier, and an
    # identifier that prints as 90001.0 will eventually be joined wrongly.
    out["permno"] = out["permno"].astype("int64")
    out["ret"] = out["ret"].astype("float64")
    return out.sort_values(["date", "permno"]).reset_index(drop=True)


def get_crsp_returns(
    start_date: str,
    end_date: str,
    permnos: list[int] | None = None,
    *,
    conn=None,
) -> pd.DataFrame:
    """Daily CRSP returns for a date range, optionally narrowed to some PERMNOs.

    Args:
        start_date: inclusive ISO date, "YYYY-MM-DD".
        end_date: inclusive ISO date, "YYYY-MM-DD".
        permnos: PERMNOs to keep. None means every security passing
            `crsp_daily`'s share-code and exchange filters. An EMPTY list means
            no securities, and returns an empty frame without touching the
            database.
        conn: an open WRDS connection. When omitted, one is opened for this call
            and closed again; a connection passed in is left open for its owner.

    Returns:
        Columns [date, permno, ret], sorted by (date, permno), with a
        RangeIndex — those columns and dtypes always, even with zero rows.

    Raises:
        ValueError: if `end_date` precedes `start_date`.

    Query and connection errors are not caught. A tool that answers a dead
    database with an empty frame teaches its caller that the market was quiet.

    PERMNO is the identifier rather than ticker because CRSP reuses tickers
    across companies over time, so a ticker-keyed result silently merges two
    different securities (see `wrds_loader.to_price_panel`).

    Known limitation: `ret` comes from `crsp.dsf` alone. CRSP records a delisted
    security's final partial-period return separately in `crsp.dsedelist`, so it
    is missing here — and those are precisely the losses survivorship bias hides
    (docs/data_sources.md). Joining `dsedelist` is follow-up work, deliberately
    not done in this commit.
    """
    # ISO dates compare correctly as strings, which is most of why the kit
    # passes them around as strings rather than parsing at every boundary.
    if end_date < start_date:
        raise ValueError(f"end_date {end_date!r} precedes start_date {start_date!r}")

    # An empty list is a request for no securities. `crsp_daily` tests
    # `if permnos:`, which would read [] as "no filter" and return the entire
    # cross-section — the exact opposite of what was asked for.
    if permnos is not None and len(permnos) == 0:
        return _empty_frame()

    tool_owns_connection = conn is None
    if tool_owns_connection:
        conn = connect()
    try:
        frame = crsp_daily(conn, start_date, end_date, permnos=permnos)
    finally:
        if tool_owns_connection:
            conn.close()

    return _to_contract(frame)
