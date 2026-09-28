"""One-time bulk pull of CRSP (CIZ / v2 format) and Compustat into data_cache/wrds/.

Usage::

    python scripts/pull_wrds.py --user YOUR_WRDS_ID [--start 1990] [--end 2025]

Resumable: each CRSP year is its own parquet file and existing files are
skipped, so an interrupted run picks up where it stopped. Everything lands in
data_cache/, which is gitignored — none of this may enter the public repo.

Why the v2 tables: legacy crsp.dsf stops at 2024-12-31; crsp.dsf_v2 continues.
In the CIZ format the delisting return is folded into dlyret on the delisting
date, so no separate dsedelist merge is needed for returns.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data_cache" / "wrds"

# CIZ equivalent of legacy shrcd in (10, 11) and exchcd in (1, 2, 3).
# conditionaltype / tradingstatusflg are kept as columns rather than filtered,
# so halted and delisting-day rows (which carry the delisting return) survive.
CRSP_FILTER = """
    sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM'
    and usincflg = 'Y' and issuertype in ('ACOR', 'CORP')
    and primaryexch in ('N', 'A', 'Q')
"""

CRSP_COLS = """
    permno, permco, dlycaldt as date, ticker, cusip, hdrcusip, siccd,
    primaryexch, conditionaltype, tradingstatusflg, dlydelflg,
    dlyprc, dlyprcflg, dlyret, dlyretx, dlyretmissflg, dlyvol, shrout, dlycap,
    dlyopen, dlyhigh, dlylow, dlyclose, dlybid, dlyask,
    dlycumfacpr, dlycumfacshr
"""

COMP_STD = "indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'"

FUNDA_WANT = """gvkey datadate fyear fyr tic cusip conm cik sich curcd
at lt ceq seq pstk pstkl pstkrv txditc txdb itcb mib sale revt cogs xsga xrd dp
oibdp oiadp ebit ebitda gp ib ibc ni pi txt xint epspx epspi csho prcc_f ajex
dvc dvt dvpsx_f che act lct dlc dltt invt rect ap ppent ppegt capx oancf fincf
ivncf re emp acominc""".split()

FUNDQ_WANT = """gvkey datadate fyearq fqtr fyr rdq tic cusip conm cik curcdq
datacqtr datafqtr atq ltq ceqq seqq pstkq txditcq mibq saleq revtq cogsq xsgaq
xrdq dpq oibdpq oiadpq ibq ibcomq niq piq txtq xintq epspxq epspiq cshoq cshprq
prccq ajexq dvpsxq cheq actq lctq dlcq dlttq invtq rectq apq ppentq req oancfy
capxy saley""".split()


def _save(frame: pd.DataFrame, name: str) -> None:
    path = OUT / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    print(f"  {name}: {len(frame):,} rows, {path.stat().st_size / 1e6:.1f} MB", flush=True)


def _cols(db, lib: str, table: str, want: list[str]) -> str:
    have = set(db.describe_table(lib, table)["name"])
    missing = [c for c in want if c not in have]
    if missing:
        print(f"  {lib}.{table}: skipping absent columns {missing}", flush=True)
    return ", ".join(c for c in want if c in have)


def pull_crsp(db, start: int, end: int) -> None:
    for year in range(start, end + 1):
        name = f"crsp_dsf_v2_{year}"
        if (OUT / f"{name}.parquet").exists():
            print(f"  {name}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        frame = db.raw_sql(
            f"select {CRSP_COLS} from crsp.dsf_v2 "
            f"where dlycaldt between '{year}-01-01' and '{year}-12-31' and {CRSP_FILTER}",
            date_cols=["date"],
        )
        _save(frame.sort_values(["date", "permno"]), name)
        print(f"    ({time.time() - t0:.0f}s)", flush=True)


def pull_reference(db, start: int) -> None:
    _save(db.raw_sql("select * from crsp.dsp500list_v2"), "crsp_sp500_membership")
    _save(db.raw_sql("select * from crsp.stkdelists"), "crsp_delists")
    _save(db.raw_sql("select * from crsp.ccmxpf_lnkhist"), "ccm_link")

    funda = _cols(db, "comp", "funda", FUNDA_WANT)
    _save(
        db.raw_sql(
            f"select {funda} from comp.funda where {COMP_STD} and datadate >= '{start - 2}-01-01'",
            date_cols=["datadate"],
        ),
        "comp_funda",
    )
    fundq = _cols(db, "comp", "fundq", FUNDQ_WANT)
    _save(
        db.raw_sql(
            f"select {fundq} from comp.fundq where {COMP_STD} and datadate >= '{start - 2}-01-01'",
            date_cols=["datadate", "rdq"],
        ),
        "comp_fundq",
    )
    _save(db.raw_sql("select * from comp.company"), "comp_company")


def add_universe_flag(daily: pd.DataFrame) -> pd.DataFrame:
    """`in_universe` = S&P 500 member that day AND a US-incorporated common stock.

    The team universe excludes REITs and foreign-incorporated members. Rows are
    flagged rather than dropped, so the exclusion stays visible and reversible.

    On a stock's delisted row CRSP blanks the descriptors (primaryexch 'X',
    tradingstatusflg 'D', sharetype 'N/A'), which would fail every member on its
    final day. Those rows are judged on the stock's previous row instead.
    """
    desc = ["sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype", "primaryexch"]
    ordered = daily.sort_values(["permno", "date"])
    delisted = (ordered["tradingstatusflg"] == "D").fillna(False)
    on_delisted = pd.DataFrame({col: delisted for col in desc})
    previous = ordered[desc].mask(on_delisted).groupby(ordered["permno"]).ffill()
    judged = ordered[desc].mask(on_delisted, previous)
    common = (
        (judged["sharetype"] == "NS")
        & (judged["securitytype"] == "EQTY")
        & (judged["securitysubtype"] == "COM")
        & (judged["usincflg"] == "Y")
        & judged["issuertype"].isin(["ACOR", "CORP"])
        & judged["primaryexch"].isin(["N", "A", "Q"])
    )
    common = common.fillna(False).reindex(daily.index)
    return daily.assign(in_universe=daily["in_sp500"] & common)


def pull_sp500(db, start: int, end: int) -> None:
    """Daily rows for every permno that was ever an S&P 500 member in [start, end].

    Pulled by permno with NO share-type filter: the common-stock filter above
    drops members that are REITs or incorporated abroad (~55 names a day by the
    2020s). Adds `in_sp500` (date inside a membership spell) and the delisting
    record on the stock's final row; `dlyret` itself is left untouched.
    """
    spells = db.raw_sql(
        "select permno, mbrstartdt, mbrenddt from crsp.dsp500list_v2 "
        f"where mbrenddt >= '{start}-01-01' and mbrstartdt <= '{end}-12-31'",
        date_cols=["mbrstartdt", "mbrenddt"],
    )
    permnos = tuple(int(p) for p in spells["permno"].unique())
    parts = []
    for year in range(start, end + 1):
        t0 = time.time()
        parts.append(
            db.raw_sql(
                f"select {CRSP_COLS}, sharetype, securitytype, securitysubtype, usincflg, "
                "issuertype from crsp.dsf_v2 "
                f"where dlycaldt between '{year}-01-01' and '{year}-12-31' "
                "and permno in %(permnos)s",
                params={"permnos": permnos},
                date_cols=["date"],
            )
        )
        print(f"  sp500 {year}: {len(parts[-1]):,} rows ({time.time() - t0:.0f}s)", flush=True)
    daily = pd.concat(parts, ignore_index=True)

    merged = daily[["date", "permno"]].merge(spells, on="permno")
    in_spell = (merged["date"] >= merged["mbrstartdt"]) & (merged["date"] <= merged["mbrenddt"])
    inside = merged[in_spell]
    keys = pd.MultiIndex.from_frame(inside[["date", "permno"]].drop_duplicates())
    daily["in_sp500"] = pd.MultiIndex.from_frame(daily[["date", "permno"]]).isin(keys)

    delists = db.raw_sql(
        "select permno, delistingdt, delret, delretmisstype, delactiontype, "
        "delstatustype, delreasontype from crsp.stkdelists where permno in %(permnos)s",
        params={"permnos": permnos},
        date_cols=["delistingdt"],
    )
    last = daily.groupby("permno")["date"].transform("max") == daily["date"]
    daily = daily.merge(delists, on="permno", how="left")
    delist_cols = [c for c in delists.columns if c != "permno"]
    daily.loc[~last.to_numpy(), delist_cols] = None
    daily = add_universe_flag(daily)

    _save(daily.sort_values(["date", "permno"]), f"sp500_daily_{start}_{end}")
    _save(spells, f"sp500_spells_{start}_{end}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", required=True, help="WRDS username (password comes from pgpass)")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--sp500", action="store_true", help="only build the S&P 500 dataset")
    args = parser.parse_args()

    import wrds

    OUT.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=args.user)
    try:
        if args.sp500:
            print(f"S&P 500 members {args.start}-{args.end}", flush=True)
            pull_sp500(db, args.start, args.end)
            return
        print("reference tables + Compustat", flush=True)
        pull_reference(db, args.start)
        print(f"CRSP daily {args.start}-{args.end}", flush=True)
        pull_crsp(db, args.start, args.end)
    finally:
        db.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
