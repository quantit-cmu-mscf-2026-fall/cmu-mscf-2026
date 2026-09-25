"""Build DATA_DICTIONARY.md and data_dictionary.csv for the shared data.

Usage::

    python scripts/build_data_dictionary.py [--user YOUR_WRDS_ID] [--refresh-meta]

Descriptions come from WRDS itself: CRSP's metadata tables (`crsp.metaiteminfo`
for column definitions, `crsp.metaflaginfo` for code meanings) and the WRDS
column comments for Compustat and the link table. That metadata is cached in
data_cache/wrds_meta/; WRDS is contacted (one Duo push) only when the cache is
missing or --refresh-meta is given.

Reads the shared files through capstone.shared_data, so it documents exactly
what teammates load. Every note stated as fact is asserted against the data
first; the build fails rather than publish a claim that stopped being true.

Output goes to data_cache/wrds_build/. Publish it with
scripts/publish_shared_data.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from capstone import shared_data as sd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data_cache" / "wrds_meta"
OUT = ROOT / "data_cache" / "wrds_build"

LABEL_TABLES = [
    ("crsp", "dsf_v2"),
    ("crsp", "stkdelists"),
    ("crsp", "dsp500list_v2"),
    ("crsp", "ccmxpf_lnkhist"),
    ("comp", "funda"),
    ("comp", "fundq"),
    ("comp", "company"),
]

COMPUSTAT = [
    ("comp_funda", "comp.funda"),
    ("comp_fundq", "comp.fundq"),
    ("comp_company", "comp.company"),
    ("ccm_link", "crsp.ccmxpf_lnkhist"),
]

UNITS = "Dollar amounts in USD millions; per-share items in dollars; share counts in millions."
NOT_AMOUNTS = {"fyear", "fyr", "fyearq", "fqtr"}

# CIZ item name for columns renamed in the pull.
ALIAS = {"date": "dlycaldt"}

DERIVED = {
    "in_sp500": "True if the stock was an S&P 500 member on this date (date within a "
    "spell in sp500_spells_1990_2025). Built by scripts/pull_wrds.py.",
    "in_universe": "Team universe: in_sp500 AND a US-incorporated ordinary common stock "
    "(sharetype NS, securitytype EQTY, securitysubtype COM, usincflg Y, issuertype "
    "ACOR/CORP) on NYSE/AMEX/NASDAQ. Excludes REITs and foreign-incorporated members. "
    "On a stock's delisting row the previous day's descriptors are used. Built by "
    "scripts/pull_wrds.py.",
}

NOTES = {
    "date": "Trading date (CRSP DlyCalDt, renamed).",
    "dlyret": "Includes dividends. On the delisting row (dlydelflg = Y) it IS the "
    "delisting return; do not compound delret on top.",
    "dlyprc": "Never negative in this format; see dlyprcflg for trade vs quote.",
    "dlycap": "In $ thousands (checked: dlycap = dlyprc x shrout, shrout in thousands).",
    "shrout": "In thousands of shares.",
    "dlybid": "Closing bid. Quoted spread = (dlyask - dlybid) / midpoint.",
    "dlyask": "Closing ask.",
    "dlycumfacpr": "Cumulative price adjustment factor; divide prices by it to compare "
    "across splits. Returns are already adjusted.",
    "permno": "Stable CRSP security id. Key on this, not ticker (tickers are reused).",
    "delret": "Delisting return; only on the delisting row, where it equals dlyret.",
    "delistingdt": "Last trading date before delisting; the delisting row itself is dated "
    "the next trading day.",
}

# Code values absent from crsp.metaflaginfo, labelled from what the data shows.
# Each is backed by an assertion in check_claims().
OBSERVED = {
    "N/A": "[not in CRSP flag table] blank placeholder; occurs only on delisting rows",
    ("dlyretmissflg", "NA"): "[not in CRSP flag table] return present; every such row has a dlyret",
    ("delretmisstype", "NA"): "[not in CRSP flag table] delisting return present; every "
    "such row has a delret",
    ("dlyprcflg", "SU"): "[not in CRSP flag table] occurs only on suspended days "
    "(tradingstatusflg S)",
}

FREE_TEXT = {"ticker", "cusip", "hdrcusip"}


def fetch_meta(user: str) -> None:
    import wrds

    META.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=user)
    try:
        labels = []
        for lib, table in LABEL_TABLES:
            frame = db.describe_table(lib, table)
            frame.insert(0, "table", f"{lib}.{table}")
            labels.append(frame)
        pd.concat(labels).to_csv(META / "wrds_column_labels.csv", index=False)
        for table in ("metaiteminfo", "metaflaginfo"):
            db.raw_sql(f"select * from crsp.{table}").to_csv(
                META / f"crsp_{table}.csv", index=False
            )
    finally:
        db.close()


def check_claims(daily: pd.DataFrame) -> None:
    """Assert every factual note above against the data."""
    cap_ratio = (daily.dlycap / (daily.dlyprc * daily.shrout)).dropna()
    assert cap_ratio.between(0.999, 1.001).mean() > 0.99, "dlycap is not prc x shrout"
    assert (daily.dlyprc.dropna() >= 0).all(), "negative prices present"
    assert (daily.loc[daily.sharetype == "N/A", "dlydelflg"] == "Y").all()
    assert daily.loc[daily.dlyretmissflg == "NA", "dlyret"].notna().all()
    assert daily.loc[daily.delretmisstype == "NA", "delret"].notna().all()
    assert (daily.loc[daily.dlyprcflg == "SU", "tradingstatusflg"] == "S").all()
    delisting = daily[daily.dlydelflg == "Y"]
    assert (delisting.date > delisting.delistingdt).all(), "delisting row dating changed"
    assert ((delisting.dlyret - delisting.delret).abs().dropna() < 1e-9).all(), (
        "dlyret no longer equals delret on delisting rows"
    )


def code_values(col: str, observed: pd.Series, items: pd.DataFrame, flags: pd.DataFrame) -> str:
    ftype = items.loc[col, "itemflagtype"] if col in items.index else None
    table = None
    if isinstance(ftype, str):
        table = flags[flags.flagtype == ftype].drop_duplicates("flagvalue").set_index("flagvalue")
    parts = []
    for value, count in observed.value_counts(dropna=True).items():
        if table is not None and value in table.index:
            desc = table.loc[value, "flagdesc"]
        else:
            desc = OBSERVED.get((col, value), OBSERVED.get(value))
            if desc is None:
                raise ValueError(f"no meaning for {col}={value!r}; add it to OBSERVED")
        parts.append(f"{value} = {desc} ({count:,})")
    return "; ".join(parts)


def build() -> pd.DataFrame:
    items = pd.read_csv(META / "crsp_metaiteminfo.csv")
    items["key"] = items.itemname.str.lower()
    items = items.drop_duplicates("key").set_index("key")
    flags = pd.read_csv(META / "crsp_metaflaginfo.csv")
    labels = pd.read_csv(META / "wrds_column_labels.csv")

    rows = []
    daily = sd.load("sp500_daily_1990_2025")
    check_claims(daily)
    for col in daily.columns:
        key = ALIAS.get(col, col)
        if col in DERIVED:
            desc, definition, source = col, DERIVED[col], "derived"
        else:
            rec = items.loc[key]
            desc, definition, source = rec.itemdesc, rec.itemdef, f"CRSP {rec.itemname}"
        codes = ""
        if daily[col].dtype == "string" and col not in FREE_TEXT:
            codes = code_values(key, daily[col], items, flags)
        rows.append(
            dict(
                file="sp500_daily_1990_2025",
                column=col,
                source=source,
                description=desc,
                definition=definition,
                notes=NOTES.get(col, ""),
                codes=codes,
            )
        )

    for col in sd.load("sp500_spells_1990_2025").columns:
        rec = items.loc[col]
        rows.append(
            dict(
                file="sp500_spells_1990_2025",
                column=col,
                source=f"CRSP {rec.itemname}",
                description=rec.itemdesc,
                definition=rec.itemdef,
                notes="",
                codes="",
            )
        )

    for name, table in COMPUSTAT:
        frame = sd.load(name)
        amounts = set(frame.select_dtypes("number").columns) - NOT_AMOUNTS
        lab = labels[labels.table == table].set_index("name")
        for col in frame.columns:
            codes = ""
            if col in ("linktype", "linkprim"):
                codes = "; ".join(f"{v} ({n:,})" for v, n in frame[col].value_counts().items())
            rows.append(
                dict(
                    file=name,
                    column=col,
                    source=table,
                    description=lab.loc[col, "comment"] if col in lab.index else "",
                    definition="",
                    notes=UNITS if name.startswith("comp_f") and col in amounts else "",
                    codes=codes,
                )
            )
    return pd.DataFrame(rows).fillna("")


def to_markdown(dd: pd.DataFrame) -> str:
    def esc(text: object) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ").strip()

    out = [
        "# Data dictionary",
        "",
        "Every column in the shared files. CRSP descriptions and code meanings come from "
        "CRSP's own metadata on WRDS (`crsp.metaiteminfo`, `crsp.metaflaginfo`); Compustat "
        "and link-table labels from the WRDS column comments. Counts in brackets are rows "
        "in our files. Machine-readable copy: `data_dictionary.csv`. Generated by "
        "`scripts/build_data_dictionary.py`.",
        "",
    ]
    for name, part in dd.groupby("file", sort=False):
        out += [f"## {name}", ""]
        if name.startswith("comp_f"):
            out += [f"_{UNITS}_", ""]
        out += ["| Column | Description | Notes |", "|---|---|---|"]
        for r in part.itertuples():
            note = r.notes if not name.startswith("comp_f") else ""
            if r.definition and r.definition != r.description and len(str(r.definition)) < 400:
                note = f"{esc(r.definition)} {esc(note)}".strip()
            out.append(f"| `{r.column}` | {esc(r.description)} | {esc(note)} |")
        coded = part[part.codes != ""]
        if len(coded):
            out += ["", "**Code values**", ""]
            out += [f"- `{r.column}`: {esc(r.codes)}" for r in coded.itertuples()]
        out.append("")
    out += [
        "## ccm_link: which links to use",
        "",
        "Standard CRSP-Compustat practice: keep `linktype` in `LU`, `LC` (researched, "
        "confirmed links) and `linkprim` in `P`, `C` (primary security), and require the "
        "CRSP date to fall within [`linkdt`, `linkenddt`] (null `linkenddt` = still "
        "active). The other codes are secondary or non-links; see the WRDS CCM "
        "documentation before using them.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", help="WRDS username; needed only to (re)fetch metadata")
    parser.add_argument("--refresh-meta", action="store_true")
    args = parser.parse_args()

    needed = ["wrds_column_labels.csv", "crsp_metaiteminfo.csv", "crsp_metaflaginfo.csv"]
    if args.refresh_meta or not all((META / f).exists() for f in needed):
        if not args.user:
            parser.error(f"metadata missing in {META}; pass --user to fetch it from WRDS")
        fetch_meta(args.user)

    dd = build()
    OUT.mkdir(parents=True, exist_ok=True)
    dd.to_csv(OUT / "data_dictionary.csv", index=False)
    (OUT / "DATA_DICTIONARY.md").write_text(to_markdown(dd), encoding="utf-8")
    print(f"{len(dd)} columns across {dd.file.nunique()} files -> {OUT}")


if __name__ == "__main__":
    main()
