# Shared data: the team's CRSP and Compustat files

One pull from WRDS, shared, so every teammate and every agent works from
identical files. The files live in a Google Drive folder shared with the five of
us; `capstone.shared_data` finds that folder on your machine, verifies every
file, and serves it. Nobody copies files or connects to WRDS to use them.

**Team only.** This is CMU-licensed data. It never enters this repository
(`data_cache/` and `*.parquet` are gitignored), and it is not uploaded to
Arkraft or shared outside the team.

## What is in it

| File | Rows | Contents |
|---|---|---|
| `sp500_daily_1990_2025` | 7.2M | Daily CRSP rows for every stock that was an S&P 500 member at any point 1990–2025, over its whole 1990–2025 history, with `in_sp500` and `in_universe` flags |
| `sp500_spells_1990_2025` | 1.4k | Membership spells: `permno`, `mbrstartdt`, `mbrenddt` |
| `comp_funda` / `comp_fundq` | 438k / 1.75M | Compustat annual and quarterly (with `rdq`, the report date) |
| `comp_company` | 59k | Compustat company header |
| `ccm_link` | 123k | CRSP–Compustat link |

What every column means, with CRSP's own definitions and code values:
`DATA_DICTIONARY.md` in the Drive folder, or `data_dictionary.csv` in your local
copy (`data_cache/wrds/`).

## One-time setup

1. Install [Google Drive for desktop](https://www.google.com/drive/download/) and
   sign in with your **andrew.cmu.edu** account.
2. Open the shared `mscf-capstone-data` folder on drive.google.com and choose
   **Add shortcut to Drive**. Shared folders do not sync until you do this.
3. In Drive for desktop, right-click the folder and choose **Available offline**,
   so the first sync is not waiting on downloads.

That is all. The folder is found automatically on Windows (any drive letter) and
macOS (`~/Library/CloudStorage/GoogleDrive-*`). If it is somewhere else, set
`CAPSTONE_DATA_DIR` to its path.

## Using it

```python
from capstone import shared_data as sd

daily = sd.load(
    "sp500_daily_1990_2025",
    columns=["date", "permno", "dlyret", "in_universe"],
    start="2010-01-01",
)
sd.data_version()   # e.g. "6c1295f853f3": identifies the exact dataset
```

`load()` syncs first: it compares the Drive folder's `SHA256SUMS` with your
local copy and copies only files that changed, verifying each before it replaces
the old one. A repeat call costs a fraction of a second.

- **Select columns.** All 40 columns of the daily file take 2.6 GB in memory;
  the ten most work needs take about 0.6 GB for all years.
- **Numbers come back as plain numpy types** (`float64`, `int64`, `bool`), text
  as compact pyarrow strings.
- **What can go wrong, and what happens:** a file still syncing, or replaced
  mid-read, fails its checksum and the previous verified copy is kept (with a
  warning); Drive not running means the local copy is used (with a warning);
  several agents syncing at once take turns. Only "no Drive folder and no local
  copy" is an error.

## Rules for using the data

These are implemented as functions so every strategy applies them identically.
Measured on the 1990–2025 file, they leave no missing return and no missing
spread on any `in_universe` row.

**Universe.** Choose holdings with `in_universe`: an S&P 500 member that day and
a US-incorporated ordinary common stock (REITs and foreign-incorporated members
excluded). **Do not drop other rows when computing returns**: a stock usually
leaves the index before it delists, and a position still held takes that loss.
Key on `permno`, not ticker. Nine companies have two share classes in the
universe (e.g. GOOG/GOOGL); group by `permco` for company-level signals.

**Returns: `sd.daily_returns(daily)`.** `dlyret` already includes dividends and,
on CRSP's delisting row (`dlydelflg == "Y"`), the delisting return; never
compound `delret` on top. Missing values are resolved as follows:

| Case | Treatment | Why |
|---|---|---|
| New security's first day (`NS`), missing price (`MP`) | 0 | Exact: nothing held it before, or the next return spans the gap |
| Bank failures with no delisting return (SVB, Signature, 2023) | −100% | Shareholders wiped out |
| Other performance delisting with no return | −50% | Average bankruptcy delisting return in this data |
| First day a stock stops being tracked (moved to OTC) | −50%, then NaN | NaN = no position can be held: exit |

**Trading costs: `sd.quoted_spread(daily)`.** The quoted spread
`(ask − bid) / midpoint` when `ask > bid > 0` and it is at most 500 bps;
otherwise the stock's own median over the previous 21 days, then that day's
median across the universe. Median spreads: ~95 bps in the 1990s, ~9 bps in the
2000s, ~2.3 bps since 2010. Before 2000, 29% of universe rows rely on the
cross-sectional fill, so treat pre-2000 cost results with care.

**Fundamentals: `sd.available_from(fund, trading_days)`.** A quarter may be used
from the first trading day **after** its report date (`rdq`), since reports
often land after the close. With no `rdq` (0.5% of quarters feeding the
universe), use the quarter end + 90 days, later than 99% of actual reports. Link
with `sd.primary_links(link)` and require the date to fall within
`[linkdt, linkenddt]`. Don't fill missing fundamentals: leave that stock out of
signals that need them.

Known limitation: Compustat values may include later restatements rather than
the figures first reported.

## Checks

`pytest -m data` runs checks against the real files on your machine: checksums,
membership counts, the delisting-return identity, completeness after the rules
above, and a value-weighted universe return that tracks the Ken French market
return (correlation above 0.98). They skip where the data is absent, including CI.

## Updating the data (maintainers)

Three scripts in `scripts/`, run from the repo root. Re-pull rather than edit
files by hand, so every change is reproducible.

1. **Pull** (one Duo push; resumable, skips files already present):

   ```bash
   python scripts/pull_wrds.py            # Compustat, link, broad CRSP
   python scripts/pull_wrds.py --sp500    # the S&P 500 daily file
   ```

   The WRDS username comes from your pgpass file (or `$WRDS_USERNAME`), so it
   never goes on the command line, where shell history and session logs would
   keep it.

   Output lands in `data_cache/wrds/`. The S&P 500 file is built from the CIZ
   tables (`crsp.dsf_v2`) because the legacy `crsp.dsf` ends at 2024-12-31. The
   broad all-stocks files filter to common stocks at the source, which drops
   CRSP's delisting rows; use the S&P 500 file for returns.

2. **Document**: `python scripts/build_data_dictionary.py` writes
   `DATA_DICTIONARY.md` and `data_dictionary.csv` to `data_cache/wrds_build/`
   from CRSP's own metadata on WRDS (cached in `data_cache/wrds_meta/`, fetched
   the first time). It asserts every stated fact against the data and
   fails rather than publish a claim that stopped being true.

3. **Check, then publish**: run `pytest -m data` against the new files, then

   ```bash
   python scripts/publish_shared_data.py data_cache/wrds/<file>.parquet ...           # preview
   python scripts/publish_shared_data.py data_cache/wrds/<file>.parquet ... --apply   # publish
   ```

   It copies each file into the Drive folder, re-verifies it, and rewrites
   `SHA256SUMS`. Teammates' next `load()` picks the change up, and
   `data_version()` changes, so runs on old and new data stay distinguishable in
   the ledger. Tell the team when you publish.
