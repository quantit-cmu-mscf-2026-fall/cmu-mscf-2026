---
name: shared-data
description: How to load and use the team's CRSP and Compustat data in cmu-mscf-2026 through capstone.shared_data - files, columns, universe, returns with delistings, spreads, point-in-time fundamentals, permno/permco keys and data_version. Use whenever code reads market or accounting data, builds a signal, or computes returns or costs.
---

# Shared data

All CRSP/Compustat access goes through `capstone.shared_data` (`sd`). Full rules:
`docs/shared_data.md`; column meanings: `data_dictionary.csv` in `sd.cache_dir()`.

## Files (`sd.load(name, ...)`, name without `.parquet`)

| Name | Contents |
|---|---|
| `sp500_daily_1990_2025` | Daily CRSP rows for every stock ever in the S&P 500, whole 1990-2025 history, with `in_sp500` and `in_universe` |
| `sp500_spells_1990_2025` | Membership spells: `permno`, `mbrstartdt`, `mbrenddt` |
| `comp_fundq` / `comp_funda` | Compustat quarterly (with `rdq`) / annual |
| `comp_company` | Compustat company header |
| `ccm_link` | CRSP-Compustat link |

## Loading

```python
from capstone import shared_data as sd

daily = sd.load(
    "sp500_daily_1990_2025",
    columns=["date", "permno", "permco", "dlyret", "dlyretmissflg", "dlydelflg",
             "delactiontype", "dlybid", "dlyask", "in_universe"],
    start="2000-01-01",
    end="2020-12-31",          # search period only; see pipeline-boundaries
)
version = sd.data_version()    # put this in every log_run params
```

- Always pass `columns`: the full daily file is ~2.6 GB in memory. `sd.daily_returns` needs
  `permno, date, dlyret, dlyretmissflg, dlydelflg, delactiontype`; `sd.quoted_spread` needs
  `permno, date, dlybid, dlyask, in_universe` (a missing column raises `KeyError`).
- Always pass `end` no later than `2020-12-31` for discovery and stages 1-3.
- `load` syncs from the Drive first; a missing Drive falls back to the local copy with a
  warning. Only "no Drive and no local copy" raises `SharedDataError`.

## The rules, as functions

| Need | Do | Never |
|---|---|---|
| Which stocks to hold | filter positions on `in_universe` | filter the rows used for returns |
| Returns | `sd.daily_returns(daily)` over every row | compound `delret` onto `dlyret`; drop rows that left the index |
| Trading costs | `sd.quoted_spread(daily)` | a flat bps guess; trust pre-2000 costs without a caveat |
| Fundamentals | `sd.available_from(fund, trading_days)` (`report_col=None` for `comp_funda`) | use a quarter on or before its `rdq`; fill missing fundamentals |
| Linking | `sd.primary_links(link)`, date within `[linkdt, linkenddt]` | join on ticker or name |
| Identity | key on `permno`; group by `permco` for company-level signals | key on ticker |

Why each matters: dropping exited rows hides delisting losses (Shumway 1997); `dlyret`
already contains the delisting return on the `dlydelflg == "Y"` row; using a report on its
own date is look-ahead because reports land after the close; nine companies have two share
classes in the universe.

## Checklist for any data-reading change

1. Loaded through `sd.load` with an explicit column list and an `end` in the search period.
2. Returns from `sd.daily_returns`, costs from `sd.quoted_spread`, fundamentals gated by
   `sd.available_from`.
3. `sd.data_version()` recorded in the run's `log_run(params=...)`.
4. No parquet, cache path or WRDS credential written into the repository.
