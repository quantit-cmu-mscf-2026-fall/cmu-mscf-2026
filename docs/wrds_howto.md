# WRDS: getting CRSP prices with your own credentials

CMU subscribes to WRDS, so CRSP is already yours. This page gets you from
nothing to a price panel. If you do not have an account yet, skip to
[Working without WRDS](#working-without-wrds) — nothing here blocks day one.

## Why CRSP and not a free price API

Free feeds give you today's survivors. CRSP gives you the historical universe:
delisted companies are present for the period they existed, with delisting
returns attached. A backtest run on today's S&P 500 members quietly excludes
Enron, Lehman, Bear Stearns and several hundred quieter failures — it is not
measuring your strategy, it is measuring the fact that you picked survivors in
advance, and it will look excellent. See `docs/data_sources.md`.

CRSP is also the reason we do not ship prices: it is licensed to CMU for your
academic use, not to us for redistribution. The loader is ours; the data comes
through your account.

## 1. Get an account

1. Go to <https://wrds-www.wharton.upenn.edu/register/> and register **with your
   `@andrew.cmu.edu` address** — the institutional match is what grants access.
2. WRDS emails you a confirmation link. Approval is usually same-day but can
   take longer; start this early.
3. Off campus, connect to CMU full VPN first. WRDS checks the network path.

CMU's subscription includes CRSP and Compustat. If a query returns a permission
error, that table is outside the subscription — ask a business librarian rather
than assuming the query is wrong.

## 2. Install and connect

```bash
pip install wrds
```

```python
import wrds
conn = wrds.Connection(wrds_username="YOUR_WRDS_ID")
```

The first connection offers to create `~/.pgpass`. **Say yes.** It stores the
credential so you never type the password again — which matters because you will
reconnect constantly, and a password typed into a notebook has a way of ending up
in a transcript or a commit.

If you create it by hand, permissions are enforced by the driver:

```bash
chmod 0600 ~/.pgpass
```

Never put your WRDS password in code, in a config file inside the repo, or in a
notebook cell. The repository is public.

## 3. Explore before you query

```python
conn.list_libraries()                    # every library, subscribed or not
conn.list_tables("crsp")                 # tables in CRSP
conn.describe_table("crsp", "dsf")       # columns of the daily stock file
```

`crsp.dsf` is the daily stock file, `crsp.dsenames` the time-varying names/
listing file, `crsp.dsp500list` the S&P 500 membership spells.

## 4. Pull a panel with this kit

```python
from capstone import wrds_loader

conn = wrds_loader.connect("YOUR_WRDS_ID")

daily = wrds_loader.cached_crsp_daily(
    conn, "2015-01-01", "2025-12-31", name="us_2015_2025"
)
prices = wrds_loader.to_price_panel(daily)   # date x ticker

members = wrds_loader.sp500_members(conn, "2015-01-01", "2025-12-31")
conn.close()
```

`cached_crsp_daily` writes to `data_cache/`, which is gitignored — CRSP data must
not enter the public repo. Delete the parquet file to refresh.

## 5. Three CRSP conventions that bite everyone once

- **`prc` can be negative.** A negative price means there was no trade that day
  and CRSP stored the bid/ask midpoint instead. Take `abs()` for a level, but
  remember those rows are quotes, not transactions.
- **`ret` already includes dividends and split adjustment.** Do not rebuild
  returns from `prc` unless you also apply `cfacpr`. Mixing the two produces
  phantom jumps on split dates that look like alpha.
- **Filter share codes and exchange codes.** `shrcd in (10, 11)` restricts to
  ordinary common shares of US-incorporated firms; without it your "US equity"
  cross-section quietly contains ADRs, REITs and closed-end funds. Our loader
  applies this by default.

A fourth, subtler one: **tickers are reused**. CRSP's stable identifier is
`permno`. A panel keyed on ticker can collide across decades — fine for a
six-month study, wrong for a twenty-year one.

## Working without WRDS

The kit ships a synthetic stand-in with the same shape, so the pipeline can be
built and tested while your account is pending:

```python
from capstone.sample_data import load_sample_prices, load_sample_fundamentals

prices = load_sample_prices()             # 126 business days x 40 tickers
facts = load_sample_fundamentals()        # tidy: ticker, period_end, filed, tag, value
```

Six months of daily bars for 40 large-cap tickers, generated from a market
factor plus sector factors plus idiosyncratic noise. Real symbols, **synthetic
prices** — never present these numbers as market data.

Two things it is good for beyond unblocking you:

- The factor structure is **known by construction**, so you can check that your
  pipeline recovers something whose answer you already have before pointing it
  at data where you do not.
- The fundamentals carry a realistic **30-75 day filing lag** between
  `period_end` and `filed`. Join on `filed`. Joining on `period_end` trades on
  numbers nobody had yet, and produces a beautiful, false backtest — the same
  trap that makes point-in-time discipline necessary on the real data.

Swapping to CRSP later is one line: `load_sample_prices()` becomes
`wrds_loader.to_price_panel(...)`. Everything downstream is unchanged.
