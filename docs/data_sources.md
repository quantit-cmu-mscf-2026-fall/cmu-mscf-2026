# Data: what we can hand you, and what you fetch yourself

The agreement between Quantit and Carnegie Mellon limits this project to
**publicly available data, data CMU already licenses for academic use (WRDS),
and synthetic data**. We provide no vendor-licensed data, and you are never
expected to access any.

That constraint is less limiting than it sounds, but it does split the world in
two, and the split is not "free vs paid" — it is **redistributable vs not**.

| | We can prepare and ship it | You fetch it yourself |
|---|---|---|
| Why | the licence permits redistribution | freely accessible, but redistribution is not granted |
| Examples | SEC EDGAR, Ken French, synthetic panels | CRSP/Compustat via WRDS, most free price APIs |

## Fundamentals — SEC EDGAR (we ship this)

The SEC states that "All Government-created content on sec.gov and EDGAR public
filing content are free to access and reuse." That is an explicit
redistribution grant, which is why this is our primary fundamentals source and
why we can hand you a prepared panel rather than a download script.

- **Financial Statement Data Sets** — quarterly ZIPs of the face financials of
  every XBRL filing (`sub.txt`, `num.txt`, `pre.txt`, `tag.txt`). Roughly 2010
  to present; by 2023 the archive held 120M+ data points across 2GB+ of zips.
- **Point-in-time by construction.** The data is "as filed" — uncorrected,
  including amendments as separate submissions. `sub.adsh` (accession number)
  carries the acceptance/filing date, which is the date the market could know
  the number. The period-end date is *not* that date, and using it is the
  single most common way to manufacture fake alpha in a fundamentals backtest.
- **Restatements are visible, not hidden.** The same fiscal period appears
  multiple times with different values across submissions. This is a feature:
  you can measure what a restatement-blind backtest would have claimed.
- **XBRL APIs** (`data.sec.gov`: `companyfacts`, `companyconcept`, `frames`)
  for per-company pulls. Rate limit is 10 requests/second and a declared
  `User-Agent` header is required — undeclared automated traffic gets blocked.

⚠️ One trap worth knowing: in December 2024 the SEC **reprocessed** the historical
datasets to include only points rendered on the primary statements, adding a
`segments` field to `num`. Vintages downloaded before that date do not match
what you download today — a point-in-time problem inside the point-in-time
source. Record which vintage you used.

## Market data — S&P 500 prices

This is where the redistribution line bites. Free price APIs (Yahoo/yfinance,
Stooq, and most aggregators) are freely *accessible* but do not grant us the
right to republish a prepared dataset — the underlying exchange data is
separately licensed regardless of what the aggregator's own terms say. So:

**Preferred: CRSP via WRDS.** CMU subscribes; you register with your CMU email
and connect through full VPN off campus. CRSP is the academic standard for US
equity prices — properly adjusted, delisting returns included, and PERMNO
identifiers that survive ticker changes. It is better than anything free, it is
already yours, and it is explicitly allowed by the agreement. We supply loader
code (`capstone.wrds_loader`); the data comes through your own credentials, and
CRSP-derived artifacts generally may not be published — check before making any
output public. Setup and the three CRSP conventions that trip everyone once:
**`docs/wrds_howto.md`**.

**No account yet? `capstone.sample_data`.** Six months of daily bars for 40
large-cap tickers, same frame shape as the CRSP loader output, plus quarterly
fundamentals carrying a realistic 30-75 day filing lag. Real symbols,
**synthetic prices** — build and test the whole pipeline on day one, then swap
one line when your account arrives.

**Fallback: a public API you pull yourself.** If you would rather not depend on
WRDS, use any free provider under your own acceptance of its terms. Our starter
kit will read whatever you produce as long as it lands in the standard frame
(date index, ticker columns, adjusted close). We do not ship the data.

**Already in the kit: Ken French's data library** — daily and monthly factor
returns and industry portfolios, redistributable, no credentials. Not a
substitute for single-name prices, but enough to build and validate an entire
cross-sectional pipeline before you touch anything harder.

## The trap nobody sees until week 9: survivorship bias

If you build a panel from **today's** S&P 500 list, every company in it survived
to 2026. Enron, Lehman, Bear Stearns, and several hundred quieter failures and
acquisitions are simply absent. A backtest on that universe is not measuring a
strategy — it is measuring the fact that you selected winners in advance, and it
will look excellent.

Fixes, in order of quality:

1. **CRSP** carries the full historical universe including delisted names and
   delisting returns. This is the real answer, and you have it.
2. **Reconstruct membership** from the Wikipedia "List of S&P 500 companies"
   change table (CC BY-SA, so a derived file must be attributed and share-alike).
   Workable back roughly two decades, with errors — treat it as a good-faith
   approximation and say so in your write-up.
3. **Fix the universe and admit it.** A backtest on today's members, clearly
   labelled as survivorship-biased, is honest. The same backtest presented as a
   result is not.

Whatever you choose, state it. "We used the current constituent list" belongs in
the methodology section, not in a footnote discovered by a reviewer.

## Synthetic data

`capstone.synth` generates panels with a **known** number of real signals. Real
data cannot tell you whether your discovery procedure works — you do not know
the right answer. Synthetic data can. Build the control arm before you trust the
instrument.

## Summary

- Fundamentals: **SEC EDGAR**, prepared by us, point-in-time, redistributable.
- Prices: **CRSP via WRDS** with your own credentials (preferred), or any public
  API you pull yourself.
- Factors/industries: **Ken French**, already in the kit.
- Ground truth: **synthetic**, already in the kit.
- Never: vendor-licensed data. We will not provide it and you do not need it.
