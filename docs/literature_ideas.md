# Literature ideas

This file is append-only. Each new candidate anomaly should record the claim, the public source reference, the publication date, the earliest allowed test-window start, and a clear literature-sourced flag that is held to the post-publication out-of-sample bar.

---

- claim: Short-horizon reversal in high-beta names predicts excess returns following earnings announcement surprises.
- source_ref: arXiv:2402.14583
- publication_date: 2024-02-14
- earliest_allowed_test_start: 2024-02-15
- source_kind: literature
- literature_sourced: true
- held_to_post_publication_oos_bar: true
- note: Open-access q-fin preprint proposing a short-horizon reversal anomaly. This candidate is literature-sourced, and it may only be tested after publication; any pre-publication mining or leakage window is invalid under the project’s post-publication out-of-sample rule.

---

- claim: Low-volatility stocks earn higher risk-adjusted returns than high-volatility stocks after controlling for market beta and sector exposure.
- rationale: Investors underappreciate safe assets in the presence of leverage constraints, benchmark crowding, and preference for lottery-like payoffs. A persistent low-volatility premium is plausible when risk is mispriced and when institutional mandates force concentrated long exposure to high-beta names.
- construction_sketch: Form a volatility-sorted cross-section from sample prices, then residualize each stock return against market and sector factors from the synthetic panel or French factor files; evaluate whether the low-vol portfolio earns a positive alpha after adjustment.
- source_flag: economic-reasoning

---

- claim: High-quality firms with stronger profitability and lower balance-sheet leverage earn superior future returns once sector and market beta are controlled for.
- rationale: Financially stronger firms can fund investment without distress risk and may be better able to convert margins into durable earnings. Quality should be rewarded when the market underprices the persistence of efficient operating models.
- construction_sketch: Use prepared fundamentals such as Revenues, NetIncomeLoss, Assets, and StockholdersEquity to build a quality score from operating profitability and book-to-equity; combine with sample prices and French factors to assess whether quality sorts survive cross-sectional controls.
- source_flag: economic-reasoning

---

- claim: Industry winners continue to outperform industry losers over intermediate horizons after accounting for market and sector factor exposure.
- rationale: Slow-moving capital allocation, sector-specific demand shocks, and gradual earnings revisions create momentum in industries even when idiosyncratic noise is heavy. A persistent industry trend can survive if information diffuses unevenly across sectors.
- construction_sketch: Use 12- or 49-industry portfolio returns from the French data to construct winner-minus-loser portfolios across 1-, 3-, and 12-month horizons, then test whether they retain spread after controlling for market beta and SMB/HML exposures.
- source_flag: economic-reasoning

---

- claim: Firms with strong gross profitability and conservative balance sheets earn a premium relative to weak-quality peers in the same sector.
- rationale: High-quality firms have lower default risk and more durable margins; investors may not fully price the persistence of their cash flow generation, especially when sector-level risk is high and market beta is noisy.
- construction_sketch: Build a profitability signal from fundamentals scaled by assets or revenues, cross-sectionally sort sample names, and compare the high- versus low-quality portfolios while partialling out the market factor and sector factor structure.
- source_flag: economic-reasoning

---

- claim: Stocks with stronger recent earnings revisions or faster operating-growth signals outperform peers with weaker revisions over the next month or quarter.
- rationale: Firms that deliver accelerating earnings and cash generation often see expected cash-flow revisions that are slow to be incorporated into prices. Information frictions and analyst underreaction create a window for profitable spread capture.
- construction_sketch: Use quarterly fundamentals and lagged filings to form an earnings-growth or profitability-change signal, then combine it with the price panel and sector exposures to evaluate whether revision-strength predicts next-period returns.
- source_flag: economic-reasoning

---

- claim: Sector-neutral value strategies based on book-to-price or asset-to-price ratios generate positive excess returns when rebalanced after accounting for market beta.
- rationale: Price pressure and slow arbitrage can leave some sectors temporarily mispriced relative to their capital structures; value can reappear when financial distress fears and book-value uncertainty are slowly resolved.
- construction_sketch: Use prepared fundamentals and sample prices to build a sector-neutral book-to-price or asset-to-price signal, then compare the long-short spread against the market factor and a sector-adjusted benchmark.
- source_flag: economic-reasoning

---
