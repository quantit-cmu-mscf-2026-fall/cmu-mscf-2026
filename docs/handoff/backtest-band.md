# Testing handoff: no-trade band in the backtest

## 1. What this PR adds
A no-trade band for `capstone/backtest.py` (QUANTIT-123): `backtest_frame(..., band=)`
and the rule itself, `band_book(weights, missing, band=, hold=)`. Tests are in
`tests/test_backtest.py` (the "No-trade band" section). It does not add `band=` to
`run_backtest`, `sweep` or `candidate_returns`, and it does not touch the factor code.

## 2. Contract
- On each rebalance (every `hold`-th date from the first) a name trades to its target only
  when |target - held| > `band`, or when it has no signal (it is closed); every other name
  keeps its held weight. The book is not rescaled, so it is never levered onto the names
  that traded. A flat book opens at its target in full. A target with no position anywhere
  takes the book flat. Between rebalances the book is held.
- `band=0` (the default) is computed exactly as before: `band_book` is not called.
- The #55 idle rule and #80's per-asset costs (including #93's refusal of an unpriced
  trade) apply to a banded book unchanged.
- A negative, NaN or infinite band raises `ValueError`; so does `hold < 1`.
- On each rebalance every name is within `band` of its target. So a banded book is not
  exactly dollar-neutral, and its gross can differ from `gross`, by at most `band` per kept
  name.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | 20 band tests in `tests/test_backtest.py` | pass | `pytest -q tests/test_backtest.py -k band` |
| band=0 unchanged | `assert_frame_equal(check_exact=True)`, hold 1 and 3, per-asset rates, warm-up | identical | `pytest -q tests/test_backtest.py -k band_zero` |
| The rule by hand | 4-name book, hold 1 and 2, empty target | matches to 1e-15 | `-k "small_book or between_rebalances or goes_flat"` |
| A band wider than 1/N | 200 names, bands 0.005 to 0.06, hold 1 and 5; a 10-name book at 0.06 | each name within `band` of its target; no flat or concentrated book | `-k "wider_than or wide_band_on"` |
| Synthetic market | `synth.make_panel(500 dates, 30 assets, seed=0)`, raw and 10-day mean, hold 1 and 5, six bands | turnover and cost fall monotonically; each name within `band` of its target | `-k widens` |
| Full suite | | pass | `pytest -q -m "not network"` |

## 4. Provenance
Ported from Cal's development sandbox, backtest part only. Left out: the factor-evaluation
`TradingSpec.band` and strategy-contract tests (they need the factor stack), and a pandas
reduction change kept there for byte-identical digests. Builds on #80 (with #93) and the
`sharpe_ratio` PR. Details: the PR description.

## 5. Known limits and areas to improve
- The loop in `band_book` is per date in Python: fine for daily books of a few thousand
  dates, slow for very wide or long ones.
- Each band tried is a variant and counts as a trial; nothing here logs it.
- Reviewer checklist: no rescale after keeping names; the flat book opening in full; closing a name with no signal; the
  `hold > 1` path, where `no_signal` is forward-filled the same way with or without a band.

## 6. Public-repo safety
Synthetic data only. No data, credentials, account ids, machine paths or private notes.
