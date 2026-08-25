# Quantit × CMU MSCF Capstone — Team Repository (46-983, Fall 2026)

This is the team's working repository: where the semester's code, experiments,
and decisions land.

It begins as a copy of the starter kit — plumbing so that week 1 goes to
research rather than to data loaders. None of that scaffolding is required.
Keep it, rewrite it, or delete it; replacing all of it with something better is
a good outcome. What you build on top of it is the project.

> **New to the project? Start with [`GETTING_STARTED.md`](GETTING_STARTED.md)** —
> day one to your first merged PR, in order, including the parts that involve
> waiting on someone else. The [open issues labelled
> `good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
> are that first week, one at a time.

## How we work here

Review protects the code that runs. It is not a gate on thinking out loud —
putting one there would only move the thinking somewhere nobody can see it.

**Goes through a pull request**: code, tests, `deploy/strategies.yml`, and the
docs describing how the code works. Green CI, one teammate's review, squash-merge.

**Post it yourself — no PR, no review, no waiting on us**:

| Where | For |
|---|---|
| [Issues → *Paper round*](../../issues/new/choose) | Your weekly one-pager, and the report-back a week later on the same issue |
| [Issues → *Topic*](../../issues/new/choose) | A research direction the team is shaping. Edit each other's freely |
| [Issues → *Work item*](../../issues/new/choose) | Something specific someone will do |
| [Wiki](../../wiki) | Notes, references, derivations, write-ups |
| [Discussions](../../discussions) | Questions and half-formed ideas |

The topics and work items are the team's own map of this project — you write it,
and you rewrite it as your understanding changes. Around week 2 your project
proposal supersedes ours entirely.

Full workflow, including what a reviewer is actually looking for:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## What is in here

| Module | What it does |
|---|---|
| `capstone/data.py` | Loads public factor and price data, with an on-disk cache |
| `capstone/sample_data.py` | Six-month synthetic panel — work before WRDS access arrives |
| `capstone/wrds_loader.py` | CRSP prices through your own CMU credentials |
| `capstone/synth.py` | Generates panels whose true signal set is known by construction |
| `capstone/backtest.py` | Cross-sectional backtest: weights → returns → summary statistics |
| `capstone/evaluate.py` | Multiple-testing corrections and the deflated Sharpe ratio |
| `capstone/runlog.py` | The experiment ledger — every run, and therefore the trial count |
| `examples/01_hello_data.py` | Load real data, confirm access works |
| `examples/02_null_experiment.py` | Run the pipeline on data with no signal in it |
| `tests/` | Tests that fail if the above is wrong |

Guides live in `docs/`: [data boundaries](docs/data_sources.md),
[WRDS setup](docs/wrds_howto.md), [reading list](docs/reading_list.md),
[working-data capture](docs/telemetry.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/01_hello_data.py      # confirms network + cache work
pytest -q                              # should be green before you change anything
```

Python 3.11+. Dependencies are pandas, numpy, scipy, statsmodels — nothing exotic.

## The one experiment worth running first

```bash
python examples/02_null_experiment.py
```

It builds a panel containing **no real signal at all**, generates a few hundred
candidate strategies on it, and reports the best Sharpe ratio found. The number
that comes back is large. Nothing produced it but chance.

Whatever selection procedure you end up building, this is the cheapest test of
it: run it where the right answer is "nothing here" and see whether it says so.
A procedure that cannot return "nothing" is not a validation procedure — it is a
ranking.

## Data sources

All public. No account, no licence, no credentials.

| Source | Contents | Notes |
|---|---|---|
| Ken French Data Library | Daily/monthly factor returns, industry and characteristic portfolios | Fetched over HTTPS, no key |
| Open Source Asset Pricing (OSAP) | ~200 published cross-sectional predictors | Large download; see `data.py` docstring |

For single-name prices there are two routes, and the full picture is in
**`docs/data_sources.md`** (what we can redistribute vs what you fetch yourself):

- **CRSP via WRDS** — CMU subscribes, so this is already yours and it is the
  right substrate: delisted names included, so a backtest is not quietly
  measuring survivorship. Loader is `capstone.wrds_loader`; setup and the CRSP
  conventions that trip everyone once are in **`docs/wrds_howto.md`**. Note that
  CRSP-derived data usually cannot be republished — check before making any
  artifact public.
- **`capstone.sample_data`** — no account needed. Six months of daily bars for
  40 large-cap tickers with a known factor structure, plus fundamentals with a
  realistic filing lag. Real symbols, **synthetic prices**. Same frame shape as
  the CRSP loader, so switching later is one line.

`data.py` caches to `./data_cache/` (gitignored). First call downloads; later
calls read from disk.

## What the tests are for

`pytest` here is not ceremony. `tests/test_evaluate.py` checks that the
multiple-testing corrections actually control what they claim to on data where
the truth is known. If you modify `evaluate.py`, those tests are what tell you
whether you broke the guarantee.

The same idea applies to whatever you build next: a test that cannot fail is not
evidence. Break the thing it protects and confirm the test goes red.

## Conventions

Returns are simple (not log), decimal (not percent) — Ken French publishes
percent, and `data.py` converts on load. Panels are `DataFrame` indexed by date
with one column per asset. Signals use the same shape as prices, and a signal at
date `t` may only use information available at `t`; `backtest.py` shifts
positions by one period for this reason.

Sharpe ratios are annualised with 252 for daily data. If you change the horizon,
change the annualisation with it — a 24-day window annualised at 252 produces
numbers that are not wrong so much as meaningless.

## Getting help

Ask us anything factual or methodological. We will generally not answer "what
should we do next?", because that decision is the part of this semester that is
yours. That is intent, not evasion.

## Working data & telemetry

Sessions run inside this repo are captured and synced — credential-redacted —
to the team's **private** archive, along with the experiment ledger; the
public repo carries deliberately PR'd code only. Project-scoped, private to
team + mentors, removable on request: see `docs/telemetry.md`.
