# Getting started — day one to your first merged PR

Written for the Group 008 team. Assumes you have a laptop, a CMU account, and
nothing else set up. Work through it in order; each step is a few minutes except
the ones marked ⏳, which involve waiting on someone else.

If a step fails, that is a normal part of this — ask in the async channel rather
than losing an hour. We would rather answer twice than have you blocked.

---

## Step 0 · Before the kickoff call ⏳

**Install Claude Code and use it a little.** Read
<https://code.claude.com/docs/en/overview>, install it, and point it at
something small of your own. You learn this by using it, so use it a lot. Come
to the kickoff with a feel for how it behaves, not a summary of the docs.

We provide a team Claude Max account (usage limits apply) — an individual
subscription alongside it is worth it.

**Start the WRDS registration now** (Step 3). Approval usually lands the same
day, but "usually" is doing work in that sentence, and nothing else in this list
depends on it.

---

## Step 1 · Accounts and access

At the kickoff you will sign two short documents and hand over one piece of
information:

| What | Why |
|---|---|
| Designation of Confidential Information (receipt) | The platform's internals are confidential; your own work is explicitly not. |
| Kickoff acknowledgment form | Accounts, GitHub username, optional media consent, working-data consent. |
| Your **GitHub username** | This is how we add you to the organization. Bring it. |

After that you will receive: an Arkraft account (public/synthetic data only),
the Claude Max team credentials, GitHub organization membership. For the execution leg later in the
semester, the team sets up its **own** Interactive Brokers paper account —
Quantit does not provision, hold, or fund it; we provide connection guidance
when that leg starts.

---

## Step 2 · Clone the repository and prove the environment works

```bash
git clone https://github.com/quantit-cmu-mscf-2026-fall/cmu-mscf-2026.git
cd cmu-mscf-2026

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q -m "not network"
```

You should see every test pass. **If they do not, stop here and ask** — a broken
environment silently poisons everything downstream, and it is much cheaper to
fix now than to debug a "wrong" backtest next week.

Then run the one example that matters most:

```bash
python examples/02_null_experiment.py
```

It builds a panel containing **no real signal at all**, runs the selection
procedure, and reports what survives. Read the output carefully. A naive
top-k-by-Sharpe rule finds "winners" in pure noise; a corrected procedure
correctly finds nothing. That gap is the entire project in one screen.

---

## Step 3 · Get WRDS access (do this early) ⏳

CMU subscribes, so CRSP and Compustat are already yours.

1. Register at <https://wrds-www.wharton.upenn.edu/register/> **with your
   `@andrew.cmu.edu` address** — the institutional match is what grants access.
2. Confirm via the email WRDS sends.
3. Off campus, connect to CMU full VPN first.
4. `pip install wrds`, then connect once and let it create `~/.pgpass`.

Full walkthrough, including the three CRSP conventions that trip everyone once:
**`docs/wrds_howto.md`**.

**While you wait — and this is a real option, not a consolation prize:**

```python
from capstone.sample_data import load_sample_prices, load_sample_fundamentals
prices = load_sample_prices()          # 126 business days x 40 tickers
facts  = load_sample_fundamentals()    # ticker, period_end, filed, tag, value
```

Real symbols, **synthetic prices**, with a known factor structure and a
realistic filing lag. Same frame shape as the CRSP loader, so switching later is
one line. You can build the entire pipeline on this.

---

## Step 4 · Understand the data boundary (five minutes, saves a week)

Read **`docs/data_sources.md`**. The short version:

- **We can hand you** SEC EDGAR fundamentals (government data, redistribution
  explicitly permitted), Ken French factors, and synthetic panels.
- **You fetch yourself** CRSP prices via your own WRDS credentials. We are not
  allowed to redistribute them, and neither are you — check before making any
  CRSP-derived artifact public.
- **Nobody uses** vendor-licensed data. You will never need it.

One trap to internalise now: a universe built from **today's** S&P 500 list
excludes every company that failed. That backtest is not measuring a strategy,
it is measuring the fact that you picked survivors in advance — and it will look
excellent. CRSP carries delisted names; use them.

---

## Step 5 · Read the brief and the plan, in that order

1. **The problem brief** — the mission, the three constraints, and what you
   decide. Short by design.
2. **The semester plan** — how the semester runs and what we will not hand over.
3. **`docs/reading_list.md`** — start with the four marked `[start here]`. Pull
   the rest when your own work needs them.

You do not need to have read everything before week 1. You do need to have an
opinion by week 2.

---

## Step 6 · Your first change, end to end

Do this in week 1 even if the change is trivial. The point is to exercise the
whole loop once while the stakes are zero.

```bash
git checkout -b yourname/first-change

# ... make a small change: a test, a docstring, a helper you needed ...

pytest -q -m "not network"
ruff check . && ruff format .

git add -A
git commit -m "Add a helper for X"
git push -u origin yourname/first-change
gh pr create --fill        # or open it in the browser
```

Then fill in the PR template. For anything research-shaped, the **trial count**
field is not optional — the number of configurations you tried is an input to
every correction this project runs, and it cannot be reconstructed in week 12.

A teammate reviews (the role rotates — do not route everything to one person),
CI must be green, then squash-merge. We read PRs as participants, not
gatekeepers.

Full workflow, including the deploy-by-registry pattern for the paper-trading
leg: **`CONTRIBUTING.md`**.

---

## Step 7 · Log every experiment, from the first one

```python
from capstone.runlog import log_run

log_run(
    "momentum_12_1",
    params={"lookback": 252, "skip": 21},
    seed=42,
    metrics={"sharpe": 0.41, "n_candidates": 300},
)
```

Log it **before** you look at the result. The ledger is the project's trial
count; an unlogged run corrupts every multiple-testing correction downstream.
`python -m capstone.runlog stats` shows where you stand.

The ledger is local and gitignored — it syncs to the team's private archive, not
to the public repo.

---

## Step 8 · The weekly rhythm

- **Weekly sync, 45 minutes.** You set the agenda.
- **The paper round** — every week, each of you brings a recent paper on
  autonomous scientific research (Google, Stanford, CMU, Meta, Anthropic) as a
  **one-page proposal, five minutes**: what it claims, what we would do
  differently if it is right, and the smallest experiment that would tell us.
  The following week, whoever's idea the team adopted reports what happened —
  including "no effect", which is the common answer and worth saying plainly.
  Post it as an issue with the **Paper round** template — no PR, no review, no
  waiting on us. Fill in the report-back section on the same issue a week later.
- **Topics and work items are yours to write.** Use the *Topic* template for a
  direction the team is chewing on and *Work item* for something specific
  someone will do. Edit each other's freely; the wiki and Discussions are open
  the same way. Only code goes through review — see `CONTRIBUTING.md`.
- **Around week 2** you write your own project proposal. From that point your
  document supersedes ours.

Sessions you run inside this repository are captured and synced, redacted, to
the team's **private** archive — for the team's own learning. Nothing outside
this directory is touched. See `docs/telemetry.md`.

---

## Checklist

```
[ ] Claude Code installed, used a little
[ ] Kickoff documents signed, GitHub username handed over
[ ] Repo cloned, pytest green, null experiment run and understood
[ ] WRDS registration submitted    (working on sample data meanwhile)
[ ] docs/data_sources.md read — you can explain survivorship bias
[ ] Brief and plan read; four [start here] papers identified
[ ] First PR opened, reviewed by a teammate, merged
[ ] First experiment in the run ledger
[ ] First paper round posted as an issue
```

When all of these are done you are set up. Everything after that is the actual
research — which is yours.
