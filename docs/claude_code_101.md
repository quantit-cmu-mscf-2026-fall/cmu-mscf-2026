# Claude Code 101

- **Audience**: capstone students. **Read before**: the Thursday session (Sep 10).
- **Time**: 20 minutes to read, and you should have the tool open while you do it.
- **Companions**: *GitHub, A to Z* (the map) and *Working with Claude Code on GitHub* (the three worked samples). This page is the tool itself.

Facts here were checked against the vendor documentation at <https://code.claude.com/docs> on
2026-09-07. When this page and the docs disagree, the docs win — and tell us, so we fix the page.

---

## 1 · What it actually is

Claude Code is an **agentic** coding environment, not a chat window with a code theme. The
difference decides how you use it: a chat answers and waits, an agent reads your files, runs
commands, edits code, runs the tests, reads the failure, and tries again — while you watch,
redirect, or walk away.

That has one consequence you should internalise before your first session:

> **The agent stops when the work *looks* done.** If it has no check it can run, "looks done" is
> the only signal available, and you become the verification loop.

Everything else in this page is downstream of that sentence.

---

## 2 · Install, log in, first session

```bash
# macOS, Linux, WSL
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

claude --version     # prints a version followed by (Claude Code)
```

Then, from inside a project directory:

```bash
cd cmu-mscf-2026
claude
```

The first run asks you to log in through the browser. Use the **team Claude Max account** we
provided; `/login` inside a session switches accounts later. On Windows, installing
[Git for Windows](https://git-scm.com/downloads/win) is recommended so the agent gets a Bash shell.

Your first three prompts in a repository you don't know should be questions, not orders:

```text
give me an overview of this codebase
what does capstone/evaluate.py do, and who calls it?
explain how run_backtest avoids lookahead
```

This is not a warm-up exercise. Asking the agent the questions you would ask a senior teammate
is the fastest way to onboard onto a codebase, and it is a documented use of the tool.

---

## 3 · The loop you will run a hundred times

```
task  →  plan  →  edit  →  run the check  →  read the evidence  →  commit  →  PR  →  review
```

**Plan before editing** when the change touches several files or you are unsure of the approach.
Press `Shift+Tab` until the status bar reads `plan mode on`, or start the session with:

```bash
claude --permission-mode plan
```

In plan mode the agent reads and proposes but does not write to disk. `Ctrl+G` opens the plan in
your editor so you can correct it before approving. For a one-line change, skip planning — if you
could describe the diff in one sentence, a plan is overhead.

**Give it a check it can run.** In this repository the checks already exist, and they are exactly
what CI runs:

```bash
ruff check . && ruff format --check . && pytest -q -m "not network"
```

So the useful prompt is not *"add a momentum baseline"* but:

```text
Add a 20-day cross-sectional momentum baseline in research/, using
capstone.sample_data.load_sample_prices and capstone.backtest.run_backtest.
Write a test that fails without it. Then run ruff and pytest and fix what breaks.
```

**Ask for the evidence, not the summary.** "Tests pass" from the agent is a claim; the pasted
`pytest` output is evidence. Get in the habit of asking for the output. It is faster to read the
evidence than to re-run the check yourself.

---

## 4 · Context is the resource you are actually managing

The context window holds the whole conversation: every message, every file read, every command
output. It fills fast, and **performance degrades as it fills** — the agent starts forgetting
earlier instructions and making mistakes it would not have made at the start.

| Situation | What to do |
|---|---|
| Moving to an unrelated task | `/clear` — reset the context entirely |
| Long session, still one task | `/compact Focus on the backtest changes` |
| You corrected the same mistake twice | `/clear`, then write a better prompt that includes what you learned |
| Wrong turn, want the code back | `Esc` to stop · `Esc Esc` or `/rewind` to restore a checkpoint |
| Big exploration you don't want in context | `use a subagent to investigate how the sweep function counts trials` |
| Want to check what loaded | `/context` |

Two of these are worth spelling out.

**Subagents.** A subagent explores in its *own* context window and reports back a summary, so
reading forty files costs you a paragraph instead of forty file reads. Use it for "how does X
work" questions in unfamiliar code, and for an independent review of a diff.

**Rewind.** Every prompt creates a checkpoint, and the agent snapshots files before editing them.
This means you can tell it to try something risky and roll back if it fails. One caveat that
matters in this project: **checkpoints only track changes made through the agent's file-editing
tools** — anything a Bash command did is not captured. Checkpoints are not a substitute for git.

---

## 5 · CLAUDE.md — the file that makes the agent behave like a team member

`CLAUDE.md` is read at the start of every session. This repository has one already; open it. It
is why the agent knows to keep the ledger, run the lint, and not commit secrets.

Rules of thumb that come straight from the vendor documentation, and that we enforce here:

- **Keep it short — under about 200 lines.** A bloated file gets ignored *in parts*, which is the
  worst failure mode because you cannot tell which parts.
- For each line ask: *would removing this cause a mistake?* If not, cut it.
- Include what the agent cannot infer: commands, conventions, gotchas, repository etiquette.
  Exclude anything it can learn by reading the code.
- `/init` generates a starter file; `/memory` lists and opens the memory files; `/context`
  confirms what actually loaded this session.
- It is checked into git, so **improving it is a pull request like any other** — and it is one of
  the highest-leverage PRs you can open, because it changes every future session for all five
  of you.

Related, and worth knowing before week 3: instructions in `CLAUDE.md` are *advisory*. If something
must happen every time with no exceptions, that is a **hook** — a script the tool runs at a fixed
point in its lifecycle (`.claude/settings.json`, browse with `/hooks`). This repository already
uses one: session capture. You will write one yourselves later in September.

---

## 5b · Harnessing — the five mechanisms, and why you keep tending them

"Harness" is the word for everything around the model that makes it behave like a member of this
team rather than a clever stranger: what it reads at startup, what it may do, what it is stopped
from doing, and what it knows only when it needs to. Five mechanisms, two axes.

| Mechanism | Who writes it | When it loads | Advisory or binding | Use it for |
|---|---|---|---|---|
| `CLAUDE.md` | you | every session | advisory | short, always-true facts: commands, conventions, hard "never" rules |
| Auto memory | the agent | every session | advisory | your corrections and preferences, accumulated without you writing them |
| `.claude/rules/*.md` | you | every session, or only for matching `paths:` | advisory | conventions that belong to one part of the repository |
| Skills | you | when relevant, or on `/name` | advisory | procedures and domain knowledge too long to keep in context always |
| Hooks | you | at a lifecycle event | **binding** | things that must happen every time, including blocking an action |

Read the two axes off the table. **Always-loaded vs on-demand** decides context cost: `CLAUDE.md`,
memory and unscoped rules are paid every session, so they must be short; skills and path-scoped
rules are paid on use, so they can be long. **Advisory vs binding** decides trust: four of the five
shape what the agent *tends* to do; only a hook decides what it *can* do.

**Why it needs tending.** A harness decays, for four reasons that all look like "the agent got
worse": the code moved and a rule now points at a function that no longer exists; `CLAUDE.md`
grew past the point where every line is read; auto memory saved a correction that was right in
July and wrong in September; a skill froze while the note it came from advanced. None of these
announce themselves. The weekly habit is prune → promote → measure: prune what the agent already
does right without the line, promote what you keep re-explaining up the ladder (chat → memory →
`CLAUDE.md` → rule → skill → hook), and after any change watch whether behaviour actually shifted.
The test of a harness line is: would removing it cause a mistake?

**Four practices that hold up**, all from the public documentation or this project:

1. **A "never" that is ignored twice becomes a hook.** The ledger rule ("log every run") lived in
   `CLAUDE.md` and was skipped; as a `PreToolUse` hook that refuses a backtest without a `run_id`, it
   cannot be. Instructions are for judgment calls; hooks are for invariants.
2. **Prune `CLAUDE.md` until every line earns its place.** A file that doubles in length halves
   its adherence. Move procedures into skills, scope conventions into `paths:` rules, and keep the
   root file to what applies to every session. `/doctor` proposes cuts; `/context` shows what loaded.
3. **A skill's description is the product.** The body is read only after invocation; the
   description is what the agent matches against. Write it as *what it does and when to use it*,
   then test from a fresh session without naming the skill. If it does not fire, rewrite the
   description, not the body.
4. **Audit auto memory like you audit a teammate's notes.** Open `/memory` monthly. Delete what is
   stale, correct what is wrong, and move anything that should be a rule into `CLAUDE.md` where you
   control it. The agent's model of you is only as good as the last correction it saved.

---

## 6 · Prompting, concretely

The pattern that separates a useful prompt from a vague one is *specificity about the source and
the check*.

| Weaker | Stronger |
|---|---|
| "add tests for synth.py" | "write a test for `candidate_frames` covering the empty-panel case, no mocks" |
| "why is the sweep slow?" | "profile `capstone.backtest.sweep` on a 300-candidate panel and show me where the time goes" |
| "fix the backtest" | "positions look shifted by a day. Check `run_backtest`, write a failing test that reproduces it, then fix it" |
| "make it better" | "@capstone/evaluate.py — is `deflated_sharpe_ratio` using the trial count I pass, or the number of rows? Show the line" |

Other levers worth using early:

- **`@` references**: `explain the logic in @capstone/runlog.py` pulls the file in directly.
- **Paste images**: drag a plot or an error screenshot into the prompt.
- **Let it interview you** for a bigger piece of work: *"I want to build X. Interview me in detail,
  ask about edge cases and tradeoffs, then write a spec to SPEC.md."* Then start a **fresh
  session** to implement the spec.

---

## 7 · Sessions do not vanish

```bash
claude --continue          # resume the most recent session in this directory
claude --resume            # pick from a list
/resume                    # same picker, from inside a session
```

Name a session with `/rename` and treat it like a branch: one workstream, one session. When a
paper-round experiment spans two evenings, resuming beats re-explaining.

For parallel work, `claude --worktree feature-name` gives the session its own git checkout so two
sessions do not collide.

---

## 8 · Failure patterns you will hit this month

The five below are documented and common. Recognising one saves an hour.

1. **The kitchen-sink session.** One task, then an unrelated question, then back. → `/clear`.
2. **Correcting over and over.** Two failed corrections means the context is polluted with failed
   approaches. → `/clear` and rewrite the prompt.
3. **The over-specified CLAUDE.md.** Too long, so half of it is ignored. → prune it.
4. **The trust-then-verify gap.** Plausible code, unhandled edge cases. → if you cannot verify it,
   do not ship it.
5. **Infinite exploration.** "Investigate the codebase" with no scope fills the window. → scope it,
   or delegate to a subagent.

And one that belongs to *this* project specifically:

6. **The unlogged run.** An agent that runs an experiment without `capstone.runlog.log_run`
   quietly lowers the bar your best candidate is later judged against. The agent will do this
   unless the instruction is somewhere it cannot miss. That is why it is in `CLAUDE.md`, and why
   we make it a hook later in the month.

---

## 9 · What the agent is not

Say this out loud once, because the semester depends on it.

- **It agrees with you.** Ask "is this signal real?" after showing it a promising backtest and you
  will usually get encouragement. Ask it instead to run the procedure on data you know is
  signal-free and report what survives. Design your questions so that *"no"* is a reachable answer.
- **It is not a source of truth for numbers.** A number in its prose is a claim; a number in the
  test output or a written artifact is a measurement. Move every number you intend to defend from
  the first category into the second.
- **It will not remember what you did not write down.** Ledger, PR description, issue, CLAUDE.md.
  Nothing else survives the session.

---

## 10 · One-page command reference

### Shell

| Command | Does |
|---|---|
| `claude` | start an interactive session |
| `claude "task"` | start with an initial prompt |
| `claude -p "query"` | one-off, non-interactive; good in scripts |
| `claude --continue` / `-c` | resume the most recent session here |
| `claude --resume` / `-r` | pick a session from a list |
| `claude --permission-mode plan` | start in plan mode |
| `claude --worktree <name>` | isolated parallel session on its own branch |

### In-session

| Command | Does |
|---|---|
| `/help` | list commands |
| `/clear` | reset the context |
| `/compact <instructions>` | summarise the conversation, keeping what you name |
| `/context` | what is loaded right now, including memory files |
| `/init` | generate or improve `CLAUDE.md` |
| `/memory` | list and open memory files |
| `/rewind` (or `Esc Esc`) | restore a previous conversation or code checkpoint |
| `/permissions` | pre-approve tools you trust |
| `/hooks` | browse configured hooks |
| `/code-review` | review the current diff in a fresh subagent |
| `/resume`, `/rename` | session management |
| `Shift+Tab` | cycle permission modes (including plan mode) |
| `Esc` | stop the agent mid-action, keeping context |

**Ask the tool about itself.** It has its own documentation: *"how do hooks work?"*,
*"what are the limitations of Claude Code?"*, *"how do I use MCP?"* — accurate answers, and
faster than searching.

---

## 11 · Before Thursday

1. `claude --version` prints a version.
2. You have run one session inside `cmu-mscf-2026` and asked it three questions about the code.
3. You have read this repository's `CLAUDE.md` and can say what it forbids.
4. You know what `/clear` does and when to press `Shift+Tab`.

Bring one question about the tool to the session. "It did something weird and I don't know why" is
an excellent question and the most useful thing you can bring.
