# Working data & telemetry — what is collected, where it goes, and why

This project deliberately keeps a record of how the work gets done — not just
what got merged. The record is **private to the team and the Quantit mentors**;
the public repositories carry code that was deliberately PR'd, and nothing else.

## What is collected

| Data | Where it is produced | Where it lives |
|---|---|---|
| Code, PRs, reviews, issues | normal git/GitHub work | public work repo |
| Agent session transcripts (including your prompts) | `.claude/settings.json` hook → `.sessions/` on your laptop | **private** `session-archive` repo, after redaction |
| Experiment runs (name, params, seed, metrics, git SHA) | `capstone.runlog.log_run` → `experiments/runs.jsonl` (gitignored) | **private** `session-archive/ledger/` |
| PR/issue/review data, consolidated | nightly workflow in the archive repo | **private** `session-archive/github-export/` |

## What is NOT collected

Anything outside this repository directory. The capture hook lives in this
repo's `.claude/settings.json`, so it fires only for sessions run **inside
this project**. Other sessions, other projects, and everything else on your
machine are untouched.

## Why we do this

1. **It is the team's learning corpus first.** You can read how each of you
   actually drives the agents — replay what worked, see the prompt that
   produced the PR that survived review. This is the fastest known way to get
   better at agentic work: study real traces, including your own.
2. **The run ledger is the trial count.** Every multiple-testing correction
   this project runs takes the number of attempts as input. A ledger that
   captures every run — automatically, before you see the result — is what
   makes the final claims honest.
3. **The mentors study how agentic research is really done.** How a team of
   five learns to steer agents over fourteen weeks is itself research, and
   the corpus is part of what this collaboration produces.

## Privacy boundary

- The archive repo is **private**: the five of you plus the Quantit mentors.
  Nothing in it is published or shared further without your consent — this is
  what you sign in section E of the kickoff form.
- Transcripts are passed through credential redaction before upload
  (`scripts/capture/sync_sessions.py`): common token shapes are replaced with
  `[REDACTED]`. Treat redaction as a backstop — don't paste secrets into a
  session in the first place.
- You may request removal of specific content from the archive at any time.

## How to sync (end of a work session)

```bash
# one-time: clone the archive next to the work repo
git clone <org>/session-archive ../session-archive

# then, whenever you finish working:
python scripts/capture/sync_sessions.py
```

## Not enabled (possible later, team's call)

Live OpenTelemetry metrics from Claude Code to a hosted dashboard (token
usage, tool-call rates). Adds infrastructure; the transcript corpus already
covers the learning use case. Revisit if the team wants live numbers.
