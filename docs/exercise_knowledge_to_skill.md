# Exercise — from your lecture notes to a skill, and into the capstone workflow

**Your own repository. Committed, visible to the team, built on your own coursework.**

This is the practice track that runs alongside the capstone through September. It has one arc:
take a lecture you already sat through, distil what you actually learned, turn it into something
the agent uses — a `CLAUDE.md`, a rule, a hook, a **skill** — tune that harness to how *you* work,
keep it in sync as the semester adds knowledge, and finally carry one skill into the capstone
workflow where it has consequences.

Eight parts. Each one ends in a commit and states exactly what to say to the agent, what to expect
back, and how to check it. Nothing here is graded by us. It is visible to the team, which is the
point: five people building five knowledge bases in the open learn faster than five people reading
the same docs.

---

## Why your own notes, and not a toy example

A skill is only worth writing when it encodes knowledge the agent does not already have. Generic
programming advice is already in the model. Your Financial Data Science lecture on purged
cross-validation, the exact convention your professor uses for a CRSP delisting return, the three
steps your stochastic-calculus course insists on before you claim a martingale — that is knowledge
with a shape, and it is yours.

Distilling a lecture is also the cheapest way to find out whether you understood it. Writing a rule
the agent can follow forces the ambiguity out into the open. If you cannot state the rule in two
sentences, you have found the part of the lecture you have not learned yet.

**One boundary, and it is not negotiable.** Your professors' slides and notes are their work. Do
not commit lecture PDFs, slide decks, problem sets, or solutions, and do not paste their text
verbatim. Write your own distillation in your own words and cite the course. If in doubt, ask the
instructor. A public repository makes this a copyright question, not a style question.

---

## Part 1 · Your repository (15 min)

Create one repository in the organization from the **`kb-template`** template, named
`kb-<your-github-username>` (for example `kb-lyz1103`). Keep it private to the organization; every
member can read every repository, so everyone can read everyone's.

```bash
gh repo create quantit-cmu-mscf-2026-fall/kb-<your-username> \
  --template quantit-cmu-mscf-2026-fall/kb-template --private --clone
cd kb-<your-username>
claude
```

The template ships this exercise as `EXERCISE.md`, a starter `CLAUDE.md`, one path-scoped rule,
one blocking hook, one skill, and an example note. Open the agent and ask, before changing anything:

```text
Read CLAUDE.md, .claude/rules, .claude/settings.json and .claude/skills. Tell me, in one
line each, what each file makes you do differently — and which of them you could ignore.
```

**Expect**: it distinguishes the advisory files (CLAUDE.md, rules, skills) from the hook, which it
cannot ignore. If it says it can ignore the hook, it is wrong; keep that answer, you will use it in
Part 4.

**Check and commit**: `/context` lists `CLAUDE.md` and the rule under memory files. Change the
README title to your name and commit: `git commit -am "kb: init"`; push.

---

## Part 2 · Distil one lecture (45 min)

Pick one lecture from any course this semester. Read your own notes first. The note you will write
has exactly four parts:

1. **The claim** — what the lecture asserts, in your words, two or three sentences.
2. **When it applies** — the conditions. Lectures state these once and students forget them.
3. **How to do it** — a numbered procedure you could follow in six months without the slides.
4. **What goes wrong** — the failure mode your professor warned about, or the one you hit in the
   homework. If none yet, say so and date it.

Use the agent as an interviewer, not an author:

```text
/distil-lecture 46-XXX purged cross-validation
```

or, without the skill:

```text
I am distilling my lecture on purged cross-validation into a four-part note. Interview me —
ask the questions that expose where my understanding is thin, one at a time. Do not write
the note until I say "write it".
```

**Expect**: five to eight questions, and at least one you cannot answer well. That question is the
deliverable — it is where your understanding stops. Answer what you can, then say "write it".

**Reject** the draft if it contains anything you did not say. The agent will helpfully add textbook
material; that is exactly what the note must not contain. Say: "Remove everything I did not tell
you. Mark gaps as `[gap]` instead of filling them."

**Check and commit**: the note has the four H2 sections, cites the course and week, and has at least
one `[gap]`. `git add notes/ && git commit -m "notes: purged cross-validation (46-XXX wk 3)"`.

---

## Part 3 · CLAUDE.md, rules, and memory (30 min)

Three mechanisms; the session is about telling them apart.

**`CLAUDE.md`** is what you write, loaded at the start of every session. Put in it what the agent
cannot infer: how your notes are structured, what it must never do, how you cite a course. Under
about 200 lines — a bloated file is ignored *in parts*, and you cannot tell which parts.

**Rules** (`.claude/rules/*.md`) are CLAUDE.md split by topic, with one extra power: a `paths:`
header makes a rule load only when the agent touches matching files. The template's
`notes-format.md` applies to `notes/**/*.md` and nothing else.

**Auto memory** is what the agent writes for itself from your corrections, stored per repository
*outside* it. You did not turn it on; it is running.

Do these three things, in order:

```text
1. Open /memory and read what auto memory has saved about me so far. Quote it.
2. Draft a second note on <another topic> following the notes-format rule, without me
   restating the format.
3. Add one line to CLAUDE.md that would have prevented the mistake you made in step 2.
```

**Expect**: in step 2 the agent follows the four-part format *because the rule loaded when it
opened `notes/`* — if it does not, the rule's wording is ambiguous; fix the wording, not the agent.
In step 3 it proposes a CLAUDE.md line; accept it only if it is specific enough to verify
("cite as `46-XXX, Fall 2026, week N`", not "cite properly").

**Check and commit**: `CLAUDE.md` under 200 lines, one rule with `paths:`, and one line in the
channel on what auto memory had saved about you. Commit: `harness: CLAUDE.md + notes-format rule`.

---

## Part 4 · A hook, because instructions are advisory (30 min)

`CLAUDE.md`, rules and skills are context: the agent reads them and usually complies. A **hook** is
a shell command the tool runs at a fixed point in its lifecycle, and it happens whether the agent
agrees or not. That distinction is the whole session.

Hooks live in `.claude/settings.json`. The events you will use: `PreToolUse` (before a tool runs —
it can block), `PostToolUse` (after), `SessionStart`, `Stop`. A hook receives the event as JSON on
stdin; a `PreToolUse` hook that exits with **code 2** blocks the action and hands your stderr text
back to the agent as the reason.

The template ships one: `.claude/hooks/no-slide-dumps.sh` refuses to write `.pdf`/`.pptx`/`.ipynb`
into the repository. **Make it fire**:

```text
Save my lecture slides into the repo as notes/week3-slides.pdf.
```

**Expect**: the agent tries, the hook blocks it with "Blocked: … looks like course material", and
the agent tells you it cannot — and suggests distilling instead. That message is the deliverable.

Then write your own. A `PostToolUse` hook on `Edit|Write` that rebuilds `notes/INDEX.md` whenever
a note changes is the one you will want by Part 7:

```text
Write a PostToolUse hook on Edit|Write that runs .claude/hooks/rebuild_index.py when the
edited file is under notes/. The script lists every notes/*.md with its first heading into
notes/INDEX.md. Register it in .claude/settings.json and show me it firing.
```

Browse what is registered with `/hooks`. **Check and commit**: two hooks registered, both seen
firing, `notes/INDEX.md` regenerated by the second. Commit: `harness: index hook`.

---

## Part 5 · Turn the knowledge into a skill (45 min)

A **skill** is a folder with a `SKILL.md`: instructions the agent loads *when they are relevant*
rather than every session. That is the difference from `CLAUDE.md` — context cost is paid on use,
so a skill can be long and specific where `CLAUDE.md` must be short and general.

The template's `distil-lecture` is a *procedure* skill. Now write a *knowledge* skill from your
Part 2 note — the thing the agent should know when it works on that topic:

```text
Create .claude/skills/purged-cv/SKILL.md from notes/purged-cross-validation.md. The skill
is for when I ask you to build or review a cross-validation split on financial time series.
Put the procedure and the failure modes in the body; keep it under 80 lines; link the note
for detail. Write the description as WHAT it does and WHEN to use it.
```

The frontmatter fields worth knowing on day one: `name` (defaults to the folder name),
`description` (what the agent matches against — write it as *what + when*), `argument-hint`,
`allowed-tools`, and `disable-model-invocation: true` when the skill should run only when you type
`/name`. Keep `SKILL.md` under about 500 lines; move detail into sibling files.

Now the test that matters. Open a **fresh session** (`/clear`, or a new terminal) and say, without
naming the skill:

```text
I need a train/test split for a monthly stock-return panel where labels are 3-month
forward returns. Set it up.
```

**Expect**: the agent invokes `purged-cv` on its own and applies your procedure — purge and embargo,
in your words. If it does not, the `description` is the defect: it is the only part the agent sees
before invoking. Rewrite it, `/clear`, try again. Two rounds is normal.

**Check and commit**: one skill invoked without being named, from a fresh session. Commit:
`skills: purged-cv` and write one paragraph in the README on what you changed in the description to
make it fire.

---

## Part 6 · Tune the harness to how you work (30 min)

The five mechanisms are the same for everyone; the *settings* are not. The agent amplifies the
habits of the person driving it, so the harness should push against your own failure mode, not a
generic one. Pick the row that is most like you — be honest — and build that first.

| If you tend to… | The agent will… | So build this first |
|---|---|---|
| Trust a clean-looking answer | ship plausible code with an unhandled edge case | a `Stop` hook that runs `pytest -q` and refuses to end the turn while it is red; ask for the output, not the summary |
| Lose the thread in long sessions | drift as context fills, forget early instructions | a `CLAUDE.md` line "when compacting, preserve the list of modified files and the open question"; `/clear` between tasks by habit |
| Skip planning and dive in | edit the wrong file confidently | start every session with `claude --permission-mode plan`; write the plan to `PLAN.md` before approving |
| Write terse prompts | guess your intent and fill gaps with defaults | a personal skill `/spec` that interviews you and writes a spec before any implementation |
| Over-explain and over-specify | ignore half of a 400-line CLAUDE.md | prune: for each line ask "would removing this cause a mistake?"; convert "never" lines to hooks |
| Prefer reading to doing | admire the plan and never run it | a `SessionStart` hook that prints the three open items from `TODO.md` — starting with the check, not the reading |

Two places to put personal settings, and the difference matters:

- `~/.claude/CLAUDE.md` and `~/.claude/rules/` — **yours, every project**. Working preferences,
  the language you want answers in, "always show me the test output".
- `./CLAUDE.md` in a repository — **the team's**. Conventions of *this* codebase. A personal
  preference committed here is noise for four other people.

```text
Look at my last five sessions in this repository (use /resume to list them). Tell me the two
corrections I made most often. Propose one CLAUDE.md line, one rule, or one hook for each —
and say which mechanism you chose and why.
```

**Expect**: the agent names patterns you did not notice ("you re-ask for the test output in every
session"). Accept one proposal, implement it, and write down which row of the table you are.

**Check and commit**: one personal setting in `~/.claude/` (not committed) and one project-level
change in your kb (committed) that targets *your* row. Commit: `harness: tuned for <row>`.

---

## Part 7 · Keep the knowledge base and the skills in sync (30 min, then weekly)

The semester keeps adding lectures. The failure mode is obvious a month in: notes grow, skills
freeze, and the agent works from a stale procedure while the accurate one sits unread in `notes/`.
Two structural fixes and one ritual.

**Structure 1 — notes are the source, skills point at them.** A skill carries the procedure and
links the note; it never duplicates the note's detail. When the note changes, the skill's link
still points at the truth.

**Structure 2 — the index loads every session.** `CLAUDE.md` can import files with `@path`. Add
`@notes/INDEX.md` to your `CLAUDE.md`: the map of what you know is now in every session, and the
Part 4 hook keeps the map current without you touching it.

**The ritual — after every new lecture, in this order:**

```text
1. /distil-lecture <course> <topic>            → new note, committed
2. "Which existing skills does this note change? Show the diff you would make."
3. Accept or reject each diff; commit "skills: sync with <topic>"
4. /kb-sync                                     → the audit
```

The template's `kb-sync` skill is the audit: every note has a `Used by:` line naming a skill or
`none yet`; every skill links at least one note; every link resolves. Its output is a table, not
prose — three columns, one row per note.

**Check and commit**: `@notes/INDEX.md` imported in `CLAUDE.md` (confirm with `/context`), one
`kb-sync` run with zero broken links, and the second note from Part 3 carrying a `Used by:` line.
Commit: `kb: sync ritual`. From here on this runs after every lecture, and it takes ten minutes.

---

## Part 8 · Carry one skill into the capstone workflow (60 min, week 4)

This is where the practice track meets the project. One skill from your knowledge base becomes
part of the team repository, is used in a real experiment, and its effect is reported back.

**Step 1 — pick the skill that changes an experiment.** The test: "if the agent had this skill
last week, which paper-round experiment would have been done differently?" If the answer is none,
pick another lecture. Validation, splitting, multiple testing, and market-data conventions are
the likely candidates for this project.

**Step 2 — port it by pull request.** The team repository's skills live in `.claude/skills/`.
Open a PR that adds yours, adapted to the team's data and modules:

```text
Port .claude/skills/purged-cv from my knowledge base into this repository. Adapt the
procedure to capstone.synth.make_panel and capstone.backtest.run_backtest — name the actual
functions. Keep my note's failure modes. Add one test under tests/ that fails if a split
leaks: a label window that overlaps a training window must raise.
```

Fill the PR template. **Results** for this PR: "no experiment yet — this adds a procedure and a
leak test". A teammate reviews, and the review question is specific: does the test actually fail
on a leaking split? Ask the reviewer to break it on purpose.

**Step 3 — use it in one experiment, logged.** Rerun one existing paper-round experiment with the
skill active. Every run through `capstone.runlog.log_run`. Compare the before and after numbers;
the honest outcome is often "the Sharpe fell, because the old split was leaking". That is the
result.

**Step 4 — report back.** In the paper-round issue's *Report back* section: what the skill
changed, the numbers before and after, the trial count, and what you would do next. One paragraph
and a table. Post it before the next sync.

**Step 5 — the loop is now the team's.** From week 5, every `kb-*` repository feeds the team
repository the same way: note → skill → PR → experiment → report back. The team's `CLAUDE.md`
gets a line pointing at the shared skills, and the skills index becomes part of the project's
`docs/`. At the final presentation, the list of skills the team built and the experiments each one
changed is a slide — evidence of a method, not a pile of code.

**Check**: one skill merged into the team repository, one experiment rerun through it with the
ledger showing both runs, one *Report back* filled. That is the week-4 deliverable.

---

## The five mechanisms, once more

| Mechanism | Written by | Loaded | Use it for |
|---|---|---|---|
| `CLAUDE.md` | you | every session | short, always-true facts: commands, conventions, hard "never" rules |
| Auto memory | the agent | every session | your corrections and preferences, accumulated without you writing them |
| `.claude/rules/*.md` | you | every session, or only for matching paths | conventions that belong to one part of the repository |
| Hooks | you | at a lifecycle event, deterministically | things that must happen every time, including blocking an action |
| Skills | you | when relevant, or when you type `/name` | procedures and domain knowledge too long to keep in context always |

The ordering principle: **advisory → deterministic**. Context (`CLAUDE.md`, rules, skills) shapes
what the agent tends to do. A hook decides what it *can* do. If you find yourself writing "always"
or "never" three times in `CLAUDE.md` and still seeing it ignored, you wanted a hook.

And the maintenance principle: **a harness decays**. Rules go stale as the code moves, `CLAUDE.md`
bloats, memory accumulates wrong facts, skills freeze while notes advance. Once a week: prune what
the agent already does right, promote what you keep re-explaining (chat → memory → `CLAUDE.md` →
rule → skill → hook), and delete what no longer changes behaviour. The test of a harness line is
the same as the test of a note: would removing it cause a mistake?

---

## Done when

- [ ] `kb-<your-username>` exists, created from the template, and the team can read it
- [ ] Two notes in `notes/`, four parts each, at least one `[gap]`, no instructor material
- [ ] `CLAUDE.md` under 200 lines with `@notes/INDEX.md`; one `paths:` rule
- [ ] Two hooks you have watched fire, one of them yours
- [ ] One knowledge skill the agent invokes on its own from a fresh session
- [ ] One personal harness change that targets your own row in the Part 6 table
- [ ] One `kb-sync` run with zero broken links
- [ ] One skill merged into the team repository, used in a logged experiment, reported back

Bring to the session: the question in Part 2 you could not answer, and the description you had to
rewrite in Part 5.
