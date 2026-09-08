# Exercise session — turn your lecture notes into a skill

**Your own repository. Committed, visible to the team, built on your own coursework.**

This is the practice track that runs alongside the capstone through September. It has one
arc: take a lecture you already sat through, distil what you actually learned, and turn that
into something the agent uses — a `CLAUDE.md` rule, a project rule, a hook, and finally a
**skill**. By the end you will have built the four extension mechanisms with your own material,
which is the fastest way to understand what each one is for.

Nothing here is graded by us. It is visible to the team, which is the point: five people
building five knowledge bases in the open learn faster than five people reading the same docs.

---

## Why your own notes, and not a toy example

A skill is only worth writing when it encodes knowledge the agent does not already have.
Generic programming advice is already in the model. Your Financial Data Science lecture on
purged cross-validation, the exact convention your professor uses for a CRSP delisting return,
the three steps your stochastic-calculus course insists on before you claim a martingale — that
is knowledge with a shape, and it is yours.

Distilling a lecture is also the cheapest way to find out whether you understood it. Writing
a rule the agent can follow forces the ambiguity out into the open. If you cannot state the
rule in two sentences, you have found the part of the lecture you have not learned yet.

**One boundary, and it is not negotiable.** Your professors' slides and notes are their work.
Do not commit lecture PDFs, slide decks, problem sets, or solutions, and do not paste their
text verbatim. Write your own distillation in your own words and cite the course. If in doubt,
ask the instructor. A public repository makes this a copyright question, not a style question.

---

## Your repository

Create one repository in the organization from the **`kb-template`** template, named
`kb-<your-github-username>` (for example `kb-lyz1103`). Keep it private to the organization; the
team already has read access to every `kb-*` repository, so everyone can read everyone's.

```bash
gh repo create quantit-cmu-mscf-2026-fall/kb-<your-username> \
  --template quantit-cmu-mscf-2026-fall/kb-template --private --clone
cd kb-<your-username>
```

The template ships this exercise as `EXERCISE.md`, so you can open Claude Code in the repository
and follow along with the document beside you.

The scaffold below is what you end the month with. Build it in the order of the four sessions;
do not create empty directories in week 1.

```text
kb-<your-username>/
├── CLAUDE.md                     # how the agent works in YOUR knowledge base
├── README.md                     # what this repo is, for a stranger
├── notes/                        # your distillations, one file per topic
│   ├── purged-cross-validation.md
│   └── crsp-delisting-returns.md
├── .claude/
│   ├── rules/                    # path-scoped conventions
│   │   └── notes-format.md
│   ├── skills/                   # the destination
│   │   └── distil-lecture/
│   │       └── SKILL.md
│   └── settings.json             # your hook
└── .gitignore
```

---

## Session 1 · Distil one lecture (this week)

Pick one lecture from any course this semester. Read your own notes, then write a distillation
in `notes/<topic>.md` with exactly these four parts:

1. **The claim** — what the lecture asserts, in your words, two or three sentences.
2. **When it applies** — the conditions. This is the part lectures state once and students forget.
3. **How to do it** — the procedure, numbered, concrete enough that you could follow it in six
   months without the slides.
4. **What goes wrong** — the failure mode your professor warned about, or the one you hit in the
   homework. If you have not hit one yet, say so; that is honest and it dates the note.

Use the agent as an interviewer rather than an author:

```text
I am distilling my lecture on purged cross-validation into a note with four parts:
the claim, when it applies, how to do it, what goes wrong. Interview me — ask me
the questions that expose where my understanding is thin. Do not write the note for me.
```

**Commit it.** One note, one commit, pushed. That is session 1.

**Deliverable**: `notes/<topic>.md` in your repo, and one sentence in the channel about the
question the agent asked that you could not answer.

---

## Session 2 · CLAUDE.md, memory, and rules

Three mechanisms, and the whole session is about telling them apart.

**`CLAUDE.md`** is what you write, loaded at the start of every session. Put in it what the agent
cannot infer: how your notes are structured, what you want it to never do, how you cite a course.
Keep it under about 200 lines — a bloated file gets ignored in parts, and you cannot tell which
parts. Check it loaded with `/context`; generate a starting one with `/init`.

**Auto memory** is what the agent writes for itself, from your corrections, stored per repository
outside it. Open `/memory` and read what it has saved about you. You will be surprised at least
once. This is the mechanism you did not know was running.

**Rules** (`.claude/rules/*.md`) are CLAUDE.md split into topics, with one extra power: a rule can
declare which files it applies to, so it only loads when relevant.

```markdown
---
paths:
  - "notes/**/*.md"
---

# Note format
- Every note has the four parts: claim, when it applies, how to do it, what goes wrong.
- Cite the course as `46-XXX, Fall 2026, week N`. Never paste slide text.
- If a claim has a condition, the condition goes in "when it applies", not in a parenthesis.
```

**Exercise**: write your `CLAUDE.md`, write one path-scoped rule for `notes/`, then ask the agent
to draft a second note. If it follows your format without being told, the rule works. If it does
not, your rule is ambiguous — fix the wording, not the agent.

**Deliverable**: `CLAUDE.md` and `.claude/rules/notes-format.md` committed, plus one line in the
channel on what auto memory had saved about you.

---

## Session 3 · A hook, because instructions are advisory

`CLAUDE.md` and rules are context: the agent reads them and usually complies. A **hook** is a
shell command the tool runs at a fixed point in its lifecycle, and it happens whether the agent
agrees or not. That distinction is the whole session.

Hooks live in `.claude/settings.json` (project) or `~/.claude/settings.json` (yours). The events
you will care about are `PreToolUse` (before a tool runs — it can block), `PostToolUse` (after),
`UserPromptSubmit`, `SessionStart`, and `Stop`. Each hook receives the event as JSON on stdin;
a `PreToolUse` hook that exits with **code 2** blocks the action and hands your stderr text back
to the agent as the reason.

A useful hook for a notes repository, in full:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | grep -q '^notes/' && python .claude/check_note_format.py || true"
          }
        ]
      }
    ]
  }
}
```

Or the blocking kind, which is the one that teaches the lesson:

```bash
#!/usr/bin/env bash
# .claude/hooks/no-slide-dumps.sh — refuse to write lecture PDFs into the repo
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
case "$FILE" in
  *.pdf|*.pptx|*.key)
    echo "Blocked: $FILE looks like course material. Distil it into notes/ instead." >&2
    exit 2 ;;
esac
exit 0
```

**Exercise**: write one hook, register it, and then **try to make the agent violate it**. A hook
you have not seen fire is a hook you have not tested. Browse what is registered with `/hooks`.

**Deliverable**: a working hook committed, and the transcript line where it fired.

---

## Session 4 · Turn the knowledge into a skill

A **skill** is a folder with a `SKILL.md`: instructions the agent loads *when they are relevant*,
rather than every session. That is the difference from `CLAUDE.md` — context cost is paid on use,
so a skill can be long and specific where `CLAUDE.md` must be short and general.

Minimum viable skill, in `.claude/skills/distil-lecture/SKILL.md`:

```markdown
---
name: distil-lecture
description: Turn raw lecture notes into a four-part note in notes/. Use when the user
  mentions distilling a lecture, a course topic, or writing up a class.
---

# Distil a lecture

1. Ask which course and week, then read the raw material the user points at.
2. Interview the user for the four parts — claim, when it applies, how to do it,
   what goes wrong. Ask about conditions and failure modes; those are the parts
   people skip.
3. Write `notes/<slug>.md` in the repository format. Never paste source text.
4. Report which of the four parts is thinnest and what question would fix it.
```

The frontmatter fields worth knowing on day one: `name` (defaults to the folder name),
`description` (this is what the agent matches against — write it as *what it does and when to
use it*), `allowed-tools` (pre-approved for that turn), `argument-hint`, and
`disable-model-invocation: true` when the skill should only run when you type `/distil-lecture`.
Keep `SKILL.md` under about 500 lines and move detail into sibling files the skill links to.

Then the exercise that proves it:

- Invoke it manually: `/distil-lecture`.
- Start a **fresh session**, say "I want to write up this week's stochastic calculus lecture",
  and see whether the agent picks the skill up on its own. If it does not, your `description` is
  the defect — it is the only part the agent sees before invoking.
- Write a second skill from a different course, so you feel the difference between a skill that
  encodes a *procedure* and one that encodes *domain knowledge*.

**Deliverable**: at least one skill in your repo that the agent invokes without being told, and a
one-paragraph note in your README on what you changed in the description to make that happen.

---

## What the four mechanisms are for

Keep this table. It is the answer to the question everyone asks in week 5.

| Mechanism | Written by | Loaded | Use it for |
|---|---|---|---|
| `CLAUDE.md` | you | every session | short, always-true facts: commands, conventions, hard "never do this" rules |
| Auto memory | the agent | every session | your corrections and preferences, accumulated without you writing them |
| `.claude/rules/*.md` | you | every session, or only for matching paths | conventions that belong to one part of the repository |
| Hooks | you | at a lifecycle event, deterministically | things that must happen every time, including blocking an action |
| Skills | you | when relevant, or when you type `/name` | procedures and domain knowledge too long to keep in context always |

The ordering principle: **advisory → deterministic**. Context (`CLAUDE.md`, rules, skills) shapes
what the agent tends to do. A hook decides what it *can* do. If you find yourself writing "always"
or "never" three times in `CLAUDE.md` and still seeing it ignored, you wanted a hook.

---

## How this connects to the capstone

Everything you build here transfers directly. The capstone repository has the same four
mechanisms: a `CLAUDE.md` you can improve by pull request, a session-capture hook already
running, and a ledger rule that would be better as a hook — which is exactly the week-3 exercise.
The knowledge base is the low-stakes place to learn the mechanisms; the capstone is where they
have consequences.

And the notes themselves are not throwaway. A team that has five knowledge bases covering
validation, time series, microstructure and asset pricing has a reference the agent can be
pointed at during the research weeks.

---

## Done when

- [ ] `kb-<your-username>` exists in the organization and the team can read it
- [ ] At least two notes in `notes/`, four parts each, no source material committed
- [ ] `CLAUDE.md` under 200 lines, and one path-scoped rule
- [ ] One hook you have watched fire
- [ ] One skill the agent invokes on its own from a fresh session
- [ ] You can say, in one sentence each, what all five mechanisms are for

Bring to the session: the skill you wrote, and the description you had to rewrite to make it fire.
