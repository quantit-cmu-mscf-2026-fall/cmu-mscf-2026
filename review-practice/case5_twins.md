# Case 5 — "The twins"

This one has no code to run, because nothing here is a numerical bug. Every
defect is process, and all of it costs the team.

## What you are reviewing

Two open pull requests from the same author, three days apart.

- Both add `research/momentum_baseline.py`. The second version is six lines longer.
- Both also add five `.DS_Store` files.
- The first one's description is still the pull request template, headings unfilled.
- CI is green on both.

Write your two lines before reading on: what you would comment, and the check that
would prove you right.

## The check

```
git fetch origin
git log --stat origin/<branch-a> -1
git log --stat origin/<branch-b> -1
git diff origin/<branch-a>..origin/<branch-b> -- research/momentum_baseline.py
```

That last command is the whole question: if the diff between the two branches is
small, these are the same change twice and one of them is noise. If it is large,
they are different changes wearing the same filename, which is worse.

## Why each of the three matters

**Two pull requests for one change.** The reviewer cannot tell which one is real,
so the rational move is to review neither. That is how a pull request sits for two
weeks. Keep the later one, close the other with a comment saying why.

**`.DS_Store`.** Your operating system's bookkeeping, not the project's. It is
noise in every future diff and it will conflict with a teammate's copy of the same
file. Add it to `.gitignore` once and it never comes back.

**A description left as template text.** This is the one worth caring about. The
body is where you state the claim the diff is supposed to support. If it is empty,
nobody re-read the change before asking someone else to read it — including the
author. A reviewer's first question is always "does the claim match the evidence",
and an unfilled body means there is no claim to check.

## What to comment

Something like:

> These two look like the same change. Can we keep the later one and close the
> other? Also `.DS_Store` is in both — worth a `.gitignore` line so it stops
> riding along. And could you fill in the description on the one we keep, mainly
> the "what this claims" part, so I know what to check the diff against?

Three things, all actionable, none of them about style. That is a good review.
