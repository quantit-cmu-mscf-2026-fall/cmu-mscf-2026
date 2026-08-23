## What

<!-- One logical change. If you need "and" to describe it, consider two PRs. -->

## Why

<!-- The problem this solves or the question it answers. Link the issue/discussion if one exists. -->

## How it was tested

<!-- Commands you ran, tests you added. "CI is green" alone is not a test plan. -->

## Results (research PRs)

<!-- Key numbers or a plot. State the seed(s) and HOW MANY configurations you tried
     to get this result — the trial count is an input to every correction we run. -->

## Checklist

- [ ] Tests pass locally (`pytest -q`)
- [ ] Lint passes (`ruff check . && ruff format --check .`)
- [ ] No credentials, tokens, or account identifiers anywhere in the diff
- [ ] Experiments: seeds fixed, trial count reported
- [ ] No large binaries or notebook outputs committed
