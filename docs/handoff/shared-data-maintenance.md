# Testing handoff: shared-data maintenance scripts

## 1. What this adds
The tooling that produces and refreshes the team's shared data (QUANTIT-56), which
`capstone.shared_data` (#26) serves. Code: `scripts/pull_wrds.py`,
`scripts/build_data_dictionary.py`, `scripts/publish_shared_data.py`, and the pgpass
username helpers in `capstone/wrds_loader.py`. Tests: `tests/test_data_scripts.py`,
`tests/test_wrds_loader.py`. Docs: "Updating the data" in `docs/shared_data.md`.
Not in scope: the loader itself (#26), and the CIZ pull for `wrds_loader` (#53).

## 2. Contract
- Every output lands in `data_cache/` (gitignored); nothing pulled enters the repo.
- The WRDS username comes from `--user`, `$WRDS_USERNAME` or the pgpass file, never
  required on a command line.
- Publishing is all or nothing: every copy is verified before anything is replaced,
  and `SHA256SUMS` is rewritten last. A failure while swapping files in puts the
  originals (kept as `.<name>.bak`) and the old `SHA256SUMS` back. Without
  `--apply` nothing is written.
- The data dictionary asserts each stated fact against the data and raises on a
  claim that fails or a code it cannot name. `N/A` in any coded column must sit on a
  delisting row.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | 18 in `test_data_scripts.py`, 3 in `test_wrds_loader.py`; hand-built frames, a fake Drive folder, no WRDS login | pass | `pytest -q tests/test_data_scripts.py tests/test_wrds_loader.py` |
| Tests catch breakage | delisting fields on every row; preview that writes; a failed swap left half-done; `N/A` off a delisting row; pgpass ignored | each turns at least one test red | edit the line, rerun the command above |
| Full suite | CI's command | pass | `pytest -q -m "not network"` |
| Real data | these scripts produced the current shared files and dictionary | `pytest -m data` passed when the PR was opened; not rerun for this refresh | `pytest -q -m data` (skips without the data) |

## 4. Provenance
Ported from Cal's development sandbox, where these files are unchanged apart from a
sandbox-only holdout guard, left out here. Lauren's review points on this PR (atomic
publish, no `--user` on the command line, delisting fields on the delisting row, a
note on concurrent publishers) are all addressed. Details: the PR description.

## 5. Known limits
- The pull and the dictionary's WRDS metadata fetch need a WRDS login, so CI
  covers only their pure functions.
- Two people publishing at once would race on `SHA256SUMS`; the docs say to publish
  one at a time.
- Worth a reviewer's look: `add_universe_flag` (delisting row judged by the day
  before) and the staged replace in `publish_shared_data.py`.

## 6. Public-repo safety
No data, credentials, account ids, machine paths or private notes.
