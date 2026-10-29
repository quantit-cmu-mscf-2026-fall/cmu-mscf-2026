"""Append-only ledger of every experiment run.

Every experiment appends one line here, and the ledger's length is the trial
count that every multiple-testing correction in `capstone.evaluate` depends
on. An unlogged experiment is an untracked hypothesis test: it lowers the bar
your best candidate is judged against without anyone knowing. Logging is one
call at the point where a result is produced, which is far cheaper than
reconstructing "how many things did we try?" in week 12.

The ledger is a plain JSONL file so it diffs cleanly, merges as appends, and
can be read with nothing but the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

LEDGER_FILENAME = "runs.jsonl"


def _ledger_dir() -> Path:
    """Directory holding the ledger.

    `CAPSTONE_LEDGER_DIR` overrides the default so tests (and anyone running
    throwaway experiments) can point the ledger elsewhere without touching the
    shared file. The default is `experiments/` at the repo root. The ledger
    file there is gitignored: it is synced to the team's private archive and
    never committed to this public repo.
    """
    env = os.environ.get("CAPSTONE_LEDGER_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "experiments"


def ledger_path() -> Path:
    """The ledger file that `log_run` writes and `read_entries` reads.

    Honours `CAPSTONE_LEDGER_DIR`, so code reading the ledger can say which
    file its trial count came from.
    """
    return _ledger_dir() / LEDGER_FILENAME


def _git_sha() -> str:
    """Short commit SHA of the code that produced the run.

    Resolved relative to this file, not the caller's cwd, so the SHA describes
    the repo the code came from. Any failure (no git, not a repo, detached
    environment) degrades to "unknown" — a missing SHA must never block
    logging, because a lost ledger entry is worse than a lost SHA.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def log_run(
    name: str,
    *,
    params: dict | None = None,
    metrics: dict | None = None,
    seed: int | None = None,
    tags: list[str] | None = None,
    notes: str = "",
) -> dict:
    """Append one experiment run to the ledger and return the entry written.

    Call this once per experiment, at the moment the result exists. The entry
    records who ran what, on which commit, with which parameters and outcome —
    enough to make the trial count auditable and each trial reconstructible.

    Args:
        name: short experiment identifier; reuse the same name across a sweep.
        params: configuration that defines the trial (lookbacks, thresholds).
        metrics: outcome numbers (sharpe, hit rate) — the tested quantity.
        seed: RNG seed, so a run can be reproduced exactly.
        tags: free-form labels for later filtering.
        notes: anything the fields above cannot express.

    Returns:
        The dict that was written, including the generated bookkeeping fields.
    """
    entry = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "user": os.environ.get("GITHUB_USER") or os.environ.get("USER") or "unknown",
        "git_sha": _git_sha(),
        "session_id": os.environ.get("CLAUDE_SESSION_ID"),
        "name": name,
        "seed": seed,
        "params": params or {},
        "metrics": metrics or {},
        "tags": tags or [],
        "notes": notes,
    }
    ledger_dir = _ledger_dir()
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / LEDGER_FILENAME
    # A run interrupted mid-write leaves a last line with no newline. Start on a
    # fresh line so this entry doesn't share it, and deleting the broken line
    # can't take a complete trial with it.
    lead = "\n" if _ends_mid_line(path) else ""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps(entry) + "\n")
    return entry


def _ends_mid_line(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) != b"\n"


def read_entries() -> list[dict]:
    """Every ledger entry, oldest first; an empty list when nothing is logged yet.

    Blank lines are skipped. A line that is not valid JSON raises `ValueError`
    naming the file and line number, rather than being skipped: a skipped entry
    is a trial that silently vanishes from the count, which under-corrects every
    downstream test and invents discoveries.
    """
    path = ledger_path()
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as fh:
        for number, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number} is not valid JSON ({exc.msg}). The ledger is "
                    "append-only, so the usual cause is a run interrupted mid-write; "
                    "fix or delete that one line. It is not skipped, because a "
                    "dropped entry silently lowers every trial count."
                ) from exc
    return entries


def trial_count(
    name: str | None = None,
    *,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
) -> int:
    """How many trials the ledger has logged, for a multiple-testing correction.

    **Scope this to the family you are correcting.** Bare `trial_count()` counts
    every entry in the ledger: every contributor's searches, and the
    methodology runs -- null calibrations, threshold sweeps, simulation studies
    -- that are not hypothesis tests on a strategy at all. Those belong in the
    ledger (`CLAUDE.md` requires it) but not in the family size of somebody
    else's correction, where they only inflate `m` and throw away power. A
    correction wants the trials that competed for the result being judged, which
    in practice means one experiment `name`, or a tag filter, or both.

    Args:
        name: count only entries logged under this experiment name (a sweep
            reuses one name). None counts every name.
        include_tags: count only entries carrying at least one of these tags.
        exclude_tags: drop entries carrying any of these tags. Use it to keep
            methodology runs out of a strategy's family size; the convention
            is to tag those "synthetic".

    Returns:
        The number of matching entries. Entries written without a `name` are
        counted by `trial_count()` but match no `name` filter, so the per-name
        counts need not sum to the total. A repeat counts: the same `name`,
        `params` and `seed` logged twice (a re-run after a crash, say) is two
        entries, because each was logged before its result was seen. That
        errs conservative; a duplicate that should not count is marked by a
        later entry, never deleted.

    Raises:
        LookupError: if nothing matches. A correction run against zero trials
            applies no correction at all, and the usual cause is a wrong
            `CAPSTONE_LEDGER_DIR`, a misspelled name or a tag filter that
            excluded everything, so it fails loudly instead of returning 0.
        TypeError: if a tag filter is a bare string. `set("synthetic")` is a
            set of letters, which would silently match nothing.
        ValueError: from `read_entries`, if a ledger line is malformed.
    """
    for arg, tags in (("include_tags", include_tags), ("exclude_tags", exclude_tags)):
        if isinstance(tags, str):
            raise TypeError(f"{arg} takes a list of tags, not a string: [{tags!r}]")
    entries = read_entries()
    if name is not None:
        entries = [entry for entry in entries if entry.get("name") == name]
    if include_tags is not None:
        wanted = set(include_tags)
        entries = [e for e in entries if wanted & set(e.get("tags") or ())]
    if exclude_tags is not None:
        unwanted = set(exclude_tags)
        entries = [e for e in entries if not unwanted & set(e.get("tags") or ())]
    if not entries:
        where = ledger_path()
        scope = f"named {name!r} " if name is not None else ""
        filters = []
        if include_tags is not None:
            filters.append(f"including tags {sorted(include_tags)}")
        if exclude_tags is not None:
            filters.append(f"excluding tags {sorted(exclude_tags)}")
        suffix = f" ({', '.join(filters)})" if filters else ""
        raise LookupError(f"no trials {scope}in the ledger at {where}{suffix}")
    return len(entries)


def _cmd_stats() -> None:
    """Print the numbers a correction needs: how many trials, by whom, of what."""
    entries = read_entries()
    names = Counter(entry.get("name", "?") for entry in entries)
    users = {entry.get("user", "unknown") for entry in entries}
    print(f"total runs: {len(entries)}")
    print(f"distinct names: {len(names)}")
    for name, count in names.most_common():
        print(f"  {name}: {count}")
    print(f"distinct users: {len(users)}")


def _cmd_list(last: int) -> None:
    """Print the most recent entries, one compact line each."""
    for entry in read_entries()[-last:]:
        metrics = json.dumps(entry.get("metrics") or {}, separators=(",", ":"))
        ts = entry.get("ts_utc", "?")
        user = entry.get("user", "?")
        name = entry.get("name", "?")
        print(f"{ts}  {user}  {name}  {metrics}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m capstone.runlog",
        description="Inspect the experiment run ledger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats", help="trial count, runs per name, distinct users")
    list_parser = subparsers.add_parser("list", help="most recent runs, compactly")
    list_parser.add_argument("--last", type=int, default=10, help="how many entries to show")
    args = parser.parse_args(argv)
    if args.command == "stats":
        _cmd_stats()
    else:
        _cmd_list(args.last)


if __name__ == "__main__":
    main()
