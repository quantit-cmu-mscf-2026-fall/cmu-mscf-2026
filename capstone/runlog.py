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
from datetime import datetime, timezone
from pathlib import Path

LEDGER_FILENAME = "runs.jsonl"


def _ledger_dir() -> Path:
    """Directory holding the ledger.

    `CAPSTONE_LEDGER_DIR` overrides the default so tests (and anyone running
    throwaway experiments) can point the ledger elsewhere without touching the
    shared file. The default lives at the repo root, not the package, so it is
    visible in the top-level tree and gets committed.
    """
    env = os.environ.get("CAPSTONE_LEDGER_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "experiments"


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
        "ts_utc": datetime.now(timezone.utc).isoformat(),
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
    with (ledger_dir / LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _read_entries() -> list[dict]:
    path = _ledger_dir() / LEDGER_FILENAME
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _cmd_stats() -> None:
    """Print the numbers a correction needs: how many trials, by whom, of what."""
    entries = _read_entries()
    names = Counter(entry.get("name", "?") for entry in entries)
    users = {entry.get("user", "unknown") for entry in entries}
    print(f"total runs: {len(entries)}")
    print(f"distinct names: {len(names)}")
    for name, count in names.most_common():
        print(f"  {name}: {count}")
    print(f"distinct users: {len(users)}")


def _cmd_list(last: int) -> None:
    """Print the most recent entries, one compact line each."""
    for entry in _read_entries()[-last:]:
        metrics = json.dumps(entry.get("metrics") or {}, separators=(",", ":"))
        print(
            f"{entry.get('ts_utc', '?')}  {entry.get('user', '?')}  {entry.get('name', '?')}  {metrics}"
        )


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
