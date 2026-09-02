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
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

LEDGER_FILENAME = "runs.jsonl"


def _stable_run_id(seed: int | None, name: str, params: dict | None, ts_utc: str) -> str:
    """Stable ledger id used to link a trial to its outcome.

    We do not attempt to mutate history. Instead, the id is a deterministic key
    based on the experiment identifier, the seed, the parameter set, and the
    timestamp of the trial entry. The outcome record then appends a second row
    referencing this id without altering the original entry.
    """
    payload = json.dumps({
        "name": name,
        "seed": seed,
        "params": params or {},
        "ts_utc": ts_utc,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    hypothesis: str | None = None,
    source: str | None = None,
    decision_ref: str | None = None,
) -> dict:
    """Append the initial trial entry for an experiment.

    A trial is logged before results are read. That entry is append-only and
    permanently preserves the original trial context. A later outcome record can
    reference the same `run_id` without mutating the original history.
    """
    ts_utc = datetime.now(UTC).isoformat()
    entry = {
        "kind": "trial",
        "run_id": _stable_run_id(seed, name, params, ts_utc),
        "ts_utc": ts_utc,
        "user": os.environ.get("GITHUB_USER") or os.environ.get("USER") or "unknown",
        "git_sha": _git_sha(),
        "session_id": os.environ.get("CLAUDE_SESSION_ID"),
        "name": name,
        "seed": seed,
        "params": params or {},
        "metrics": metrics or {},
        "tags": tags or [],
        "notes": notes,
        "hypothesis": hypothesis,
        "source": source,
        "decision_ref": decision_ref,
    }
    ledger_dir = _ledger_dir()
    ledger_dir.mkdir(parents=True, exist_ok=True)
    with (ledger_dir / LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def log_outcome(
    *,
    trial_ref: str,
    decision: str,
    metrics: dict | None = None,
    tags: list[str] | None = None,
    notes: str = "",
    hypothesis: str | None = None,
    source: str | None = None,
    decision_ref: str | None = None,
) -> dict:
    """Append the final outcome for a previously logged trial.

    This preserves append-only history: the original trial entry remains intact,
    and a second JSONL row references it via `trial_ref` and records the final
    decision after the gate has run.
    """
    ts_utc = datetime.now(UTC).isoformat()
    entry = {
        "kind": "outcome",
        "run_id": _stable_run_id(None, "outcome", {"trial_ref": trial_ref}, ts_utc),
        "trial_ref": trial_ref,
        "ts_utc": ts_utc,
        "user": os.environ.get("GITHUB_USER") or os.environ.get("USER") or "unknown",
        "git_sha": _git_sha(),
        "session_id": os.environ.get("CLAUDE_SESSION_ID"),
        "decision": decision,
        "metrics": metrics or {},
        "tags": tags or [],
        "notes": notes,
        "hypothesis": hypothesis,
        "source": source,
        "decision_ref": decision_ref,
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
        for raw in fh:
            line = raw.strip()
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
        ts = entry.get("ts_utc", "?")
        user = entry.get("user", "?")
        name = entry.get("name", "?")
        print(f"{ts}  {user}  {name}  {metrics}")


def _cmd_digest() -> None:
    """Print a human-readable synopsis grouped by tag.

    Each trial is reported with its hypothesis, source, and the final decision
    from its linked outcome record, so a reviewer can follow the narrative in one
    place without reading the raw JSONL.
    """
    entries = _read_entries()
    grouped: dict[str, list[dict]] = {}
    trials_by_id = {entry.get("run_id"): entry for entry in entries if entry.get("kind") == "trial"}
    outcomes_by_trial_ref = {}
    for entry in entries:
        if entry.get("kind") == "outcome":
            outcomes_by_trial_ref.setdefault(entry.get("trial_ref"), []).append(entry)
        tags = entry.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if not tags:
            tags = ["untagged"]
        for tag in tags:
            grouped.setdefault(tag, []).append(entry)

    if not entries:
        print("No runs in the ledger.")
        return

    for tag in sorted(grouped):
        print(f"Tag: {tag}")
        seen_trial_ids = set()
        for entry in grouped[tag]:
            if entry.get("kind") == "outcome":
                continue
            trial_id = entry.get("run_id")
            if trial_id in seen_trial_ids:
                continue
            seen_trial_ids.add(trial_id)
            trial = entry
            outcome = (outcomes_by_trial_ref.get(trial_id) or [{}])[-1]
            decision = str(outcome.get("decision", (trial.get("metrics") or {}).get("decision", "unknown")))
            hypothesis = trial.get("hypothesis") or outcome.get("hypothesis") or "-"
            source = trial.get("source") or outcome.get("source") or "-"
            decision_ref = outcome.get("decision_ref") or trial.get("decision_ref") or "-"
            name = trial.get("name", "?")
            print(f"  - Name: {name}")
            print(f"    Hypothesis: {hypothesis}")
            print(f"    Source: {source}")
            print(f"    Decision: {decision}")
            print(f"    Decision ref: {decision_ref}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m capstone.runlog",
        description="Inspect the experiment run ledger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats", help="trial count, runs per name, distinct users")
    list_parser = subparsers.add_parser("list", help="most recent runs, compactly")
    list_parser.add_argument("--last", type=int, default=10, help="how many entries to show")
    subparsers.add_parser("digest", help="readable summary grouped by tag")
    args = parser.parse_args(argv)
    if args.command == "stats":
        _cmd_stats()
    elif args.command == "list":
        _cmd_list(args.last)
    else:
        _cmd_digest()


if __name__ == "__main__":
    main()
