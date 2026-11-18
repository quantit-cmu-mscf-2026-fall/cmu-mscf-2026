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
# The name of an entry that can leave earlier unseen entries out of the count.
CORRECTION_NAME = "ledger-correction"


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
        tags: free-form labels for later filtering. A bare string is one tag,
            stored as a one-item list, so `trial_count`'s tag filters see it.
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
        "tags": [tags] if isinstance(tags, str) else list(tags or []),
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


def _tag_set(tags) -> set[str]:
    """An entry's tags as a set. A bare string is one tag, not a set of letters.

    `log_run` stores a list, but an entry written with `tags="synthetic"` before
    it normalised strings holds the string itself, and `set("synthetic")` would
    match no tag filter: an `include_tags` count would silently drop the entry
    and under-count the family.
    """
    if not tags:
        return set()
    if isinstance(tags, str):
        return {tags}
    return set(tags)


def _ends_mid_line(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) != b"\n"


def _read_numbered() -> list[tuple[int, dict]]:
    """Every ledger entry with its 1-based line number, oldest first.

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
                entries.append((number, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number} is not valid JSON ({exc.msg}). The ledger is "
                    "append-only, so the usual cause is a run interrupted mid-write; "
                    "fix or delete that one line. It is not skipped, because a "
                    "dropped entry silently lowers every trial count."
                ) from exc
    return entries


def read_entries() -> list[dict]:
    """Every ledger entry, oldest first; an empty list when nothing is logged yet.

    Blank lines are skipped. A line that is not valid JSON raises `ValueError`
    naming the file and line number (see `_read_numbered`).
    """
    return [entry for _, entry in _read_numbered()]


def _same_json(a, b) -> bool:
    """Equal as JSON values: `1`, `1.0` and `true` are three different values."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _malformed(line: int, why: str) -> ValueError:
    return ValueError(
        f"{ledger_path()}:{line} is a malformed ledger correction: {why}. Fix or "
        "delete that one line; it is not ignored, because that would quietly count "
        "runs someone meant to exclude."
    )


class _Corrections:
    """The ledger's entries sorted into trials and exclusions.

    The ledger is append-only: a mistaken or duplicate run is never deleted.
    A later entry named `CORRECTION_NAME` marks it as excluded instead, and says why
    (a seen target still counts as a trial; see below)::

        log_run(CORRECTION_NAME, params={
            "ledger_name": "evaluate-factor",
            "excludes": [{"ts_utc": "...", "factor_id": "..."}, ...],
            "reason": "duplicates of the 2026-10-05 run",
        })

    A correction is itself never deleted either, so a mistaken one is undone by
    a later retraction, which names the correction's `ts_utc`::

        log_run(CORRECTION_NAME, params={
            "retracts": ["<the correction's ts_utc>"],
            "reason": "it excluded the wrong runs",
        })

    A retraction can be retracted in turn, which restores what it undid.

    Each exclusion item targets one entry logged under `ledger_name` at `ts_utc`,
    *before* the correction, whose params hold every other key of the item as
    the same JSON value (`1` does not match `true`): the timestamp names the
    run, and the params it lists tell apart runs logged in the same microsecond.
    Matching consumes the earliest unmatched entry for each item. A target with
    recorded metrics remains a trial: seen results always count. A correction
    line repeated verbatim (the same entry, `ts_utc` included, as when a sync
    appends a line twice) is read once, so the copy cannot consume a second run.

    Attributes:
        trials: entries that count, in ledger order.
        excluded: targets of active corrections, in ledger order; seen targets
            also remain in `trials`.
        corrections: how many correction entries the ledger holds.
        retracted: how many of those a later active retraction undid.
        unmatched: active exclusion items that matched no earlier run.

    Raises:
        ValueError: naming the file and line, if a correction is malformed: no
            non-empty `reason`; a `ledger_name` that is not a non-empty string;
            `excludes` not a list of items each with a string `ts_utc`;
            `retracts` not a list of timestamps of earlier corrections; or both
            kinds in one entry. Ignoring a malformed correction would quietly
            count runs someone meant to exclude.
    """

    def __init__(self, numbered: list[tuple[int, dict]]):
        # position -> ("exclude", ledger_name, items) or ("retract", target positions)
        parsed: dict[int, tuple] = {}
        seen_lines: set[str] = set()
        for pos, (line, entry) in enumerate(numbered):
            if entry.get("name") != CORRECTION_NAME:
                continue
            try:
                key = json.dumps(entry, sort_keys=True)
            except (TypeError, ValueError):
                key = None
            if key is not None:
                if key in seen_lines:
                    continue  # a verbatim repeat of an earlier correction line
                seen_lines.add(key)
            params = entry.get("params") or {}
            if not isinstance(params, dict):
                raise _malformed(line, "params is not an object")
            if not _is_text(params.get("reason")):
                raise _malformed(line, "params.reason must say, in a non-empty string, why")
            if "retracts" in params:
                if "ledger_name" in params or "excludes" in params:
                    raise _malformed(line, "one entry either retracts or excludes, not both")
                targets = params["retracts"]
                if not (
                    isinstance(targets, list) and targets and all(_is_text(x) for x in targets)
                ):
                    raise _malformed(line, "retracts must be a list of correction ts_utc strings")
                positions = []
                for ts in targets:
                    found = [
                        q for q in range(pos) if q in parsed and numbered[q][1].get("ts_utc") == ts
                    ]
                    if not found:
                        raise _malformed(line, f"retracts {ts!r}, which is no earlier correction")
                    positions += found
                parsed[pos] = ("retract", positions)
                continue
            items = params.get("excludes")
            if not _is_text(params.get("ledger_name")):
                raise _malformed(line, "ledger_name must be a non-empty string")
            if not (
                isinstance(items, list)
                and all(isinstance(i, dict) and _is_text(i.get("ts_utc")) for i in items)
            ):
                raise _malformed(line, "excludes must be a list of items each with a ts_utc string")
            parsed[pos] = ("exclude", params["ledger_name"], items)

        # A retraction is only undone by a later one, so walk newest first.
        undone: set[int] = set()
        active: dict[int, tuple] = {}
        for pos in sorted(parsed, reverse=True):
            if pos in undone:
                continue
            active[pos] = parsed[pos]
            if parsed[pos][0] == "retract":
                undone.update(parsed[pos][1])
        self.corrections = len(parsed)
        self.retracted = len(undone)

        rules: dict[tuple[str, str], list[tuple[int, int, dict]]] = {}
        for pos, rule in active.items():
            if rule[0] == "exclude":
                for n, item in enumerate(rule[2]):
                    rules.setdefault((rule[1], item["ts_utc"]), []).append((pos, n, item))
        used: set[tuple[int, int]] = set()
        self.trials: list[dict] = []
        self.excluded: list[dict] = []
        for pos, (_, entry) in enumerate(numbered):
            name, ts = entry.get("name"), entry.get("ts_utc")
            if name == CORRECTION_NAME:
                continue
            hit = None
            if isinstance(name, str) and isinstance(ts, str):
                params = entry.get("params")
                params = params if isinstance(params, dict) else {}
                for at, n, item in rules.get((name, ts), ()):
                    if (
                        (at, n) not in used
                        and at > pos
                        and all(
                            k in params and _same_json(params[k], v)
                            for k, v in item.items()
                            if k != "ts_utc"
                        )
                    ):
                        hit = (at, n)
                        used.add(hit)
                        break
            if hit:
                self.excluded.append(entry)
            if not hit or _is_seen(entry):
                self.trials.append(entry)
        self.unmatched = sum(len(r[2]) for r in active.values() if r[0] == "exclude") - len(used)


def _is_seen(entry: dict) -> bool:
    """Whether the run recorded an outcome, so its result may have been seen.

    A run logged with `metrics` carries its result, and a result that was seen
    is the one a "best of N" correction must not quietly drop.
    """
    return bool(entry.get("metrics"))


def _in_scope(entry, name, include_tags, exclude_tags) -> bool:
    if name is not None and entry.get("name") != name:
        return False
    if include_tags is not None and not set(include_tags) & _tag_set(entry.get("tags")):
        return False
    return not (exclude_tags is not None and set(exclude_tags) & _tag_set(entry.get("tags")))


def trial_count(
    name: str | None = None,
    *,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    count_seen_exclusions: bool = True,
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
        count_seen_exclusions: retained for callers that explicitly passed True.
            Seen runs always count; False is refused rather than returning an
            unsafe count for inference. Inspect `stats` for correction targets.

    Corrections (`_Corrections`) apply first: an entry an active ledger
    correction excludes is not a trial only if it recorded no metrics. Seen
    results always count. The correction itself is never a trial.

    Returns:
        The number of matching entries. Entries written without a `name` are
        counted by `trial_count()` but match no `name` filter, so the per-name
        counts need not sum to the total. A repeat counts: the same `name`,
        `params` and `seed` logged twice (a re-run after a crash, say) is two
        entries, because each was logged before its result was seen. That
        errs conservative; a duplicate that should not count is marked by a
        later correction entry, never deleted.

    Raises:
        LookupError: if nothing matches. A correction run against zero trials
            applies no correction at all, and the usual cause is a wrong
            `CAPSTONE_LEDGER_DIR`, a misspelled name or a tag filter that
            excluded everything, so it fails loudly instead of returning 0.
        TypeError: if a tag filter is a bare string. `set("synthetic")` is a
            set of letters, which would silently match nothing.
        ValueError: if `count_seen_exclusions` is not True, a ledger line is
            malformed, or a ledger correction is.
    """
    for arg, tags in (("include_tags", include_tags), ("exclude_tags", exclude_tags)):
        if isinstance(tags, str):
            raise TypeError(f"{arg} takes a list of tags, not a string: [{tags!r}]")
    if count_seen_exclusions is not True:
        raise ValueError("seen runs always count; count_seen_exclusions must be True")
    ledger = _Corrections(_read_numbered())
    entries = [e for e in ledger.trials if _in_scope(e, name, include_tags, exclude_tags)]
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
    """Print the numbers a correction needs: how many trials, by whom, of what.

    Runs per name are trials: ledger corrections applied, as in `trial_count`.
    Correction targets are listed separately: unseen targets are excluded,
    while targets with recorded metrics remain in the trial count.
    """
    numbered = _read_numbered()
    entries = [entry for _, entry in numbered]
    ledger = _Corrections(numbered)
    trials = ledger.trials
    names = Counter(entry.get("name", "?") for entry in trials)
    users = {entry.get("user", "unknown") for entry in entries}
    seen = [e for e in ledger.excluded if _is_seen(e)]
    print(f"total runs: {len(entries)}")
    print(f"trials (corrections applied): {len(trials)}")
    print(f"targeted by corrections: {len(ledger.excluded)}")
    print(f"excluded unseen runs: {len(ledger.excluded) - len(seen)}")
    print(f"seen correction targets retained as trials: {len(seen)}")
    print(f"correction entries: {ledger.corrections} ({ledger.retracted} retracted)")
    print(f"correction items matching no earlier run: {ledger.unmatched}")
    print(f"distinct names: {len(names)}")
    for name, count in names.most_common():
        print(f"  {name}: {count}")
    if ledger.excluded:
        print("correction targets per name (of which seen, retained):")
        excluded = Counter(entry.get("name", "?") for entry in ledger.excluded)
        seen_names = Counter(entry.get("name", "?") for entry in seen)
        for name, count in excluded.most_common():
            print(f"  {name}: {count} ({seen_names[name]})")
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
