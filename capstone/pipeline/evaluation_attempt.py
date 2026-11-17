"""Per-opportunity checkpointed attempts for frozen evaluation and portfolio runs."""

import json

from capstone import runlog
from capstone.artifacts import atomic_json
from capstone.execution import replay_identity
from capstone.pipeline.study import file_digest

# v1 attempts were `{"manifest_id", "status": "started"}` with no checkpoints; they stay refused.
ATTEMPT_SCHEMA = "opportunity-checkpointed-attempt-v2"


def _files(root, folders):
    return {
        p.relative_to(root).as_posix(): file_digest(p)
        for folder in folders
        for p in sorted((root / folder).rglob("*"))
        if p.is_file()
    }


def _ledger_lines():
    """The run ledger's lines (a torn last line included, never parsed here)."""
    path = runlog.ledger_path()
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def _logged_since(attempt):
    """`replay_key`s of the `reference-replay` rows logged since this attempt began: the
    killed attempt's own registrations. Empty for an attempt that predates the
    count (it resumes as before, logging its replays again)."""
    start = attempt.get("ledger_lines")
    if not isinstance(start, int):
        return []
    keys = []
    for line in _ledger_lines()[start:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn write: not a logged trial
        if isinstance(row, dict) and row.get("name") == replay_identity.REPLAY_ROW:
            keys.append(replay_identity.replay_key(row.get("params") or {}))
    return keys


def open_attempt(root, manifest_id, units, outputs, *, legacy):
    """Start an attempt, or resume one after verifying every checkpointed opportunity.

    `units` is `(opportunity id, output folders)` in frozen manifest order; `outputs` are the
    folders a run writes. Checkpointed opportunities must match their saved digests exactly
    (changed, missing or added files are refused, never recomputed). Any other file under
    `outputs` was never committed (the interrupted opportunity, or controls); it is recorded
    with its digest in `recoveries`, removed, and recomputed from the same frozen manifest.

    The ledger's line count when the attempt began is recorded (`ledger_lines`); a resumed
    attempt names the replays logged since (`replay_identity.already_logged`), so inside
    the caller's `replay_identity.resume_scope()` the recomputed scenarios do not log their
    trials a second time: the ledger is exactly once across a kill.
    """
    path = root / "attempt.json"
    # atomic_json writes <name>.tmp then replaces; a leftover one is an unfinished write.
    for partial in root.glob("*.json.tmp"):
        partial.unlink()
    if not path.exists():
        attempt = {
            "schema": ATTEMPT_SCHEMA,
            "manifest_id": manifest_id,
            "status": "started",
            "completed_opportunities": [],
            "recoveries": [],
            "ledger_lines": len(_ledger_lines()),
        }
        atomic_json(path, attempt)
        return attempt
    attempt = json.loads(path.read_text(encoding="utf-8"))
    if attempt.get("schema") != ATTEMPT_SCHEMA:
        raise ValueError(legacy)
    done = attempt["completed_opportunities"]
    if attempt["manifest_id"] != manifest_id or [e["id"] for e in done] != [
        unit for unit, _ in units[: len(done)]
    ]:
        raise ValueError("attempt checkpoints do not belong to this frozen registration")
    committed = {k: v for entry in done for k, v in entry["artifacts"].items()}
    owned = {folder for _, folders in units[: len(done)] for folder in folders}
    if _files(root, sorted(owned)) != committed:
        raise ValueError("checkpointed opportunity evidence changed or incomplete")
    replay_identity.already_logged(_logged_since(attempt))
    discarded = {k: v for k, v in _files(root, outputs).items() if k not in committed}
    if discarded:
        attempt["recoveries"].append({"resumed_after": len(done), "discarded": discarded})
        atomic_json(path, attempt)
        for relative in discarded:
            (root / relative).unlink()
    for folder in outputs:  # replay evidence folders are created exclusively
        for item in sorted((root / folder).rglob("*"), reverse=True):
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()
    return attempt


def checkpoint_opportunity(root, attempt, unit, folders, row):
    """Commit one finished opportunity (evaluated, blocked or quarantined) with its digests."""
    entry = {"id": unit, "row": row, "artifacts": _files(root, folders)}
    attempt["completed_opportunities"].append(entry)
    atomic_json(root / "attempt.json", attempt)
