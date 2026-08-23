"""Sync captured working data to the team's PRIVATE session archive.

Run after a work session:  python scripts/capture/sync_sessions.py

What it moves, and where (nothing here ever enters the public repo):
- `.sessions/transcripts/*.jsonl`  ->  ../session-archive/transcripts/<user>/
- `experiments/runs.jsonl`         ->  ../session-archive/ledger/<user>-runs.jsonl

Every file is passed through `redact()` first: conservative patterns for
common credential shapes are replaced with [REDACTED] before anything leaves
the machine. Redaction is a backstop, not a license — never paste secrets in
a session in the first place.

The archive must be cloned as a sibling directory:
    git clone <org>/session-archive ../session-archive
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT.parent / "session-archive"

REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"gh[pos]_[A-Za-z0-9]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"),
    re.compile(r"""(?i)(api[_-]?key|secret|password|token)["']?\s*[:=]\s*["']?[^\s"']{8,}"""),
]


def redact(text: str) -> tuple[str, int]:
    """Replace credential-shaped substrings with [REDACTED]; return (clean, count)."""
    total = 0
    for pattern in REDACT_PATTERNS:
        text, n = pattern.subn("[REDACTED]", text)
        total += n
    return text, total


def _sync_file(src: Path, dest: Path) -> int:
    clean, n = redact(src.read_text(encoding="utf-8", errors="replace"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(clean, encoding="utf-8")
    return n


def main() -> int:
    if not ARCHIVE.exists():
        print(
            "session-archive clone not found — run: "
            "git clone <org>/session-archive ../session-archive"
        )
        return 1

    user = os.environ.get("GITHUB_USER") or os.environ.get("USER") or "unknown"
    synced = 0
    redactions = 0

    transcripts = REPO_ROOT / ".sessions" / "transcripts"
    if transcripts.exists():
        for src in sorted(transcripts.glob("*.jsonl")):
            owner = src.name.split("-")[0] or user
            redactions += _sync_file(src, ARCHIVE / "transcripts" / owner / src.name)
            synced += 1

    ledger = REPO_ROOT / "experiments" / "runs.jsonl"
    if ledger.exists():
        redactions += _sync_file(ledger, ARCHIVE / "ledger" / f"{user}-runs.jsonl")
        synced += 1

    staged = subprocess.run(
        ["git", "-C", str(ARCHIVE), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if staged or synced:
        subprocess.run(["git", "-C", str(ARCHIVE), "add", "-A"], check=False)
        if subprocess.run(
            ["git", "-C", str(ARCHIVE), "diff", "--cached", "--quiet"], check=False
        ).returncode:
            subprocess.run(
                ["git", "-C", str(ARCHIVE), "commit", "-m", "sync sessions"], check=False
            )
            push = subprocess.run(["git", "-C", str(ARCHIVE), "push"], check=False)
            if push.returncode:
                print("warning: committed locally but push failed — push manually")

    print(f"synced {synced} file(s), {redactions} redaction(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
