"""Claude Code hook: capture the session transcript into `.sessions/`.

Registered in `.claude/settings.json` for the Stop and SessionEnd events, so it
only ever fires for sessions run inside this repository — nothing else on the
machine is touched. Stop fires many times per session; the copy is keyed by
session id and overwritten each time, so it converges to the final transcript
even if the laptop dies before SessionEnd.

`.sessions/` is gitignored: captured data never enters the public repository.
It reaches the team's PRIVATE archive only via `sync_sessions.py`, which
redacts credential patterns first. See docs/telemetry.md.

Fail-safe by design: a capture problem must never disrupt the person working,
so every path exits 0 and prints nothing on success.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    try:
        transcript = payload.get("transcript_path")
        session_id = str(payload.get("session_id") or "unknown")
        if not transcript:
            return 0
        src = Path(str(transcript)).expanduser()
        if not src.exists():
            return 0

        user = os.environ.get("GITHUB_USER") or os.environ.get("USER") or "unknown"
        sid8 = session_id[:8]
        out_dir = REPO_ROOT / ".sessions" / "transcripts"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out_dir / f"{user}-{sid8}.jsonl")

        index_entry = {
            "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": session_id,
            "user": user,
            "file": f"transcripts/{user}-{sid8}.jsonl",
            "event": payload.get("hook_event_name"),
        }
        with open(REPO_ROOT / ".sessions" / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(index_entry) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
