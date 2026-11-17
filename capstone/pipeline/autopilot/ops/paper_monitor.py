"""Declares the optional `[paper_monitor]` config section: a folder of paper results to show.

`root` names a paper tick's output folder. Lifecycle recommendations read the monitor
store under `<root>/monitor` as well as `<run>/monitor`, so the same frozen rules can
recommend retiring or suspending a strategy on paper results (a person still confirms).
Unset (the default), nothing is read and a run is byte for byte what it was. An ops
section: not part of any stage's identity. No after-tick hook, and nothing here connects
to a broker or an account: it only reads files that a paper run left on disk.

With `root` set, `status.json` / `status.md` also show that folder's last alert
(`<root>/ALERT.json`) as `paper_alert`. Unset, neither document changes.
"""

import json
from pathlib import Path

from capstone.pipeline.autopilot.config import Key

SECTION = "paper_monitor"
SECTIONS = {
    SECTION: {
        "root": Key(
            "path",
            None,
            "a paper tick's output folder; lifecycle recommendations read <root>/monitor too",
        ),
    }
}


def root(ctx):
    """The configured paper root, or None."""
    return (ctx.config.sections.get(SECTION) or {}).get("root")


def status_fields(config):
    """`{"paper_alert": <root>/ALERT.json or None}` with `[paper_monitor] root` set, else
    `{}` (the status documents are then unchanged)."""
    paper = (config.sections.get(SECTION) or {}).get("root")
    if paper is None:
        return {}
    try:
        alert = json.loads((Path(paper) / "ALERT.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        alert = None
    return {"paper_alert": alert}
