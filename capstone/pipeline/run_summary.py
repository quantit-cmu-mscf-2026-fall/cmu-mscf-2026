"""A finished search's run summary: usage, cost estimate, papers, decisions, variants.

`run_real_search` is `search.run_search` with every model call's usage recorded
(`usage.jsonl`, written even if the run fails) and `run_summary.json` written next to
the manifest: calls, an estimated cost (`usage.cost_estimate`, list prices; the
console's bill is the real one), the trial count, what each paper produced, the
deepen decisions with the agent's reason and the evidence it cited, and the variants
ranked by net Sharpe.

`summarize_workdir` rebuilds the summary from the files a finished search left (manifest,
factor store, saved summary frame, usage) and the ledger: no model, no data. It reads
the summary frame through `manifest.read_frames`, so an edited frame is refused.
Stage 3 is reported as the manifest has it ("unmeasured": it is not registered for this
run, with no `factors_digest` or `alpha3`, until `alpha3` is calibrated).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd

from capstone import gate, runlog
from capstone.pipeline import manifest, search
from capstone.pipeline.config import E2EConfig
from capstone.pipeline.usage import UsageRecorder, cost_estimate

__all__ = [
    "RUN_SUMMARY_FILE",
    "USAGE_FILE",
    "run_real_search",
    "summarize_workdir",
    "write_summary",
]

USAGE_FILE = "usage.jsonl"
RUN_SUMMARY_FILE = "run_summary.json"
TOP_VARIANTS = 10
PVALUE_NOTE = "pvalue_screen is unadjusted; the gate adjusts for trial_count"
_VARIANT_COLUMNS = (
    "factor_id",
    "hypothesis_id",
    "hold",
    "neutral",
    "expression",
    "sharpe_net",
    "sharpe_gross",
    "mean_turnover",
    "pvalue_screen",
)
_UNADJUSTED = {"pvalue_screen": "pvalue_screen_unadjusted"}  # the top table's column label


def _papers(con: sqlite3.Connection, seeds: tuple[str, ...]) -> list[dict]:
    """Per paper, in the order the store took them (not by `added_at`, a wall-clock
    second that ties or ticks by chance): its hypotheses and the factors they stored
    before deepening."""
    seed_set = set(seeds)
    out = []
    for key, title in con.execute("SELECT key, title FROM papers ORDER BY rowid"):
        hypotheses = [
            h for (h,) in con.execute("SELECT id FROM hypotheses WHERE paper_key = ?", (key,))
        ]
        stored = set()
        if hypotheses:
            marks = ",".join("?" * len(hypotheses))
            found = con.execute(
                "SELECT DISTINCT factor_id FROM proposals WHERE status = 'stored' AND "
                f"hypothesis_id IN ({marks})",
                hypotheses,
            )
            stored = {fid for (fid,) in found if fid in seed_set}
        out.append(
            {"key": key, "title": title, "hypotheses": len(hypotheses), "factors": len(stored)}
        )
    return out


def _decisions(con: sqlite3.Connection) -> list[dict]:
    """Each deepen round's decision, in order: who decided, the parent and edit, and
    (the agent's) reason and the memory evidence it cited."""
    rows = con.execute("SELECT from_id, reason FROM moves WHERE action = 'deepen' ORDER BY id")
    out = []
    for parent, reason in rows:
        record = json.loads(reason)
        out.append(
            {
                "by": record.get("by"),
                "parent": parent,
                "motif": record.get("motif"),
                "reason": record.get("reason", ""),
                "evidence_used": record.get("evidence_used", ""),
            }
        )
    return out


def _variants(summary: pd.DataFrame) -> dict:
    if summary.empty:
        return {"count": 0, "positive_net": 0, "top": [], "note": PVALUE_NOTE}
    ranked = summary.sort_values("sharpe_net", ascending=False, kind="stable")
    keep = [c for c in _VARIANT_COLUMNS if c in summary.columns]
    top = ranked.head(TOP_VARIANTS)[keep].reset_index()
    top = top.rename(columns=_UNADJUSTED)
    return {
        "count": int(len(summary)),
        "positive_net": int((summary["sharpe_net"] > 0).sum()),
        "top": json.loads(top.to_json(orient="records")),
        "note": PVALUE_NOTE,
    }


def summarize_workdir(workdir: str | Path) -> dict:
    """The run's summary (module docstring), from a finished search's files and the ledger."""
    workdir = Path(workdir)
    m = manifest.read_manifest(workdir / manifest.MANIFEST_FILE)
    lines = (workdir / USAGE_FILE).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line]
    _, frame = manifest.read_frames(workdir, m)
    con = sqlite3.connect(workdir / "factors.db")
    try:
        papers, decisions = _papers(con, m.factor_ids), _decisions(con)
        found = con.execute("SELECT status FROM memory_events WHERE run_id = ?", (m.run_id,))
        statuses = Counter(status for (status,) in found)
    finally:
        con.close()
    looks = [
        e
        for e in runlog.read_entries()
        if e.get("name") == gate.LOOK_NAME and m.gate_config_hash in json.dumps(e)
    ]
    return {
        "run_id": m.run_id,
        "run_tag": m.run_tag,
        "data_version": m.data_version,
        "model_calls": m.model_calls,
        "usage": cost_estimate(records),
        "trial_count": m.trial_count,
        "papers": papers,
        "seed_factors": len(m.factor_ids),
        "deepen": {
            "rounds": m.deepen_rounds,
            "stopped": m.deepen_stopped,
            "children": dict(sorted(statuses.items())),
            "decisions": decisions,
        },
        "families": list(m.families),
        "n_variants": m.n_variants,
        "variants": _variants(frame),
        "gate_config_hash": m.gate_config_hash,
        "n_boot": m.n_boot,
        "stage3": m.stage3,
        "thresholds_source": m.thresholds_source,
        "holdout_opened": bool(looks),
    }


def write_summary(workdir: str | Path) -> dict:
    """Rebuild the summary from `workdir` and write `run_summary.json` there."""
    workdir = Path(workdir)
    summary = summarize_workdir(workdir)
    (workdir / RUN_SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_real_search(
    config: E2EConfig,
    *,
    client,
    workdir: str | Path,
    editor=None,
    loader=None,
    source_version: str | None = None,
    allow_default_thresholds: bool = False,
) -> dict:
    """`run_search` with every call's usage recorded; returns (and writes) the summary.

    A workdir that already holds a search is refused before anything runs. The usage file
    is written even if the search fails, once the search has made its workdir."""
    workdir = Path(workdir)
    search.refuse_existing(workdir)
    recorder = UsageRecorder(client)
    try:
        search.run_search(
            config,
            client=recorder,
            workdir=workdir,
            editor=editor,
            loader=loader,
            source_version=source_version,
            allow_default_thresholds=allow_default_thresholds,
        )
    finally:
        if workdir.is_dir():
            lines = "".join(json.dumps(r) + "\n" for r in recorder.records)
            (workdir / USAGE_FILE).write_text(lines, encoding="utf-8")
    return write_summary(workdir)
