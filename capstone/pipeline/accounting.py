"""Read-only reconciliation of original adaptive evaluations and exploratory baskets.

The saved return matrix is only identified here (by digest) and by its column
names. Names come from an injected `matrix_columns` loader, so this module never
opens an archive itself; callers supply a loader that reads through
`capstone.shared_data` or a team-approved derived-archive reader.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from capstone import runlog
from capstone.pipeline.study import STUDY_VERSION, file_digest, reconcile, record_reconciliation


@contextmanager
def ledger_at(directory):
    previous = os.environ.get("CAPSTONE_LEDGER_DIR")
    os.environ["CAPSTONE_LEDGER_DIR"] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CAPSTONE_LEDGER_DIR", None)
        else:
            os.environ["CAPSTONE_LEDGER_DIR"] = previous


def reconcile_sources(registration, original_ledger, matrix, basket_ledger, out, *, matrix_columns):
    for label, path in (("original_ledger", original_ledger), ("original_return_matrix", matrix)):
        if registration["sources"].get(label) != file_digest(path):
            raise ValueError("legacy source differs from the frozen registration")
    with ledger_at(Path(original_ledger).parent):
        original = [e for e in runlog.read_entries() if "e2e-run:crsp-v1" in e.get("tags", [])]
        original_count = runlog.trial_count(include_tags=["e2e-run:crsp-v1"])
    with ledger_at(Path(basket_ledger).parent):
        basket = runlog.read_entries()
    variants = [
        name for name in matrix_columns(matrix) if name not in {"date", "__index_level_0__"}
    ]
    decisions = {}
    entries = original + basket
    for i, entry in enumerate(entries):
        name, params = entry.get("name"), entry.get("params", {})
        if name in {"evaluate-factor", "deepen"} and params.get("variant_id"):
            reason = (
                "Saved evaluated variant"
                if name == "evaluate-factor"
                else "Adaptive reward evaluation retained conservatively even when saved again"
            )
            decision = {
                "classification": "strategy_trial",
                "reason": reason,
                "variant_ids": [params["variant_id"]],
            }
        elif name == "crsp-v1-basket-exploration-registration":
            decision = {
                "classification": "strategy_trial",
                "reason": "Registered exploratory basket informed subsequent research",
                "variant_ids": [],
            }
        elif name == "crsp-v1-basket-exploration-result":
            decision = {
                "classification": "diagnostic",
                "reason": "Result of the registered basket, not another evaluation",
                "variant_ids": [],
            }
        else:
            decision = {}
        decisions[str(i)] = decision
    result = reconcile(registration, entries, variants, decisions)
    result["source_summary"] = {
        "original_runlog_count": original_count,
        "original_entries": len(original),
        "basket_records": len(basket),
        "basket_ledger_hash": file_digest(basket_ledger),
        "saved_return_columns": len(variants),
        "deepen_entries": sum(e.get("name") == "deepen" for e in original),
    }
    record_reconciliation(out, result)
    return result


def import_opportunities(accounting):
    if not accounting.get("reconciled") or accounting.get("blockers"):
        raise ValueError("unresolved accounting cannot authorize a research screen")
    eligible = [
        row
        for row in accounting["classified_entries"]
        if row["classification"] in {"strategy_trial", "rerun", "superseded"}
    ]
    if len(eligible) != accounting["conservative_trial_entries"]:
        raise ValueError("accounting count and classified opportunities disagree")
    for row in eligible:
        runlog.log_run(
            "qualification-imported-opportunity",
            params={
                "source_entry_hash": row["source_entry_hash"],
                "source_index": row["source_index"],
                "original_study_hash": accounting["study_hash"],
            },
            tags=[STUDY_VERSION, "eligible-research-trial"],
            notes=row["reason"],
        )
    return runlog.trial_count("qualification-imported-opportunity") if eligible else 0
