"""Immutable, bounded discovery study. Registration never reads performance results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capstone import runlog
from capstone.release import release_identity, runtime_versions
from capstone.strategy.research import canonical, contracts, digest

STUDY_VERSION = "historical-qualification-1"
SIDES = ("long_only", "long_short")
NEW_METHODS = {
    "value": ((0.1, "equal"), (0.2, "equal"), (0.1, "capitalization")),
    "profitability": ((0.3, "capitalization"), (0.2, "capitalization"), (0.3, "equal")),
}


def definition():
    candidates = []
    for claim in contracts()["claims"]:
        if claim["status"] != "executable":
            continue
        for construction in ("reference", "alternate_weighting"):
            for side in SIDES:
                candidates.append(
                    {
                        "family": claim["id"],
                        "method": claim["method"],
                        "construction": construction,
                        "side": side,
                    }
                )
    for method, variations in NEW_METHODS.items():
        for tail, weighting in variations:
            for side in SIDES:
                candidates.append(
                    {
                        "family": f"paper-{method}",
                        "method": method,
                        "construction": "fundamental",
                        "tail": tail,
                        "weighting": weighting,
                        "side": side,
                    }
                )
    for candidate in candidates:
        candidate["candidate_id"] = digest(candidate)[:16]
    return {
        "version": STUDY_VERSION,
        "seed": 404,
        "bootstrap_draws": 1999,
        "research_q": 0.05,
        "candidate_limit": 36,
        "candidates": candidates,
        "history": ["1990-01-01", "1995-12-31"],
        "evaluation": ["1996-01-01", "2020-12-31"],
        "anchor": "1992-12-31",
        "capital_per_profile": 100000.0,
        "walk_forward": [[2005, 2006, 2010], [2010, 2011, 2015], [2015, 2016, 2020]],
        "minimum_training_observations": 252,
        "costs": {
            "commission_bps": 1,
            "slippage_bps": 5,
            "spread": "previous_completed_session_qualified_half_spread",
            "short_borrow_bps_annual": 200,
            "financing_bps_annual": 500,
        },
        "cost_scenarios": {"base": 1, "stress": 2},
        "zero_cost": "diagnostic_counterfactual_never_selected",
        "basket_sets": ["reference", "expanded", "training_selected"],
        "allocations": ["equal_family", "correlation_groups"],
        "correlation": {"positive_threshold": 0.8, "minimum_overlap": 252},
        "benchmarks": [
            "cash",
            "cap_weight_universe_proxy",
            "equal_weight_universe_proxy",
            "known_momentum_12_1/long_only",
            "known_momentum_12_1/long_short",
        ],
        "diagnostics": [
            "five_year_periods",
            "factor_and_sector_exposures",
            "rolling_correlation",
            "effective_dimensions",
            "family_contributions",
            "leave_one_family_out",
            "cost_components",
            "legacy_saved_results_separate",
            "cohort_january",
        ],
        "selection": "prior_training_sharpe_max_then_id; family_bootstrap_BH_0.05; empty_cash",
        "new_sources": {
            "value": "https://doi.org/10.1111/j.1540-6261.1992.tb04398.x",
            "profitability": "https://mysimon.rochester.edu/novy-marx/data_lib/OSoV/pmu_details.html",
        },
        "fundamentals": {
            "vintage_required": True,
            "exclude_sic": [6000, 6999],
            "formation": "June",
            "holding_months": 12,
            "rebalance": "monthly_within_annual_selection",
            "universe": "point_in_time_sp500_nonfinancial_companies",
        },
        "blocked_slots_replaced": False,
        "reused_discovery": True,
        "holdout_opened": False,
        "formal_admission": False,
        "broker_connection": False,
    }


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
    return target


def freeze(path, *, sources=None):
    target = Path(path)
    if target.exists():
        raise FileExistsError("study registration cannot be overwritten")
    value = {
        "schema_version": 1,
        "study": definition(),
        "release": release_identity(),
        "runtime": runtime_versions(),
        "sources": {label: file_digest(p) for label, p in sorted((sources or {}).items())},
    }
    value["study_hash"] = digest(value)
    runlog.log_run(
        "qualification-study-registration",
        seed=404,
        params={"study_hash": value["study_hash"], "candidate_limit": 36},
        tags=[STUDY_VERSION, "registration"],
        notes="Before historical results.",
    )
    write_once(target, value)
    return value


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest({k: v for k, v in value.items() if k != "study_hash"}) != value["study_hash"]:
        raise ValueError("study registration changed")
    study = value["study"]
    ids = [c["candidate_id"] for c in study["candidates"]]
    if len(ids) != 36 or len(set(ids)) != 36 or study["evaluation"][1] != "2020-12-31":
        raise ValueError("invalid frozen study budget or boundary")
    return value


def reconcile(registration, entries, variant_ids, decisions):
    """Explain every source row; repeats remain trials unless explicitly corrected upstream."""
    if not registration.get("study_hash"):
        raise ValueError("register the study before reconciling research")
    variants = set(map(str, variant_ids))
    classified, blockers, covered = [], [], set()
    allowed = {"strategy_trial", "rerun", "diagnostic", "methodology", "correction", "superseded"}
    for i, entry in enumerate(entries):
        decision = decisions.get(str(i), {})
        kind = decision.get("classification")
        matched = set(decision.get("variant_ids", []))
        if kind not in allowed or not decision.get("reason"):
            blockers.append(f"unclassified_source_row:{i}")
        if matched - variants:
            blockers.append(f"unknown_return_column:{i}")
        covered.update(matched)
        classified.append(
            {
                "source_index": i,
                "source_entry_hash": digest(entry),
                "name": entry.get("name"),
                "timestamp": entry.get("ts_utc"),
                **decision,
            }
        )
    blockers += [f"unmapped_return_column:{v}" for v in sorted(variants - covered)]
    count = sum(
        row.get("classification") in {"strategy_trial", "rerun", "superseded"} for row in classified
    )
    return {
        "study_hash": registration["study_hash"],
        "source_entries": len(entries),
        "return_columns": len(variants),
        "classified_entries": classified,
        "conservative_trial_entries": count,
        "blockers": blockers,
        "reconciled": not blockers,
        "count_policy": "Entry count is not automatically the family bootstrap variant count.",
    }


def record_reconciliation(path, result):
    target = Path(path)
    if target.exists():
        raise FileExistsError("reconciliation records are append-only")
    runlog.log_run(
        "qualification-ledger-reconciliation",
        params={
            "study_hash": result["study_hash"],
            "record_hash": digest(result),
            "reconciled": result["reconciled"],
        },
        tags=[STUDY_VERSION, "accounting"],
        notes="Original records remain unchanged.",
    )
    return write_once(target, result)
