"""Typed program evidence routing: admission only from a verified program gate report.

A program or saved portfolio is never admitted by inherited or supplied evidence. Its
report (`capstone.program_gate`) is re-verified against its procedure, registered
thresholds and compatible reconciliation every time: a search-stage report that passes
it is `search_stage_compatible` with the single blocker `final_holdout_not_authorized`;
only a final report under a spent, authorized holdout protocol admits. Anything else is
`program_validation_evidence_incompatible`.

`thresholds` is the registered-threshold file a bundle recorded for the report; a
changed recorded file is not `incompatible` evidence but the refusal
`ThresholdRecordMismatch`, raised to the caller.

A search-stage report whose only blocker is the sealed holdout is `incubation_eligible`.
That never admits it (`admitted` and `formal_admission` stay false); it only lets a plan
be built for it on the incubation transmission route: a paper account, capped capital,
the report re-verified at send. The final route is unchanged.
"""

import sqlite3
from pathlib import Path

from capstone.reference import admission as reference_admission
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec

PENDING = "program_financial_admission_pending"
FINAL_PENDING = "final_holdout_not_authorized"
INCOMPATIBLE = "program_validation_evidence_incompatible"
SEARCH_STAGE = "search_stage_compatible"
FINAL = "final_holdout_accepted"
INCUBATION_ROUTE = "incubation-only"  # the registry role this route serves
INCUBATION_UNQUALIFIED = "incubation_route_unqualified"


def _gated(spec, report_path, identity, missing, thresholds=None):
    result = identity | {
        "admitted": False,
        "formal_admission": False,
        "validation_status": "missing",
        "report_hash": None,
        "blockers": [PENDING, missing],
    }
    if report_path is None:
        return result
    # lazy: the gate module imports the execution bundle, which imports this module
    from capstone import program_gate
    from capstone.pipeline.registered_thresholds import ThresholdRecordMismatch

    try:
        report = program_gate.verify(report_path, spec, thresholds)
    except ThresholdRecordMismatch:
        raise  # a typed refusal, never silently incompatible evidence
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, sqlite3.Error):
        return result | {"validation_status": "incompatible", "blockers": [PENDING, INCOMPATIBLE]}
    final = report["holdout_opened"] is True
    return result | {
        "admitted": final,
        "formal_admission": final,
        "validation_status": FINAL if final else SEARCH_STAGE,
        "report_hash": report["id"],
        "report_path": str(Path(report_path).resolve()),
        "procedure_id": report["procedure_id"],
        "blockers": [] if final else [FINAL_PENDING],
    }


def program_admission(spec, report_path=None, thresholds=None):
    if not isinstance(spec, ProgramStrategySpec):
        raise ValueError("program evidence requires a schema-six program")
    identity = {
        "strategy_id": spec.id,
        "strategy_schema": 6,
        "implementation": spec.implementation,
        "claim_id": spec.claim_id,
        "method_spec_id": spec.method_spec_id,
        "family_id": spec.hypothesis_id,
    }
    return _gated(spec, report_path, identity, "reviewed_gate_report_missing", thresholds)


def admission(spec, report_path=None, thresholds=None):
    if isinstance(spec, SavedProgramPortfolioSpec):
        identity = {
            "strategy_id": spec.id,
            "strategy_schema": 7,
            "implementation": spec.implementation,
        }
        missing = "saved_portfolio_financial_admission_pending"
        return _gated(spec, report_path, identity, missing, thresholds)
    if isinstance(spec, ProgramStrategySpec):
        return program_admission(spec, report_path, thresholds)
    return reference_admission(spec, report_path)


def plan_admission(spec, saved, thresholds=None):
    """Saved bundle evidence is re-verified from its report; it never grants admission itself.

    `thresholds` is the registered-threshold file the bundle recorded for this strategy."""
    if not isinstance(spec, (ProgramStrategySpec, SavedProgramPortfolioSpec)):
        if not isinstance(saved, dict) or not isinstance(saved.get("blockers"), list):
            raise ValueError("saved financial evidence is missing or malformed")
        return saved
    report = saved.get("report_path") if isinstance(saved, dict) else None
    expected = admission(spec, report, thresholds)
    return (
        expected
        if saved is None or saved == expected
        else admission(spec, "unsupported-supplied-evidence")
    )


def incubation_eligible(value):
    """A re-verified search-stage admission whose one blocker is the sealed holdout.

    Only such evidence may travel the incubation route; a missing, incompatible or final
    report may not (a final one takes the final route)."""
    return (
        isinstance(value, dict)
        and value.get("validation_status") == SEARCH_STAGE
        and value.get("blockers") == [FINAL_PENDING]
        and value.get("admitted") is False
        and value.get("formal_admission") is False
        and isinstance(value.get("report_path"), str)
    )
