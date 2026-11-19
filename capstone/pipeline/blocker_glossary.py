"""Typed blocker -> the step or decision that clears it.

Single source of truth for what a status report says clears a blocker. A blocker is
named by a stable code (or message); each row says what a person or a later stage has
to supply for it to go away. Rows describe the clearing step in plain capability
names, so the table reads without any internal plan.
"""

from __future__ import annotations

ADMISSION = "a program gate report and an active registry state"
BROKER = "broker acceptance"
DAILY = "a panel that holds the as-of session"
QUOTE = "an eligibility rule that excludes the name; a data gap is a data request"
HOLDOUT = "never in discovery; only the final protocol plus manager authorization"
UNSUPPORTED = "not supported in order plans"

CLEARED_BY = {
    # Sources
    "source_missing": "supply the file; start a new batch, or resume with a refresh",
    "pdf_requires_ocr": "supply an `inline_text` excerpt; there is no OCR engine",
    "pdf_partial_requires_ocr": (
        "supply an `inline_text` excerpt; a page subset is never archived as the source"
    ),
    # Claims, programs and model calls
    "incomplete_specification": (
        "declared defaults; re-register (a bounded number of times per packet)"
    ),
    "constructor_unavailable": (
        "a calendar schedule version that supports it; event/cohort/cross-entity ideas stay blocked"
    ),
    "needs_data": "freeze with an input archive; else a data request",
    "budget_exhausted": "a new registration within the budget ceilings",
    "request_abandoned": "none: an explicit operator abandon is the recorded outcome",
    "ResearchPaused": (
        "explicit journal retry/abandon; two pauses in a row: report, do not restart"
    ),
    # Historical evaluation
    "missing_qualified_quote": QUOTE,
    "nonpositive_sizing_price": QUOTE,
    "nonpositive_formation_capitalization": QUOTE + " (cap-weighted constructions only)",
    "UnqualifiedCost": "cost eligibility before ranking",
    "program_corporate_action_transition_unqualified": (
        "the default cash-in-lieu-following-open treatment; observed terms via a data request"
    ),
    "HoldoutError": HOLDOUT,
    "HoldoutSealed": HOLDOUT,
    # Incremental value
    "factor_provenance_unqualified": "a qualified daily factor frame (a data request)",
    "threshold_unregistered": "registered thresholds (a gate registration for this frame)",
    # Portfolios
    "empty_pool_cash_control": (
        "qualified pool members from a real evaluation; proves nothing about combinations"
    ),
    "insufficient_sessions": "more overlapping sessions",
    "benchmark_control_missing": "a completed benchmark control",
    "benchmark_control_unavailable_on_prefix_route": (
        "the prefix run's benchmark control receipts (restart the prefix run)"
    ),
    # Daily plans: validation blockers
    "program_financial_admission_pending": ADMISSION,
    "reviewed_gate_report_missing": "a reviewed gate report",
    "saved_portfolio_financial_admission_pending": "a reviewed gate report",
    "program_validation_evidence_incompatible": (
        "a gate report bound to this procedure and reconciliation"
    ),
    "final_holdout_not_authorized": (
        "a final gate report under an authorized final holdout protocol (who may authorize "
        "opening the holdout is a manager decision)"
    ),
    # Daily plans: planning blockers
    "execution_cost_assumption_missing": "pass a cost in bps when building the bundle",
    "outstanding_orders_require_reconciliation": "reconcile open orders in the account file",
    "rounded_cash_deficit": "capital or limits; whole-share rounding overspent",
    "rounded_gross_limit": "limits file or capital",
    "rounded_single_name_limit": "limits file or capital",
    "long_only_short_position": "fix the account or the profile",
    "order_notional_limit": "limits file",
    # Daily plans: broker blockers
    "broker_receivable_recovery_unqualified": BROKER,
    "reviewed_deployment_release_required": "a clean tree at a `deploy-*` tag",
    "paper_account_verified": BROKER,
    "permissions_verified": BROKER,
    "funds_verified": BROKER,
    "data_verified": BROKER,
    "shorting_verified": BROKER,
    "contracts_unqualified": "broker contract qualification (`con_id`)",
    # Execution contracts
    "variable_program_allocations_unqualified": UNSUPPORTED,
    "asynchronous_program_formations_unqualified": UNSUPPORTED,
    "selected_portfolio_retired_use_saved_portfolio": "register a saved portfolio instead",
    "PaperGuardError": "never cleared here: the broker stays disconnected",
    # Lifecycle stages and operator errors
    "missing input": "run the command named after `produce it with:`",
    "missing_input": "run the command the failure's `error` names after `produce it with:`",
    "stage_refused": "read the failure's `error`; fix the input and rerun the same command",
    "price_missing": (
        "a halted name's carried price from the session's marks (bounded by the export's "
        "halt-session limit), or a fixed export or universe"
    ),
    "snapshot must contain the completed decision session": DAILY,
    "UnresolvedInstructions": (
        "reconcile the prepared instructions only (or abandon them), then carry again"
    ),
    "journal does not chain": (
        "record the event in the account's corporate events or carry the account"
    ),
    "fault proof refused": "run the fault scenarios; read the fault results `refusal`",
    "receipt_stale": "fresh prefix receipts (restart the prefix run)",
    "current_authorization_refused": "an active current authorization covering the window",
    "current_authorization_invalid": "re-decide from a version whose decisions verify",
    "decide_refused": "read the failure's `error`; an on-schedule date after the last decision",
    "no_stage_artifact": "the stage never wrote a key artifact; rerun it",
    "retirement recommended": "a person retires the strategy; a registry commit",
    "promotion eligible": "a person promotes it with a confirm record",
}
UNMAPPED = "unmapped: no glossary row for this blocker"


def cleared_by(blocker):
    """What clears `blocker`: an exact row, else the longest row contained in it."""
    text = str(blocker)
    if text in CLEARED_BY:
        return CLEARED_BY[text]
    hits = [key for key in CLEARED_BY if key in text]
    return CLEARED_BY[max(hits, key=len)] if hits else UNMAPPED
