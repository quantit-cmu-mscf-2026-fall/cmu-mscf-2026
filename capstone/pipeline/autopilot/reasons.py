"""Typed blocker and STOP reasons, grouped into operator classes.

A stage declares every code it may return (`Stage.reasons`); the runner refuses an
undeclared one. A code that is already a row of `blocker_glossary.CLEARED_BY` takes its
`cleared_by` from there, so the glossary stays the single source for shared blockers.
"""

from __future__ import annotations

from dataclasses import dataclass

from capstone.pipeline import blocker_glossary

CLASSES = (
    "Source",
    "Claim",
    "Spend",
    "Data",
    "Statistics",
    "Admission",
    "Planning",
    "Broker",
    "Operator",
)


@dataclass(frozen=True)
class ReasonSpec:
    """One declared reason: its operator class and what clears it."""

    klass: str
    cleared_by: str | None = None  # None: the blocker glossary row of the same code

    def __post_init__(self):
        if self.klass not in CLASSES:
            raise ValueError(f"unknown blocker class {self.klass!r}")


# Reasons the core itself raises; every stage may also return these.
CORE = {
    "stage_not_implemented": ReasonSpec("Operator", "the stage's owner implements it"),
    "stage_crashed": ReasonSpec("Operator", "read `detail`; fix the cause; the next tick retries"),
    "stage_crash_limit": ReasonSpec(
        "Operator", "fix the crash, then `autopilot clear --stage <name>`"
    ),
    "undeclared_reason": ReasonSpec("Operator", "declare the code in the stage's `reasons`"),
    "no_stage_artifact": ReasonSpec("Operator"),
    "artifact_changed": ReasonSpec(
        "Operator", "restore the recorded artifact, or start a new run root"
    ),
    "config_section_changed": ReasonSpec(
        "Operator", "restore the section the stage started with, or start a new run root"
    ),
    "policy_changed": ReasonSpec(
        "Operator", "restore `[autopilot] policy_id`, or start a new run root"
    ),
    "autopilot_locked": ReasonSpec("Operator", "another tick owns the run; wait for it"),
    "ops_hook_failed": ReasonSpec("Operator", "read `detail`; fix the operations hook"),
    # Spend guard and model-client guard (Context.model_client): STOPs
    "spend_guard_missing": ReasonSpec(
        "Spend", "the caller supplies a spend guard to `tick` before a spending stage runs"
    ),
    "spend_run_ceiling_exceeded": ReasonSpec(
        "Spend", "lower the ceiling, or set `[spend] run_ceiling` in a new run root"
    ),
    "spend_global_ceiling_exceeded": ReasonSpec(
        "Spend", "a new standing authorization; the global ceiling is never raised here"
    ),
    "spend_accounting_unreadable": ReasonSpec(
        "Spend",
        "make the registration and every run in `[spend] runs_dir` / `accounting_roots` readable",
    ),
    "spend_policy_changed": ReasonSpec(
        "Spend", "restore the `[spend]` ceilings the run started with, or start a new run root"
    ),
    "spend_lock_busy": ReasonSpec(
        "Spend", "another run held `<runs_dir>/.spend-guard.lock`; the next tick retries"
    ),
    "model_client_not_authorized": ReasonSpec(
        "Spend",
        "a person sets `[autopilot] real_model_calls = true` (never under pytest), then clears",
    ),
}


def record(code, detail, declared):
    """The journal form of a reason."""
    spec = declared[code]
    return {
        "code": code,
        "class": spec.klass,
        "detail": str(detail),
        "cleared_by": spec.cleared_by or blocker_glossary.cleared_by(code),
    }


class Blocked(Exception):
    """The stage cannot proceed until something outside it changes; the next tick retries."""

    status = "blocked"

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


class Stop(Blocked):
    """A STOP: the whole run halts until a person runs `autopilot clear`."""

    status = "stopped"


class NotImplementedBlocker(Blocked):
    """Raised by a stub stage whose owner has not implemented it yet."""

    def __init__(self, stage, unit):
        super().__init__("stage_not_implemented", f"{stage} is owned by {unit}")
