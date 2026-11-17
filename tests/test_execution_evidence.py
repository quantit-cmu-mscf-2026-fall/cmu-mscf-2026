"""Evidence routing: nothing admits without a re-verified gate report (synthetic fakes)."""

import sys
import types

import pytest

from capstone.execution import evidence
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec


class ThresholdRecordMismatch(Exception):
    pass


class SavedShell(SavedProgramPortfolioSpec):
    """Routing reads only identity, so the book machinery is not built."""

    id = "s" * 64
    implementation = "impl"


@pytest.fixture
def program():
    return ProgramStrategySpec(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="high",
        data_version="evidence-fixture-v1",
        anchor="2000-01-31",
        side="long_short",
        tail=0.5,
        schedule=ProgramSchedule(kind="month_end", interval=1),
    )


@pytest.fixture
def saved():
    return object.__new__(SavedShell)


@pytest.fixture
def gate(monkeypatch):
    """Fake the gate verifier and threshold-record modules, which are lazy imports."""
    state = types.SimpleNamespace(report=None, error=None)
    module = types.ModuleType("capstone.program_gate")

    def verify(path, spec, thresholds=None):
        if state.error is not None:
            raise state.error
        return state.report

    module.verify = verify
    thresholds = types.ModuleType("capstone.pipeline.registered_thresholds")
    thresholds.ThresholdRecordMismatch = ThresholdRecordMismatch
    monkeypatch.setitem(sys.modules, "capstone.program_gate", module)
    monkeypatch.setitem(sys.modules, "capstone.pipeline.registered_thresholds", thresholds)
    monkeypatch.setattr(sys.modules["capstone"], "program_gate", module, raising=False)
    return state


def report(holdout_opened):
    return {"id": "r" * 64, "procedure_id": "p" * 64, "holdout_opened": holdout_opened}


def test_program_without_report_is_pending_never_admitted(program):
    result = evidence.admission(program)
    assert result["admitted"] is False and result["formal_admission"] is False
    assert result["blockers"] == [evidence.PENDING, "reviewed_gate_report_missing"]
    assert result["strategy_schema"] == 6 and result["report_hash"] is None


def test_program_admission_rejects_other_specs():
    with pytest.raises(ValueError, match="schema-six program"):
        evidence.program_admission(object())


def test_saved_portfolio_without_report_is_pending(saved):
    result = evidence.admission(saved)
    assert result["strategy_schema"] == 7
    assert result["blockers"] == [evidence.PENDING, "saved_portfolio_financial_admission_pending"]
    assert result["admitted"] is False


def test_search_stage_report_is_compatible_but_not_admitted(program, gate, tmp_path):
    gate.report = report(holdout_opened=False)
    result = evidence.admission(program, tmp_path / "r.json")
    assert result["validation_status"] == evidence.SEARCH_STAGE
    assert result["blockers"] == [evidence.FINAL_PENDING]
    assert result["admitted"] is False and result["formal_admission"] is False
    assert evidence.incubation_eligible(result)


def test_only_a_final_report_admits(program, gate, tmp_path):
    gate.report = report(holdout_opened=True)
    result = evidence.admission(program, tmp_path / "r.json")
    assert result["admitted"] is True and result["formal_admission"] is True
    assert result["validation_status"] == evidence.FINAL and result["blockers"] == []
    assert not evidence.incubation_eligible(result)


@pytest.mark.parametrize("error", [ValueError("x"), KeyError("x"), OSError("x")])
def test_unverifiable_report_is_incompatible(program, gate, tmp_path, error):
    gate.error = error
    result = evidence.admission(program, tmp_path / "r.json")
    assert result["validation_status"] == "incompatible"
    assert result["blockers"] == [evidence.PENDING, evidence.INCOMPATIBLE]
    assert not result["admitted"] and not evidence.incubation_eligible(result)


def test_threshold_record_mismatch_is_raised_not_swallowed(program, gate, tmp_path):
    gate.error = ThresholdRecordMismatch("changed")
    with pytest.raises(ThresholdRecordMismatch):
        evidence.admission(program, tmp_path / "r.json")


def test_supplied_evidence_never_grants_admission(program, gate):
    gate.error = ValueError("unsupported")
    forged = evidence.admission(program) | {"admitted": True, "formal_admission": True}
    result = evidence.plan_admission(program, forged)
    assert result["admitted"] is False
    assert result["blockers"] == [evidence.PENDING, evidence.INCOMPATIBLE]


def test_plan_admission_reverifies_the_saved_report(program, gate, tmp_path):
    gate.report = report(holdout_opened=False)
    saved_evidence = evidence.admission(program, tmp_path / "r.json")
    assert evidence.plan_admission(program, saved_evidence) == saved_evidence
    gate.error = ValueError("report changed")
    result = evidence.plan_admission(program, saved_evidence)
    assert result["admitted"] is False
    assert result["blockers"] == [evidence.PENDING, evidence.INCOMPATIBLE]


def test_plan_admission_passes_reference_evidence_through():
    saved_evidence = {"blockers": ["reviewed_gate_report_missing"]}
    assert evidence.plan_admission(object(), saved_evidence) is saved_evidence
    for bad in (None, {}, {"blockers": "x"}):
        with pytest.raises(ValueError, match="missing or malformed"):
            evidence.plan_admission(object(), bad)


@pytest.mark.parametrize(
    "change",
    [
        {"validation_status": "missing"},
        {"blockers": []},
        {"admitted": True},
        {"formal_admission": True},
        {"report_path": None},
    ],
)
def test_incubation_eligible_is_strict(change):
    base = {
        "validation_status": evidence.SEARCH_STAGE,
        "blockers": [evidence.FINAL_PENDING],
        "admitted": False,
        "formal_admission": False,
        "report_path": "r.json",
    }
    assert evidence.incubation_eligible(base)
    assert not evidence.incubation_eligible(base | change)
    assert not evidence.incubation_eligible(None)
