"""Synthetic registration records only; no calibration or financial admission."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier

import pytest

from capstone import runlog
from capstone.factors import procedure_records
from capstone.factors.procedure_contracts import SCOPE_RESOLUTION, ProcedureSpec
from capstone.factors.procedure_records import (
    ARM_CALLS,
    CALIBRATION,
    CALIBRATION_LABEL,
    COMPLETION,
    INHERITED,
    OPPORTUNITY,
    PREFIX,
    RECONCILIATION,
    RUN,
    calibration_qualified,
    complete_opportunity,
    inherit_reconciliation,
    nothing_shown,
    opportunity_record,
    reconcile_opportunities,
    record_arm_calls,
    record_prefix,
    register_calibration,
    register_procedure,
    register_run,
    registered_opportunity,
    require_compatible_reconciliation,
    reserve_opportunity,
    save_scope_resolution,
)
from capstone.factors.research_contracts import canonical, digest
from capstone.factors.research_records import load_artifact, put_artifact

COUNTS = [12, 9]


def h(value):
    return digest({"synthetic": value})


class Controller:
    """The two things procedure records need from a research controller: an append-only
    event table plus artifacts in one SQLite connection, and a transaction."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level=None, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS research_events
              (seq INTEGER PRIMARY KEY, brief TEXT NOT NULL, state TEXT NOT NULL,
               reason TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON research_events
              BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON research_events
              BEGIN SELECT RAISE(ABORT,'append only'); END;
        """)

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def events(self):
        return list(self.db.execute("SELECT * FROM research_events ORDER BY seq"))


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    result = Controller(tmp_path / "controller.db")
    yield result
    result.db.close()


def inherited(controller, tmp_path, **changes):
    row = {
        "study_hash": h("old-study"),
        "source_entries": 2,
        "return_columns": 1,
        "classified_entries": [
            {
                "source_index": 0,
                "classification": "strategy_trial",
                "reason": "synthetic earlier trial",
                "variant_ids": ["old"],
                "source_entry_hash": h(0),
            },
            {
                "source_index": 1,
                "classification": "diagnostic",
                "reason": "synthetic repeated metadata",
                "variant_ids": [],
                "source_entry_hash": h(1),
            },
        ],
        "conservative_trial_entries": 1,
        "blockers": [],
        "reconciled": True,
        "count_policy": "Entry count is not automatically the family bootstrap variant count.",
        **changes,
    }
    path = tmp_path / "inherited.json"
    path.write_text(canonical(row), encoding="utf-8")
    return inherit_reconciliation(controller, path), path


def spec(controller, tmp_path, *, resolved=False, model_calls=6, budgets=None):
    old, _ = inherited(controller, tmp_path)
    legacy = {
        "state": "unresolved",
        "reported_counts": COUNTS,
        "inherited_classification_id": old,
        "resolution_rule_id": None,
        "justification": "",
        "evidence_ids": [],
        "effective_scope": None,
    }
    if resolved:
        justification = "Synthetic structural fixture only; no real effective scope resolved."
        effective_scope = {
            "combination_rule": "add_registered_prior_count",
            "prior_effective_trials": 7,
            "calibration_rule_id": None,
            "scope_description": "Seven synthetic inherited alternatives under a fixture rule.",
        }
        rule_id = digest(
            {
                "reported_counts": COUNTS,
                "inherited_classification_id": old,
                "justification": justification,
                "effective_scope": effective_scope,
            }
        )
        proof = save_scope_resolution(
            controller,
            {
                "rule_id": rule_id,
                "inherited_classification_id": old,
                "reported_counts": COUNTS,
                "unresolved_scope_items": [],
                "justification": justification,
                "effective_scope": effective_scope,
            },
        )
        legacy.update(
            state="registered_resolution",
            resolution_rule_id=rule_id,
            justification=justification,
            evidence_ids=[proof],
            effective_scope=effective_scope,
        )
    return ProcedureSpec.freeze(
        generator={
            "id": "synthetic-generator-v1",
            "models": {"model": "fixture-only", "temperature": 0},
            **{
                name: h(name)
                for name in ("prompt_id", "grammar_id", "compiler_id", "code_id", "runtime_id")
            },
        },
        execution={
            "implementation_id": h("execution"),
            "costs": {"spread": 1},
            "return_basis": "funded_total_return",
            "risk_policy_id": h("risk"),
        },
        ownership=[
            {
                "method_key": "liquidity-method",
                "family": "liquidity_supply",
                "budget_owner": "liquidity",
                "aliases": ["old-label", "new-label"],
            }
        ],
        sources={
            "knowledge_cutoff": "2026-10-29",
            "knowledge_policy_id": h("knowledge"),
            "reused_discovery": True,
            "prior_study_ids": [h("old-study")],
        },
        qualification={
            "receipt_policy_id": h("receipt"),
            "membership_clock": "prior_qualified_prefix",
            "decision_clock": "month_end",
            "availability_rule": "strictly_before_decision",
        },
        allocator={
            "implementation_id": h("allocator"),
            "membership_policy_id": h("membership"),
            "policy": {"mode": "equal_weight"},
        },
        feedback={
            "policy_id": h("feedback"),
            "schema_repairs": 1,
            "transient_retries": 1,
            "max_refinements_per_parent": 1,
            "max_refinement_depth": 1,
            "max_feedback_rounds": 1,
            "stopping_rule_id": h("stop"),
        },
        arms=[
            {
                "name": "research",
                "classification": "research",
                "budgets": [
                    {
                        "owner": "liquidity",
                        **(
                            budgets or {"proposal": 2, "refinement": 0, "comparison": 0, "rerun": 0}
                        ),
                    }
                ],
                "model_call_limit": model_calls,
            }
        ],
        statistics={
            "counting_rule": "classified-logical-opportunities-v1",
            "bootstrap_rule": "verified-return-columns-v1",
            "compatibility_rule": "exact-procedure-and-scope-v1",
            "final_test_rule_id": h("final-test"),
            "legacy_scope": legacy,
        },
    )


def start(controller, procedure, **changes):
    key = register_procedure(controller, procedure)
    kwargs = {
        "run_key": "run1",
        "world_id": h("world"),
        "input_scope_id": h("inputs"),
        "namespace": "synthetic",
        "seed": 1,
        "registered_at": "2000-01-01",
        **changes,
    }
    return register_run(controller, key, **kwargs)


def procedure_entries():
    return [e for e in runlog.read_entries() if e["name"] == procedure_records.STAGE]


def test_semantic_identity_and_alias_order_are_immutable(controller, tmp_path):
    original = spec(controller, tmp_path)
    payload = original.to_record()
    payload["ownership"][0]["aliases"].reverse()
    assert ProcedureSpec.from_record(payload).identity == original.identity
    payload["allocator"]["policy"]["mode"] = "shrunk_max_sharpe"
    assert original.to_record()["allocator"]["policy"]["mode"] == "equal_weight"
    assert ProcedureSpec.from_record(payload).identity != original.identity
    assert register_procedure(controller, original) == original.identity
    before = len(controller.events()), len(runlog.read_entries())
    assert register_procedure(controller, original) == original.identity
    assert before == (len(controller.events()), len(runlog.read_entries()))


@pytest.mark.parametrize(
    "section,key",
    [
        ("generator", "prompt_id"),
        ("generator", "grammar_id"),
        ("generator", "compiler_id"),
        ("generator", "code_id"),
        ("execution", "implementation_id"),
        ("feedback", "policy_id"),
        ("feedback", "stopping_rule_id"),
        ("allocator", "membership_policy_id"),
    ],
)
def test_each_scientific_semantic_change_has_a_new_procedure(controller, tmp_path, section, key):
    old = spec(controller, tmp_path)
    payload = old.to_record()
    payload[section][key] = h("changed")
    assert ProcedureSpec.from_record(payload).identity != old.identity


def test_registration_is_audited_once_and_replay_repairs_a_missing_mirror(
    controller, tmp_path, monkeypatch
):
    procedure = spec(controller, tmp_path)
    original = runlog.log_run
    monkeypatch.setattr(
        runlog,
        "log_run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic interrupted mirror")),
    )
    with pytest.raises(OSError):
        register_procedure(controller, procedure)
    assert len(controller.events()) == 2  # inherited record, then the procedure; SQLite wins
    assert procedure_entries() == []
    monkeypatch.setattr(runlog, "log_run", original)
    register_procedure(controller, procedure)
    register_procedure(controller, procedure)
    assert len(controller.events()) == 2
    (entry,) = procedure_entries()
    assert entry["params"]["procedure_id"] == procedure.identity
    assert entry["params"]["kind"] == "research_procedure_v1"


def test_inherited_row_classifications_do_not_resolve_effective_scope(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    legacy = procedure.to_record()["statistics"]["legacy_scope"]
    saved = load_artifact(controller.db, legacy["inherited_classification_id"], kind=INHERITED)
    assert saved["record"]["reconciled"] and saved["record"]["conservative_trial_entries"] == 1
    assert legacy["state"] == "unresolved" and legacy["effective_scope"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"reconciled": False},
        {"blockers": ["unclassified"]},
        {"source_entries": 3},
        {"conservative_trial_entries": 2},
    ],
)
def test_unreconciled_or_inconsistent_inherited_rows_refuse(controller, tmp_path, change):
    with pytest.raises(ValueError, match="not reconciled|inventory differs"):
        inherited(controller, tmp_path, **change)
    assert controller.events() == []


def test_inherited_rows_need_a_reason_and_a_known_classification(controller, tmp_path):
    row = canonical(
        {
            "study_hash": h("s"),
            "source_entries": 1,
            "return_columns": 0,
            "classified_entries": [
                {
                    "source_index": 0,
                    "classification": "made-up",
                    "reason": "x",
                    "variant_ids": [],
                    "source_entry_hash": h(0),
                }
            ],
            "conservative_trial_entries": 0,
            "blockers": [],
            "reconciled": True,
            "count_policy": "x",
        }
    )
    (tmp_path / "odd.json").write_text(row, encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        inherit_reconciliation(controller, tmp_path / "odd.json")


def test_legacy_resolution_must_be_bound_to_saved_evidence(controller, tmp_path):
    procedure = spec(controller, tmp_path, resolved=True)
    assert register_procedure(controller, procedure) == procedure.identity
    record = procedure.to_record()
    proof_id = record["statistics"]["legacy_scope"]["evidence_ids"][0]
    proof = load_artifact(controller.db, proof_id, kind=SCOPE_RESOLUTION)
    other = save_scope_resolution(controller, {**proof, "unresolved_scope_items": ["gap"]})
    record["statistics"]["legacy_scope"]["evidence_ids"] = [other]
    with pytest.raises(ValueError, match="not bound to saved evidence"):
        register_procedure(controller, ProcedureSpec.from_record(record))
    with pytest.raises(ValueError, match="typed immutable procedure"):
        register_procedure(controller, procedure.to_record())


def test_unsaved_inherited_classification_refuses_registration(tmp_path, controller):
    procedure = spec(controller, tmp_path)
    other = Controller(tmp_path / "other.db")
    try:
        with pytest.raises(ValueError, match="research artifact is missing"):
            register_procedure(other, procedure)
    finally:
        other.db.close()


def test_resolution_rule_cannot_be_relabelled_or_have_no_count_rule(controller, tmp_path):
    original = spec(controller, tmp_path, resolved=True).to_record()
    original["statistics"]["legacy_scope"]["effective_scope"]["prior_effective_trials"] = 0
    with pytest.raises(ValueError, match="positive inherited count"):
        ProcedureSpec.from_record(original)
    original = spec(controller, tmp_path, resolved=True).to_record()
    original["statistics"]["legacy_scope"]["effective_scope"]["scope_description"] = "changed scope"
    with pytest.raises(ValueError, match="rule identity"):
        ProcedureSpec.from_record(original)


def test_a_run_needs_a_registered_procedure_and_explicit_inputs(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    kwargs = {
        "run_key": "run1",
        "world_id": h("world"),
        "input_scope_id": h("inputs"),
        "namespace": "synthetic",
        "seed": 1,
        "registered_at": "2000-01-01",
    }
    with pytest.raises(ValueError, match="missing"):
        register_run(controller, procedure.identity, **kwargs)
    raw = put_artifact(controller.db, "research_procedure_v1", procedure.to_record())
    with pytest.raises(ValueError, match="registration event"):
        register_run(controller, raw, **kwargs)
    key = register_procedure(controller, procedure)
    for bad in ({"namespace": "paper"}, {"seed": True}, {"seed": 1.0}):
        with pytest.raises(ValueError, match="namespace and seed"):
            register_run(controller, key, **{**kwargs, **bad})
    with pytest.raises(ValueError):
        register_run(controller, key, **{**kwargs, "registered_at": "2000-1-1"})


def test_run_aliases_share_one_statistical_scope_but_not_frozen_inputs(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    first = start(controller, procedure)
    assert first == start(controller, procedure)
    second = register_run(
        controller,
        procedure.identity,
        run_key="another-process",
        world_id=h("world"),
        input_scope_id=h("new-input-revision"),
        namespace="synthetic",
        seed=1,
        registered_at="2000-01-01",
    )
    assert second != first
    one, two = (load_artifact(controller.db, k, kind=RUN) for k in (first, second))
    assert one["statistical_scope_id"] == two["statistical_scope_id"]
    other_world = start(controller, procedure, run_key="run3", world_id=h("other-world"))
    three = load_artifact(controller.db, other_world, kind=RUN)
    assert three["statistical_scope_id"] != one["statistical_scope_id"]
    with pytest.raises(ValueError, match="different frozen inputs"):
        start(controller, procedure, input_scope_id=h("changed"), seed=2)


def test_arm_calls_are_bounded_by_the_registered_equal_total(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    key = record_arm_calls(controller, run, {"research": {"general": 3, "refinement": 2}})
    saved = load_artifact(controller.db, key, kind=ARM_CALLS)
    assert saved["arms"]["research"] == {"limit": 6, "general": 3, "refinement": 2, "total": 5}
    assert key == record_arm_calls(controller, run, {"research": {"general": 3, "refinement": 2}})
    with pytest.raises(ValueError, match="exceeds"):
        record_arm_calls(controller, run, {"research": {"general": 4, "refinement": 3}})
    with pytest.raises(ValueError, match="every registered arm"):
        record_arm_calls(controller, run, {"other": {"general": 1, "refinement": 0}})
    with pytest.raises(ValueError):
        record_arm_calls(controller, run, {"research": {"general": -1, "refinement": 0}})


def calibration_record(procedure, **changes):
    row = {
        "kind": CALIBRATION,
        "label": CALIBRATION_LABEL,
        "namespace": "synthetic",
        "model_calls": 0,
        "procedure_id": procedure.identity,
        "calibration_rule_id": h("rule"),
        "selected": {"threshold": 1},
        **changes,
    }
    return {"id": digest(row), **row}


def test_a_calibration_record_is_verified_before_it_is_saved(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    register_procedure(controller, procedure)
    assert not calibration_qualified(controller, procedure)
    record = calibration_record(procedure)
    path = tmp_path / "calibration.json"
    path.write_text(canonical(record), encoding="utf-8")
    key = register_calibration(controller, path)
    assert load_artifact(controller.db, key, kind=CALIBRATION) == record
    assert calibration_qualified(controller, procedure)
    for tampered in (
        {**record, "selected": {"threshold": 2}},
        calibration_record(procedure, namespace="real"),
        calibration_record(procedure, model_calls=1),
        calibration_record(procedure, label="a different claim"),
    ):
        path.write_text(canonical(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="verified scripted-client"):
            register_calibration(controller, path)


def test_calibration_without_a_selection_does_not_qualify(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    path = tmp_path / "calibration.json"
    path.write_text(canonical(calibration_record(procedure, selected=None)), encoding="utf-8")
    register_calibration(controller, path)
    assert not calibration_qualified(controller, procedure)


def test_whole_procedure_calibration_must_use_the_registered_rule(controller, tmp_path):
    procedure = spec(controller, tmp_path, resolved=True)
    record = procedure.to_record()
    legacy = record["statistics"]["legacy_scope"]
    legacy["effective_scope"] = {
        "combination_rule": "whole_procedure_calibration",
        "prior_effective_trials": None,
        "calibration_rule_id": h("whole rule"),
        "scope_description": "Replay the inherited synthetic discovery and every adaptive step.",
    }
    rule = {
        name: legacy[name]
        for name in (
            "reported_counts",
            "inherited_classification_id",
            "justification",
            "effective_scope",
        )
    }
    legacy["resolution_rule_id"] = digest(rule)
    legacy["evidence_ids"] = [
        put_artifact(
            controller.db,
            SCOPE_RESOLUTION,
            {"rule_id": legacy["resolution_rule_id"], **rule, "unresolved_scope_items": []},
        )
    ]
    whole = ProcedureSpec.from_record(record)
    register_procedure(controller, whole)
    path = tmp_path / "calibration.json"
    path.write_text(
        canonical(calibration_record(whole, calibration_rule_id=h("a different rule"))),
        encoding="utf-8",
    )
    register_calibration(controller, path)
    assert not calibration_qualified(controller, whole)
    path.write_text(
        canonical(calibration_record(whole, calibration_rule_id=h("whole rule"))),
        encoding="utf-8",
    )
    register_calibration(controller, path)
    assert calibration_qualified(controller, whole)


def reserve(controller, run_id, ordinal=0, **kwargs):
    return reserve_opportunity(
        controller,
        run_id,
        arm="research",
        method_key=kwargs.pop("method_key", "liquidity-method"),
        kind=kwargs.pop("kind", "proposal"),
        ordinal=ordinal,
        assigned_at=kwargs.pop("assigned_at", "2000-01-02"),
        **kwargs,
    )


def evidence(controller, opportunity_id, *, as_of="2000-01-03", columns=(), **extra):
    with controller.transaction():
        key = put_artifact(
            controller.db,
            "synthetic_outcome_v1",
            {
                "opportunity_id": opportunity_id,
                "available_at": as_of,
                "history_end": as_of,
                "return_columns": list(columns),
                "financial_observed": bool(columns),
                **extra,
            },
        )
    return [{"kind": "synthetic_outcome_v1", "artifact_id": key}]


def finish(
    controller,
    opportunity_id,
    *,
    status="evaluated",
    as_of="2000-01-03",
    columns=("variant-a",),
    duplicate_of=None,
    financial_observed=None,
):
    return complete_opportunity(
        controller,
        opportunity_id,
        status=status,
        as_of=as_of,
        evidence=evidence(
            controller,
            opportunity_id,
            as_of=as_of,
            columns=columns,
            **(
                {"financial_observed": financial_observed} if financial_observed is not None else {}
            ),
        ),
        return_columns=columns,
        duplicate_of=duplicate_of,
    )


def audit(
    controller,
    run_id,
    classifications,
    *,
    previous_id=None,
    unexplained=False,
    unreserved_trial=False,
    columns=("variant-a",),
):
    entries = runlog.read_entries()
    source_decisions = {}
    completion_by_id = {}
    for entry in entries:
        key = entry["params"]["artifact_id"]
        if entry["params"]["kind"] == COMPLETION:
            completion_by_id[key] = load_artifact(controller.db, key, kind=COMPLETION)
    for index, entry in enumerate(entries):
        result = completion_by_id.get(entry["params"]["artifact_id"])
        financial = result is not None and result["financial_observed"]
        source_decisions[str(index)] = {
            "classification": "strategy_trial" if financial else "diagnostic",
            "reason": "synthetic explicit classification",
            "variant_ids": result["return_columns"] if financial else [],
            "opportunity_ids": [result["opportunity_id"]] if result else [],
        }
    if unexplained:
        entries.append({"name": "unexplained-future-row"})
    if unreserved_trial:
        source_decisions[str(len(entries))] = {
            "classification": "strategy_trial",
            "reason": "synthetic unreserved financial look",
            "variant_ids": [],
            "opportunity_ids": [],
        }
        entries.append({"name": "synthetic-extra-charged-trial"})
    return reconcile_opportunities(
        controller,
        run_id,
        entries=entries,
        return_columns=columns,
        source_decisions=source_decisions,
        classifications=classifications,
        previous_id=previous_id,
    )


def test_unresolved_legacy_scope_reconciles_without_a_statistical_count(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    saved = load_artifact(
        controller.db,
        procedure.to_record()["statistics"]["legacy_scope"]["inherited_classification_id"],
        kind=INHERITED,
    )
    assert saved["record"]["reconciled"] and saved["record"]["conservative_trial_entries"] == 1
    run = start(controller, procedure)
    a, b = reserve(controller, run), reserve(controller, run, 1)
    finish(controller, a)
    finish(controller, b, status="malformed", columns=())
    key = audit(
        controller,
        run,
        {
            a: {"classification": "financial_alternative", "reason": "evaluated"},
            b: {"classification": "nonfinancial_failure", "reason": "no valid hypothesis"},
        },
    )
    row = load_artifact(controller.db, key, kind=RECONCILIATION)
    assert row["current_financial_opportunities"] == 1 and row["procedural_slots"] == 2
    assert row["statistical_trial_count"] is None
    assert row["bootstrap_candidate_columns"] == ["variant-a"]
    assert row["source_audit"]["source_entries"] > row["current_financial_opportunities"]
    assert row["blockers"] == ["legacy_effective_statistical_scope_unresolved:12/9"]
    with pytest.raises(ValueError, match="incompatible"):
        require_compatible_reconciliation(
            controller,
            key,
            procedure_id=procedure.identity,
            statistical_scope_id=row["statistical_scope_id"],
            source_inventory_id=row["source_inventory_id"],
            current_financial_opportunities=1,
            statistical_trial_count=None,
            bootstrap_candidate_columns=["variant-a"],
        )


def test_cached_receipt_resume_and_run_aliases_never_renew_slots(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    run = start(controller, procedure)
    a = reserve(controller, run, method_key="old-label")
    assert a == reserve(controller, run, method_key="new-label")
    result = finish(controller, a)
    before = len(controller.events()), len(runlog.read_entries())
    assert finish(controller, a) == result
    assert before == (len(controller.events()), len(runlog.read_entries()))
    second = register_run(
        controller,
        procedure.identity,
        run_key="another-process",
        world_id=h("world"),
        input_scope_id=h("new-input-revision"),
        namespace="synthetic",
        seed=1,
        registered_at="2000-01-01",
    )
    first_row = load_artifact(controller.db, run, kind=RUN)
    assert (
        first_row["statistical_scope_id"]
        == load_artifact(controller.db, second, kind=RUN)["statistical_scope_id"]
    )
    with pytest.raises(ValueError, match="immutable assignment"):
        reserve(controller, second)
    with pytest.raises(ValueError, match="slot"):
        reserve(controller, run, 2)
    with pytest.raises(ValueError, match="overwritten"):
        finish(controller, a, columns=("different",))


def test_completion_commit_survives_runlog_failure_without_a_second_opportunity(
    controller, tmp_path, monkeypatch
):
    procedure = spec(controller, tmp_path)
    run = start(controller, procedure)
    a = reserve(controller, run)
    original = runlog.log_run
    monkeypatch.setattr(
        runlog,
        "log_run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic interrupted mirror")),
    )
    with pytest.raises(OSError):
        finish(controller, a)
    assert len([e for e in controller.events() if e["state"] == "procedure:" + COMPLETION]) == 1
    monkeypatch.setattr(runlog, "log_run", original)
    finish(controller, a)
    finish(controller, a)
    assert len([e for e in runlog.read_entries() if e["params"]["kind"] == COMPLETION]) == 1


def test_future_outcomes_and_specs_never_enter_earlier_prefix(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    a = reserve(controller, run)
    finish(controller, a)
    january = record_prefix(controller, run, as_of="2000-01-31")
    before = load_artifact(controller.db, january, kind=PREFIX)
    b = reserve(controller, run, 1, assigned_at="2000-12-01")
    finish(controller, b, status="execution_failed", as_of="2000-12-31", columns=())
    assert record_prefix(controller, run, as_of="2000-01-31") == january
    assert load_artifact(controller.db, january, kind=PREFIX) == before
    assert record_prefix(controller, run, as_of="2000-12-31") != january


def test_late_backdating_foreign_evidence_and_unregistered_artifacts_refuse(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    a, b = reserve(controller, run), reserve(controller, run, 1)
    foreign = evidence(controller, b)
    with pytest.raises(ValueError, match="another opportunity"):
        complete_opportunity(
            controller,
            a,
            status="evaluated",
            as_of="2000-01-03",
            evidence=foreign,
            return_columns=["variant-a"],
        )
    future = evidence(controller, a, as_of="2000-12-31", columns=["variant-a"])
    with pytest.raises(ValueError, match="future"):
        complete_opportunity(
            controller,
            a,
            status="evaluated",
            as_of="2000-01-03",
            evidence=future,
            return_columns=["variant-a"],
        )
    record_prefix(controller, run, as_of="2000-01-31")
    with pytest.raises(ValueError, match="backdate"):
        finish(controller, a)
    raw = put_artifact(controller.db, OPPORTUNITY, {"run_id": run})
    with pytest.raises(ValueError, match="registration event"):
        complete_opportunity(controller, raw, status="malformed", as_of="2000-01-03", evidence=[])


def test_refinements_keep_owner_and_have_bounded_earlier_feedback(controller, tmp_path):
    procedure = spec(
        controller, tmp_path, budgets={"proposal": 1, "refinement": 2, "comparison": 0, "rerun": 1}
    )
    run = start(controller, procedure)
    parent = reserve(controller, run)
    with pytest.raises(ValueError, match="earlier"):
        reserve(
            controller,
            run,
            kind="refinement",
            parent_id=parent,
            change_id=h("change"),
            shown_to_agent=nothing_shown("2000-01-02"),
        )
    with pytest.raises(ValueError, match="shown to the agent"):
        reserve(controller, run, kind="refinement", parent_id=parent, change_id=h("change"))
    finish(controller, parent)
    child = reserve(
        controller,
        run,
        kind="refinement",
        parent_id=parent,
        change_id=h("change"),
        assigned_at="2000-02-01",
        method_key="new-label",
        shown_to_agent=nothing_shown("2000-02-01"),
    )
    row = load_artifact(controller.db, child, kind=OPPORTUNITY)
    assert row["family"] == "liquidity_supply" and row["budget_owner"] == "liquidity"
    assert row["shown_to_agent"] == nothing_shown("2000-02-01")
    assert load_artifact(controller.db, parent, kind=OPPORTUNITY)["shown_to_agent"] == {
        "receipt_ids": [],
        "before": "2000-01-02",
        "summary_digest": nothing_shown("x")["summary_digest"],
    }
    with pytest.raises(ValueError, match="budget"):
        reserve(
            controller,
            run,
            1,
            kind="refinement",
            parent_id=parent,
            change_id=h("change2"),
            assigned_at="2000-02-01",
            shown_to_agent=nothing_shown("2000-02-01"),
        )
    with pytest.raises(ValueError, match="identity"):
        reserve(
            controller,
            run,
            kind="rerun",
            parent_id=parent,
            assigned_at="2000-02-01",
            shown_to_agent=nothing_shown("2000-02-01"),
        )


def test_rows_saved_before_feedback_bounds_keep_their_slot_and_nothing_shown(tmp_path, monkeypatch):
    from capstone.factors.procedure_records import _save

    keys, rows = [], {}
    for name in ("new", "legacy"):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / name / "ledger"))
        (tmp_path / name).mkdir()
        c = Controller(tmp_path / name / "controller.db")
        try:
            run = start(c, spec(c, tmp_path / name))
            if name == "legacy":
                row = {k: v for k, v in rows.items() if k != "shown_to_agent"}
                with c.transaction():
                    saved = _save(c, OPPORTUNITY, row, run)
                assert reserve(c, run) == saved
                with pytest.raises(ValueError, match="missing or has a different kind"):
                    reserve(
                        c,
                        run,
                        shown_to_agent={**nothing_shown("2000-01-02"), "receipt_ids": [h(1)]},
                    )
            else:
                keys.append(reserve(c, run))
                rows = load_artifact(c.db, keys[0], kind=OPPORTUNITY)
                assert rows["shown_to_agent"] == nothing_shown("2000-01-02")
        finally:
            c.db.close()


def test_reconciliation_accounts_duplicates_failures_and_append_only_corrections(
    controller, tmp_path
):
    procedure = spec(controller, tmp_path, resolved=True)
    run = start(controller, procedure)
    a, b = reserve(controller, run), reserve(controller, run, 1)
    finish(controller, a)
    finish(controller, b, status="duplicate", columns=(), duplicate_of=a)
    classes = {
        a: {"classification": "financial_alternative", "reason": "actual synthetic result"},
        b: {"classification": "duplicate", "reason": "same construction; no new variant"},
    }
    bad = audit(controller, run, classes, unexplained=True)
    assert not load_artifact(controller.db, bad, kind=RECONCILIATION)["reconciled"]
    good = audit(controller, run, classes, previous_id=bad)
    row = load_artifact(controller.db, good, kind=RECONCILIATION)
    assert row["reconciled"] and row["previous_id"] == bad
    assert row["current_financial_opportunities"] == 1 and row["procedural_slots"] == 2
    assert row["statistical_trial_count"] == 8
    assert not row["calibration_qualified"] and not row["formal_admission"]
    assert load_artifact(controller.db, bad, kind=RECONCILIATION)["blockers"]
    assert (
        require_compatible_reconciliation(
            controller,
            good,
            procedure_id=procedure.identity,
            statistical_scope_id=row["statistical_scope_id"],
            source_inventory_id=row["source_inventory_id"],
            current_financial_opportunities=1,
            statistical_trial_count=8,
            bootstrap_candidate_columns=["variant-a"],
        )
        == row
    )
    changed = procedure.to_record()
    changed["allocator"]["policy"]["mode"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        require_compatible_reconciliation(
            controller,
            good,
            procedure_id=ProcedureSpec.from_record(changed).identity,
            statistical_scope_id=row["statistical_scope_id"],
            source_inventory_id=row["source_inventory_id"],
            current_financial_opportunities=1,
            statistical_trial_count=8,
            bootstrap_candidate_columns=["variant-a"],
        )
    with pytest.raises(Exception, match="append only"):
        controller.db.execute("DELETE FROM research_events")


def test_missing_columns_and_unspent_planned_slots_stay_blocked(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path, resolved=True))
    a = reserve(controller, run)
    finish(controller, a)
    key = audit(
        controller,
        run,
        {a: {"classification": "financial_alternative", "reason": "synthetic result"}},
        columns=(),
    )
    row = load_artifact(controller.db, key, kind=RECONCILIATION)
    assert any("missing_return_column" in item for item in row["blockers"])
    assert any("unaccounted_planned_slots" in item for item in row["blockers"])


@pytest.mark.parametrize(
    "status", ["malformed", "source_blocked", "data_blocked", "execution_failed", "not_reached"]
)
def test_no_result_slots_are_retained_without_inventing_statistical_variants(
    controller, tmp_path, status
):
    procedure = spec(
        controller,
        tmp_path,
        resolved=True,
        budgets={"proposal": 1, "refinement": 0, "comparison": 0, "rerun": 0},
    )
    run = start(controller, procedure)
    opportunity = reserve(controller, run)
    finish(controller, opportunity, status=status, columns=())
    role = "unused" if status == "not_reached" else "nonfinancial_failure"
    key = audit(
        controller,
        run,
        {opportunity: {"classification": role, "reason": "synthetic no-result accounting"}},
        columns=(),
    )
    row = load_artifact(controller.db, key, kind=RECONCILIATION)
    assert row["reconciled"] and row["procedural_slots"] == 1
    assert row["current_financial_opportunities"] == 0 and row["bootstrap_candidate_columns"] == []
    assert row["statistical_trial_count"] == 7


def test_reservation_rollback_leaves_no_partial_result_or_opportunity(
    controller, tmp_path, monkeypatch
):
    from capstone.factors import procedure_records

    run = start(controller, spec(controller, tmp_path))
    before = len(controller.events())
    original = procedure_records._save

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic crash before transaction commit")

    monkeypatch.setattr(procedure_records, "_save", interrupted)
    with pytest.raises(RuntimeError, match="before transaction commit"):
        reserve(controller, run)
    assert len(controller.events()) == before
    assert not controller.db.execute(
        "SELECT 1 FROM research_artifacts WHERE kind=?", (OPPORTUNITY,)
    ).fetchall()
    monkeypatch.setattr(procedure_records, "_save", original)
    opportunity = reserve(controller, run)
    assert load_artifact(controller.db, opportunity, kind=OPPORTUNITY)["run_id"] == run


def test_concurrent_reservations_have_one_durable_slot(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    barrier = Barrier(2)
    path = controller.db.execute("PRAGMA database_list").fetchone()[2]

    def worker():
        other = Controller(path)
        try:
            barrier.wait(timeout=5)
            try:
                return reserve(other, run)
            except RuntimeError:
                # The audit mirror lock may contend after the one SQLite commit.
                return None
        finally:
            other.db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: worker(), range(2)))
    saved = reserve(controller, run)
    assert all(value in {None, saved} for value in values)
    assert (
        len([row for row in controller.events() if row["state"] == "procedure:" + OPPORTUNITY]) == 1
    )


def test_non_scalar_calibration_rule_cannot_supply_a_legacy_gate_count(controller, tmp_path):
    procedure = spec(
        controller,
        tmp_path,
        resolved=True,
        budgets={"proposal": 1, "refinement": 0, "comparison": 0, "rerun": 0},
    )
    record = procedure.to_record()
    legacy = record["statistics"]["legacy_scope"]
    legacy["effective_scope"] = {
        "combination_rule": "whole_procedure_calibration",
        "prior_effective_trials": None,
        "calibration_rule_id": h("synthetic whole procedure replay rule"),
        "scope_description": "Replay the inherited synthetic discovery and all adaptive decisions.",
    }
    rule = {
        name: legacy[name]
        for name in (
            "reported_counts",
            "inherited_classification_id",
            "justification",
            "effective_scope",
        )
    }
    legacy["resolution_rule_id"] = digest(rule)
    proof = put_artifact(
        controller.db,
        "procedure_scope_resolution_v1",
        {
            "rule_id": legacy["resolution_rule_id"],
            **rule,
            "unresolved_scope_items": [],
        },
    )
    legacy["evidence_ids"] = [proof]
    procedure = ProcedureSpec.from_record(record)
    run = start(controller, procedure)
    opportunity = reserve(controller, run)
    finish(controller, opportunity)
    key = audit(
        controller,
        run,
        {opportunity: {"classification": "financial_alternative", "reason": "synthetic outcome"}},
    )
    row = load_artifact(controller.db, key, kind=RECONCILIATION)
    assert row["reconciled"] and row["current_financial_opportunities"] == 1
    assert row["statistical_trial_count"] is None and not row["calibration_qualified"]
    with pytest.raises(ValueError, match="incompatible"):
        require_compatible_reconciliation(
            controller,
            key,
            procedure_id=procedure.identity,
            statistical_scope_id=row["statistical_scope_id"],
            source_inventory_id=row["source_inventory_id"],
            current_financial_opportunities=1,
            statistical_trial_count=1,
            bootstrap_candidate_columns=["variant-a"],
        )


def compatible(controller, key):
    row = load_artifact(controller.db, key, kind=RECONCILIATION)
    return require_compatible_reconciliation(
        controller,
        key,
        procedure_id=row["procedure_id"],
        statistical_scope_id=row["statistical_scope_id"],
        source_inventory_id=row["source_inventory_id"],
        current_financial_opportunities=row["current_financial_opportunities"],
        statistical_trial_count=row["statistical_trial_count"],
        bootstrap_candidate_columns=row["bootstrap_candidate_columns"],
    )


def test_unreserved_source_trial_blocks_and_supersedes_an_older_clean_handoff(controller, tmp_path):
    procedure = spec(
        controller,
        tmp_path,
        resolved=True,
        budgets={"proposal": 1, "refinement": 0, "comparison": 0, "rerun": 0},
    )
    run = start(controller, procedure)
    opportunity = reserve(controller, run)
    finish(controller, opportunity)
    classes = {
        opportunity: {
            "classification": "financial_alternative",
            "reason": "synthetic financial observation",
        }
    }
    good = audit(controller, run, classes)
    assert compatible(controller, good)["reconciled"]
    blocked = audit(controller, run, classes, previous_id=good, unreserved_trial=True)
    row = load_artifact(controller.db, blocked, kind=RECONCILIATION)
    assert "unreserved_source_trial" in row["blockers"]
    with pytest.raises(ValueError, match="incompatible"):
        compatible(controller, good)
    with pytest.raises(ValueError, match="incompatible"):
        compatible(controller, blocked)


def test_changed_accounting_frontier_invalidates_an_old_clean_handoff(controller, tmp_path):
    procedure = spec(
        controller,
        tmp_path,
        resolved=True,
        budgets={"proposal": 1, "refinement": 0, "comparison": 0, "rerun": 0},
    )
    run = start(controller, procedure)
    opportunity = reserve(controller, run)
    finish(controller, opportunity)
    good = audit(
        controller,
        run,
        {opportunity: {"classification": "financial_alternative", "reason": "synthetic result"}},
    )
    assert compatible(controller, good)["reconciled"]
    register_run(
        controller,
        procedure.identity,
        run_key="same-world-restart",
        world_id=h("world"),
        input_scope_id=h("input-revision"),
        namespace="synthetic",
        seed=1,
        registered_at="2000-01-01",
    )
    with pytest.raises(ValueError, match="incompatible"):
        compatible(controller, good)


def test_same_world_new_run_alias_cannot_backdate_into_a_sealed_prefix(controller, tmp_path):
    procedure = spec(controller, tmp_path)
    run = start(controller, procedure)
    opportunity = reserve(controller, run)
    finish(controller, opportunity)
    earlier = record_prefix(controller, run, as_of="2000-01-10")
    alias = register_run(
        controller,
        procedure.identity,
        run_key="new-output-process",
        world_id=h("world"),
        input_scope_id=h("future-input-revision"),
        namespace="synthetic",
        seed=1,
        registered_at="2000-01-01",
    )
    assert record_prefix(controller, alias, as_of="2000-01-10") == earlier
    with pytest.raises(ValueError, match="backdate"):
        reserve(controller, alias, 1, assigned_at="2000-01-02")
    later = reserve(controller, alias, 1, assigned_at="2000-02-01")
    finish(controller, later, as_of="2000-02-02")
    assert record_prefix(controller, alias, as_of="2000-01-10") == earlier


def test_failed_after_financial_observation_still_counts_without_promoting_return_columns(
    controller, tmp_path
):
    procedure = spec(
        controller,
        tmp_path,
        resolved=True,
        budgets={"proposal": 1, "refinement": 0, "comparison": 0, "rerun": 0},
    )
    run = start(controller, procedure)
    opportunity = reserve(controller, run)
    finish(controller, opportunity, status="execution_failed", columns=(), financial_observed=True)
    key = audit(
        controller,
        run,
        {
            opportunity: {
                "classification": "financial_alternative",
                "reason": "valid earlier observations preceded a later failure",
            }
        },
        columns=(),
    )
    row = load_artifact(controller.db, key, kind=RECONCILIATION)
    assert row["reconciled"] and row["current_financial_opportunities"] == 1
    assert row["statistical_trial_count"] == 8 and row["bootstrap_candidate_columns"] == []
    assert not row["formal_admission"]


def test_unknown_financial_observation_state_and_observed_duplicate_refuse(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    opportunity = reserve(controller, run)
    refs = evidence(controller, opportunity, financial_observed=None)
    with pytest.raises(ValueError, match="observation state"):
        complete_opportunity(
            controller, opportunity, status="data_blocked", as_of="2000-01-03", evidence=refs
        )
    with pytest.raises(ValueError, match="free duplicate"):
        finish(controller, opportunity, status="not_reached", columns=(), financial_observed=True)


def test_public_opportunity_lookup_verifies_authority_without_reserving_alias_slots(
    controller, tmp_path
):
    run = start(controller, spec(controller, tmp_path))
    opportunity = reserve(controller, run)
    before = len(controller.events())
    assert opportunity_record(controller, opportunity)["run_id"] == run
    assert (
        registered_opportunity(
            controller, run, arm="research", method_key="old-label", kind="proposal", ordinal=0
        )[0]
        == opportunity
    )
    assert (
        registered_opportunity(
            controller, run, arm="research", method_key="new-label", kind="proposal", ordinal=0
        )[0]
        == opportunity
    )
    with pytest.raises(ValueError, match="not been reserved"):
        registered_opportunity(
            controller, run, arm="research", method_key="new-label", kind="proposal", ordinal=1
        )
    assert len(controller.events()) == before


def test_return_evidence_cannot_be_relabelled_as_no_financial_observation(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    opportunity = reserve(controller, run)
    refs = evidence(controller, opportunity, columns=["variant-a"], financial_observed=False)
    with pytest.raises(ValueError, match="saved outcome evidence"):
        complete_opportunity(
            controller, opportunity, status="execution_failed", as_of="2000-01-03", evidence=refs
        )


def test_evidence_availability_cannot_predate_its_reserved_opportunity(controller, tmp_path):
    run = start(controller, spec(controller, tmp_path))
    opportunity = reserve(controller, run, assigned_at="2000-02-01")
    refs = evidence(controller, opportunity, as_of="2000-01-31", columns=["variant-a"])
    with pytest.raises(ValueError, match="saved outcome evidence"):
        complete_opportunity(
            controller,
            opportunity,
            status="evaluated",
            as_of="2000-02-02",
            evidence=refs,
            return_columns=["variant-a"],
        )
