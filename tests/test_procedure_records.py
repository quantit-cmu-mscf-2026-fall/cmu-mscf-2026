"""Synthetic registration records only; no calibration or financial admission."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from capstone import runlog
from capstone.factors import procedure_records
from capstone.factors.procedure_contracts import SCOPE_RESOLUTION, ProcedureSpec
from capstone.factors.procedure_records import (
    ARM_CALLS,
    CALIBRATION,
    CALIBRATION_LABEL,
    INHERITED,
    RUN,
    calibration_qualified,
    inherit_reconciliation,
    record_arm_calls,
    register_calibration,
    register_procedure,
    register_run,
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
            CREATE TABLE research_events
              (seq INTEGER PRIMARY KEY, brief TEXT NOT NULL, state TEXT NOT NULL,
               reason TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TRIGGER no_update BEFORE UPDATE ON research_events
              BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER no_delete BEFORE DELETE ON research_events
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


def spec(controller, tmp_path, *, resolved=False, model_calls=6):
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
                        "proposal": 2,
                        "refinement": 0,
                        "comparison": 0,
                        "rerun": 0,
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
