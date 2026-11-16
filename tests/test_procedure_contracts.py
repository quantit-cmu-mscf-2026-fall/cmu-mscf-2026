"""Procedure identity: what a registered procedure fixes, and what it refuses to leave open.

Synthetic ids only; nothing here runs a model or reads data.
"""

from __future__ import annotations

import copy

import pytest

from capstone.factors.procedure_contracts import ProcedureSpec
from capstone.factors.research_contracts import canonical, digest

COUNTS = [11, 7]  # caller-supplied scope counts; any positive integers are accepted


def h(value):
    return digest({"synthetic": value})


def spec_record(*, resolved=False, budgets=None):
    legacy = {
        "state": "unresolved",
        "reported_counts": COUNTS,
        "inherited_classification_id": h("old-classification"),
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
                "inherited_classification_id": legacy["inherited_classification_id"],
                "justification": justification,
                "effective_scope": effective_scope,
            }
        )
        legacy.update(
            state="registered_resolution",
            resolution_rule_id=rule_id,
            justification=justification,
            evidence_ids=[h("saved-evidence")],
            effective_scope=effective_scope,
        )
    return {
        "generator": {
            "id": "synthetic-generator-v1",
            "models": {"model": "fixture-only", "temperature": 0},
            **{
                name: h(name)
                for name in ("prompt_id", "grammar_id", "compiler_id", "code_id", "runtime_id")
            },
        },
        "execution": {
            "implementation_id": h("execution"),
            "costs": {"spread": 1},
            "return_basis": "funded_total_return",
            "risk_policy_id": h("risk"),
        },
        "ownership": [
            {
                "method_key": "liquidity-method",
                "family": "liquidity_supply",
                "budget_owner": "liquidity",
                "aliases": ["old-label", "new-label"],
            }
        ],
        "sources": {
            "knowledge_cutoff": "2026-10-29",
            "knowledge_policy_id": h("knowledge"),
            "reused_discovery": True,
            "prior_study_ids": [h("old-study")],
        },
        "qualification": {
            "receipt_policy_id": h("receipt"),
            "membership_clock": "prior_qualified_prefix",
            "decision_clock": "month_end",
            "availability_rule": "strictly_before_decision",
        },
        "allocator": {
            "implementation_id": h("allocator"),
            "membership_policy_id": h("membership"),
            "policy": {"mode": "equal_weight"},
        },
        "feedback": {
            "policy_id": h("feedback"),
            "schema_repairs": 1,
            "transient_retries": 1,
            "max_refinements_per_parent": 1,
            "max_refinement_depth": 1,
            "max_feedback_rounds": 1,
            "stopping_rule_id": h("stop"),
        },
        "arms": [
            {
                "name": "research",
                "classification": "research",
                "budgets": [
                    dict(
                        owner="liquidity",
                        **(
                            budgets or {"proposal": 2, "refinement": 0, "comparison": 0, "rerun": 0}
                        ),
                    )
                ],
                "model_call_limit": 6,
            }
        ],
        "statistics": {
            "counting_rule": "classified-logical-opportunities-v1",
            "bootstrap_rule": "verified-return-columns-v1",
            "compatibility_rule": "exact-procedure-and-scope-v1",
            "final_test_rule_id": h("final-test"),
            "legacy_scope": legacy,
        },
    }


def spec(**kwargs):
    return ProcedureSpec.freeze(**spec_record(**kwargs))


def test_semantic_identity_and_alias_order_are_immutable():
    original = spec()
    payload = original.to_record()
    payload["ownership"][0]["aliases"].reverse()
    assert ProcedureSpec.from_record(payload).identity == original.identity
    payload["allocator"]["policy"]["mode"] = "shrunk_max_sharpe"
    assert original.to_record()["allocator"]["policy"]["mode"] == "equal_weight"
    assert ProcedureSpec.from_record(payload).identity != original.identity


def test_the_spec_owns_its_bytes_and_replays_exactly():
    original = spec()
    original.to_record()["generator"]["id"] = "mutated"
    assert original.to_record()["generator"]["id"] == "synthetic-generator-v1"
    assert ProcedureSpec.from_record(original.to_record()) == original
    assert original.payload_json == canonical(original.to_record())
    with pytest.raises(ValueError, match="canonical"):
        ProcedureSpec(original.payload_json.replace("{", "{ ", 1))


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
def test_each_scientific_semantic_change_has_a_new_procedure(section, key):
    old = spec()
    payload = old.to_record()
    payload[section][key] = h("changed")
    assert ProcedureSpec.from_record(payload).identity != old.identity


def test_owner_and_arm_lookup_resolve_aliases_and_refuse_strangers():
    procedure = spec()
    assert procedure.owner("new-label")["method_key"] == "liquidity-method"
    assert procedure.arm("research")["model_call_limit"] == 6
    with pytest.raises(ValueError, match="no registered canonical owner"):
        procedure.owner("unregistered")
    with pytest.raises(ValueError, match="arm was not registered"):
        procedure.arm("other")


def refused(match, mutate, **kwargs):
    record = spec_record(**kwargs)
    mutate(record)
    with pytest.raises(ValueError, match=match):
        ProcedureSpec.freeze(**record)


def test_incomplete_or_unknown_fields_and_unknown_versions_are_refused():
    refused("incomplete or unknown", lambda r: r.pop("allocator"))
    refused("incomplete or unknown", lambda r: r.update(extra={}))
    refused("incomplete or unknown", lambda r: r["feedback"].update(extra=1))
    with pytest.raises(ValueError, match="unsupported procedure version"):
        ProcedureSpec.from_record({**spec().to_record(), "version": 3})
    with pytest.raises(ValueError, match="invalid procedure structure"):
        ProcedureSpec.freeze(generator=None)


def test_generator_and_execution_must_be_exact():
    refused("content-addressed", lambda r: r["generator"].update(prompt_id="not-a-digest"))
    refused("exact configured models", lambda r: r["generator"].update(models={}))
    refused("stable logical name", lambda r: r["generator"].update(id="a path/with slash"))
    refused("funded return", lambda r: r["execution"].update(return_basis="price_return"))
    refused("funded return", lambda r: r["execution"].update(costs={}))


def test_ownership_must_name_families_owners_and_aliases_without_overlap():
    refused("unknown broad coverage family", lambda r: r["ownership"][0].update(family="other"))
    refused(
        "duplicate method",
        lambda r: r["ownership"].append(copy.deepcopy(r["ownership"][0])),
    )

    def second_owner(r, **change):
        r["ownership"].append(
            {
                "method_key": "second",
                "family": "forced_flows",
                "budget_owner": "flows",
                "aliases": [],
                **change,
            }
        )
        r["arms"][0]["budgets"].append(
            {"owner": "flows", "proposal": 0, "refinement": 0, "comparison": 0, "rerun": 0}
        )

    refused("more than one canonical owner", lambda r: second_owner(r, aliases=["old-label"]))
    refused("alias cannot replace", lambda r: second_owner(r, aliases=["liquidity-method"]))
    refused(
        "another coverage family",
        lambda r: second_owner(r, budget_owner="liquidity", family="forced_flows"),
    )


def test_every_arm_must_bound_every_budget_owner_with_nonnegative_integers():
    refused("explicitly bound every", lambda r: r["arms"][0]["budgets"].clear())
    refused("unowned or duplicate", lambda r: r["arms"][0]["budgets"][0].update(owner="ghost"))
    refused("duplicate or unclassified", lambda r: r["arms"][0].update(classification="maybe"))
    refused("bounded nonnegative", lambda r: r["arms"][0].update(model_call_limit=-1))
    refused("bounded nonnegative", lambda r: r["arms"][0]["budgets"][0].update(proposal=1.5))
    refused("bounded nonnegative", lambda r: r["arms"][0]["budgets"][0].update(rerun=True) or None)
    refused("duplicate or unclassified", lambda r: r["arms"].append(copy.deepcopy(r["arms"][0])))


def test_feedback_cannot_exceed_the_controller_repair_and_retry_bounds():
    refused("cannot exceed", lambda r: r["feedback"].update(schema_repairs=2))
    refused("cannot exceed", lambda r: r["feedback"].update(transient_retries=2))
    refused("bounded nonnegative", lambda r: r["feedback"].update(max_feedback_rounds=-1))


def test_sources_and_qualification_timing_must_be_explicit():
    refused("canonical decision date", lambda r: r["sources"].update(knowledge_cutoff="20261029"))
    refused("reused-discovery", lambda r: r["sources"].update(reused_discovery="yes"))
    refused("content-addressed", lambda r: r["sources"].update(prior_study_ids=["x"]))
    refused("timing", lambda r: r["qualification"].update(membership_clock="whenever"))
    refused("timing", lambda r: r["qualification"].update(availability_rule="same_day"))
    refused("exact allocator policy", lambda r: r["allocator"].update(policy={}))


def test_inventories_are_normalized_so_order_never_changes_identity():
    record = spec_record()
    record["sources"]["prior_study_ids"] = [h("b"), h("a"), h("b")]
    shuffled = ProcedureSpec.freeze(**record)
    record["sources"]["prior_study_ids"] = [h("a"), h("b")]
    assert ProcedureSpec.freeze(**record).identity == shuffled.identity


def test_statistics_rules_are_fixed_and_the_original_scope_stays_visible():
    refused("unsupported statistical", lambda r: r["statistics"].update(counting_rule="other"))
    refused(
        "must remain visible",
        lambda r: r["statistics"]["legacy_scope"].update(reported_counts=[]),
    )
    refused(
        "must remain visible",
        lambda r: r["statistics"]["legacy_scope"].update(reported_counts=[3, 0]),
    )
    refused("unknown original", lambda r: r["statistics"]["legacy_scope"].update(state="dropped"))
    refused(
        "cannot imply a resolution",
        lambda r: r["statistics"]["legacy_scope"].update(justification="because"),
    )


def test_unresolved_scope_registers_without_inventing_a_count():
    legacy = spec().to_record()["statistics"]["legacy_scope"]
    assert legacy["state"] == "unresolved" and legacy["effective_scope"] is None


def test_resolution_rule_cannot_be_relabelled_or_have_no_count_rule():
    original = spec(resolved=True).to_record()
    original["statistics"]["legacy_scope"]["effective_scope"]["prior_effective_trials"] = 0
    with pytest.raises(ValueError, match="positive inherited count"):
        ProcedureSpec.from_record(original)
    original = spec(resolved=True).to_record()
    original["statistics"]["legacy_scope"]["effective_scope"]["scope_description"] = "changed scope"
    with pytest.raises(ValueError, match="rule identity"):
        ProcedureSpec.from_record(original)


def test_a_resolution_needs_a_justification_evidence_and_a_known_combination_rule():
    refused(
        "scientific justification",
        lambda r: r["statistics"]["legacy_scope"].update(justification=" "),
        resolved=True,
    )
    refused(
        "saved evidence",
        lambda r: r["statistics"]["legacy_scope"].update(evidence_ids=[]),
        resolved=True,
    )
    refused(
        "combination rule",
        lambda r: r["statistics"]["legacy_scope"]["effective_scope"].update(
            combination_rule="average"
        ),
        resolved=True,
    )
