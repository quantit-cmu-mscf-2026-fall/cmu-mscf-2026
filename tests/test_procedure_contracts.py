"""Procedure identity: what a registered procedure fixes, and what it refuses to leave open.

Synthetic ids only; nothing here runs a model or reads data.
"""

from __future__ import annotations

import copy

import pytest

from capstone.factors.procedure_contracts import (
    SCOPE_RESOLUTION,
    TWO_TIER_RULE,
    TWO_TIER_STATISTIC,
    ProcedureSpec,
    scope_evidence,
    whole_procedure_resolution,
)
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


# Pinned from a version-1 record: adding the two-tier section must not move this identity.
VERSION_1_IDENTITY = "8e8c30bfa37b08b8f4121433fa555f1698888e53dd97ebad57883a0f3ceaf688"


def two_tier(**change):
    return {
        "rule": TWO_TIER_RULE,
        "tier1_id": h("tier1"),
        "statistic": TWO_TIER_STATISTIC,
        "survivors_per_family": 2,
        "agreement_sample": 5,
        "agreement_seed": 3,
        "min_rank_agreement": 0.5,
        "calibration_tier1_error": 0.25,
    } | change


def test_a_version_1_procedure_keeps_its_exact_identity():
    procedure = spec()
    assert procedure.identity == VERSION_1_IDENTITY
    assert procedure.to_record()["version"] == 1 and procedure.two_tier is None
    assert '"evaluation"' not in procedure.payload_json


def test_a_two_tier_section_makes_a_version_2_procedure_with_its_own_identity():
    old = spec()
    new = ProcedureSpec.freeze(**spec_record(), evaluation=two_tier())
    assert new.to_record()["version"] == 2 and new.two_tier == two_tier()
    assert new.identity != old.identity
    assert ProcedureSpec.from_record(new.to_record()) == new
    changed = ProcedureSpec.freeze(**spec_record(), evaluation=two_tier(agreement_seed=4))
    assert changed.identity != new.identity


def test_two_tier_version_and_section_must_agree():
    record = spec().to_record()
    with pytest.raises(ValueError, match="needs procedure version 2"):
        ProcedureSpec.from_record({**record, "evaluation": two_tier()})
    with pytest.raises(ValueError, match="needs procedure version 2"):
        ProcedureSpec.from_record({**record, "version": 2})


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"rule": "other"}, "unsupported two-tier"),
        ({"statistic": "other"}, "unsupported two-tier"),
        ({"tier1_id": "not-a-digest"}, "content-addressed"),
        ({"survivors_per_family": 0}, "at least one"),
        ({"agreement_sample": 2}, "at least three"),
        ({"agreement_seed": -1}, "nonnegative"),
        ({"min_rank_agreement": 1.5}, "out of range"),
        ({"min_rank_agreement": "high"}, "must be a number"),
        ({"min_rank_agreement": True}, "must be a number"),
        ({"calibration_tier1_error": -0.1}, "out of range"),
        ({"calibration_tier1_error": 101}, "out of range"),
    ],
)
def test_a_malformed_two_tier_section_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        ProcedureSpec.freeze(**spec_record(), evaluation=two_tier(**change))


def test_a_two_tier_section_with_missing_or_extra_fields_is_refused():
    row = two_tier()
    row.pop("agreement_seed")
    with pytest.raises(ValueError, match="incomplete or unknown"):
        ProcedureSpec.freeze(**spec_record(), evaluation=row)
    with pytest.raises(ValueError, match="incomplete or unknown"):
        ProcedureSpec.freeze(**spec_record(), evaluation=two_tier() | {"extra": 1})


def unresolved_legacy():
    return spec().to_record()["statistics"]["legacy_scope"]


def test_whole_procedure_resolution_invents_no_scalar_count():
    legacy = unresolved_legacy()
    resolved, evidence = whole_procedure_resolution(
        legacy,
        calibration_rule_id=h("calibration-rule"),
        justification="Synthetic fixture justification.",
        scope_description="Measured end to end under a fixture rule.",
    )
    scope = resolved["effective_scope"]
    assert resolved["state"] == "registered_resolution"
    assert scope["prior_effective_trials"] is None
    assert scope["combination_rule"] == "whole_procedure_calibration"
    assert evidence == scope_evidence(resolved)
    assert resolved["evidence_ids"] == [digest({"kind": SCOPE_RESOLUTION, "payload": evidence})]
    record = spec_record()
    record["statistics"]["legacy_scope"] = resolved
    registered = ProcedureSpec.freeze(**record)
    assert registered.to_record()["statistics"]["legacy_scope"]["effective_scope"] == scope


def test_only_an_unresolved_scope_can_be_resolved_and_the_rule_must_be_an_identity():
    resolved, _ = whole_procedure_resolution(
        unresolved_legacy(),
        calibration_rule_id=h("rule"),
        justification="j",
        scope_description="s",
    )
    with pytest.raises(ValueError, match="only an unresolved"):
        whole_procedure_resolution(
            resolved, calibration_rule_id=h("rule"), justification="j", scope_description="s"
        )
    with pytest.raises(ValueError, match="content-addressed"):
        whole_procedure_resolution(
            unresolved_legacy(),
            calibration_rule_id="not-a-digest",
            justification="j",
            scope_description="s",
        )


def test_a_scalar_count_cannot_ride_along_a_whole_procedure_rule():
    resolved, _ = whole_procedure_resolution(
        unresolved_legacy(), calibration_rule_id=h("rule"), justification="j", scope_description="s"
    )
    resolved["effective_scope"]["prior_effective_trials"] = 9
    record = spec_record()
    record["statistics"]["legacy_scope"] = resolved
    with pytest.raises(ValueError, match="cannot invent a scalar count"):
        ProcedureSpec.freeze(**record)
