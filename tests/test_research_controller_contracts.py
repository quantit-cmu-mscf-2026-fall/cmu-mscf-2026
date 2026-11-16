from dataclasses import asdict, replace

import pytest

from capstone.factors.research_contracts import (
    AssignedBrief,
    MethodSpec,
    ResearchPolicy,
    SourceClaim,
    digest,
    qualify,
)


def policy():
    return ResearchPolicy((("liquidity_supply", "discovery", 2),))


def claim(**kwargs):
    return SourceClaim(
        "https://example.org/primary",
        "section 2",
        "short reversal",
        "no reversal",
        ("daily_returns", "dated_flows"),
        True,
        **kwargs,
    )


def test_identity_is_mapping_order_independent_and_policy_is_frozen():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert policy().identity == ResearchPolicy(**asdict(policy())).identity
    with pytest.raises(ValueError, match="rejection"):
        replace(policy(), model_rejection=True)
    with pytest.raises(ValueError):
        replace(policy(), discovery_end="2021-01-01")
    with pytest.raises(ValueError):
        AssignedBrief("renamed", "agent_invented_family", "discovery", "method")


def test_rich_claim_retains_missing_dependencies_without_proxy():
    original = claim()
    assert qualify(original, frozenset({"daily_returns"})) == ("needs_data", ("dated_flows",))
    assert original.required_inputs == ("daily_returns", "dated_flows")
    assert qualify(replace(original, full_methods=False), frozenset())[0] == "methods_review_needed"
    assert qualify(claim(universe="futures"), frozenset())[0] == "outside_scope"


def test_new_method_semantics_get_new_identity():
    spec = MethodSpec(
        "claim",
        "reference",
        "buy losers",
        "bottom decile",
        "equal",
        "weekly",
        "one week",
        "next formation",
        "completed session",
        "long_only",
    )
    assert replace(spec, holding="one month").identity != spec.identity
    with pytest.raises(ValueError, match="deviations"):
        replace(spec, variant="adaptation")
    with pytest.raises(ValueError, match="incomplete"):
        replace(spec, formation="")


def test_json_loaded_reservations_do_not_retain_mutable_inner_lists():
    rows = [["liquidity_supply", "discovery", 2]]
    frozen = ResearchPolicy(rows)
    original = frozen.identity
    rows[0][2] = 200
    assert frozen.identity == original
    assert frozen.reservations[0][2] == 2
