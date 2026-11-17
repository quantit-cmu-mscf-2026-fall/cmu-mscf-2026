from dataclasses import asdict

import pytest

from capstone import artifacts
from capstone.factors import research_contracts as rc


def test_canonical_and_digest_are_the_shared_artifact_identity():
    assert rc.canonical is artifacts.canonical
    assert rc.digest is artifacts.digest


def test_policy_identity_ignores_reservation_order_and_rejects_bad_scope():
    a = rc.ResearchPolicy((("forced_flows", "discovery", 2), ("liquidity_supply", "refinement", 1)))
    b = rc.ResearchPolicy((("liquidity_supply", "refinement", 1), ("forced_flows", "discovery", 2)))
    assert a.identity == b.identity and len(a.identity) == 64
    for bad in (
        {"discovery_end": "2021-06-30"},
        {"max_repairs": 2},
        {"model_rejection": True},
        {"universe": "russell"},
    ):
        with pytest.raises(ValueError):
            rc.ResearchPolicy((), **bad)
    with pytest.raises(ValueError, match="duplicate"):
        rc.ResearchPolicy((("forced_flows", "discovery", 1), ("forced_flows", "discovery", 2)))
    with pytest.raises(ValueError, match="invalid"):
        rc.ResearchPolicy((("nonsense", "discovery", 1),))
    with pytest.raises(ValueError, match="invalid"):
        rc.ResearchPolicy((("forced_flows", "discovery", True),))


def test_brief_claim_and_method_validation():
    with pytest.raises(ValueError):
        rc.AssignedBrief("p", "forced_flows", "other", "m")
    with pytest.raises(ValueError):
        rc.AssignedBrief("", "forced_flows", "discovery", "m")
    kwargs = dict(
        source_url="https://example.org/paper",
        method_location="s2",
        claim="c",
        falsifier="f",
        required_inputs=("returns", "returns", "membership"),
        full_methods=True,
    )
    claim = rc.SourceClaim(**kwargs)
    assert claim.required_inputs == ("membership", "returns")
    with pytest.raises(ValueError, match="primary"):
        rc.SourceClaim(**{**kwargs, "source_url": "http://example.org"})
    with pytest.raises(ValueError, match="boolean"):
        rc.SourceClaim(**{**kwargs, "full_methods": 1})
    with pytest.raises(ValueError, match="required inputs"):
        rc.SourceClaim(**{**kwargs, "required_inputs": ()})


def _method(**over):
    base = dict(
        claim_id="c",
        variant="reference",
        direction="long",
        selection="top",
        weighting="equal",
        formation="monthly",
        holding="1m",
        exits="none",
        availability="lagged",
        profile="long_only",
    )
    return rc.MethodSpec(**{**base, **over})


def test_method_identity_changes_with_semantics_and_adaptations_disclose():
    assert _method().identity == _method().identity
    assert _method().identity != _method(weighting="cap").identity
    assert asdict(_method())["deviations"] == ()
    with pytest.raises(ValueError, match="deviations"):
        _method(variant="adaptation")
    assert _method(variant="adaptation", deviations=["drops microcaps"]).deviations == (
        "drops microcaps",
    )
    with pytest.raises(ValueError, match="incomplete"):
        _method(exits="")


def test_qualify_never_trusts_the_claim_for_availability():
    def claim(**over):
        base = dict(
            source_url="https://example.org/p",
            method_location="s2",
            claim="c",
            falsifier="f",
            required_inputs=("returns", "short_interest"),
            full_methods=True,
        )
        return rc.SourceClaim(**{**base, **over})

    have = frozenset({"returns"})
    assert rc.qualify(claim(full_methods=False), have) == ("methods_review_needed", ())
    assert rc.qualify(claim(universe="global"), have) == ("outside_scope", ())
    assert rc.qualify(claim(), have) == ("needs_data", ("short_interest",))
    assert rc.qualify(claim(), have | {"short_interest"}) == ("specification_ready", ())
    assert not rc.ConstructionEvidence("s", "v1", True, True, False).passed
    assert rc.ConstructionEvidence("s", "v1", True, True, True).passed
