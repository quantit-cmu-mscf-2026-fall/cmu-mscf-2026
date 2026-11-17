"""Claim and methods-review contracts: identity, versions and construction refusals."""

from dataclasses import asdict, replace

import pytest

from capstone.factors.idea_contracts import (
    ClaimAssessment,
    DeclaredDefault,
    IdeaClaimV2,
    LocatedQuote,
    MethodsReview,
    ResearchAssumption,
    ReviewedSourceInventory,
)
from capstone.factors.research_contracts import MethodSpec, SourceClaim, digest

REVISION = "a" * 64
SPAN = "b" * 64


def hypothesis(kind="exploratory_original", **changes):
    fields = dict(
        kind=kind,
        mechanism="Temporary liquidity demand can move prices away from fundamentals.",
        prediction="The lowest prior-month return decile has higher next-month returns.",
        horizon="One calendar month after formation.",
        universe="Point-in-time S&P 500 common stocks.",
        falsifier="No positive incremental net return relative to the frozen cash control.",
        mathematics="s(i,t) = -r(i,t-1); w(i,t) = 1/n for the top signal decile.",
        required_inputs=("adjusted_prices", "membership"),
        assumptions=(
            ResearchAssumption("Liquidity pressure is temporary.", "Testable conjecture."),
        ),
    )
    return IdeaClaimV2(**(fields | changes))


def review(**changes):
    fields = dict(
        revision_id=REVISION,
        span_ids=(SPAN,),
        disposition="sufficient",
        reviewer="independent-human-review",
        review_record_id="review-log:record-7",
        method_key="short_reversal",
    )
    return MethodsReview(**(fields | changes))


def assessment(**changes):
    fields = dict(
        claim_id=hypothesis().identity,
        budget_scope_id="controller-reservation:liquidity:discovery",
        inventory_id="inventory",
        input_inventory_id="qualified-input-review:fixture-v1",
        qualified_inputs=("membership", "adjusted_prices"),
        evidence_status="uncited_original",
        completeness_status="specification_ready",
        data_status="ready",
        evidence_issues=(),
        completeness_issues=(),
        missing_inputs=(),
    )
    return ClaimAssessment(**(fields | changes))


def test_quote_offsets_are_exact_character_positions_and_purposes_are_closed():
    quote = LocatedQuote(REVISION, SPAN, "αx", start=3, purpose="method")
    assert quote.end == 5
    with pytest.raises(ValueError, match="purpose"):
        LocatedQuote(REVISION, SPAN, "x", purpose="proof")
    with pytest.raises(ValueError, match="start"):
        LocatedQuote(REVISION, SPAN, "x", start=-1)
    with pytest.raises(ValueError, match="revision identity"):
        LocatedQuote("short", SPAN, "x")
    with pytest.raises(ValueError, match="quote"):
        LocatedQuote(REVISION, SPAN, "   ")


def test_researcher_assumptions_cannot_be_encoded_as_source_statements():
    assumption = ResearchAssumption("The signal always works.", "Unverified researcher conjecture.")
    with pytest.raises(ValueError, match="LocatedQuote"):
        hypothesis(source_statements=(assumption,))
    payload = asdict(hypothesis())
    payload["source_statements"] = [{"text": assumption.text, "rationale": assumption.rationale}]
    with pytest.raises(TypeError):
        IdeaClaimV2.from_dict(payload)
    assert hypothesis(assumptions=(assumption,)).source_statements == ()


def test_agent_full_methods_flags_are_not_claim_fields():
    payload = asdict(hypothesis("reference", method_key="short_reversal"))
    payload["full_methods"] = True
    with pytest.raises(TypeError):
        IdeaClaimV2.from_dict(payload)


def test_partial_specs_persist_with_explicit_unresolved_rules():
    claim = hypothesis(
        prediction="", required_inputs=(), unresolved_rules=("Exit timing unknown.",)
    )
    assert IdeaClaimV2.from_dict(asdict(claim)) == claim


def test_declared_defaults_change_identity_and_need_claim_version_three():
    default = DeclaredDefault(
        "formation lookback", "252 sessions", "The source leaves the window open."
    )
    claim = hypothesis(declared_defaults=(default,))
    restored = IdeaClaimV2.from_dict(asdict(claim))
    assert restored == claim and restored.declared_defaults == (default,)
    assert claim.identity != hypothesis().identity
    legacy = replace(hypothesis(), semantics_version=2)
    with pytest.raises(ValueError, match="version"):
        replace(legacy, declared_defaults=(default,))


def test_v2_claims_keep_identity():
    saved = {k: v for k, v in asdict(hypothesis()).items() if k != "declared_defaults"}
    legacy = IdeaClaimV2.from_dict(saved | {"semantics_version": 2})
    assert legacy.semantics_version == 2 and legacy.declared_defaults == ()
    # Identity computed by the v2-only implementation.
    assert legacy.identity == "864b7c037b33e466d4562c32cc518bd3c7b33e424a8f617bf57455b4c0241925"


def test_v3_claims_keep_identity():
    default = DeclaredDefault("lookback", "252 sessions", "Researcher choice.")
    plain = replace(hypothesis(), semantics_version=3)
    defaulted = replace(hypothesis(declared_defaults=(default,)), semantics_version=3)
    # Identities computed by the claim-v3 implementation.
    assert plain.identity == "6bf6cf6ae970c110e83472ab66fa27cc77142305675da64527302d77ab110c32"
    assert defaulted.identity == "f20044c7391cc25d3d971ccdfa7238b9b5b63eaa6f39407a5882a633fe0d50db"
    assert IdeaClaimV2.from_dict(asdict(defaulted)) == defaulted
    for claim in (plain, defaulted):
        assert claim.identity != replace(claim, semantics_version=4).identity


def test_aliases_order_and_mutable_inputs_cannot_mint_new_identity():
    inputs = ["membership", "adjusted_prices"]
    first = hypothesis(required_inputs=inputs, aliases=("paper-title", "web-title"))
    original = first.identity
    inputs.append("invented_after_construction")
    alias = replace(
        first, aliases=("new alias",), required_inputs=tuple(reversed(first.required_inputs))
    )
    assert first.identity == alias.identity == original
    assert replace(first, horizon="Two months.").identity != original
    assert first.semantics_version == 4
    with pytest.raises(ValueError, match="version"):
        replace(first, semantics_version=1)


def test_math_text_is_preserved_as_data_without_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    text = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    restored = IdeaClaimV2.from_dict(asdict(hypothesis(mathematics=text)))
    assert restored.mathematics == text and not marker.exists()


def test_legacy_v1_source_and_method_identities_are_unchanged():
    source = SourceClaim(
        "https://example.org/paper",
        "page 4",
        "Reversal predicts next month",
        "No incremental net returns",
        ("prices",),
        True,
    )
    method = MethodSpec(
        "claim",
        "reference",
        "positive",
        "top decile",
        "equal",
        "month end",
        "one month",
        "rebalance",
        "lagged",
        "long_only",
    )
    assert (
        digest(asdict(source)) == "88e421dda91f6c9ed7ddb7c92275cc7c5113008efc2662b0507604c66823e3bd"
    )
    assert method.identity == "47fb82b44375d91e04266e5f51ed7c835f747259bfad7c03fddd112c9ea600dc"


def test_conflicting_inventory_reviews_are_rejected_and_inventory_identity_follows_reviews():
    original = review()
    conflicting = replace(original, disposition="insufficient", review_record_id="review-log:8")
    with pytest.raises(ValueError, match="ambiguous"):
        ReviewedSourceInventory("controller", (original, conflicting))
    with pytest.raises(ValueError, match="located"):
        replace(original, span_ids=())
    with pytest.raises(ValueError, match="disposition"):
        review(disposition="approved")
    first = ReviewedSourceInventory("controller", (original,))
    second = ReviewedSourceInventory("controller", (conflicting,))
    assert first.identity != second.identity
    assert first.identity == ReviewedSourceInventory("controller", (original,)).identity


@pytest.mark.parametrize("identity", [None, "", "   "])
def test_qualified_input_names_cannot_claim_readiness_without_inventory_identity(identity):
    with pytest.raises(ValueError, match="inventory identity"):
        assessment(input_inventory_id=identity)


def test_assessment_canonicalizes_names_and_never_admits_financially():
    first = assessment()
    assert first.qualified_inputs == ("adjusted_prices", "membership")
    assert first.assessment_version == 3 and first.financial_status == "pending"
    reordered = assessment(qualified_inputs=("adjusted_prices", "membership"))
    assert digest(asdict(first)) == digest(asdict(reordered))
    changed = assessment(input_inventory_id="qualified-input-review:fixture-v2")
    assert digest(asdict(first)) != digest(asdict(changed))
    with pytest.raises(ValueError, match="inventory identity"):
        replace(first, input_inventory_id=None)
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(first, financial_status="admitted")


def test_stored_assessment_v2_record_still_loads():
    saved = {
        k: v
        for k, v in asdict(assessment(adaptation_record=("declared_default:lookback",))).items()
        if k != "financial_status"
    }
    saved |= {"assessment_version": 2}
    stored = ClaimAssessment(**saved)
    assert stored.assessment_version == 2
    assert stored.adaptation_record == ("declared_default:lookback",)
    assert asdict(stored) == saved | {"financial_status": "pending"}
    with pytest.raises(ValueError, match="assessment version"):
        ClaimAssessment(**(saved | {"assessment_version": 4}))
    with pytest.raises(ValueError, match="version 2"):
        assessment(assessment_version=1, adaptation_record=("declared_default:x",))
