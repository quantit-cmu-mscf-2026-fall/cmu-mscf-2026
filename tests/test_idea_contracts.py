"""Private offline draft checks; no shared repository files are modified."""

from dataclasses import asdict, replace

import pytest

from capstone.factors.idea_contracts import (
    DeclaredDefault,
    IdeaClaimV2,
    LocatedQuote,
    MethodsReview,
    ResearchAssumption,
    ReviewedSourceInventory,
    assess_claim,
)
from capstone.factors.research_contracts import digest
from capstone.factors.source_acquisition import acquire_source
from capstone.factors.source_archive import SourceArchive
from capstone.factors.source_intake import SourceRequest

SCOPE = "controller-reservation:liquidity:discovery"
EMPTY_INVENTORY = ReviewedSourceInventory("independent-reviewer")


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


def assess(claim, archive, **changes):
    options = dict(
        budget_scope_id=SCOPE,
        reviewed_methods=EMPTY_INVENTORY,
        qualified_inputs=frozenset({"adjusted_prices", "membership"}),
        input_inventory_id="qualified-input-review:fixture-v1",
    )
    return assess_claim(claim, archive, **(options | changes))


def add_source(
    archive,
    text="Rank prior returns; hold the selected decile for one month.",
    key="paper",
    **changes,
):
    request = SourceRequest("inline_text", text=text, key=key, role="reference")
    outcome = archive.intake(request, acquire_source, **changes)
    artifact = archive.load(outcome.revision_id)
    span = artifact.spans[0]
    quote = LocatedQuote(artifact.revision_id, span.span_id, span.text, purpose="method")
    return artifact, quote


def inventory(artifact, *, method_key="short_reversal", **changes):
    review = MethodsReview(
        artifact.revision_id,
        tuple(span.span_id for span in artifact.spans),
        "sufficient",
        "independent-human-review",
        "review-log:record-7",
        method_key,
    )
    return ReviewedSourceInventory("controller-owned-source-review", (replace(review, **changes),))


@pytest.fixture
def archive(tmp_path):
    return SourceArchive(tmp_path / "private-sources")


def test_original_without_primary_paper_can_be_specified_but_finance_stays_pending(archive):
    claim = hypothesis()
    result = assess(claim, archive)
    assert result.evidence_status == "uncited_original"
    assert result.completeness_status == "specification_ready"
    assert result.data_status == "ready" and result.financial_status == "pending"
    assert result.budget_scope_id == SCOPE
    assert archive.history() == ()


def test_same_specification_cannot_claim_reference_sufficiency_without_review(archive):
    claim = hypothesis("reference", method_key="short_reversal")
    result = assess(claim, archive)
    assert result.evidence_status == "missing"
    assert result.completeness_status == "incomplete"
    assert "methods_review_required" in result.completeness_issues


def test_exact_multisource_evidence_and_trusted_methods_review_are_separate(archive):
    paper, quote = add_source(archive)
    _, corroboration = add_source(
        archive, "A separate observation motivates the mechanism.", key="note"
    )
    claim = hypothesis(
        "reference",
        source_statements=(quote, corroboration),
        method_sources=(paper.revision_id,),
        method_key="short_reversal",
    )
    unreviewed = assess(claim, archive)
    assert unreviewed.evidence_status == "quote_integrity_verified"
    assert unreviewed.completeness_status == "incomplete"
    reviewed = assess(claim, archive, reviewed_methods=inventory(paper))
    assert reviewed.evidence_status == "quote_integrity_verified"
    assert reviewed.completeness_status == "specification_ready"
    assert reviewed.inventory_id != unreviewed.inventory_id
    assert reviewed.financial_status == "pending"


@pytest.mark.parametrize(
    "mutation,issue",
    [
        ({"revision_id": "0" * 64}, "source_revision_invalid:"),
        ({"span_id": "0" * 64}, "source_span_missing:"),
        ({"quote": "An invented method absent from the source."}, "source_quote_mismatch:"),
        ({"start": 1}, "source_quote_mismatch:"),
    ],
)
def test_invented_or_mislocated_evidence_is_invalid_not_silently_repaired(archive, mutation, issue):
    _, quote = add_source(archive)
    result = assess(hypothesis(source_statements=(replace(quote, **mutation),)), archive)
    assert result.evidence_status == "invalid"
    assert any(item.startswith(issue) for item in result.evidence_issues)
    assert result.completeness_status == "specification_ready"
    assert result.financial_status == "pending"


def test_quote_offsets_are_exact_unicode_character_positions_in_the_bound_span(archive):
    artifact, full = add_source(archive, "αx αx")
    second = replace(full, quote="αx", start=3, purpose="mechanism")
    assert second.end == 5
    result = assess(hypothesis(source_statements=(second,)), archive)
    assert result.evidence_status == "quote_integrity_verified"
    bad = replace(second, start=2)
    assert assess(hypothesis(source_statements=(bad,)), archive).evidence_status == "invalid"
    assert archive.load(artifact.revision_id).spans[0].text == "αx αx"


def test_changed_source_revision_cannot_inherit_old_quotes_or_methods_review(archive):
    first, old_quote = add_source(archive)
    second, new_quote = add_source(
        archive, "Changed method: rank volume, not prior returns.", refresh=True
    )
    assert first.revision_id != second.revision_id
    result = assess(hypothesis(source_statements=(old_quote,)), archive)
    assert result.evidence_status == "quote_integrity_verified"
    forged = replace(old_quote, revision_id=second.revision_id)
    assert assess(hypothesis(source_statements=(forged,)), archive).evidence_status == "invalid"
    claim = hypothesis(
        "reference",
        source_statements=(new_quote,),
        method_sources=(second.revision_id,),
        method_key="short_reversal",
    )
    result = assess(claim, archive, reviewed_methods=inventory(first))
    assert result.completeness_status == "incomplete"
    assert f"methods_review_required:{second.revision_id}" in result.completeness_issues


def test_mutated_archived_bytes_fail_closed_without_refetch(archive):
    artifact, quote = add_source(archive)
    path = archive.root / "objects" / (artifact.raw_sha256 + ".bin")
    path.write_bytes(b"Changed data cannot verify under the old content identity.")
    result = assess(hypothesis(source_statements=(quote,)), archive)
    assert result.evidence_status == "invalid"
    assert result.evidence_issues == (f"source_revision_invalid:{artifact.revision_id}",)
    assert len(archive.history()) == 1


def test_researcher_assumptions_cannot_be_encoded_as_source_statements(archive):
    assumption = ResearchAssumption("The signal always works.", "Unverified researcher conjecture.")
    with pytest.raises(ValueError, match="LocatedQuote"):
        hypothesis(source_statements=(assumption,))
    payload = asdict(hypothesis())
    payload["source_statements"] = [{"text": assumption.text, "rationale": assumption.rationale}]
    with pytest.raises(TypeError):
        IdeaClaimV2.from_dict(payload)
    result = assess(hypothesis(assumptions=(assumption,)), archive)
    assert result.evidence_status == "uncited_original"


def test_agent_full_methods_or_inventory_flags_are_not_claim_fields(archive):
    payload = asdict(hypothesis("reference", method_key="short_reversal"))
    payload["full_methods"] = True
    with pytest.raises(TypeError):
        IdeaClaimV2.from_dict(payload)
    with pytest.raises(ValueError, match="separately"):
        assess(hypothesis(), archive, reviewed_methods={"full_methods": True})


def test_review_of_another_method_or_uncited_method_spans_cannot_bless_claim(archive):
    artifact, quote = add_source(archive)
    claim = hypothesis(
        "reference",
        source_statements=(quote,),
        method_sources=(artifact.revision_id,),
        method_key="different_method",
    )
    assert (
        assess(claim, archive, reviewed_methods=inventory(artifact)).completeness_status
        == "incomplete"
    )
    context = replace(quote, purpose="context")
    claim = replace(claim, method_key="short_reversal", source_statements=(context,))
    result = assess(claim, archive, reviewed_methods=inventory(artifact))
    assert f"reviewed_method_evidence_invalid:{artifact.revision_id}" in result.completeness_issues


def test_insufficient_review_and_undeclared_adaptation_remain_blocked(archive):
    artifact, quote = add_source(archive)
    claim = hypothesis(
        "adaptation",
        source_statements=(quote,),
        method_sources=(artifact.revision_id,),
        method_key="short_reversal",
    )
    result = assess(
        claim, archive, reviewed_methods=inventory(artifact, disposition="insufficient")
    )
    assert "adaptation_notes_required" in result.completeness_issues
    assert f"methods_review_required:{artifact.revision_id}" in result.completeness_issues
    complete = replace(
        claim, adaptation_notes=("Use point-in-time S&P 500 instead of all equities.",)
    )
    reviewed = assess(complete, archive, reviewed_methods=inventory(artifact))
    assert reviewed.completeness_status == "specification_ready"
    assert reviewed.financial_status == "pending"


def test_missing_inputs_are_retained_without_changing_other_status_axes(archive):
    result = assess(hypothesis(), archive, qualified_inputs=frozenset({"membership"}))
    assert result.missing_inputs == ("adjusted_prices",)
    assert result.data_status == "needs_data"
    assert result.evidence_status == "uncited_original"
    assert result.completeness_status == "specification_ready"
    with pytest.raises(ValueError, match="collection"):
        assess(hypothesis(), archive, qualified_inputs="adjusted_prices")


def test_partial_specs_and_unresolved_rules_persist_as_explicit_completeness_gaps(archive):
    claim = hypothesis(
        prediction="", required_inputs=(), unresolved_rules=("Exit timing unknown.",)
    )
    restored = IdeaClaimV2.from_dict(asdict(claim))
    assert restored == claim
    result = assess(restored, archive)
    assert result.completeness_status == "incomplete" and result.data_status == "undeclared"
    assert "missing_field:prediction" in result.completeness_issues
    assert "missing_field:required_inputs" in result.completeness_issues
    assert "unresolved_rule:Exit timing unknown." in result.completeness_issues


def test_declared_defaults_persist_without_blocking(archive):
    default = DeclaredDefault(
        "formation lookback", "252 sessions", "The source leaves the window open."
    )
    claim = hypothesis(declared_defaults=(default,))
    restored = IdeaClaimV2.from_dict(asdict(claim))
    assert restored == claim and restored.declared_defaults == (default,)
    assert claim.identity != hypothesis().identity
    result = assess(restored, archive)
    assert result.completeness_status == "specification_ready" and result.data_status == "ready"
    assert result.completeness_issues == ()
    assert result.adaptation_record == ("declared_default:formation lookback",)
    assert result.assessment_version == 3
    # A declared default never resolves an unresolved rule; unresolved still blocks.
    blocked = assess(replace(claim, unresolved_rules=("Exit timing unknown.",)), archive)
    assert blocked.completeness_status == "incomplete"
    assert blocked.completeness_issues == ("unresolved_rule:Exit timing unknown.",)
    assert blocked.adaptation_record == ("declared_default:formation lookback",)


def test_v2_claims_keep_identity_and_assess_under_v1_semantics(archive):
    saved = {k: v for k, v in asdict(hypothesis()).items() if k != "declared_defaults"}
    legacy = IdeaClaimV2.from_dict(saved | {"semantics_version": 2})
    assert legacy.semantics_version == 2 and legacy.declared_defaults == ()
    # Identity computed by the v2-only implementation (main @ b3588d8).
    assert legacy.identity == "864b7c037b33e466d4562c32cc518bd3c7b33e424a8f617bf57455b4c0241925"
    result = assess(legacy, archive)
    assert result.assessment_version == 1 and result.adaptation_record == ()
    assert result.completeness_status == "specification_ready"
    default = DeclaredDefault("lookback", "252 sessions", "Researcher choice.")
    with pytest.raises(ValueError, match="version"):
        replace(legacy, declared_defaults=(default,))


def test_aliases_order_and_mutable_inputs_cannot_mint_new_identity_or_budget_scope(archive):
    inputs = ["membership", "adjusted_prices"]
    first = hypothesis(required_inputs=inputs, aliases=("paper-title", "web-title"))
    original = first.identity
    inputs.append("invented_after_construction")
    alias = replace(
        first, aliases=("new alias",), required_inputs=tuple(reversed(first.required_inputs))
    )
    assert first.identity == alias.identity == original
    changed = replace(first, horizon="Two months.")
    assert changed.identity != original
    assert {assess(claim, archive).budget_scope_id for claim in (first, alias, changed)} == {SCOPE}
    assert first.semantics_version == 4
    with pytest.raises(ValueError, match="version"):
        replace(first, semantics_version=1)


def test_v3_claims_keep_identity_and_assess_exactly_as_before(archive):
    default = DeclaredDefault("lookback", "252 sessions", "Researcher choice.")
    plain = replace(hypothesis(), semantics_version=3)
    defaulted = replace(hypothesis(declared_defaults=(default,)), semantics_version=3)
    # Identities computed by the claim-v3 implementation (main @ 37da276).
    assert plain.identity == "6bf6cf6ae970c110e83472ab66fa27cc77142305675da64527302d77ab110c32"
    assert defaulted.identity == "f20044c7391cc25d3d971ccdfa7238b9b5b63eaa6f39407a5882a633fe0d50db"
    assert IdeaClaimV2.from_dict(asdict(defaulted)) == defaulted
    for claim in (plain, defaulted):
        old, new = assess(claim, archive), assess(replace(claim, semantics_version=4), archive)
        assert old.assessment_version == new.assessment_version == 3
        assert asdict(old) == asdict(new) | {"claim_id": claim.identity}
        assert claim.identity != replace(claim, semantics_version=4).identity
    # No inventory can qualify an explicit unavailable declaration.
    with pytest.raises(ValueError, match="unavailable"):
        assess(plain, archive, qualified_inputs=frozenset({"unavailable:membership"}))


def test_math_text_is_preserved_as_data_without_execution_or_financial_admission(archive, tmp_path):
    marker = tmp_path / "must-not-exist"
    text = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    claim = hypothesis(mathematics=text)
    restored = IdeaClaimV2.from_dict(asdict(claim))
    assert restored.mathematics == text and not marker.exists()
    result = assess(restored, archive)
    assert result.financial_status == "pending"
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(result, financial_status="admitted")
    assert not marker.exists()


def test_conflicting_inventory_reviews_are_rejected_and_review_revisions_are_bound(archive):
    artifact, quote = add_source(archive)
    reviewed = inventory(artifact)
    original = reviewed.reviews[0]
    conflicting = replace(original, disposition="insufficient", review_record_id="review-log:8")
    with pytest.raises(ValueError, match="ambiguous"):
        ReviewedSourceInventory("controller", (original, conflicting))
    with pytest.raises(ValueError, match="located"):
        replace(original, span_ids=())
    claim = hypothesis(
        "reference",
        method_key="short_reversal",
        source_statements=(quote,),
        method_sources=(artifact.revision_id,),
    )
    old_assessment = assess(claim, archive, reviewed_methods=reviewed)
    new_inventory = replace(reviewed, reviews=(conflicting,))
    new_assessment = assess(claim, archive, reviewed_methods=new_inventory)
    assert old_assessment.inventory_id != new_assessment.inventory_id
    assert old_assessment.completeness_status == "specification_ready"
    assert new_assessment.completeness_status == "incomplete"


def test_invented_method_quote_cannot_satisfy_the_reviewed_method_binding(archive):
    artifact, quote = add_source(archive)
    invented = replace(quote, quote="Use an invented rule absent from the reviewed method.")
    claim = hypothesis(
        "reference",
        method_key="short_reversal",
        source_statements=(invented,),
        method_sources=(artifact.revision_id,),
    )
    result = assess(claim, archive, reviewed_methods=inventory(artifact))
    assert result.evidence_status == "invalid" and result.completeness_status == "incomplete"
    assert f"reviewed_method_evidence_invalid:{artifact.revision_id}" in result.completeness_issues


@pytest.mark.parametrize("identity", [None, "", "   "])
def test_qualified_input_names_cannot_claim_readiness_without_inventory_identity(archive, identity):
    with pytest.raises(ValueError, match="inventory identity"):
        assess(hypothesis(), archive, input_inventory_id=identity)


def test_empty_qualified_inventory_can_report_missing_data_without_inventing_authority(archive):
    result = assess_claim(
        hypothesis(), archive, budget_scope_id=SCOPE, reviewed_methods=EMPTY_INVENTORY
    )
    assert result.data_status == "needs_data" and result.input_inventory_id is None
    assert result.qualified_inputs == ()
    assert result.missing_inputs == ("adjusted_prices", "membership")


def test_assessments_bind_input_inventory_identity_and_canonical_qualified_names(archive):
    first = assess(hypothesis(), archive, qualified_inputs=["membership", "adjusted_prices"])
    reordered = assess(hypothesis(), archive, qualified_inputs=["adjusted_prices", "membership"])
    changed = assess(hypothesis(), archive, input_inventory_id="qualified-input-review:fixture-v2")
    assert first.qualified_inputs == ("adjusted_prices", "membership")
    assert first.assessment_version == 3
    assert digest(asdict(first)) == digest(asdict(reordered))
    assert first.data_status == changed.data_status == "ready"
    assert digest(asdict(first)) != digest(asdict(changed))
    with pytest.raises(ValueError, match="inventory identity"):
        replace(first, input_inventory_id=None)


def long_source(archive, paragraphs=66):
    paragraph = "Method paragraph {}: rank stocks by trailing risk."
    text = "\n\n".join(paragraph.format(n) for n in range(paragraphs))
    request = SourceRequest("inline_text", text=text, key="long-paper", role="reference")
    return archive.load(archive.intake(request, acquire_source).revision_id)


def cite(artifact, index, purpose):
    span = artifact.spans[index]
    return LocatedQuote(artifact.revision_id, span.span_id, span.text, purpose=purpose)


def scoped_inventory(artifact, reviewed=64):
    spans = tuple(span.span_id for span in artifact.spans[:reviewed])
    return inventory(artifact, method_key="low_risk", span_ids=spans)


def test_large_review_needs_cited_subset_within_scope(archive):
    paper = long_source(archive)
    reviews = scoped_inventory(paper)
    assert len(reviews.reviews[0].span_ids) == 64 and len(paper.spans) == 66
    issue = f"reviewed_method_evidence_invalid:{paper.revision_id}"
    claim = hypothesis(
        "reference",
        method_key="low_risk",
        method_sources=(paper.revision_id,),
        source_statements=(cite(paper, 3, "method"), cite(paper, 40, "mathematics")),
    )
    ready = assess(claim, archive, reviewed_methods=reviews)
    assert ready.assessment_version == 3
    assert (
        ready.completeness_status == "specification_ready"
        and issue not in ready.completeness_issues
    )
    # A cited method or mathematics span outside the reviewed scope is not covered.
    for purpose in ("method", "mathematics"):
        outside = replace(
            claim, source_statements=(*claim.source_statements, cite(paper, 65, purpose))
        )
        assert issue in assess(outside, archive, reviewed_methods=reviews).completeness_issues
    # Context citations of reviewed spans do not cite the reviewed method.
    uncited = replace(claim, source_statements=(cite(paper, 3, "context"),))
    assert issue in assess(uncited, archive, reviewed_methods=reviews).completeness_issues
    # A review span missing from the archived revision still invalidates the review.
    forged = inventory(
        paper, method_key="low_risk", span_ids=(*reviews.reviews[0].span_ids, "f" * 64)
    )
    assert issue in assess(claim, archive, reviewed_methods=forged).completeness_issues


def test_large_reviewed_method_replays_under_scope_rule(archive):
    """A low-risk-control shape: 64 reviewed spans, 2 cited method spans.

    The saved v2 claim keeps its assessment-v1 outcome (old runs are immutable); the
    same content as a v3 claim clears the method-evidence axis. The Cohen-Frazzini
    shape (reference citing no method or mathematics span) stays invalid.
    """
    paper = long_source(archive)
    reviews = scoped_inventory(paper)
    issue = f"reviewed_method_evidence_invalid:{paper.revision_id}"
    low_risk = hypothesis(
        "adaptation",
        method_key="low_risk",
        method_sources=(paper.revision_id,),
        adaptation_notes=("Point-in-time S&P 500 instead of CRSP top 1000.",),
        source_statements=(
            cite(paper, 10, "method"),
            cite(paper, 5, "prediction"),
            cite(paper, 10, "prediction"),
            cite(paper, 20, "mathematics"),
            cite(paper, 20, "method"),
            cite(paper, 10, "universe"),
        ),
    )
    legacy = replace(low_risk, semantics_version=2)
    old = assess(legacy, archive, reviewed_methods=reviews)
    assert old.assessment_version == 1 and issue in old.completeness_issues
    new = assess(low_risk, archive, reviewed_methods=reviews)
    assert new.assessment_version == 3 and new.completeness_issues == ()
    assert new.completeness_status == "specification_ready"
    network = hypothesis(
        "reference",
        method_key="low_risk",
        method_sources=(paper.revision_id,),
        source_statements=tuple(
            cite(paper, 7, purpose)
            for purpose in ("mechanism", "prediction", "context", "universe")
        ),
    )
    assert issue in assess(network, archive, reviewed_methods=reviews).completeness_issues
