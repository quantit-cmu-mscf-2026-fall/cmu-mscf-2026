"""Idea catalogue entries and the assignment checks run before a claim is assessed."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from capstone.factors.idea_contracts import IdeaClaimV2, LocatedQuote, ReviewedSourceInventory
from capstone.factors.idea_routes import (
    IdeaCatalogueEntry,
    _entry,
    _scope,
    assess_entry_claim,
    catalogue_rows,
)
from capstone.factors.research_contracts import AssignedBrief, ResearchPolicy
from capstone.factors.source_acquisition import acquire_source
from capstone.factors.source_archive import SourceArchive
from capstone.factors.source_intake import SourceRequest

POLICY = ResearchPolicy((("liquidity_supply", "discovery", 2),))
INVENTORY = ReviewedSourceInventory("trusted-controller-review")


def entry(packet="idea", method="reversal", **changes):
    fields = dict(
        brief=AssignedBrief(packet, "liquidity_supply", "discovery", method),
        kind="exploratory_original",
        aliases=("web-alias", "note-alias"),
    )
    return IdeaCatalogueEntry(**(fields | changes))


def claim(**changes):
    fields = dict(
        kind="exploratory_original",
        mechanism="Temporary demand reverses.",
        prediction="Prior losers outperform next month.",
        horizon="One month.",
        universe="sp500",
        falsifier="No incremental net return.",
        mathematics="signal = -prior_return",
        required_inputs=("prices",),
    )
    return IdeaClaimV2(**(fields | changes))


def stub(*entries, inputs=("prices", "flows")):
    rows = {e.brief.packet_id: (e.brief, e, "program") for e in entries}
    return SimpleNamespace(catalogue=rows, policy=POLICY, inputs=frozenset(inputs))


def assess(controller, row, proposal, archive, **changes):
    options = dict(
        archive=archive,
        reviewed_methods=INVENTORY,
        qualified_inputs=("prices",),
        input_inventory_id="trusted-data-v1",
    )
    return assess_entry_claim(controller, row, proposal, **(options | changes))


@pytest.fixture
def archive(tmp_path):
    return SourceArchive(tmp_path / "private-sources")


def test_entry_validates_controller_owned_fields_and_round_trips_with_a_version():
    row = entry(required_inputs=("flows", "prices", "prices"))
    assert row.required_inputs == ("flows", "prices")
    payload = row.to_dict()
    assert payload["version"] == 2 and payload["constructor"] == "program"
    assert IdeaCatalogueEntry.from_dict(payload) == row
    with pytest.raises(ValueError, match="version"):
        IdeaCatalogueEntry.from_dict(payload | {"version": 1})
    with pytest.raises(ValueError, match="constructor"):
        IdeaCatalogueEntry.from_dict(payload | {"constructor": "formula"})
    with pytest.raises(ValueError, match="kind"):
        entry(kind="invented")
    with pytest.raises(ValueError, match="brief"):
        entry(brief={"packet_id": "idea"})
    with pytest.raises(ValueError, match="revision"):
        entry(source_revisions=("not-an-identity",))
    with pytest.raises(ValueError, match="sequence"):
        entry(aliases="web-alias")


def test_catalogue_rows_keep_legacy_rows_and_reject_ambiguous_aliases():
    legacy = {"brief": asdict(AssignedBrief("old", "liquidity_supply", "discovery", "m")), "x": 1}
    rows = catalogue_rows([legacy, entry()])
    assert rows[0] == legacy and rows[1]["version"] == 2
    with pytest.raises(ValueError, match="ambiguous"):
        catalogue_rows([entry(), entry("other", method="m2", aliases=("web-alias",))])
    with pytest.raises(ValueError, match="ambiguous"):
        catalogue_rows([legacy, entry("other", method="m2", aliases=("old",))])


def test_aliases_resolve_to_one_packet_and_legacy_rows_need_the_v1_route():
    controller = stub(entry())
    assert _entry(controller, "note-alias")[0] == "idea"
    assert _entry(controller, "idea")[0] == "idea"
    with pytest.raises(ValueError, match="alias"):
        _entry(controller, "unknown")
    legacy = SimpleNamespace(catalogue={"old": (None, object(), "reference")})
    with pytest.raises(ValueError, match="V1"):
        _entry(legacy, "old")


def test_budget_scope_binds_policy_family_and_track_not_packet_or_alias():
    first = entry()
    other = entry("second", method="weekly", aliases=())
    controller = stub(first, other)
    assert _scope(controller, first) == _scope(controller, other)
    elsewhere = entry(
        "third", brief=AssignedBrief("third", "liquidity_supply", "replication_control", "m3")
    )
    assert _scope(controller, elsewhere) != _scope(controller, first)
    changed = SimpleNamespace(policy=ResearchPolicy((("liquidity_supply", "discovery", 3),)))
    assert _scope(changed, first) != _scope(controller, first)


def test_assignment_adds_required_inputs_and_keeps_the_assigned_scope(archive):
    row = entry(required_inputs=("flows",))
    controller = stub(row)
    effective, assessment = assess(controller, row, claim(), archive)
    assert effective.required_inputs == ("flows", "prices")
    assert effective.identity != claim().identity
    assert assessment.missing_inputs == ("flows",) and assessment.data_status == "needs_data"
    assert assessment.budget_scope_id == _scope(controller, row)
    assert assessment.financial_status == "pending"


def test_claim_cannot_change_kind_method_or_exceed_frozen_capabilities(archive):
    row = entry()
    controller = stub(row)
    with pytest.raises(ValueError, match="kind"):
        assess(controller, row, claim(kind="adaptation"), archive)
    with pytest.raises(ValueError, match="canonical method"):
        assess(controller, row, claim(method_key="different"), archive)
    assess(controller, row, claim(method_key="reversal"), archive)
    with pytest.raises(ValueError, match="capabilities"):
        assess(controller, row, claim(), archive, qualified_inputs=("prices", "futures"))
    with pytest.raises(ValueError, match="explicit collection"):
        assess(controller, row, claim(), archive, qualified_inputs="prices")


def test_claim_may_cite_only_approved_source_revisions(archive):
    request = SourceRequest(
        "inline_text", text="Rank prior returns.", key="paper", role="reference"
    )
    artifact = archive.load(archive.intake(request, acquire_source).revision_id)
    span = artifact.spans[0]
    quote = LocatedQuote(artifact.revision_id, span.span_id, span.text, purpose="method")
    proposal = claim(source_statements=(quote,))
    unapproved = entry()
    with pytest.raises(ValueError, match="unapproved source"):
        assess(stub(unapproved), unapproved, proposal, archive)
    approved = replace(unapproved, source_revisions=(artifact.revision_id,))
    _, assessment = assess(stub(approved), approved, proposal, archive)
    assert assessment.evidence_status == "quote_integrity_verified"
