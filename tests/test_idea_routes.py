from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from capstone.factors.idea_contracts import (
    IdeaClaimV2,
    LocatedQuote,
    MethodsReview,
    ReviewedSourceInventory,
)
from capstone.factors.idea_routes import (
    IdeaCatalogueEntry,
    assess_idea,
    begin_idea_attempt,
    complete_idea_attempt,
    idea_binding,
    record_idea_feedback,
    request_context,
    request_idea_repair,
    reserve_idea,
)
from capstone.factors.research_contracts import (
    AssignedBrief,
    ConstructionEvidence,
    MethodSpec,
    ResearchPolicy,
    SourceClaim,
    digest,
)
from capstone.factors.research_controller import ResearchController
from capstone.factors.research_controller import ResearchController as LegacyController
from capstone.factors.research_records import load_artifact
from capstone.factors.research_requests import RequestJournal
from capstone.factors.source_acquisition import acquire_source
from capstone.factors.source_archive import SourceArchive
from capstone.factors.source_intake import SourceRequest
from capstone.pipeline.budget import BudgetedClient, CallBudgetExceeded

INVENTORY = ReviewedSourceInventory("trusted-controller-review")
POLICY = ResearchPolicy((("liquidity_supply", "discovery", 2),))


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


def controller(path, rows=None):
    return ResearchController(
        path, POLICY, rows or [entry()], ("prices", "flows"), generator_version="private-routing-v1"
    )


def assess(c, archive, proposal=None, packet="idea", **changes):
    fields = dict(
        archive=archive,
        reviewed_methods=INVENTORY,
        qualified_inputs=("prices",),
        input_inventory_id="trusted-data-v1",
    )
    return assess_idea(c, packet, proposal or claim(), **(fields | changes))


def feedback(c, current, *, reason="construction_failed"):
    return record_idea_feedback(c, "idea", kind=reason, reason="Typed verification failure.")


class Client:
    max_retries = 0
    calls = 0

    def __init__(self, proposal=None, tool_name="claim"):
        self.messages = self
        self.proposal = asdict(claim()) if proposal is None else proposal
        self.tool_name = tool_name

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name=self.tool_name, input=self.proposal)],
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )


def parse(response):
    if not isinstance(response.content[0].input, dict):
        raise ValueError("claim must be an object")
    return IdeaClaimV2.from_dict(response.content[0].input)


def attempt(c, archive, journal, proposal=None):
    context = request_context(c, "note-alias")
    request = {"model": "fixture", "messages": [{"role": "user", "content": "frozen"}]}
    input_id = digest("frozen-generation-input")
    attempt_id = begin_idea_attempt(
        c, "web-alias", frozen_input_id=input_id, request=request, journal=journal
    )
    response = BudgetedClient(
        Client(None if proposal is None else asdict(proposal)), 2, journal=journal, **context
    ).create(**request)
    request_id = journal.report()["requests"][-1]["id"]
    outcome = complete_idea_attempt(
        c,
        "idea",
        attempt_id=attempt_id,
        frozen_input_id=input_id,
        journal=journal,
        request_id=request_id,
        response=response,
        parse_response=parse,
        archive=archive,
        reviewed_methods=INVENTORY,
        qualified_inputs=("prices",),
        input_inventory_id="trusted-data-v1",
    )
    return outcome, response, request_id, request


def repairs(c):
    return c.db.execute("SELECT repairs FROM research_briefs WHERE id='idea'").fetchone()[0]


@pytest.fixture
def archive(tmp_path):
    return SourceArchive(tmp_path / "private-sources")


def test_original_route_has_no_fake_paper_and_identical_binding_replays_idempotently(
    tmp_path, archive
):
    c = controller(tmp_path / "controller.db")
    assert reserve_idea(c, "web-alias")
    first = assess(c, archive)
    events = c.events()
    assert assess(c, archive, packet="note-alias") == first
    assert c.events() == events and repairs(c) == 0
    assert c.state("idea") == "idea_assessed"
    assert first["assessment"]["evidence_status"] == "uncited_original"
    assert first["assessment"]["financial_status"] == "pending"
    assert c.db.execute("SELECT count(*) FROM research_briefs").fetchone()[0] == 1
    assert (
        c.db.execute(
            "SELECT count(*) FROM research_artifacts WHERE kind='source_claim'"
        ).fetchone()[0]
        == 0
    )
    with pytest.raises(ValueError, match="unsupported transition"):
        c.replay_handoff("idea")
    c.close()


def test_aliases_resume_and_parameter_changes_cannot_reset_reservations(tmp_path, archive):
    path = tmp_path / "controller.db"
    rows = [entry(), entry("renamed", aliases=())]
    c = controller(path, rows)
    assert reserve_idea(c, "note-alias")
    first = assess(c, archive)
    with pytest.raises(ValueError, match="counted"):
        assess(c, archive, replace(claim(), horizon="Two months."))
    assert not c.reserve("renamed") and not reserve_idea(c, "web-alias")
    c.close()
    c = controller(path, list(reversed(rows)))
    assert idea_binding(c, "web-alias")["binding_id"] == first["binding_id"]
    assert not reserve_idea(c, "note-alias") and not c.reserve("renamed")
    assert repairs(c) == 0
    c.close()


def test_changed_trusted_inventories_preserve_original_claim_reservation_and_evidence(
    tmp_path, archive
):
    c = controller(tmp_path / "controller.db")
    reserve_idea(c, "idea")
    first = assess(c, archive)
    before = len(c.events())
    second = assess(
        c,
        archive,
        reviewed_methods=ReviewedSourceInventory("review-v2"),
        input_inventory_id="trusted-data-v2",
    )
    assert len(c.events()) == before + 1 and repairs(c) == 0
    assert first["claim_record_id"] == second["claim_record_id"]
    assert first["declared_claim_record_id"] == second["declared_claim_record_id"]
    assert first["binding_id"] != second["binding_id"]
    assert first["assessment_record_id"] != second["assessment_record_id"]
    assert first["budget_scope_id"] == second["budget_scope_id"]
    assert (
        load_artifact(c.db, first["assessment_record_id"], kind="idea_assessment_v1")
        == first["assessment"]
    )
    assert c.db.execute("SELECT count(*) FROM research_briefs").fetchone()[0] == 1
    c.close()


def test_frozen_capabilities_and_curator_dependencies_cannot_be_bypassed(tmp_path, archive):
    c = controller(tmp_path / "controller.db", [entry(required_inputs=("flows",))])
    reserve_idea(c, "idea")
    result = assess(c, archive)
    assert result["assessment"]["missing_inputs"] == ["flows"]
    assert result["assessment"]["data_status"] == "needs_data"
    assert result["declared_claim_id"] != result["claim_id"]
    with pytest.raises(ValueError, match="frozen"):
        assess(c, archive, qualified_inputs=("prices", "agent-invented-capability"))
    assert repairs(c) == 0
    c.close()


def test_source_backed_route_requires_approved_revisions_and_external_method_review(
    tmp_path, archive
):
    source = SourceRequest("inline_text", text="Rank prior returns and hold for one month.")
    artifact = archive.load(archive.intake(source, acquire_source).revision_id)
    quote = LocatedQuote(
        artifact.revision_id, artifact.spans[0].span_id, artifact.spans[0].text, purpose="method"
    )
    c = controller(
        tmp_path / "controller.db",
        [entry(kind="reference", source_revisions=(artifact.revision_id,))],
    )
    reserve_idea(c, "idea")
    proposal = claim(
        kind="reference",
        method_key="reversal",
        source_statements=(quote,),
        method_sources=(artifact.revision_id,),
    )
    unreviewed = assess(c, archive, proposal)
    assert unreviewed["assessment"]["completeness_status"] == "incomplete"
    reviewed = ReviewedSourceInventory(
        "trusted-review",
        (
            MethodsReview(
                artifact.revision_id,
                (quote.span_id,),
                "sufficient",
                "reviewer",
                "review:1",
                "reversal",
            ),
        ),
    )
    current = assess(c, archive, proposal, reviewed_methods=reviewed)
    assert current["assessment"]["completeness_status"] == "specification_ready"
    assert current["assessment"]["financial_status"] == "pending"
    forbidden = replace(proposal, source_statements=(replace(quote, revision_id="0" * 64),))
    with pytest.raises(ValueError, match="unapproved"):
        assess(c, archive, forbidden)
    with pytest.raises(ValueError, match="canonical method"):
        assess(c, archive, replace(proposal, method_key="agent-new-family"))
    c.close()


def test_legacy_catalogue_identity_and_v1_transitions_are_unchanged(tmp_path):
    source = SourceClaim(
        "https://example.org/paper", "page 4", "Reversal", "No reversal", ("prices",), True
    )
    legacy = {
        "brief": asdict(AssignedBrief("old", "liquidity_supply", "discovery", "old-method")),
        "claim": asdict(source),
        "constructor": "reference",
    }
    original = LegacyController(
        tmp_path / "old.db", POLICY, [legacy], ("prices",), generator_version="v1"
    )
    c = ResearchController(
        tmp_path / "private.db", POLICY, [legacy], ("prices",), generator_version="v1"
    )
    assert c.identity == original.identity
    assert c.identity == "27bbc41a2cfd7a5f6fc81ae391de07ac69bf97e55396b0f796c805e66e41061a"
    assert c.reserve("old") and original.reserve("old")
    assert c.ground("old", source) == original.ground("old", source) == "specification_ready"
    assert c.claim_id("old") == "7af088a39248a548697c2ddd31e05625304f985155a01bc6215f2c099d9d8638"
    spec = MethodSpec(
        c.claim_id("old"),
        "reference",
        "positive",
        "decile",
        "equal",
        "month end",
        "one month",
        "rebalance",
        "lagged",
        "long_only",
    )
    assert c.freeze_spec("old", spec) == original.freeze_spec("old", spec) == spec.identity
    assert spec.identity == "9dba3244b50338faa8323b8fbe599c3edf1bf6e8a018e80c21b58d2d6b933aa7"

    def verifier(specification):
        return ConstructionEvidence(specification.identity, "verified-v1", True, True, True)

    assert (
        c.construct("old", verifier)
        == original.construct("old", verifier)
        == "construction_verified"
    )
    assert c.replay_handoff("old") == original.replay_handoff("old")
    assert c.events() == original.events()
    original.close()
    c.close()


def test_frozen_catalogue_cannot_change_source_scope_family_or_alias_ownership(tmp_path):
    path = tmp_path / "controller.db"
    c = controller(path)
    assert reserve_idea(c, "web-alias")
    c.close()
    with pytest.raises(ValueError, match="manifest changed"):
        controller(path, [entry(source_revisions=("0" * 64,))])
    with pytest.raises(ValueError, match="ambiguous"):
        controller(tmp_path / "ambiguous.db", [entry(), entry("other", method="other")])
    other_family = entry(
        "other", aliases=(), brief=AssignedBrief("other", "forced_flows", "discovery", "reversal")
    )
    with pytest.raises(ValueError, match="change family"):
        controller(tmp_path / "family.db", [entry(), other_family])
    c = controller(path)
    assert not reserve_idea(c, "note-alias")
    assert c.db.execute("SELECT count(*) FROM research_briefs").fetchone()[0] == 1
    c.close()


def test_counted_repair_survives_resume_and_exhaustion_preserves_previous_claims(tmp_path, archive):
    path = tmp_path / "controller.db"
    c = controller(path)
    reserve_idea(c, "idea")
    manifest = {"call_limits": {"liquidity_supply:discovery": 2}, "controller": c.identity}
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        first, _, _, _ = attempt(c, archive, journal)
    failed = feedback(c, first)
    assert (
        request_idea_repair(c, "web-alias", feedback_id=failed, reason="Verifier rejected it.")
        == "idea_repair_requested"
    )
    assert repairs(c) == 1
    before = len(c.events())
    assert (
        request_idea_repair(c, "idea", feedback_id=failed, reason="Same replay.")
        == "idea_repair_requested"
    )
    assert len(c.events()) == before
    c.close()
    c = controller(path)
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        second, _, _, _ = attempt(c, archive, journal, replace(claim(), horizon="Two months."))
    assert second["claim_id"] != first["claim_id"] and repairs(c) == 1
    second_failure = feedback(c, second)
    assert (
        request_idea_repair(c, "idea", feedback_id=second_failure, reason="Still invalid.")
        == "idea_repair_exhausted"
    )
    assert repairs(c) == 1
    with pytest.raises(ValueError, match="counted"):
        assess(c, archive, replace(claim(), horizon="Three months."))
    first_binding = load_artifact(c.db, first["idea_binding_id"], kind="idea_binding_v1")
    assert (
        load_artifact(c.db, first_binding["claim_record_id"], kind="idea_claim_v2")["horizon"]
        == "One month."
    )
    c.close()


def test_unchanged_repaired_claim_finishes_and_identical_failure_exhausts(tmp_path, archive):
    c = controller(tmp_path / "controller.db")
    reserve_idea(c, "idea")
    manifest = {"call_limits": {"liquidity_supply:discovery": 2}, "controller": c.identity}
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        first, response, request_id, request = attempt(c, archive, journal)
        failed = feedback(c, first)
        request_idea_repair(c, "idea", feedback_id=failed, reason="First failure.")
        before = c.events()
        assert assess(c, archive)["claim_id"] == first["claim_id"]
        assert c.state("idea") == "idea_repair_requested" and c.events() == before
        c.close()
        c = controller(tmp_path / "controller.db")
        completion = dict(
            attempt_id=first["attempt_id"],
            frozen_input_id=first["frozen_input_id"],
            journal=journal,
            request_id=request_id,
            response=response,
            parse_response=parse,
            archive=archive,
            reviewed_methods=INVENTORY,
        )
        assert complete_idea_attempt(c, "web-alias", **completion) == first
        assert c.events() == before and repairs(c) == 1
        second, _, _, _ = attempt(c, archive, journal)
        assert second["claim_id"] == first["claim_id"]
        assert second["attempt_id"] != first["attempt_id"]
        assert c.state("idea") == "idea_assessed"
        second_failure = feedback(c, second)
        assert second_failure != failed
        assert (
            request_idea_repair(
                c, "idea", feedback_id=second_failure, reason="Same typed failure again."
            )
            == "idea_repair_exhausted"
        )
        before = c.events()
        assert complete_idea_attempt(c, "idea", **completion) == first
        assert c.events() == before and repairs(c) == 1
        assert journal.count("liquidity_supply:discovery") == 2
        with pytest.raises(CallBudgetExceeded):
            BudgetedClient(
                Client(), 2, journal=journal, bucket="liquidity_supply:discovery", task_id="extra"
            ).create(**request)
    c.close()


def test_initial_malformed_response_retains_schema_failure_and_permits_bounded_repair(
    tmp_path, archive
):
    c = controller(tmp_path / "controller.db")
    reserve_idea(c, "idea")
    context = request_context(c, "idea")
    manifest = {"call_limits": {context["bucket"]: 2}, "controller": c.identity}
    request = {"model": "fixture", "messages": []}
    input_id = digest("frozen-generation-input")
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        attempt_id = begin_idea_attempt(
            c, "idea", frozen_input_id=input_id, request=request, journal=journal
        )
        raw = BudgetedClient(Client([]), 2, journal=journal, **context).create(**request)
        failure = complete_idea_attempt(
            c,
            "idea",
            attempt_id=attempt_id,
            frozen_input_id=input_id,
            journal=journal,
            request_id=journal.report()["requests"][0]["id"],
            response=raw,
            parse_response=parse,
            archive=archive,
            reviewed_methods=INVENTORY,
        )
        assert failure["failure"]["kind"] == "schema_invalid" and failure["claim_id"] is None
        assert c.state("idea") == "idea_schema_failed"
        with pytest.raises(ValueError, match="missing"):
            idea_binding(c, "idea")
        assert (
            c.db.execute(
                "SELECT count(*) FROM research_artifacts WHERE kind='idea_claim_v2'"
            ).fetchone()[0]
            == 0
        )
        fid = feedback(c, failure, reason="schema_invalid")
        assert (
            request_idea_repair(c, "idea", feedback_id=fid, reason="Repair malformed response.")
            == "idea_repair_requested"
        )
        fixed, _, _, _ = attempt(c, archive, journal)
        assert fixed["failure"] is None and c.state("idea") == "idea_assessed"
        assert repairs(c) == 1
    with pytest.raises(ValueError, match="schema6"):
        request_context(c, "idea", phase="program")
    c.close()


@pytest.mark.parametrize(
    "mutation", ["attempt", "input", "request", "request_digest", "response", "receipt", "journal"]
)
def test_attempt_completion_rejects_mismatched_receipts_and_frozen_inputs(
    tmp_path, archive, mutation
):
    c = controller(tmp_path / "controller.db")
    reserve_idea(c, "idea")
    manifest = {"call_limits": {"liquidity_supply:discovery": 2}, "controller": c.identity}
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        first, response, request_id, request = attempt(c, archive, journal)
        args = dict(
            attempt_id=first["attempt_id"],
            frozen_input_id=first["frozen_input_id"],
            journal=journal,
            request_id=request_id,
            response=response,
            parse_response=parse,
            archive=archive,
            reviewed_methods=INVENTORY,
        )
        if mutation == "attempt":
            args["attempt_id"] = "0" * 64
        elif mutation == "input":
            args["frozen_input_id"] = "0" * 64
        elif mutation == "request":
            context = request_context(c, "idea") | {"task_id": "unrelated"}
            BudgetedClient(Client(), 2, journal=journal, **context).create(**request)
            args["request_id"] = journal.report()["requests"][-1]["id"]
        elif mutation == "response":
            args["response"] = SimpleNamespace(content=[], usage=SimpleNamespace())
        elif mutation == "request_digest":
            journal.db.execute("UPDATE model_requests SET request=? WHERE id=?", ("{}", request_id))
        elif mutation == "receipt":
            journal.db.execute(
                "UPDATE response_receipts SET digest=? WHERE request_id=?", ("0" * 64, request_id)
            )
        before = c.events()
        if mutation == "journal":
            with RequestJournal(tmp_path / "other.db", manifest) as other:
                args["journal"] = other
                with pytest.raises(ValueError):
                    complete_idea_attempt(c, "idea", **args)
        else:
            with pytest.raises(ValueError):
                complete_idea_attempt(c, "idea", **args)
        assert c.events() == before and repairs(c) == 0
    c.close()


def test_started_attempt_freezes_request_input_journal_and_parser_errors_are_not_hidden(
    tmp_path, archive
):
    c = controller(tmp_path / "controller.db")
    reserve_idea(c, "idea")
    manifest = {"call_limits": {"liquidity_supply:discovery": 2}, "controller": c.identity}
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        request = {"model": "fixture", "messages": []}
        input_id = digest("frozen-input")
        attempt_id = begin_idea_attempt(
            c, "web-alias", frozen_input_id=input_id, request=request, journal=journal
        )
        before = c.events()
        assert (
            begin_idea_attempt(
                c, "note-alias", frozen_input_id=input_id, request=request, journal=journal
            )
            == attempt_id
        )
        for fields in (
            {"frozen_input_id": "0" * 64},
            {"request": request | {"model": "changed-model"}},
        ):
            args = dict(frozen_input_id=input_id, request=request, journal=journal) | fields
            with pytest.raises(ValueError, match="frozen attempt"):
                begin_idea_attempt(c, "idea", **args)
        response = BudgetedClient(
            Client(), 2, journal=journal, **request_context(c, "idea")
        ).create(**request)

        def programming_error(response):
            raise TypeError("internal parser bug")

        args = dict(
            attempt_id=attempt_id,
            frozen_input_id=input_id,
            journal=journal,
            request_id=journal.report()["requests"][0]["id"],
            response=response,
            parse_response=programming_error,
            archive=archive,
            reviewed_methods=INVENTORY,
        )
        with pytest.raises(TypeError, match="internal parser bug"):
            complete_idea_attempt(c, "idea", **args)
        assert c.events() == before and c.state("idea") == "assigned"
        args["parse_response"] = lambda raw: SimpleNamespace(claim=parse(raw), input_id="0" * 64)
        with pytest.raises(ValueError, match="parsed proposal"):
            complete_idea_attempt(c, "idea", **args)
        assert c.events() == before and repairs(c) == 0
    c.close()


def test_interrupted_completion_rolls_back_assessment_and_replays_saved_response(
    tmp_path, archive, monkeypatch
):
    c = controller(tmp_path / "controller.db")
    reserve_idea(c, "idea")
    manifest = {"call_limits": {"liquidity_supply:discovery": 2}, "controller": c.identity}
    with RequestJournal(tmp_path / "requests.db", manifest) as journal:
        request = {"model": "fixture", "messages": []}
        input_id = digest("frozen-input")
        attempt_id = begin_idea_attempt(
            c, "idea", frozen_input_id=input_id, request=request, journal=journal
        )
        response = BudgetedClient(
            Client(), 2, journal=journal, **request_context(c, "idea")
        ).create(**request)
        args = dict(
            attempt_id=attempt_id,
            frozen_input_id=input_id,
            journal=journal,
            request_id=journal.report()["requests"][0]["id"],
            response=response,
            parse_response=parse,
            archive=archive,
            reviewed_methods=INVENTORY,
        )
        original_event = c._event

        def interrupt_outcome(packet, state, reason, payload):
            if "idea_outcome_id" in payload:
                raise RuntimeError("interrupted completion")
            return original_event(packet, state, reason, payload)

        before = c.events()
        with monkeypatch.context() as patch:
            patch.setattr(c, "_event", interrupt_outcome)
            with pytest.raises(RuntimeError, match="interrupted completion"):
                complete_idea_attempt(c, "idea", **args)
        assert c.events() == before and c.state("idea") == "assigned"
        with pytest.raises(ValueError, match="missing"):
            idea_binding(c, "idea")
        outcome = complete_idea_attempt(c, "idea", **args)
        before = c.events()
        assert complete_idea_attempt(c, "idea", **args) == outcome
        assert c.events() == before and c.state("idea") == "idea_assessed"
        assert journal.count("liquidity_supply:discovery") == 1
    c.close()
