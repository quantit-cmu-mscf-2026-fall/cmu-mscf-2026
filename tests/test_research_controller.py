from dataclasses import asdict, replace

import pytest

from capstone.factors.research_contracts import (
    ConstructionEvidence,
    MethodSpec,
    ResearchPolicy,
    SourceClaim,
)
from capstone.factors.research_controller import ResearchController


def packet(name="one", family="liquidity_supply", track="discovery", method="reversal", **claim):
    source = SourceClaim(
        "https://example.org/paper",
        "section 2",
        "reversal",
        "no reversal",
        ("returns", "dated_flows"),
        True,
        **claim,
    )
    return {
        "brief": {
            "packet_id": name,
            "family": family,
            "track": track,
            "method_key": method,
        },
        "claim": asdict(source),
        "constructor": "reference",
    }


def controller(path, catalogue=None, inputs=("returns", "dated_flows")):
    p = ResearchPolicy(
        (
            ("liquidity_supply", "discovery", 2),
            ("liquidity_supply", "replication_control", 1),
            ("forced_flows", "discovery", 1),
        )
    )
    return ResearchController(path, p, catalogue or [packet()], inputs, generator_version="test-v1")


def specification(claim_id):
    return MethodSpec(
        claim_id,
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


def test_missing_dependencies_and_scope_cannot_be_overridden_by_agent(tmp_path):
    c = controller(tmp_path / "r.db", inputs=("returns",))
    c.reserve("one")
    proposal = replace(SourceClaim(**packet()["claim"]), required_inputs=("returns",))
    assert c.ground("one", proposal) == "needs_data"
    assert c.events()[-1]["payload"]["missing"] == ["dated_flows"]
    with pytest.raises(ValueError, match="unsupported transition"):
        c.freeze_spec("one", specification(c._latest("one", "claim_id")))
    c.close()
    source = packet(universe="futures")
    c = controller(tmp_path / "scope.db", [source])
    c.reserve("one")
    assert (
        c.ground("one", replace(SourceClaim(**source["claim"]), universe="sp500"))
        == "outside_scope"
    )
    c.close()


def test_bounded_repairs_and_only_verified_handoff(tmp_path):
    c = controller(tmp_path / "r.db")
    c.reserve("one")
    assert c.ground("one", SourceClaim(**packet()["claim"])) == "specification_ready"
    spec = specification(c._latest("one", "claim_id"))
    assert c.freeze_spec("one", spec) == spec.identity

    def fail(s):
        return ConstructionEvidence(s.identity, "independent-v1", True, True, False)

    assert c.construct("one", fail) == "repair_requested"
    c.freeze_spec("one", spec)
    assert c.construct("one", fail) == "construction_failed"
    with pytest.raises(ValueError):
        c.freeze_spec("one", spec)
    with pytest.raises(ValueError):
        c.replay_handoff("one")
    c.close()


def test_valid_reference_can_proceed_but_never_self_admit(tmp_path):
    c = controller(tmp_path / "r.db", [packet(track="replication_control")])
    c.reserve("one")
    c.ground("one", SourceClaim(**packet()["claim"]))
    spec = specification(c._latest("one", "claim_id"))
    c.freeze_spec("one", spec)
    assert (
        c.construct(
            "one",
            lambda s: ConstructionEvidence(s.identity, "example-v1", True, True, True),
        )
        == "construction_verified"
    )
    assert c.replay_handoff("one")["formal_admission"] is False
    with pytest.raises(Exception, match="append only"):
        c.db.execute("DELETE FROM research_events")
    c.close()


def test_schedule_is_row_order_invariant_and_unusable_batch_terminates(tmp_path):
    catalogue = [
        packet(),
        packet("flows", family="forced_flows", method="flows"),
        packet("two", method="weekly"),
    ]
    sequences = []
    for i, rows in enumerate((catalogue, list(reversed(catalogue)))):
        c = controller(tmp_path / f"{i}.db", rows, inputs=())
        order = []
        while (brief := c.next_brief()) is not None:
            order.append(brief.packet_id)
            claim = SourceClaim(
                **next(p["claim"] for p in rows if p["brief"]["packet_id"] == brief.packet_id)
            )
            assert c.ground(brief.packet_id, claim) == "needs_data"
        assert not any(e["state"] == "replay_ready" for e in c.events())
        sequences.append(order)
        c.close()
    assert sequences[0] == sequences[1]


def test_existing_source_bridge_does_not_mark_blocked_leads_globally_ingested(tmp_path):
    class Source:
        def papers(self):
            return [{"key": "two"}, {"key": "one"}]

        def done(self, keys):
            raise AssertionError("research consumer must not overwrite legacy ingestion")

    c = controller(tmp_path / "source.db", [packet(), packet("two", method="other")])
    assert [p["key"] for p in c.pending_papers(Source())] == ["one", "two"]
    c.reserve("one")
    c.ground("one", SourceClaim(**packet()["claim"]))
    assert c.pending_papers(Source()) == [{"key": "two"}]
    c.close()


def test_failure_memory_cannot_turn_missing_data_into_an_economic_rejection(tmp_path):
    c = controller(tmp_path / "failure.db", inputs=())
    c.reserve("one")
    c.ground("one", SourceClaim(**packet()["claim"]))
    with pytest.raises(ValueError):
        c.record_failure("one", "validation_failed", {})
    with pytest.raises(ValueError, match="unsupported transition"):
        c.record_failure("one", "interpretation_failed", {})
    assert c.state("one") == "needs_data"
    c.close()


def test_proposal_cannot_expand_an_approved_stock_scope(tmp_path):
    c = controller(tmp_path / "scope-expansion.db")
    c.reserve("one")
    proposal = replace(SourceClaim(**packet()["claim"]), universe="futures")
    assert c.ground("one", proposal) == "outside_scope"
    c.close()
