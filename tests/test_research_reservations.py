"""Reservation capacity, restart safety and deterministic scheduling of the controller."""

from dataclasses import asdict

import pytest

from capstone.factors.research_contracts import ResearchPolicy, SourceClaim
from capstone.factors.research_controller import RecordsController, ResearchController


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


def test_reservations_survive_restart_and_aliases_cannot_reset_them(tmp_path):
    path = tmp_path / "research.db"
    catalogue = [
        packet(),
        packet("renamed"),
        packet("control", track="replication_control"),
    ]
    c = controller(path, catalogue)
    assert c.reserve("one")
    assert not c.reserve("one") and not c.reserve("renamed")
    assert c.reserve("control")
    c.close()
    c = controller(path, list(reversed(catalogue)))
    assert not c.reserve("one") and not c.reserve("renamed")
    assert len(c.events()) == 2
    c.close()
    with pytest.raises(ValueError, match="manifest changed"):
        controller(path, catalogue, inputs=("returns",))
    with pytest.raises(ValueError, match="change family"):
        controller(tmp_path / "bad.db", [packet(), packet("alias", family="forced_flows")])


def test_track_capacity_is_a_hard_limit_per_family(tmp_path):
    catalogue = [packet("a", method="m1"), packet("b", method="m2"), packet("c", method="m3")]
    c = controller(tmp_path / "capacity.db", catalogue)
    assert c.reserve("a") and c.reserve("b")
    assert not c.reserve("c")
    assert c.state("c") is None and c.state("a") == "assigned"
    c.close()


def test_schedule_is_row_order_invariant_and_terminates_when_capacity_is_spent(tmp_path):
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
        assert c.next_brief() is None
        sequences.append(order)
        c.close()
    assert sequences[0] == sequences[1] and sorted(sequences[0]) == ["flows", "one", "two"]


def test_events_are_append_only(tmp_path):
    c = controller(tmp_path / "r.db")
    c.reserve("one")
    with pytest.raises(Exception, match="append only"):
        c.db.execute("DELETE FROM research_events")
    with pytest.raises(Exception, match="append only"):
        c.db.execute("UPDATE research_events SET state='x'")
    c.close()


def test_records_controller_reads_an_existing_study_without_rebinding_its_manifest(tmp_path):
    path = tmp_path / "r.db"
    c = controller(path)
    c.reserve("one")
    c.close()
    records = RecordsController(path)
    with records.transaction():
        count = records.db.execute("SELECT count(*) FROM research_events").fetchone()[0]
    assert count == 1
    records.db.close()
