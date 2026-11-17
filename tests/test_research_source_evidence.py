from copy import deepcopy
from hashlib import sha256

import pytest
from test_research_controller import controller as make_controller
from test_research_controller import packet as catalogue_packet

from capstone.factors.research_contracts import SourceClaim, canonical


def test_absent_excerpt_keeps_legacy_identity_claim_and_event_bytes(tmp_path):
    row = catalogue_packet()
    c = make_controller(tmp_path / "legacy.db", [row])
    try:
        c.reserve("one")
        c.ground("one", SourceClaim(**row["claim"]))
        assert c.identity == "aa85edbc319ff8491df8150caf340f8633ca35eb2ba5af30950f7d82fe09af00"
        assert (
            c.claim_id("one") == "61a4603d51ff80b6eb882e5df3fabbe0ce34ce16dfd50fcc7cecb7a9797dcc16"
        )
        payload = c.events()[-1]["payload"]
        assert sha256(canonical(payload).encode()).hexdigest() == (
            "10c23de61f35b995c17c16ee10ba15b1da00ac471443d29dfa14c57518e9d40f"
        )
        assert c.source_evidence("one") == {
            "catalogue": row["claim"],
            "grounded": payload["effective"],
        }
    finally:
        c.close()


def test_curator_excerpt_is_frozen_at_grounding_and_survives_reopening(tmp_path):
    row = catalogue_packet()
    row["method_excerpt"] = "Retained source excerpt supplies separately curated method evidence."
    original = deepcopy(row)
    path = tmp_path / "excerpt.db"
    c = make_controller(path, [row])
    try:
        c.reserve("one")
        row["method_excerpt"] = "Caller mutation cannot rewrite frozen source evidence."
        c.ground("one", SourceClaim(**row["claim"]))
        evidence = c.source_evidence("one")
        assert evidence["method_excerpt"] == original["method_excerpt"]
        assert c.events()[-1]["payload"]["method_excerpt"] == original["method_excerpt"]
        assert c.identity != "aa85edbc319ff8491df8150caf340f8633ca35eb2ba5af30950f7d82fe09af00"
        claim_id = c.claim_id("one")
    finally:
        c.close()
    reopened = make_controller(path, [original])
    try:
        assert reopened.source_evidence("one") == evidence
        assert reopened.claim_id("one") == claim_id
    finally:
        reopened.close()
    with pytest.raises(ValueError, match="manifest changed"):
        make_controller(path, [row])


@pytest.mark.parametrize("value", [None, "", " ", False, []])
def test_explicit_excerpt_requires_nonempty_curator_text(tmp_path, value):
    row = catalogue_packet()
    row["method_excerpt"] = value
    with pytest.raises(ValueError, match="excerpt"):
        make_controller(tmp_path / "invalid.db", [row])


def test_source_evidence_requires_a_retained_grounding(tmp_path):
    c = make_controller(tmp_path / "ungrounded.db")
    try:
        c.reserve("one")
        with pytest.raises(ValueError, match="grounded source"):
            c.source_evidence("one")
    finally:
        c.close()
