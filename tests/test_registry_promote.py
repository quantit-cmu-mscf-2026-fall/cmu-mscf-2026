"""Promotion: an incubating entry becomes a deploy entry only on a confirming record.

The record is a synthetic `ConfirmState` (a constant positive return series); the
registry is a temp folder written through the same `_commit` registration uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_registry_lifecycle import AT, lines, registered

from capstone.artifacts import canonical, digest
from capstone.incubation.confirm import ConfirmConfig, ConfirmState, confirm
from capstone.incubation.monitor import MonitorConfig
from capstone.strategy.registry import (
    INCUBATION_ONLY,
    LIFECYCLE_FILE,
    REGISTRY_FILE,
    LifecycleRewritePending,
    check_confirm_record,
    promote,
    read_lifecycle,
    read_registry,
    set_state,
    verify_lifecycle,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "deploy").mkdir()
    (tmp_path / REGISTRY_FILE).write_bytes(b"# registry\nstrategies: []\n")
    return tmp_path


def promoting_record(strategy_id: str, per_period: float = 0.5) -> dict:
    state = confirm(
        np.full(2_000, per_period),
        ConfirmState.start(
            ConfirmConfig(scale=1.0, horizon=1260),
            MonitorConfig(expected_mean=0.0, scale=1.0, horizon=1260),
        ),
    )
    return {"strategy_id": strategy_id, "confirm_state": state.to_dict()}


def test_the_promoting_record_really_promotes():
    state = check_confirm_record(promoting_record("momo"), "momo")
    assert state.decision == "promote" and state.e_value >= 1 / state.config.alpha


def test_promotion_requires_a_confirm_record_that_promotes(root):
    registered(root, role=INCUBATION_ONLY)
    path = root / REGISTRY_FILE
    before = path.read_bytes()
    record = promoting_record("momo")
    continuing = {"strategy_id": "momo", "confirm_state": dict(record["confirm_state"])}
    continuing["confirm_state"]["decision"] = "continue"
    continuing["confirm_state"]["reason"] = ""
    forged = {"strategy_id": "momo", "confirm_state": dict(record["confirm_state"])}
    forged["confirm_state"]["log_wealth"] = [0.0] * len(record["confirm_state"]["log_wealth"])
    garbage = {"strategy_id": "momo", "confirm_state": {"nope": 1}}
    for bad in (None, {}, continuing, forged, garbage, promoting_record("someone-else")):
        with pytest.raises(ValueError, match="confirm"):
            promote(root, "momo", confirm_record=bad, reviewer="reviewer-a", at=AT)
    assert path.read_bytes() == before and len(read_lifecycle(root)) == 1
    entry = promote(root, "momo", confirm_record=record, reviewer="reviewer-a", at=AT)
    assert entry.role == "deploy" and entry.state == "active" and entry.enabled is False
    assert read_registry(root) == (entry,)
    event = read_lifecycle(root)[-1]
    assert event["field"] == "role" and (event["from"], event["to"]) == (INCUBATION_ONLY, "deploy")
    assert event["confirm_record"] == json.loads(canonical(record))  # the evidence is kept
    with pytest.raises(ValueError, match="incubation-only"):
        promote(root, "momo", confirm_record=record, reviewer="reviewer-a", at=AT)
    # a hand-restored role contradicts the promotion event
    path.write_bytes(path.read_bytes() + b'    role: "incubation-only"\n')
    with pytest.raises(ValueError, match="lifecycle"):
        read_registry(root)


def test_promotion_leaves_enabled_alone_and_needs_an_active_entry(root):
    registered(root, role=INCUBATION_ONLY)
    path = root / REGISTRY_FILE
    path.write_bytes(path.read_bytes().replace(b"enabled: false", b"enabled: true"))
    set_state(root, "momo", "suspended", reason="r", reviewer="reviewer-a", at=AT)
    with pytest.raises(ValueError, match="suspended"):
        promote(root, "momo", confirm_record=promoting_record("momo"), reviewer="reviewer-a", at=AT)
    set_state(root, "momo", "active", reason="r", reviewer="reviewer-a", at=AT)
    entry = promote(
        root, "momo", confirm_record=promoting_record("momo"), reviewer="reviewer-a", at=AT
    )
    assert entry.enabled is True and entry.role == "deploy"
    with pytest.raises(ValueError, match="reviewer and a timestamp"):
        promote(root, "momo", confirm_record=promoting_record("momo"), reviewer="", at=AT)
    registered(root, "plain")
    with pytest.raises(ValueError, match="only an incubation-only entry"):
        promote(
            root, "plain", confirm_record=promoting_record("plain"), reviewer="reviewer-a", at=AT
        )


def test_a_role_event_is_checked_when_the_log_is_read(root):
    registered(root, role=INCUBATION_ONLY)
    promote(root, "momo", confirm_record=promoting_record("momo"), reviewer="reviewer-a", at=AT)
    entries = read_registry(root)
    genesis, role_line = lines(root)
    again = json.loads(role_line)
    again["prev_hash"] = digest(json.loads(role_line))
    with pytest.raises(ValueError, match="one incubation-only -> deploy"):
        verify_lifecycle(genesis + role_line + canonical(again) + chr(10), entries)
    weak = json.loads(role_line)
    weak["confirm_record"] = promoting_record("momo", per_period=0.0)
    weak["prev_hash"] = digest(json.loads(genesis))
    with pytest.raises(ValueError, match="confirm record does not promote"):
        verify_lifecycle(genesis + canonical(weak) + chr(10), entries)
    edited = role_line.replace('"reason":"confirmed"', '"reason":"other"')
    with pytest.raises(ValueError, match="lifecycle") as refused:
        verify_lifecycle(genesis + edited, entries)
    assert not isinstance(refused.value, LifecycleRewritePending)


def test_a_deploy_entry_cannot_have_a_role_event(root):
    registered(root)  # registered as deploy
    entries = read_registry(root)
    genesis = lines(root)[0]
    event = {
        "field": "role",
        "id": "momo",
        "from": INCUBATION_ONLY,
        "to": "deploy",
        "reason": "confirmed",
        "reviewer": "reviewer-a",
        "at": AT,
        "confirm_record": promoting_record("momo"),
        "prev_hash": digest(json.loads(genesis)),
    }
    with pytest.raises(ValueError, match="one incubation-only -> deploy"):
        verify_lifecycle(genesis + canonical(event) + chr(10), entries)
    assert (root / LIFECYCLE_FILE).read_text(encoding="utf-8") == genesis


def test_a_confirm_record_must_meet_the_policy_alpha_and_have_observations():
    loose = promoting_record("momo")
    loose["confirm_state"]["config"]["alpha"] = 0.5  # e-value is far above 1/0.5
    with pytest.raises(ValueError, match="promotion needs alpha"):
        check_confirm_record(loose, "momo")
    empty = ConfirmState.start(
        ConfirmConfig(scale=1.0, horizon=1260),
        MonitorConfig(expected_mean=0.0, scale=1.0, horizon=1260),
    )
    with pytest.raises(ValueError, match="confirm"):
        check_confirm_record({"strategy_id": "momo", "confirm_state": empty.to_dict()}, "momo")


def test_replay_refuses_a_role_event_on_a_suspended_strategy(root):
    registered(root, role=INCUBATION_ONLY)
    promote(root, "momo", confirm_record=promoting_record("momo"), reviewer="reviewer-a", at=AT)
    entries = read_registry(root)
    genesis, role_line = lines(root)
    state = {
        "field": "state",
        "id": "momo",
        "from": "active",
        "to": "suspended",
        "reason": "r",
        "reviewer": "reviewer-a",
        "at": AT,
        "prev_hash": digest(json.loads(genesis)),
    }
    role = json.loads(role_line)
    role["prev_hash"] = digest(state)
    with pytest.raises(ValueError, match="only an active one is promoted"):
        verify_lifecycle(genesis + canonical(state) + chr(10) + canonical(role) + chr(10), entries)
