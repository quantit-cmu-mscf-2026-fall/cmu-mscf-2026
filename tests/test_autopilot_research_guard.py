"""The research loop-guard file and the request journal's open pauses.

Synthetic request journals and controller records only; no model call is made (the fake
client fails every send with a typed status).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from capstone.factors.research_contracts import digest
from capstone.factors.research_records import put_artifact
from capstone.factors.research_requests import RequestJournal, ResearchPaused
from capstone.pipeline.autopilot import ap3_guard

BUCKET = "generation"
MANIFEST = {"call_limits": {BUCKET: 4}, "model": "fixture", "packets": ["frozen"]}


class Failing:
    """A client whose every send is refused with an HTTP status (never a real call)."""

    max_retries = 0

    def __init__(self, status):
        self.status = status
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        error = RuntimeError("synthetic failure")
        error.status_code = self.status
        raise error


def pause(root, task="idea-0:attempt", ordinal=0, status=402):
    """Leave one request paused in `<root>/requests.db`; returns its logical id."""
    root.mkdir(parents=True, exist_ok=True)
    with RequestJournal(root / "requests.db", MANIFEST) as journal:
        with pytest.raises(ResearchPaused):
            journal.call(Failing(status), BUCKET, task, ordinal, {"model": "fixture"})
    return digest({"bucket": BUCKET, "task": task, "ordinal": ordinal})


def attempt(root, attempt_id, packet_id):
    with closing(sqlite3.connect(root / "controller.db")) as db, db:
        put_artifact(db, "idea_attempt_v1", {"attempt_id": attempt_id, "packet_id": packet_id})


def test_a_new_guard_is_blank_and_a_saved_one_round_trips_atomically(tmp_path):
    assert ap3_guard.load(tmp_path) == ap3_guard.blank()
    guard = {**ap3_guard.blank(), "consecutive_pauses": 2, "last_pause": ["r1"]}
    ap3_guard.save(tmp_path, guard)
    assert ap3_guard.load(tmp_path) == guard
    folder = tmp_path.joinpath(*ap3_guard.GUARD).parent
    assert [p.name for p in folder.iterdir()] == ["research-guard.json"]  # no temporary left


def test_an_unknown_schema_version_is_refused(tmp_path):
    ap3_guard.save(tmp_path, {**ap3_guard.blank(), "schema_version": 2})
    with pytest.raises(ValueError, match="unsupported schema_version"):
        ap3_guard.load(tmp_path)


def test_an_older_file_without_the_newer_keys_reads_with_their_defaults(tmp_path):
    path = tmp_path.joinpath(*ap3_guard.GUARD)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "consecutive_pauses": 1}), encoding="utf-8")
    guard = ap3_guard.load(tmp_path)
    assert guard["consecutive_pauses"] == 1 and guard["restart_count"] is False
    assert guard["decisions"] == [] and guard["cleared_pause"] is None


def test_retries_count_only_explicit_retries_of_one_packet():
    guard = {
        "decisions": [
            {"packet": "a", "decision": "retry"},
            {"packet": "a", "decision": "abandon"},
            {"packet": "a", "decision": "retry"},
            {"packet": "b", "decision": "retry"},
        ]
    }
    assert ap3_guard.retries(guard, "a") == 2 and ap3_guard.retries(guard, "b") == 1
    assert ap3_guard.retries(guard, "c") == 0


def test_without_a_journal_nothing_is_awaiting(tmp_path):
    assert ap3_guard.awaiting(tmp_path) == []
    assert ap3_guard.resolution(tmp_path, "any") is None


def test_a_paused_request_awaits_until_a_person_resolves_it(tmp_path):
    logical = pause(tmp_path)
    [row] = ap3_guard.awaiting(tmp_path)
    assert (row["logical"], row["bucket"], row["state"]) == (logical, BUCKET, "paused")
    assert "http=402" in row["error"]
    assert ap3_guard.resolution(tmp_path, row["id"]) is None
    with RequestJournal(tmp_path / "requests.db", MANIFEST) as journal:
        journal.resolve(row["id"], "retry", "synthetic operator decision")
    assert ap3_guard.awaiting(tmp_path) == []
    assert ap3_guard.resolution(tmp_path, row["id"]) == "retry"


def test_the_guard_reads_the_journal_read_only(tmp_path):
    pause(tmp_path)
    before = (tmp_path / "requests.db").read_bytes()
    ap3_guard.awaiting(tmp_path)
    assert (tmp_path / "requests.db").read_bytes() == before


def test_packets_map_requests_to_the_attempt_that_made_them(tmp_path):
    generation = pause(tmp_path, task="attempt-1", ordinal=0)
    advisory = pause(tmp_path, task="attempt-2:source-rule-alignment", ordinal=1)
    attempt(tmp_path, "attempt-1", "idea-0")
    attempt(tmp_path, "attempt-2", "idea-1")
    requests = [{"logical": generation}, {"logical": advisory}]
    owners = ap3_guard.packets(tmp_path, requests, [BUCKET])
    assert owners == {generation: "idea-0", advisory: "idea-1"}


def test_a_request_no_attempt_explains_is_its_own_packet(tmp_path):
    stray = {"logical": "0" * 64}
    assert ap3_guard.packets(tmp_path, [stray], [BUCKET]) == {"0" * 64: "request:" + "0" * 64}
    assert ap3_guard.packets(tmp_path, [], [BUCKET]) == {}


def test_the_logical_id_in_the_guard_is_the_one_the_journal_computes(tmp_path):
    """If the two digests drifted apart, every pause would be keyed as an unknown request."""
    logical = pause(tmp_path, task="attempt-9", ordinal=2)
    attempt(tmp_path, "attempt-9", "idea-9")
    assert ap3_guard.packets(tmp_path, [{"logical": logical}], [BUCKET]) == {logical: "idea-9"}
