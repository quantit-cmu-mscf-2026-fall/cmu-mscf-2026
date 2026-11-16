from types import SimpleNamespace as NS

import pytest

from capstone.factors.research_requests import RequestJournal, ResearchPaused
from capstone.pipeline.budget import BudgetedClient, CallBudgetExceeded


class Client:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        result = next(self.outcomes)
        if isinstance(result, BaseException):
            raise result
        return result


def answer():
    return NS(
        content=[NS(type="tool_use", name="record", input={"value": 3})],
        usage=NS(input_tokens=8, output_tokens=9),
    )


def manifest():
    return {"call_limits": {"A": 2, "B": 1}, "model": "fixture", "packets": ["frozen"]}


def call(journal, client, task="one", bucket="A"):
    return BudgetedClient(
        client, journal.limits[bucket], journal=journal, bucket=bucket, task_id=task
    ).messages.create(model="fixture", messages=[])


def test_saved_response_replays_after_restart_without_another_call(tmp_path):
    path = tmp_path / "requests.db"
    client = Client([answer()])
    with RequestJournal(path, manifest()) as j:
        assert call(j, client).content[0].input == {"value": 3}
    with RequestJournal(path, manifest()) as j:
        assert call(j, client).usage.output_tokens == 9
        assert client.calls == j.count("A") == 1
        with pytest.raises(ValueError, match="inputs changed"):
            BudgetedClient(client, 2, journal=j, bucket="A", task_id="one").create(model="changed")


def test_the_send_is_counted_before_it_goes_out(tmp_path):
    seen = []

    class Watching(Client):
        def create(self, **kwargs):
            seen.append((journal.count("A"), journal.report()["requests"][0]["state"]))
            return super().create(**kwargs)

    with RequestJournal(tmp_path / "r.db", manifest()) as journal:
        call(journal, Watching([answer()]))
        assert seen == [(1, "inflight")]
        assert [e["state"] for e in journal.report()["events"]] == ["reserved", "completed"]


def test_ceiling_refuses_the_call_past_it_without_sending(tmp_path):
    client = Client([answer(), answer(), answer()])
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        call(j, client, task="one")
        call(j, client, task="two")
        with pytest.raises(CallBudgetExceeded):
            call(j, client, task="three")
        assert client.calls == j.count("A") == 2


def test_failed_send_is_saved_uncertain_and_never_silently_resent(tmp_path):
    path = tmp_path / "r.db"
    client = Client([TimeoutError(), answer()])
    with RequestJournal(path, manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, client)
        assert client.calls == 1
        assert [r["state"] for r in j.report()["requests"]] == ["uncertain"]
    with RequestJournal(path, manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, client)
        assert client.calls == j.count("A") == 1
        assert "TimeoutError" in j.report()["requests"][0]["error"]


def test_manifest_lock_and_change_are_refused(tmp_path):
    path = tmp_path / "r.db"
    with RequestJournal(path, manifest()):
        with pytest.raises(RuntimeError, match="another dispatcher"):
            RequestJournal(path, manifest())
    with pytest.raises(ValueError, match="manifest changed"):
        RequestJournal(path, manifest() | {"model": "different"})
    with pytest.raises(ValueError, match="nonnegative frozen"):
        RequestJournal(tmp_path / "bad.db", {"call_limits": {"A": -1}})


def test_event_log_is_append_only(tmp_path):
    import sqlite3

    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        call(j, Client([answer()]))
        with pytest.raises(sqlite3.DatabaseError, match="append only"):
            j.db.execute("DELETE FROM request_events")
        with pytest.raises(sqlite3.DatabaseError, match="append only"):
            j.db.execute("UPDATE request_events SET detail='x'")


def test_sdk_retries_are_forbidden_and_completed_malformed_tools_are_cached(tmp_path):
    client = Client([NS(content=[], usage=NS(input_tokens=1, output_tokens=0))])
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        client.max_retries = 2
        with pytest.raises(ValueError, match="hidden SDK"):
            call(j, client)
        assert client.calls == 0
        client.max_retries = 0
        assert call(j, client).content == []
        assert call(j, client).content == []
        assert client.calls == 1


def test_full_bucket_still_replays_a_completed_response(tmp_path):
    client = Client([answer()])
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        call(j, client, bucket="B")
        assert j.count("B") == 1
        assert call(j, client, bucket="B").content[0].name == "record"
        assert client.calls == 1
        with pytest.raises(CallBudgetExceeded):
            call(j, client, bucket="B", task="different")


def test_altered_response_cannot_be_replayed_as_original_evidence(tmp_path):
    client = Client([answer()])
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        call(j, client)
        with j.db:
            j.db.execute("UPDATE model_requests SET response='{}' WHERE state='completed'")
        with pytest.raises(ValueError, match="contents changed"):
            call(j, client)
        assert client.calls == 1


def test_bucket_limits_cannot_be_mutated_or_transferred_after_registration(tmp_path):
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        with pytest.raises(TypeError):
            j.limits["A"] = 99
        j.manifest["call_limits"]["A"] = 99
        assert j.limits["A"] == 2
        with pytest.raises(ValueError, match="frozen bucket"):
            BudgetedClient(Client([]), 99, journal=j, bucket="A", task_id="one")
