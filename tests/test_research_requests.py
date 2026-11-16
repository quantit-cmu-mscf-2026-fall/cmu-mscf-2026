import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from capstone import faults
from capstone.factors.research_requests import RequestAbandoned, RequestJournal, ResearchPaused
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


def test_uncertain_send_pauses_all_new_work_until_explicit_resolution(tmp_path):
    path = tmp_path / "r.db"
    client = Client([TimeoutError(), answer()])
    with RequestJournal(path, manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, client)
        with pytest.raises(ResearchPaused, match="study paused"):
            call(j, client, bucket="B", task="other")
        rid = j.report()["requests"][0]["id"]
        assert client.calls == 1
    with RequestJournal(path, manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, client)
        j.resolve(
            rid, "retry", "operator resolved uncertain request; accepts possible repeat billing"
        )
        assert call(j, client).content[0].name == "record"
        assert client.calls == j.count("A") == 2
        assert [r["state"] for r in j.report()["requests"]] == ["uncertain", "completed"]
        with pytest.raises(CallBudgetExceeded):
            call(j, client, task="two")


def test_resolution_needs_a_decision_a_reason_and_a_waiting_request(tmp_path):
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, Client([TimeoutError()]))
        rid = j.report()["requests"][0]["id"]
        with pytest.raises(ValueError, match="decision and reason"):
            j.resolve(rid, "maybe", "unclear")
        with pytest.raises(ValueError, match="decision and reason"):
            j.resolve(rid, "retry", "  ")
        with pytest.raises(ValueError, match="not awaiting"):
            j.resolve("missing", "retry", "no such request")
        j.resolve(rid, "abandon", "operator keeps the incomplete case")
        assert j.report()["events"][-1]["detail"].startswith("abandon: ")
        with pytest.raises(Exception, match="UNIQUE"):
            j.resolve(rid, "retry", "second decision")


def test_inflight_reservations_and_abandoned_calls_cannot_disappear(tmp_path):
    path = tmp_path / "r.db"
    with RequestJournal(path, manifest()) as j:
        with j.db:
            j.db.execute(
                "INSERT INTO model_requests VALUES ('id','logical','A',0,'{}',NULL,'inflight',NULL)"
            )
    with RequestJournal(path, manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, Client([answer()]))
        j.resolve("id", "abandon", "response unavailable; retain reservation")
        assert j.count("A") == 1
        assert call(j, Client([answer()])).content[0].input["value"] == 3


def test_abandoned_request_does_not_resend_or_block_other_cases(tmp_path):
    client = Client([TimeoutError(), answer()])
    with RequestJournal(tmp_path / "r.db", manifest()) as j:
        with pytest.raises(ResearchPaused):
            call(j, client)
        rid = j.report()["requests"][0]["id"]
        j.resolve(rid, "abandon", "operator retains an incomplete case")
        with pytest.raises(RequestAbandoned):
            call(j, client)
        assert client.calls == 1
        assert call(j, client, task="other").content[0].input["value"] == 3
        assert j.count("A") == 2


def test_interrupted_inflight_is_marked_uncertain_when_owner_restarts(tmp_path):
    path = tmp_path / "r.db"
    with RequestJournal(path, manifest()) as j, j.db:
        j.db.execute(
            "INSERT INTO model_requests VALUES ('id','logical','A',0,'{}',NULL,'inflight',NULL)"
        )
    with RequestJournal(path, manifest()) as j:
        assert j.report()["requests"][0]["state"] == "uncertain"
        assert j.report()["events"][-1]["state"] == "uncertain"
        assert j.report()["blocked"]


@pytest.fixture
def fault(monkeypatch):
    def arm(at):
        monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
        monkeypatch.setenv("CAPSTONE_FAULT_AT", at)
        faults._fault_plan.cache_clear()
        faults._fault_hits.clear()

    yield arm
    faults._fault_plan.cache_clear()
    faults._fault_hits.clear()


def test_kill_after_reserve_never_resends_and_waits_for_resolution(tmp_path, fault):
    path = tmp_path / "r.db"
    client = Client([answer(), answer()])
    fault("runner:after_reserve")
    with RequestJournal(path, manifest()) as j:
        with pytest.raises(faults.InjectedFault):
            call(j, client)
        assert [r["state"] for r in j.report()["requests"]] == ["inflight"]
    assert client.calls == 0
    with RequestJournal(path, manifest()) as j:
        report = j.report()
        assert report["requests"][0]["state"] == "uncertain" and report["blocked"]
        assert (
            report["events"][-1]["detail"] == "previous owner exited before a response checkpoint"
        )
        with pytest.raises(ResearchPaused):
            call(j, client)
        assert client.calls == 0 and j.count("A") == 1
        j.resolve(report["requests"][0]["id"], "retry", "fixture: nothing was received")
        assert call(j, client).content[0].name == "record"
        assert client.calls == 1 and j.count("A") == 2


def test_a_killed_process_leaves_a_journal_a_fresh_process_resumes_stably(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    path = tmp_path / "r.db"
    script = textwrap.dedent(
        """
        import json, sys
        from capstone.factors.research_requests import RequestJournal
        path, mode = sys.argv[1:]
        with RequestJournal(path, {"call_limits": {"A": 2}}) as j:
            if mode == "send":
                class C:
                    messages = None
                    def create(self, **kw):
                        raise AssertionError("must not be sent")
                c = C(); c.messages = c
                j.call(c, "A", "t", 0, {"model": "fixture"})
            else:
                print(json.dumps(j.report(), sort_keys=True))
        """
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE_FAULT")}
    base |= {"PYTHONPATH": str(repo), "PYTHONUTF8": "1"}

    def run(mode, **extra):
        return subprocess.run(
            [sys.executable, "-c", script, str(path), mode],
            env=base | extra,
            capture_output=True,
            text=True,
            check=False,
        )

    killed = run("send", CAPSTONE_FAULT_INJECTION="1", CAPSTONE_FAULT_AT="runner:after_reserve")
    assert killed.returncode != 0 and "injected fault" in killed.stderr
    first, second = run("report"), run("report")
    assert first.returncode == 0 == second.returncode, first.stderr
    assert first.stdout == second.stdout
    assert '"state": "uncertain"' in first.stdout and '"blocked": true' in first.stdout
