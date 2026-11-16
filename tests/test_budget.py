"""A budgeted client counts every sent call and refuses the one past the ceiling."""

import threading
from types import SimpleNamespace

import pytest

from capstone.factors.deepen import BudgetExhausted
from capstone.pipeline.budget import BudgetedClient, CallBudgetExceeded


class Recorder:
    def __init__(self):
        self.sent = []
        self.messages = self

    def create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(n=len(self.sent))


def test_calls_pass_through_until_the_ceiling_then_are_refused_unsent():
    recorder = Recorder()
    budget = BudgetedClient(recorder, 2)
    assert budget.messages.create(prompt="a").n == 1
    assert budget.create(prompt="b").n == 2
    with pytest.raises(CallBudgetExceeded, match="2 model calls"):
        budget.create(prompt="c")
    assert [m["prompt"] for m in recorder.sent] == ["a", "b"]
    assert (budget.calls, budget.refused) == (2, 1)


def test_the_exception_is_a_budget_exhausted_so_deepen_loops_stop():
    assert issubclass(CallBudgetExceeded, BudgetExhausted)


def test_check_reports_a_refusal_a_caller_swallowed():
    budget = BudgetedClient(Recorder(), 1)
    budget.check()
    budget.create(x=1)
    budget.check()
    with pytest.raises(CallBudgetExceeded):
        budget.create(x=2)
    with pytest.raises(CallBudgetExceeded, match=r"1 call\(s\) refused"):
        budget.check()


@pytest.mark.parametrize("bad", [0, -3])
def test_ceiling_must_allow_a_call(bad):
    with pytest.raises(ValueError, match="at least 1"):
        BudgetedClient(Recorder(), bad)


def test_concurrent_callers_never_send_past_the_ceiling():
    recorder = Recorder()
    budget = BudgetedClient(recorder, 5)
    outcomes = []

    def worker():
        try:
            budget.create(x=1)
            outcomes.append("sent")
        except CallBudgetExceeded:
            outcomes.append("refused")

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("sent") == 5 and outcomes.count("refused") == 15
    assert len(recorder.sent) == 5


class StubJournal:
    limits = {"search": 3}

    def __init__(self):
        self.replayed = 0

    def count(self, bucket):
        return self.replayed

    def call(self, client, bucket, task_id, ordinal, kwargs):
        self.replayed += 1
        return (bucket, task_id, ordinal, kwargs)


def test_a_journal_needs_the_frozen_bucket_allowance_and_a_task_id():
    for kwargs in (
        {"bucket": "other", "task_id": "t"},
        {"bucket": "search", "task_id": ""},
    ):
        with pytest.raises(ValueError, match="frozen bucket allowance"):
            BudgetedClient(Recorder(), 3, journal=StubJournal(), **kwargs)
    with pytest.raises(ValueError, match="frozen bucket allowance"):
        BudgetedClient(Recorder(), 2, journal=StubJournal(), bucket="search", task_id="t")


def test_with_a_journal_each_call_goes_through_it_with_a_running_ordinal():
    journal = StubJournal()
    budget = BudgetedClient(Recorder(), 3, journal=journal, bucket="search", task_id="t")
    assert budget.create(a=1) == ("search", "t", 0, {"a": 1})
    assert budget.create(a=2)[2] == 1
    assert budget.calls == 2


class YieldingBudget(BudgetedClient):
    """Yield inside the read-modify-write window, not in the outgoing call."""

    @property
    def calls(self):
        import time

        value = self._calls
        time.sleep(0.01)
        return value

    @calls.setter
    def calls(self, value):
        self._calls = value


def ceiling_outcomes(budget, n=20):
    from concurrent.futures import ThreadPoolExecutor

    start = threading.Barrier(n)

    def call(i):
        start.wait()
        try:
            budget.create(x=i)
            return "sent"
        except CallBudgetExceeded:
            return "refused"

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(call, range(n)))


def test_concurrency_regression_detects_a_removed_budget_lock():
    import contextlib

    for locked in (True, False):
        recorder = Recorder()
        budget = YieldingBudget(recorder, 5)
        if not locked:
            budget._lock = contextlib.nullcontext()
        outcomes = ceiling_outcomes(budget)
        if locked:
            assert outcomes.count("sent") == 5 and len(recorder.sent) == 5
            assert outcomes.count("refused") == 15
        else:
            assert outcomes.count("sent") > 5  # the same exercise kills the mutation


def test_journal_ordinals_are_reserved_once_while_calls_overlap():
    import time
    from concurrent.futures import ThreadPoolExecutor

    class Journal:
        limits = {"search": 20}

        def __init__(self):
            self.records = []
            self.lock = threading.Lock()

        def count(self, bucket):
            return len(self.records)

        def call(self, client, bucket, task, ordinal, kwargs):
            time.sleep(0.02)
            with self.lock:
                self.records.append((ordinal, kwargs["x"]))
            return ordinal

    journal = Journal()
    budget = BudgetedClient(Recorder(), 20, journal=journal, bucket="search", task_id="t")
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda i: budget.create(x=i), range(10)))
    assert sorted(results) == list(range(10))
    assert sorted(ordinal for ordinal, _ in journal.records) == list(range(10))
    assert budget.ordinal == budget.calls == 10


@pytest.mark.parametrize("bad", [True, 2.5, float("nan")])
def test_ceiling_is_an_integer_not_a_boolean_or_unbounded_float(bad):
    with pytest.raises(ValueError, match="positive integer"):
        BudgetedClient(Recorder(), bad)


@pytest.mark.parametrize("failure", ["overloaded", "timeout"])
def test_sdk_wire_attempts_do_not_exceed_the_budget(monkeypatch, failure):
    import socket

    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    from capstone.factors import llm

    def no_network(*args, **kwargs):
        pytest.fail("real socket connections are forbidden in this synthetic test")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    sent = []

    def transport(request):
        sent.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return httpx.Response(
            529,
            headers={"retry-after": "0"},
            json={"error": {"type": "overloaded_error", "message": "synthetic"}},
        )

    actual = anthropic.Anthropic

    def client(**kwargs):
        return actual(
            api_key="synthetic",
            base_url="https://fake.invalid",
            http_client=httpx.Client(transport=httpx.MockTransport(transport)),
            **kwargs,
        )

    monkeypatch.setattr(anthropic, "Anthropic", client)
    sdk = llm.make_client()
    try:
        budget = BudgetedClient(sdk, 1)
        with pytest.raises(anthropic.APIError):
            budget.create(
                model="synthetic",
                max_tokens=1,
                messages=[{"role": "user", "content": "synthetic"}],
            )
        assert len(sent) == budget.calls == 1
        with pytest.raises(CallBudgetExceeded):
            budget.create(model="synthetic", max_tokens=1, messages=[])
        assert len(sent) == 1
    finally:
        sdk.close()


def test_a_client_with_hidden_retries_is_refused_before_any_send():
    recorder = Recorder()
    recorder.max_retries = 2
    with pytest.raises(ValueError, match="disable hidden SDK retries"):
        BudgetedClient(recorder, 3)
    assert recorder.sent == []


def test_a_completed_journal_response_replays_at_the_bucket_ceiling():
    class ReplayJournal(StubJournal):
        def count(self, bucket):
            return 3

        def call(self, client, bucket, task_id, ordinal, kwargs):
            if ordinal >= 3:
                raise CallBudgetExceeded("frozen search allowance exhausted")
            return ordinal

    recorder = Recorder()
    budget = BudgetedClient(recorder, 3, journal=ReplayJournal(), bucket="search", task_id="t")
    assert [budget.create(x=i) for i in range(3)] == [0, 1, 2]
    with pytest.raises(CallBudgetExceeded):
        budget.create(x=3)
    assert budget.calls == 3 and budget.refused == 1 and recorder.sent == []


def test_a_failed_journal_call_does_not_reuse_a_reserved_ordinal():
    class FailingJournal(StubJournal):
        def call(self, client, bucket, task_id, ordinal, kwargs):
            if ordinal == 0:
                raise RuntimeError("synthetic uncertain send")
            return ordinal

    budget = BudgetedClient(Recorder(), 3, journal=FailingJournal(), bucket="search", task_id="t")
    with pytest.raises(RuntimeError, match="uncertain send"):
        budget.create(x=1)
    assert budget.create(x=2) == 1
