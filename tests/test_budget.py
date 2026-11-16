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
