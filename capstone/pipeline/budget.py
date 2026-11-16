"""Shared model-call budget; an optional durable request journal replays completed calls."""

from __future__ import annotations

import threading

from capstone.factors.deepen import BudgetExhausted


class CallBudgetExceeded(BudgetExhausted):
    """The run's agreed number of model calls is used up."""


class BudgetedClient:
    """Count every sent call, or replay its durable checkpoint when a journal is supplied."""

    def __init__(self, client, max_calls: int, *, journal=None, bucket=None, task_id=None):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if journal is not None and (
            bucket not in journal.limits or journal.limits[bucket] != max_calls or not task_id
        ):
            raise ValueError("durable requests require the frozen bucket allowance and task id")
        self.client = client
        self.max_calls = max_calls
        self.journal = journal
        self.bucket = bucket
        self.task_id = task_id
        self.ordinal = 0
        self.calls = 0 if journal is None else journal.count(bucket)
        self.refused = 0
        self.messages = self
        # Concurrent papers share this client: the ceiling check and the
        # count are one step, so a call past the ceiling never goes out.
        self._lock = threading.Lock()

    def create(self, **kwargs):
        if self.journal is not None:
            try:
                result = self.journal.call(
                    self.client, self.bucket, self.task_id, self.ordinal, kwargs
                )
            except CallBudgetExceeded:
                self.refused += 1
                raise
            finally:
                self.calls = self.journal.count(self.bucket)
            self.ordinal += 1
            return result
        with self._lock:
            if self.calls >= self.max_calls:
                self.refused += 1
                raise CallBudgetExceeded(f"the run's {self.max_calls} model calls are used up")
            self.calls += 1
        return self.client.messages.create(**kwargs)

    def check(self) -> None:
        """Raise if a call was refused, including when a legacy paper loop caught it."""
        if self.refused:
            raise CallBudgetExceeded(
                f"the run's {self.max_calls} model calls are used up "
                f"({self.refused} call(s) refused)"
            )
