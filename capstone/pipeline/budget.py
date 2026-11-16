"""Shared model-call budget: a hard ceiling on the calls a run may send."""

from __future__ import annotations

import threading


# Follow-up: when `factors/deepen.py` (which defines `BudgetExhausted`) is public, make
# this subclass it so the deepening loop's handler also catches the ceiling.
# The durable-request (journal) branch of the source is not ported here (follow-up).
class CallBudgetExceeded(RuntimeError):
    """The run's agreed number of model calls is used up."""


class BudgetedClient:
    """Count every sent call; refuse the ones past `max_calls` before they go out."""

    def __init__(self, client, max_calls: int):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.client = client
        self.max_calls = max_calls
        self.calls = 0
        self.refused = 0
        self.messages = self
        # Concurrent papers (paper_pool) share this client: the ceiling check and the
        # count are one step, so a call past the ceiling never goes out.
        self._lock = threading.Lock()

    def create(self, **kwargs):
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
