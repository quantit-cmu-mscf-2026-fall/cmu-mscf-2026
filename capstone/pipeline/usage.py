"""Model usage for a search run: a per-call recorder, prompt caching and a cost estimate.

`UsageRecorder` keeps each call's model and token counts; `PromptCaching` marks the
system text as a cacheable prefix; `cost_estimate` prices the records at `PRICES`.
"""

from __future__ import annotations

import threading

from capstone.factors.paper_pool import current_slot

__all__ = ["PRICES", "PromptCaching", "UsageRecorder", "cost_estimate"]

# USD per million tokens, (input, output): list prices, an estimate only.
# Source constants, not a team-approved rate table; the console's bill is the real one.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

_TOKENS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class UsageRecorder:
    """`client` that keeps each call's model and token usage in `records`.

    Calls made by concurrent papers (`paper_pool`, each in its own slot) are kept in
    paper order, then call order within the paper, not in the order they returned:
    they go in before the next call made outside a slot, or when `records` is read.
    """

    def __init__(self, client):
        self.client = client
        self._records: list[dict] = []
        self._pending: list[tuple[int, int, dict]] = []
        self._lock = threading.Lock()
        self.messages = self

    @property
    def max_retries(self) -> int:
        """The wrapped client's hidden SDK retries, visible to a call budget."""
        return getattr(self.client, "max_retries", 0)

    @property
    def records(self) -> list[dict]:
        with self._lock:
            self._flush()
            return self._records

    def _flush(self) -> None:
        self._records.extend(r for *_, r in sorted(self._pending, key=lambda p: p[:2]))
        self._pending.clear()

    def create(self, **kwargs):
        response = self.client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        record = {"model": kwargs.get("model", "")}
        for name in _TOKENS:
            record[name] = int(getattr(usage, name, 0) or 0)
        where = current_slot()
        with self._lock:
            if where is None:
                self._flush()
                self._records.append(record)
            else:  # a paper's calls are sequential: arrival order within its slot
                self._pending.append((where, len(self._pending), record))
        return response


class PromptCaching:
    """`client` that marks each request's system text as a cacheable prefix.

    The text is unchanged: a string `system` becomes one text block with
    `cache_control`, so the tools and system prompt (the stable prefix of every
    factor, repair and alignment call) can be read from the cache on repeats.
    Requests whose `system` is already a list of blocks pass through unchanged."""

    def __init__(self, client):
        self.client = client
        self.messages = self

    @property
    def max_retries(self) -> int:
        """The wrapped client's hidden SDK retries, visible to a call budget."""
        return getattr(self.client, "max_retries", 0)

    def create(self, **kwargs):
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            block = {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            kwargs = {**kwargs, "system": [block]}
        return self.client.messages.create(**kwargs)


def cost_estimate(records: list[dict], prices: dict[str, tuple[float, float]] = PRICES) -> dict:
    """Tokens per model and their cost at `prices`; models without a price are named."""
    by_model: dict[str, dict] = {}
    for r in records:
        m = by_model.setdefault(r["model"], dict(calls=0, input_tokens=0, output_tokens=0))
        m["calls"] += 1
        m["input_tokens"] += r["input_tokens"]
        m["output_tokens"] += r["output_tokens"]
    usd = sum(
        (m["input_tokens"] * prices[name][0] + m["output_tokens"] * prices[name][1]) / 1e6
        for name, m in by_model.items()
        if name in prices
    )
    unpriced = sorted(name for name in by_model if name not in prices)
    return {"usd": round(usd, 4), "by_model": by_model, "unpriced": unpriced}
