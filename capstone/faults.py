"""A test-only, env-gated crash hook for resumable stages.

Production never sets `CAPSTONE_FAULT_INJECTION=1`, so `maybe_fault` is then a no-op.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import cache


class InjectedFault(RuntimeError):
    """A test-only simulated crash at a named stage point (see `maybe_fault`)."""


@cache
def _fault_plan() -> dict[str, set[int]]:
    """`CAPSTONE_FAULT_AT` parsed once; empty unless `CAPSTONE_FAULT_INJECTION=1`."""
    if os.environ.get("CAPSTONE_FAULT_INJECTION") != "1":
        return {}
    plan: dict[str, set[int]] = {}
    for item in os.environ.get("CAPSTONE_FAULT_AT", "").split(","):
        point, _, nth = item.strip().rpartition(":")
        if not nth.isdigit():
            point, nth = item.strip(), "1"
        if point:
            plan.setdefault(point, set()).add(int(nth))
    return plan


_fault_hits: Counter[str] = Counter()


def maybe_fault(point: str) -> None:
    """Raise `InjectedFault` on the listed hit of `point`; a no-op in production.

    `CAPSTONE_FAULT_AT` is a comma-separated list of `<stage>:<step>[:N]` (N-th hit,
    default 1). Tests only: production never sets `CAPSTONE_FAULT_INJECTION=1`.
    """
    plan = _fault_plan()
    if point in plan:
        _fault_hits[point] += 1
        if _fault_hits[point] in plan[point]:
            raise InjectedFault(f"injected fault at {point} (hit {_fault_hits[point]})")
