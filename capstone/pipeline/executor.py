"""Worker sizing for concurrent stages: the names `capstone.factors.paper_pool` uses.

- `detect()` reads the machine: the logical cores this process may use.
- `env_cap()` reads `CAPSTONE_MAX_WORKERS`, an operator's cap on automatic sizing.

This is the reduced form: no process pool and no memory probing. The full resource-aware
executor is a separate follow-up (tasks S-16 / C-01) and replaces this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

INLINE = 1  # one worker: run in this process, no pool
AUTO = 0  # a config value of 0 means "size it from the machine"
ENV_CAP = "CAPSTONE_MAX_WORKERS"  # an operator's (or the test suite's) cap on auto sizing


@dataclass(frozen=True)
class Machine:
    cores: int


def _cores():
    counted = getattr(os, "process_cpu_count", None)  # 3.13+: honours affinity
    if counted is not None:
        return counted() or INLINE
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or INLINE
    return os.cpu_count() or INLINE


def detect():
    """This machine's usable cores."""
    return Machine(cores=_cores())


def env_cap():
    """`CAPSTONE_MAX_WORKERS` as a positive int, or None when unset or `AUTO`."""
    text = os.environ.get(ENV_CAP, "").strip()
    if not text:
        return None
    value = int(text)
    if value < AUTO:
        raise ValueError(f"{ENV_CAP}={text} must be >= {AUTO}")
    return value or None
