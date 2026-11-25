"""A resource-aware process executor for the search's concurrent stages.

- `detect()` reads the machine: logical cores (the ones this process may use) and total /
  available RAM, standard library only, on Windows and Linux (other systems: total RAM
  from `sysconf`, available taken as total).
- `plan(tasks, task_mb, resources)` sizes a pool: the fewest of the task count, the
  cores, `[resources] max_workers`, the `CAPSTONE_MAX_WORKERS` environment cap and the
  memory budget divided by the per-task estimate; never fewer than one (`INLINE`).
- `ordered(fn, jobs, workers)` runs `fn(*args)` for every `(key, args)` and yields
  `(key, result)` in key order, never in completion order. At one worker it runs inline,
  lazily, in this process (no pool): exactly the sequential code path.
- Every worker process runs one BLAS/OpenMP thread (`THREADS_PER_WORKER`): the thread
  variables are set in the environment the children are spawned with and again in the
  worker initializer.

Literal worker and thread counts live only here (`INLINE`, `THREADS_PER_WORKER`);
`tests/test_executor.py` scans `capstone/` for any other.

This replaces the reduced form that `capstone.factors.paper_pool` first used (cores and the
`CAPSTONE_MAX_WORKERS` cap only); `detect()` now also reads the RAM.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.connection
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

INLINE = 1  # one worker: run in this process, no pool
AUTO = 0  # a config value of 0 means "size it from the machine"
THREADS_PER_WORKER = 1  # BLAS/OpenMP threads inside each worker process
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
ENV_CAP = "CAPSTONE_MAX_WORKERS"  # an operator's (or the test suite's) cap on auto sizing
WORKER_BASE_MB = 256  # one idle worker: interpreter, numpy, pandas
AVAILABLE_SHARE = 0.75  # auto budget: this share of the RAM available now
ORPHANED = 3  # exit status of a worker whose parent died
MB = 1024 * 1024


@dataclass(frozen=True)
class Machine:
    cores: int
    total_mb: int
    available_mb: int


@dataclass(frozen=True)
class Resources:
    """The `[resources]` caps; `AUTO` (0) leaves a cap to the machine."""

    max_workers: int = AUTO
    memory_budget_mb: int = AUTO

    @classmethod
    def from_section(cls, section):
        section = section or {}
        values = cls(
            max_workers=section.get("max_workers", AUTO),
            memory_budget_mb=section.get("memory_budget_mb", AUTO),
        )
        if values.max_workers < AUTO or values.memory_budget_mb < AUTO:
            raise ValueError(f"[resources] values must be >= {AUTO}: {values}")
        return values


def _cores():
    counted = getattr(os, "process_cpu_count", None)  # 3.13+: honours affinity
    if counted is not None:
        return counted() or INLINE
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or INLINE
    return os.cpu_count() or INLINE


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory_windows():
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullTotalPhys // MB, status.ullAvailPhys // MB


def _memory_linux(path="/proc/meminfo"):
    fields = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        name, _, rest = line.partition(":")
        fields[name] = int(rest.split()[0]) * 1024  # kB
    total = fields["MemTotal"]
    return total // MB, fields.get("MemAvailable", total) // MB


def _memory_sysconf():
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return total // MB, total // MB


def detect():
    """This machine's cores and RAM (MB)."""
    if sys.platform == "win32":
        total, available = _memory_windows()
    elif Path("/proc/meminfo").is_file():
        total, available = _memory_linux()
    else:
        total, available = _memory_sysconf()
    return Machine(cores=_cores(), total_mb=int(total), available_mb=int(available))


def env_cap():
    """`CAPSTONE_MAX_WORKERS` as a positive int, or None when unset or `AUTO`."""
    text = os.environ.get(ENV_CAP, "").strip()
    if not text:
        return None
    value = int(text)
    if value < AUTO:
        raise ValueError(f"{ENV_CAP}={text} must be >= {AUTO}")
    return value or None


def footprint_mb(*paths):
    """On-disk size (MB) of the files under `paths`: the base of a per-task estimate."""
    size = 0
    for path in map(Path, paths):
        files = [path] if path.is_file() else path.rglob("*") if path.is_dir() else []
        size += sum(p.stat().st_size for p in files if p.is_file())
    return size / MB


def plan(tasks, task_mb, resources=None, *, machine=None):
    """Workers for `tasks` tasks of about `task_mb` MB each (at least `INLINE`)."""
    if tasks <= INLINE:
        return INLINE
    resources = resources or Resources()
    machine = machine or detect()
    budget = resources.memory_budget_mb or int(machine.available_mb * AVAILABLE_SHARE)
    caps = [tasks, machine.cores, int(budget // max(task_mb, INLINE))]
    caps += [c for c in (resources.max_workers, env_cap()) if c]
    return max(INLINE, min(caps))


@contextmanager
def _thread_environment():
    """Children are spawned with one BLAS/OpenMP thread each; restored afterwards."""
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: str(THREADS_PER_WORKER) for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _exit_with_parent():
    """End this worker as soon as the process that owns the pool is gone (a killed tick):
    an orphan would keep running, hold the parent's console pipes and write late."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch():
        multiprocessing.connection.wait([parent.sentinel])
        os._exit(ORPHANED)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _initialize_worker():
    _exit_with_parent()
    os.environ.update({name: str(THREADS_PER_WORKER) for name in THREAD_VARIABLES})


@contextmanager
def process_pool(workers):
    """A process pool of `workers` one-thread workers; pending work is cancelled when the
    block exits on an error, and running work is waited for, so nothing outlives it."""
    with _thread_environment():
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker)
        try:
            yield pool
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)


@contextmanager
def ordered(fn, jobs, workers):
    """An iterator of `(key, fn(*args))` over `jobs` (`(key, args)` pairs), in key order.

    At `INLINE` workers `fn` runs in this process when the next item is taken, exactly
    as a sequential loop would. Otherwise every job is submitted to a pool and results
    are handed out in key order as they become available; leaving the block early
    cancels what has not started and waits for what has."""
    jobs = sorted(jobs, key=lambda job: job[0])
    if workers <= INLINE or len(jobs) <= INLINE:
        yield ((key, fn(*args)) for key, args in jobs)
        return
    with process_pool(min(workers, len(jobs))) as pool:
        futures = [(key, pool.submit(fn, *args)) for key, args in jobs]
        yield ((key, future.result()) for key, future in futures)
