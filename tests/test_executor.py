"""The resource-aware executor (`capstone.pipeline.executor`).

Machine detection, pool sizing from a per-task memory estimate and the config caps,
results in key order at any worker count, one BLAS thread per worker, and a scan of
`capstone/` for hard-coded worker or thread counts. No data, no model, no network.
"""

from __future__ import annotations

import ast
import os
import re
import time
from pathlib import Path

import pytest

from capstone.pipeline import executor

REPO = Path(__file__).resolve().parents[1]
GB = 1024
SMALL_8 = executor.Machine(cores=4, total_mb=8 * GB, available_mb=8 * GB)
SMALL_16 = executor.Machine(cores=4, total_mb=16 * GB, available_mb=16 * GB)
BIG = executor.Machine(cores=64, total_mb=512 * GB, available_mb=512 * GB)


@pytest.fixture
def uncapped(monkeypatch):
    """No `CAPSTONE_MAX_WORKERS` (the suite's conftest caps auto pools at one)."""
    monkeypatch.delenv(executor.ENV_CAP, raising=False)


# --- detection ------------------------------------------------------------------------------


def test_detect_reads_cores_and_memory():
    machine = executor.detect()
    assert machine.cores >= 1
    assert machine.total_mb > 0 and 0 < machine.available_mb <= machine.total_mb


def test_linux_meminfo_is_parsed(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16384000 kB\nMemFree:  100 kB\nMemAvailable:    8192000 kB\n",
        encoding="ascii",
    )
    assert executor._memory_linux(meminfo) == (16000, 8000)
    meminfo.write_text("MemTotal:       2048000 kB\n", encoding="ascii")  # old kernels
    assert executor._memory_linux(meminfo) == (2000, 2000)


# --- sizing ---------------------------------------------------------------------------------


def test_plan_takes_the_smallest_cap(uncapped):
    def plan(tasks, task_mb=100, machine=BIG, **resources):
        return executor.plan(tasks, task_mb, executor.Resources(**resources), machine=machine)

    assert plan(0) == plan(1) == 1  # nothing to share: inline
    assert plan(6) == 6  # the task count
    assert plan(100, machine=SMALL_8) == 4  # the cores
    assert plan(100, max_workers=3) == 3  # [resources] max_workers
    assert plan(100, memory_budget_mb=250) == 2  # the budget over the per-task estimate
    assert plan(100, memory_budget_mb=50) == 1  # never fewer than one
    # auto budget: 75% of the RAM available now
    assert plan(100, task_mb=1024, machine=SMALL_8) == 4
    tight = executor.Machine(cores=4, total_mb=8 * GB, available_mb=3 * GB)
    assert plan(100, task_mb=1024, machine=tight) == 2


def test_the_environment_caps_auto_sizing(monkeypatch):
    monkeypatch.setenv(executor.ENV_CAP, "2")
    assert executor.plan(100, 1, machine=BIG) == 2
    monkeypatch.setenv(executor.ENV_CAP, "0")  # 0 = no cap
    assert executor.plan(100, 1, machine=BIG) == 64
    monkeypatch.setenv(executor.ENV_CAP, "-1")
    with pytest.raises(ValueError):
        executor.plan(100, 1, machine=BIG)


@pytest.mark.parametrize("machine", [SMALL_8, SMALL_16], ids=["8GB", "16GB"])
def test_small_machine_profiles_stay_within_cores_and_budget(uncapped, machine):
    for task_mb in (256, 1024, 3 * GB, 64 * GB):
        workers = executor.plan(720, task_mb, machine=machine)
        assert 1 <= workers <= machine.cores
        assert workers == 1 or workers * task_mb <= machine.available_mb


def test_footprint_counts_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"0" * executor.MB)
    (tmp_path / "y.bin").write_bytes(b"0" * executor.MB)
    assert executor.footprint_mb(tmp_path) == 2
    assert executor.footprint_mb(tmp_path / "y.bin", tmp_path / "missing") == 1


# --- ordered results ------------------------------------------------------------------------


def slow_square(x, delay):
    time.sleep(delay)
    return x * x, os.getpid(), os.environ.get("OMP_NUM_THREADS")


def test_inline_runs_lazily_in_this_process():
    seen = []

    def record(x):
        seen.append(x)
        return x

    with executor.ordered(record, [(2, (2,)), (1, (1,)), (3, (3,))], 1) as results:
        assert seen == []
        assert next(results) == (1, 1) and seen == [1]  # one at a time, in key order
        assert list(results) == [(2, 2), (3, 3)]


def test_a_pool_hands_out_results_in_key_order_not_completion_order():
    # the first keys finish last
    jobs = [(k, (k, 0.05 * (4 - k))) for k in (3, 0, 2, 1)]
    with executor.ordered(slow_square, jobs, 3) as results:
        got = list(results)
    assert [key for key, _ in got] == [0, 1, 2, 3]
    assert [value for _, (value, _, _) in got] == [0, 1, 4, 9]
    assert {pid for _, (_, pid, _) in got} - {os.getpid()}  # ran in workers
    assert {threads for _, (_, _, threads) in got} == {str(executor.THREADS_PER_WORKER)}


def test_the_parent_environment_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with executor.ordered(slow_square, [(0, (0, 0)), (1, (1, 0))], 2) as results:
        list(results)
    assert os.environ["OMP_NUM_THREADS"] == "7" and "MKL_NUM_THREADS" not in os.environ


def marked(folder, key, delay):
    time.sleep(delay)
    (Path(folder) / str(key)).write_text("ran", encoding="utf-8")
    return key


def test_leaving_early_cancels_what_has_not_started_and_waits_for_the_rest(tmp_path):
    jobs = [(k, (str(tmp_path), k, 0.3)) for k in range(12)]
    with pytest.raises(RuntimeError):
        with executor.ordered(marked, jobs, 2) as results:
            next(results)
            raise RuntimeError("stop")
    ran = sorted(int(p.name) for p in tmp_path.iterdir())
    assert 1 <= len(ran) < len(jobs)  # the queue was cancelled
    time.sleep(0.5)
    assert len(list(tmp_path.iterdir())) == len(ran)  # nothing still running afterwards


# --- config ---------------------------------------------------------------------------------


def test_resources_section_defaults_to_auto_and_refuses_negatives():
    assert executor.Resources.from_section(None) == executor.Resources()
    assert executor.Resources.from_section({"max_workers": 3}) == executor.Resources(max_workers=3)
    with pytest.raises(ValueError, match="must be >= 0"):
        executor.Resources.from_section({"max_workers": -2})
    with pytest.raises(ValueError, match="must be >= 0"):
        executor.Resources.from_section({"memory_budget_mb": -1})


def test_the_worker_initializer_sets_one_thread_per_worker(monkeypatch):
    monkeypatch.setattr(executor, "_exit_with_parent", lambda: None)
    for name in executor.THREAD_VARIABLES:
        monkeypatch.setenv(name, "7")
    executor._initialize_worker()
    assert {os.environ[n] for n in executor.THREAD_VARIABLES} == {str(executor.THREADS_PER_WORKER)}


# --- no hard-coded worker or thread counts --------------------------------------------------

COUNT = re.compile(r"(?i)(^|_)(workers?|threads?|processes|n_jobs|jobs|cores|pool_size)$")
POOLS = {"ProcessPoolExecutor", "ThreadPoolExecutor", "Pool", "ThreadPool"}
# The only literal counts allowed in capstone/: the executor's named constants.
ALLOWED = {
    ("capstone/pipeline/executor.py", "THREADS_PER_WORKER"),
}


def _int(node):
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _count(node):
    """An int literal, or an expression built around one: `min(cores, 8)`, `cores or 4`
    ."""
    if _int(node):
        return True
    if isinstance(node, ast.Call) and _name(node.func) in {"min", "max"}:
        return any(_count(arg) for arg in node.args)
    if isinstance(node, ast.BoolOp):
        return any(_count(value) for value in node.values)
    return False


def _option(node):
    """The dest an `add_argument("--n-workers", ...)` call names, or ""."""
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value.startswith("--"):
                return arg.value.lstrip("-").replace("-", "_")
    return ""


def _digits(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isdigit()


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _thread_variable(name):
    return name.upper().endswith(("_THREADS", "_NUM_THREADS", "MAXIMUM_THREADS"))


def literal_counts(source, relative="<snippet>"):
    """(file, line, what) for every literal worker or thread count in `source`."""
    found = []

    def hit(node, what):
        if (relative, what) not in ALLOWED:
            found.append((relative, node.lineno, what))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            callee = _name(node.func)
            if callee in POOLS and node.args and _count(node.args[0]):
                hit(node, callee)
            for keyword in node.keywords:
                if keyword.arg and COUNT.search(keyword.arg) and _count(keyword.value):
                    hit(node, keyword.arg)
            if callee == "add_argument" and COUNT.search(_option(node)):  # a CLI default
                for keyword in node.keywords:
                    if keyword.arg == "default" and _count(keyword.value):
                        hit(node, _option(node))
            if callee in {"get", "setdefault", "getenv"} and len(node.args) == 2:
                key, default = node.args
                if COUNT.search(_name(key)) and _count(default):
                    hit(node, _name(key))
                if _thread_variable(_name(key)) and _digits(default):
                    hit(node, _name(key))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            tail = positional[len(positional) - len(arguments.defaults) :]
            pairs = list(zip(tail, arguments.defaults, strict=True))
            pairs += [
                (a, d)
                for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
                if d
            ]
            for argument, default in pairs:
                if COUNT.search(argument.arg) and _count(default):
                    hit(node, argument.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = _name(target.slice) if isinstance(target, ast.Subscript) else _name(target)
                if COUNT.search(name) and node.value is not None and _count(node.value):
                    hit(node, name)
                if _thread_variable(name) and node.value is not None and _digits(node.value):
                    hit(node, name)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                name = _name(key) if key is not None else ""
                if _thread_variable(name) and _digits(value):
                    hit(node, name)
                if not COUNT.search(name):
                    continue
                if _count(value):
                    hit(node, name)
                if isinstance(value, ast.Call):  # a config Key(kind, default, ...)
                    values = [*value.args, *(k.value for k in value.keywords)]
                    if any(_count(v) for v in values):
                        hit(node, name)
    return found


def test_the_scan_finds_literal_counts():
    snippets = [
        "ProcessPoolExecutor(4)",
        "ProcessPoolExecutor(max_workers=8)",
        "run(workers=2)",
        "def f(x, workers=1): pass",
        "def f(*, n_jobs=4): pass",
        "max_workers = 16",
        "cfg.get('workers', 1)",
        "os.environ['OMP_NUM_THREADS'] = '1'",
        "env = {'MKL_NUM_THREADS': '4'}",
        "os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')",
        "schema = {'workers': Key('int', 1, 'help')}",
        "schema = {'workers': Key('int', AUTO, 'help', legacy=1)}",
        # caps built around a literal, and CLI defaults
        "p.add_argument('--workers', type=int, default=min(os.cpu_count() or 1, 8))",
        "p.add_argument('-w', '--n-jobs', default=4)",
        "run(workers=min(cores, 8))",
        "workers = os.cpu_count() or 4",
        "def f(threads=max(2, n)): pass",
    ]
    for snippet in snippets:
        assert literal_counts(snippet), snippet
    clean = [
        "ProcessPoolExecutor(max_workers=workers)",
        "def f(workers=INLINE): pass",
        "if workers <= 1: pass",
        "schema = {'workers': Key('int', AUTO, 'help', legacy=INLINE)}",
        "count = 3",
        "p.add_argument('--workers', type=int, default=None)",
        "p.add_argument('--timeout', type=float, default=3600)",
        "workers = max(INLINE, min(caps))",
    ]
    for snippet in clean:
        assert not literal_counts(snippet), snippet


def test_capstone_has_no_hard_coded_worker_or_thread_counts():
    found = []
    for path in sorted((REPO / "capstone").rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        found += literal_counts(path.read_text(encoding="utf-8"), relative)
    assert found == [], "literal worker/thread counts (use capstone.pipeline.executor)"
    # the allowlisted constant is still where the scan expects it
    source = (REPO / "capstone/pipeline/executor.py").read_text(encoding="utf-8")
    assert re.search(r"^THREADS_PER_WORKER = 1\b", source, flags=re.M)


ORPHAN = """import os, sys, time
from capstone.pipeline import executor

if __name__ == "__main__":
    with executor.process_pool(2) as pool:
        pool.submit(time.sleep, 120)
        pool.submit(time.sleep, 120)
        time.sleep(5)  # both workers started and busy
        os._exit(9)  # the parent dies (a killed tick)
"""


def test_workers_exit_when_their_parent_is_killed(tmp_path):
    """A killed parent leaves no orphan worker: the console pipes it shares close, so a
    caller waiting on them (subprocess.run, a scheduler) returns at once."""
    import subprocess
    import sys

    script = tmp_path / "orphan.py"
    script.write_text(ORPHAN, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "tests")])}
    done = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=90
    )  # an orphan would hold the pipes for its whole 120 s sleep: TimeoutExpired
    assert done.returncode == 9, done.stdout + done.stderr
