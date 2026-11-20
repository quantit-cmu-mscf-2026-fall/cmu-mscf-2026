"""The variant grid on worker processes, sharing one panel.

`evaluation.evaluate_factors` spends its compute in `factor_outcome`: every stored
factor traded by every holds x neutral spec, a pure function of a factor tree and the
search panel. `PanelPool` runs those side by side:

- **The panel is shared once.** The parent pickles it into its own directory under
  `<workdir>/.parallel/` (so two pools on one workdir never share a file) and each
  worker loads it on its first task (one copy per worker process, `_worker_panel`).
- **Workers compute, the parent writes.** A worker returns a factor's `FactorOutcome`; the
  parent records the variant rows, logs the trials and builds the frames in the trees'
  order, exactly as the sequential loop does (`evaluate_factors(outcomes=...)`). The
  ledger, `factors.db` and the frames are the same bytes at one worker and at N.
- **Worker count** is `executor.plan` on `[resources]` (`max_workers`,
  `memory_budget_mb`) and `CAPSTONE_MAX_WORKERS`; at `executor.INLINE` nothing is
  staged or spawned and the path is the sequential code itself.
- **A failing worker** surfaces as `EvaluationWorkerError` (naming the factor), raised
  where the sequential loop would have raised.

Not here: fetching the deepen scores ahead (a `PooledScorer`) waits for the
evaluation-backed deepen scorer.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from pathlib import Path

from capstone.factors.evaluation import Panel, TradingSpec, factor_outcome
from capstone.pipeline import executor

__all__ = ["STAGING", "EvaluationWorkerError", "PanelPool", "panel_mb", "workers"]

STAGING = ".parallel"  # under the workdir: a directory per pool, removed when the pool closes
PANEL_FILE = "panel.pkl"
PANEL_COPIES = 3  # per worker: the loaded panel plus a factor's signal and traded frames


class EvaluationWorkerError(RuntimeError):
    """A factor's evaluation failed in a worker process (or the pool broke)."""


def panel_mb(panel: Panel) -> float:
    """The panel's in-memory size (MB): its frames and series, deep."""
    size = 0
    for frame in (*panel.fields.values(), panel.universe, panel.half_spread, panel.sic):
        if frame is None:
            continue
        usage = frame.memory_usage(deep=True)
        size += int(usage.sum()) if hasattr(usage, "sum") else int(usage)
    return size / executor.MB


def workers(tasks: int, panel: Panel, resources: executor.Resources | None) -> int:
    """`executor.plan` for `tasks` evaluations, each worker holding `PANEL_COPIES` panels."""
    task_mb = executor.WORKER_BASE_MB + PANEL_COPIES * panel_mb(panel)
    return executor.plan(tasks, task_mb, resources)


# ---------------------------------------------------------------------------
# Worker side

_WORKER_PANEL: dict = {}  # one loaded panel per worker process


def _worker_panel(path: str) -> Panel:
    if path not in _WORKER_PANEL:
        _WORKER_PANEL.clear()
        with open(path, "rb") as handle:
            _WORKER_PANEL[path] = pickle.load(handle)
    return _WORKER_PANEL[path]


def remote_outcome(path: str, tree, specs: tuple[TradingSpec, ...]):
    """`evaluation.factor_outcome` on the shared panel, in a worker."""
    return factor_outcome(tree, _worker_panel(path), specs)


# ---------------------------------------------------------------------------
# Parent side


def _taken(future, what: str):
    try:
        return future.result()
    except BrokenProcessPool as exc:
        raise EvaluationWorkerError(f"the evaluation pool broke while {what}: {exc}") from exc
    except Exception as exc:
        raise EvaluationWorkerError(f"{what} failed in a worker: {exc!r}") from exc


class PanelPool:
    """A process pool sharing one panel; a context manager. At `executor.INLINE`
    workers it is the sequential code: nothing staged, nothing spawned."""

    def __init__(self, panel: Panel, workers: int, workdir: str | Path):
        self.panel = panel
        self.workers = max(executor.INLINE, int(workers))
        self.staging = Path(workdir) / STAGING  # holds one directory per open pool
        self._dir: Path | None = None
        self._path: str | None = None
        self._pool = None
        self._stack = ExitStack()

    @property
    def parallel(self) -> bool:
        return self.workers > executor.INLINE

    def __enter__(self) -> PanelPool:
        if self.parallel:
            self.staging.mkdir(parents=True, exist_ok=True)
            self._dir = Path(tempfile.mkdtemp(prefix="pool-", dir=self.staging))  # its own
            path = self._dir / PANEL_FILE
            with open(path, "wb") as handle:
                pickle.dump(self.panel, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self._path = str(path)
            self._stack.callback(self._unstage)
            self._pool = self._stack.enter_context(executor.process_pool(self.workers))
        return self

    def _unstage(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)
        try:
            self.staging.rmdir()  # only once no other pool is staged under it
        except OSError:
            pass

    def __exit__(self, *exc_info) -> bool:
        self._pool = None
        return self._stack.__exit__(*exc_info)

    def outcomes(self, specs):
        """An `evaluate_factors(outcomes=...)` callable: `[(fid, tree)]` -> `(fid,
        factor_outcome)` in that order, computed on the pool (inline: lazily, here)."""
        specs = tuple(specs)

        def run(items):
            if not self.parallel:
                return ((fid, factor_outcome(tree, self.panel, specs)) for fid, tree in items)
            futures = [
                (fid, self._pool.submit(remote_outcome, self._path, tree, specs))
                for fid, tree in items
            ]
            return ((fid, _taken(f, f"evaluating factor {fid}")) for fid, f in futures)

        return run
