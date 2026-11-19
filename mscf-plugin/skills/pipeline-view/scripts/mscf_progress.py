"""Report a running experiment's position in the pipeline to the pipeline view (stdlib only).

In an experiment script::

    from mscf_progress import track

    with track("momentum-15-1") as run:
        run.stage("data")                 # station ids: see stations.json
        run.wait("queued for a CPU")      # optional: shown as waiting, not working
        inputs = load_inputs()           # your experiment code
        run.stage("signal")
        for i, chunk in enumerate(chunks):
            run.progress((i + 1) / len(chunks), f"chunk {i + 1}")
        run.stage("backtest")
        ...
        runlog.log_run("momentum-15-1", ..., tags=["stage:screen", ...])   # the view picks this up

Or run an unmodified script under `pipeline_run.py`, which tracks start, the ledger entry and
exit. Both write to $MSCF_PROGRESS_DIR (default ~/.cache/mscf-pipeline/progress), outside
the repository. Files hold the experiment name, station, progress and timestamps only:
nothing about the session, the data or the results. If the directory cannot be written,
tracking turns itself off and the experiment carries on.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

HEARTBEAT_SECONDS = 2.0


def progress_dir() -> Path:
    root = os.environ.get("MSCF_PROGRESS_DIR")
    return Path(root) if root else Path.home() / ".cache" / "mscf-pipeline" / "progress"


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


class Run:
    """One tracked experiment. Use `track(...)` rather than constructing this directly."""

    def __init__(self, name: str, run_id: str | None = None, *, owns_meta: bool = True) -> None:
        self.name = name
        self.run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.owns_meta = owns_meta
        self.dir = progress_dir()
        self.meta_path = self.dir / f"{self.run_id}.json"
        self.events_path = self.dir / f"{self.run_id}.events.jsonl"
        self.enabled = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.meta = {
            "run_id": self.run_id,
            "name": name,
            "pid": os.getpid(),
            "user": os.environ.get("GITHUB_USER") or os.environ.get("USER") or "unknown",
            "started": time.time(),
            "heartbeat": time.time(),
            "status": "running",
            "exit_code": None,
            "ended": None,
        }

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> Run:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.owns_meta:
                _atomic_write(self.meta_path, self.meta)
                self._thread = threading.Thread(target=self._beat, daemon=True)
                self._thread.start()
        except OSError:
            self.enabled = False
        return self

    def finish(self, status: str = "done", exit_code: int | None = 0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=HEARTBEAT_SECONDS * 2)
        if self.enabled and self.owns_meta:
            self.meta.update(
                status=status, exit_code=exit_code, ended=time.time(), heartbeat=time.time()
            )
            try:
                _atomic_write(self.meta_path, self.meta)
            except OSError:
                pass

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self.meta["heartbeat"] = time.time()
            try:
                _atomic_write(self.meta_path, self.meta)
            except OSError:
                return

    # -- reporting ---------------------------------------------------------
    def _event(self, **fields) -> None:
        if not self.enabled:
            return
        fields["ts"] = time.time()
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(fields) + "\n")
        except OSError:
            self.enabled = False

    def stage(self, station: str, note: str = "") -> None:
        """Move to a station (an id from stations.json); progress within it restarts at 0."""
        self._event(station=station, pct=0.0, waiting=False, note=note)

    def wait(self, note: str = "") -> None:
        """Waiting for a shared resource (a CPU slot, a WRDS connection) rather than working.

        The run sits in the station's queue instead of at the instrument until the next
        stage() or progress() call.
        """
        self._event(waiting=True, note=note or "waiting")

    def progress(self, fraction: float, note: str = "") -> None:
        """Progress within the current station, 0 to 1."""
        self._event(pct=max(0.0, min(1.0, float(fraction))), waiting=False, note=note)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is SystemExit and exc.code in (0, None):
            self.finish("done", 0)
        elif exc_type is KeyboardInterrupt:
            self.finish("failed", 130)
        elif exc_type is not None:
            self._event(note=f"{exc_type.__name__}: {str(exc)[:120]}")
            self.finish("failed", 1)
        else:
            self.finish("done", 0)
        return False


def track(name: str) -> Run:
    """Track an experiment. Under pipeline_run.py, attach to the wrapper's run instead."""
    wrapped = os.environ.get("MSCF_RUN_ID")
    if wrapped:
        return Run(name, wrapped, owns_meta=False).start()
    return Run(name).start()
