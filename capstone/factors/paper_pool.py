"""Papers side by side: each paper's model calls run concurrently, its store writes
land in paper order.

`run_isolated(con, source, client, config, concurrency)` is `pipeline.run` with one
change of meaning: every paper is processed against the store **as it was when the
paper phase started** (a private in-memory copy per paper), not against the factors
the papers before it stored. That is what makes the papers independent, so their
hypotheses, factor, repair and alignment calls can be in flight together. It is a
different procedure from `pipeline.run` (a later paper's prompt no longer names the
structures an earlier paper made frequent, and its originality screen does not see
an earlier paper's factors), so it is opt-in (`--paper-mode isolated` on the command line);
the sequential default is unchanged.

As each source-ordered result arrives, its new rows (paper, hypotheses, proposals with
their factors, usage) are copied into `con` in source order, exactly as the store's
own writers would have made them: a factor a previous paper already stored becomes a
"duplicate" proposal ("already in the store"). The source's hand-off (`done`) is
called in the same order. So the store, and everything computed from it, is the
same at every concurrency, 1 included, for the same model answers.

On interruption, no new papers are scheduled. Already-running papers are joined
and their completed or partial rows and recorded usage are merged before the
exception is re-raised. A partial paper is not handed off as done.

`slot()` names the paper a call belongs to (a context variable set inside each
paper's task), so a recorder can keep its records in paper order
(a usage recorder) instead of completion order.
"""

from __future__ import annotations

import contextvars
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock

from capstone.factors.llm import FactorConfig
from capstone.factors.pipeline import RunSummary, process_paper
from capstone.factors.zoo import default_zoo
from capstone.pipeline import executor

__all__ = ["copy_new_rows", "current_slot", "resolve_concurrency", "run_isolated", "slot"]

_SLOT: contextvars.ContextVar[int | None] = contextvars.ContextVar("paper_slot", default=None)
_PAPER_TABLES = ("papers", "hypotheses", "factors", "proposals", "usage", "sqlite_sequence")


def current_slot() -> int | None:
    """The source position of the paper whose task is running here, or None."""
    return _SLOT.get()


@contextmanager
def slot(index: int) -> Iterator[None]:
    token = _SLOT.set(index)
    try:
        yield
    finally:
        _SLOT.reset(token)


def resolve_concurrency(configured: int, papers: int) -> int:
    """`configured`, or (at `executor.AUTO`) the machine's cores under the operator's
    `CAPSTONE_MAX_WORKERS` cap (also applied to explicit sizes); never more than
    the papers, never fewer than one."""
    if configured < executor.AUTO:
        raise ValueError(f"concurrency must be >= {executor.AUTO}, got {configured}")
    caps = [papers, configured or executor.detect().cores]
    caps += [c for c in (executor.env_cap(),) if c]
    return max(executor.INLINE, min(caps))


def _snapshot(con: sqlite3.Connection) -> sqlite3.Connection:
    """An in-memory copy of the store, usable from one worker thread."""
    private = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        con.backup(private)
    except BaseException:
        private.close()
        raise
    private.row_factory = sqlite3.Row
    private.execute("PRAGMA foreign_keys = ON")
    return private


def _counts(con: sqlite3.Connection) -> dict[str, int]:
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in sorted(tables)}


def _max_id(con: sqlite3.Connection, table: str) -> int:
    return con.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0]


def _insert(con: sqlite3.Connection, table: str, row: dict, verb: str = "INSERT") -> int:
    columns = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    sql = f"{verb} INTO {table} ({columns}) VALUES ({marks})"
    return con.execute(sql, tuple(row.values())).rowcount


def copy_new_rows(
    private: sqlite3.Connection,
    con: sqlite3.Connection,
    base: dict[str, int],
    keys: dict,
    *,
    _marker: tuple[str, int] | None = None,
) -> Counter:
    """Copy one paper's new rows from `private` into `con`, as the store's writers would
    have written them there; returns how many "stored" proposals became "duplicate".
    `base` holds the snapshot's last proposal and usage ids, `keys` its paper and
    hypothesis keys."""
    changed: Counter = Counter()
    with con:
        for row in private.execute("SELECT * FROM papers ORDER BY rowid"):
            if row["key"] not in keys["papers"]:
                _insert(con, "papers", dict(row), "INSERT OR IGNORE")
        for row in private.execute("SELECT * FROM hypotheses ORDER BY rowid"):
            if row["id"] not in keys["hypotheses"]:
                _insert(con, "hypotheses", dict(row), "INSERT OR IGNORE")
        query = "SELECT * FROM proposals WHERE id > ? ORDER BY id"
        for row in private.execute(query, (base["proposals"],)).fetchall():
            proposal = {k: row[k] for k in row.keys() if k != "id"}
            if proposal["status"] == "stored":
                factor = private.execute(
                    "SELECT * FROM factors WHERE id = ?", (proposal["factor_id"],)
                ).fetchone()
                if not _insert(con, "factors", dict(factor), "INSERT OR IGNORE"):
                    proposal.update(status="duplicate", reason="already in the store")
                    changed["stored"] -= 1
                    changed["duplicate"] += 1
            _insert(con, "proposals", proposal)
        query = "SELECT * FROM usage WHERE id > ? ORDER BY id"
        for row in private.execute(query, (base["usage"],)).fetchall():
            _insert(con, "usage", {k: row[k] for k in row.keys() if k != "id"})
        if _marker is not None:
            table, index = _marker
            con.execute(f'INSERT INTO "{table}" (paper_index) VALUES (?)', (index,))
    return changed


def run_isolated(
    con: sqlite3.Connection, source, client, config: FactorConfig, concurrency: int
) -> RunSummary:
    """`pipeline.run` with every paper against the starting store, `concurrency` papers
    in flight at once (see the module docstring). `client` must be safe to call from
    several threads. Interrupted runs persist already-started work before re-raising."""
    papers = list(source.papers())
    workers = resolve_concurrency(concurrency, len(papers))
    zoo = default_zoo()
    name = type(source).__name__
    before = _counts(con)
    base = {"proposals": _max_id(con, "proposals"), "usage": _max_id(con, "usage")}
    keys = {
        "papers": {r[0] for r in con.execute("SELECT key FROM papers")},
        "hypotheses": {r[0] for r in con.execute("SELECT id FROM hypotheses")},
    }
    starting = _snapshot(con)
    snapshot_lock = Lock()
    stopped = Event()
    summary = RunSummary()
    completed = {}
    consumed = set()
    marker = f"paper_pool_{uuid.uuid4().hex}"
    con.execute(f'CREATE TEMP TABLE "{marker}" (paper_index INTEGER PRIMARY KEY)')

    def copied(index):
        return (
            con.execute(f'SELECT 1 FROM "{marker}" WHERE paper_index=?', (index,)).fetchone()
            is not None
        )

    def task(index: int):
        private = None
        out = RunSummary()
        error = None
        try:
            if stopped.is_set():
                return None, out, None
            with snapshot_lock:
                private = _snapshot(starting)
            with slot(index):
                process_paper(private, papers[index], client, config, zoo, name, out)
        except BaseException as exc:
            stopped.set()
            error = exc
        completed[index] = (private, out, error)
        return completed[index]

    def merge(index, outcome):
        private, result, error = outcome
        try:
            if private is None:
                if error is not None:
                    raise error
                return  # Cancelled before starting: no rows, calls or handoff.
            touched = {t for t, n in _counts(private).items() if n != before.get(t, 0)}
            if not touched <= set(_PAPER_TABLES):
                raise RuntimeError(f"a paper wrote outside the paper tables: {sorted(touched)}")
            # The TEMP marker commits atomically with the batch. After an
            # interrupted commit, retry only if both marker and batch rolled back.
            changed = (
                Counter()
                if copied(index)
                else copy_new_rows(private, con, base, keys, _marker=(marker, index))
            )
            summary.hypotheses.update(result.hypotheses)
            summary.outcomes.update(result.outcomes)
            summary.outcomes.update(changed)
            summary.outcomes = +summary.outcomes
            summary.usage.update(result.usage)
            if error is not None:
                raise error
            key = papers[index]["key"]
            if key in result.papers_failed:
                summary.papers_failed[key] = result.papers_failed[key]
                return
            try:
                source.done([key])
                summary.papers_done.append(key)
            except Exception as exc:  # noqa: BLE001 - as pipeline.run
                summary.papers_failed[key] = f"{type(exc).__name__}: {exc}"
        finally:
            if private is None or copied(index):
                consumed.add(index)
                if private is not None:
                    private.close()

    try:
        if workers <= executor.INLINE:
            for index in range(len(papers)):
                merge(index, task(index))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {}
                try:
                    for i in range(min(workers, len(papers))):
                        pending[i] = pool.submit(task, i)
                    next_index = len(pending)
                    while pending:
                        index = next(iter(pending))
                        outcome = pending[index].result()
                        del pending[index]
                        merge(index, outcome)
                        if not stopped.is_set() and next_index < len(papers):
                            pending[next_index] = pool.submit(task, next_index)
                            next_index += 1
                except BaseException:
                    stopped.set()
                    for future in pending.values():
                        future.cancel()
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
    except BaseException as exc:
        stopped.set()
        errors = []
        for index in sorted(completed):
            if index in consumed:
                continue
            try:
                merge(index, completed[index])
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "paper interruption and drain errors", [exc, *errors]
            ) from None
        raise
    finally:
        for private, _, _ in completed.values():
            if private is not None:
                private.close()
        con.execute(f'DROP TABLE "{marker}"')
        starting.close()
    return summary
