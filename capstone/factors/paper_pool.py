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

After the concurrent part, each paper's new rows (paper, hypotheses, proposals with
their factors, usage) are copied into `con` in source order, exactly as the store's
own writers would have made them: a factor a previous paper already stored becomes a
"duplicate" proposal ("already in the store"). The source's hand-off (`done`) is
called in the same order. So the store, and everything computed from it, is the
same at every concurrency, 1 included, for the same model answers.

`slot()` names the paper a call belongs to (a context variable set inside each
paper's task), so a recorder can keep its records in paper order
(a usage recorder) instead of completion order.
"""

from __future__ import annotations

import contextvars
import sqlite3
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

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
    `CAPSTONE_MAX_WORKERS` cap; never more than the papers, never fewer than one."""
    if configured < executor.AUTO:
        raise ValueError(f"concurrency must be >= {executor.AUTO}, got {configured}")
    caps = [papers, configured or executor.detect().cores]
    caps += [c for c in (executor.env_cap(),) if c]
    return max(executor.INLINE, min(caps))


def _snapshot(con: sqlite3.Connection) -> sqlite3.Connection:
    """An in-memory copy of the store, usable from one worker thread."""
    private = sqlite3.connect(":memory:", check_same_thread=False)
    con.backup(private)
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
    private: sqlite3.Connection, con: sqlite3.Connection, base: dict[str, int], keys: dict
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
    return changed


def run_isolated(
    con: sqlite3.Connection, source, client, config: FactorConfig, concurrency: int
) -> RunSummary:
    """`pipeline.run` with every paper against the starting store, `concurrency` papers
    in flight at once (see the module docstring). `client` must be safe to call from
    several threads (`BudgetedClient` is)."""
    papers = list(source.papers())
    zoo = default_zoo()
    name = type(source).__name__
    before = _counts(con)
    base = {"proposals": _max_id(con, "proposals"), "usage": _max_id(con, "usage")}
    keys = {
        "papers": {r[0] for r in con.execute("SELECT key FROM papers")},
        "hypotheses": {r[0] for r in con.execute("SELECT id FROM hypotheses")},
    }
    privates = [_snapshot(con) for _ in papers]

    def task(index: int) -> RunSummary:
        out = RunSummary()
        with slot(index):
            process_paper(privates[index], papers[index], client, config, zoo, name, out)
        return out

    workers = resolve_concurrency(concurrency, len(papers))
    try:
        if workers <= executor.INLINE:
            results = [task(i) for i in range(len(papers))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(task, range(len(papers))))
        summary = RunSummary()
        for index, (paper, result) in enumerate(zip(papers, results, strict=True)):
            private = privates[index]
            touched = {t for t, n in _counts(private).items() if n != before.get(t, 0)}
            if not touched <= set(_PAPER_TABLES):
                raise RuntimeError(f"a paper wrote outside the paper tables: {sorted(touched)}")
            changed = copy_new_rows(private, con, base, keys)
            summary.hypotheses.update(result.hypotheses)
            summary.outcomes.update(result.outcomes)
            summary.outcomes.update(changed)
            summary.outcomes = +summary.outcomes
            summary.usage.update(result.usage)
            key = paper["key"]
            if key in result.papers_failed:
                summary.papers_failed[key] = result.papers_failed[key]
                continue
            try:
                source.done([key])
                summary.papers_done.append(key)
            except Exception as exc:  # noqa: BLE001 - as pipeline.run
                summary.papers_failed[key] = f"{type(exc).__name__}: {exc}"
    finally:
        for private in privates:
            private.close()
    return summary
