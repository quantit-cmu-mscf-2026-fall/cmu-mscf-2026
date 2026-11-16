"""Optional discovery-row cache filled only through shared_data.

``CachedRows(file_name, columns)`` is a ``(start, end) -> rows`` loader. Each
call syncs the shared manifest, then keys by ``sd.data_version()``, file,
columns and period. Derived parquet files live under gitignored
``data_cache/search_rows`` (or ``CAPSTONE_SEARCH_CACHE``); no licensed data
belongs in git. This is the S-15d exception documented in docs/shared_data.md.

Missing/unbounded ends are refused before I/O. Loaded and cached rows must
have discovery dates only. A cache entry is served only if its digest and
dtypes match; a frame parquet cannot round-trip exactly is never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import pandas as pd
from pyarrow import ArrowException

from capstone import shared_data as sd
from capstone.factors.evaluation import DISCOVERY_END, HoldoutError
from capstone.pipeline.rows import REPO, rows_digest

__all__ = ["CACHE_ENV", "CachedRows", "default_directory"]

CACHE_ENV = "CAPSTONE_SEARCH_CACHE"
FORMAT = 2


def default_directory() -> Path:
    override = os.environ.get(CACHE_ENV)
    return Path(override) if override else REPO / "data_cache" / "search_rows"


def _check_dates(rows: pd.DataFrame) -> None:
    dates = pd.to_datetime(rows["date"], errors="raise")
    if dates.isna().any() or (dates > pd.Timestamp(DISCOVERY_END)).any():
        raise HoldoutError("search rows must have known dates within the discovery period")


class CachedRows:
    """Cache a shared file's selected columns; ``hits``/``misses`` count reads.

    The former arbitrary ``loader, source_version`` arguments are deliberately
    unsupported: callers supply the shared file name and explicit columns instead.
    """

    def __init__(self, name: str, columns: list[str], directory: str | Path | None = None):
        if not isinstance(name, str) or not name:
            raise TypeError("name must identify a shared file, not a loader callable")
        if isinstance(columns, str) or not columns or "date" not in columns:
            raise ValueError("columns must explicitly select shared-file columns including date")
        self.name, self.columns = name, list(columns)
        self.directory = Path(directory) if directory is not None else default_directory()
        self.hits = self.misses = 0

    def key(self, start, end, source_version: str) -> str:
        payload = {
            "format": FORMAT,
            "file": self.name,
            "columns": self.columns,  # order is observable in the returned frame
            "source_version": source_version,
            "start": None if start is None else str(start),
            "end": str(end),
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def __call__(self, start, end) -> pd.DataFrame:
        end_date = pd.Timestamp(end)
        if pd.isna(end_date) or end_date > pd.Timestamp(DISCOVERY_END):
            raise HoldoutError(f"end must be bounded by the discovery period ({DISCOVERY_END})")
        sd.sync()
        version = sd.data_version()
        key = self.key(start, end, version)
        data, meta = self.directory / f"{key}.parquet", self.directory / f"{key}.json"
        if data.is_file() and meta.is_file():
            try:
                rows = pd.read_parquet(data)
                info = json.loads(meta.read_text(encoding="utf-8"))
                valid = info == {"digest": rows_digest(rows), "dtypes": list(map(str, rows.dtypes))}
            except (OSError, ValueError, ArrowException):
                valid = False
            if valid:
                _check_dates(rows)
                self.hits += 1
                return rows
        self.misses += 1
        rows = sd.load(self.name, columns=self.columns, start=start, end=end, auto_sync=False)
        _check_dates(rows)
        # Another process may have synced while loading; don't label those rows with an old key.
        if sd.data_version() != version:
            return rows
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = data.with_name(f".{key}.{uuid.uuid4().hex}.partial.parquet")
        meta_partial = partial.with_suffix(".json")
        try:
            rows.to_parquet(partial)
            try:
                pd.testing.assert_frame_equal(rows, pd.read_parquet(partial), check_exact=True)
            except AssertionError:
                return rows
            info = {"digest": rows_digest(rows), "dtypes": list(map(str, rows.dtypes))}
            meta_partial.write_text(json.dumps(info) + "\n", encoding="utf-8")
            os.replace(partial, data)
            os.replace(meta_partial, meta)
        finally:
            partial.unlink(missing_ok=True)
            meta_partial.unlink(missing_ok=True)
        return rows
