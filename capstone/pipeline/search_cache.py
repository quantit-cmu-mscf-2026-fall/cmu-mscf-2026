"""The search period's rows, cached on disk between runs.

`CachedRows(loader, source_version)` is a panel loader (`(start, end) -> rows`): the first
call for a (loader, `source_version`, start, end) reads through `loader` and keeps
the rows as parquet under `data_cache/search_rows/` (gitignored; `CAPSTONE_SEARCH_CACHE`
moves it); a later call with the same key reads the file instead. A changed key
(another `sd.data_version()`, period or loader) is another file, so it rebuilds.

The cache is lossless where it matters: next to each file is the digest of its rows
(`rows.rows_digest`, the digest the data version names), checked on every read; a
file whose rows no longer give it is ignored and rebuilt. So a cached run registers
the same data version as an uncached one. Like the search's own loader, it refuses any
end after `DISCOVERY_END` before reading anything: it never holds holdout rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from capstone.factors.evaluation import DISCOVERY_END, HoldoutError
from capstone.pipeline.rows import REPO, rows_digest

__all__ = ["CACHE_ENV", "CachedRows", "default_directory"]

CACHE_ENV = "CAPSTONE_SEARCH_CACHE"
FORMAT = 1  # bump when the cached file's layout changes


def default_directory() -> Path:
    override = os.environ.get(CACHE_ENV)
    return Path(override) if override else REPO / "data_cache" / "search_rows"


class CachedRows:
    """A caching panel loader (module docstring); `hits` and `misses` count reads."""

    def __init__(self, loader, source_version: str, directory: str | Path | None = None):
        self.loader = loader
        self.source_version = source_version
        self.directory = Path(directory) if directory is not None else default_directory()
        self.hits = self.misses = 0

    def key(self, start, end) -> str:
        module = getattr(self.loader, "__module__", "")
        name = f"{module}.{getattr(self.loader, '__qualname__', type(self.loader).__name__)}"
        payload = {
            "format": FORMAT,
            "loader": name,
            "source_version": self.source_version,
            "start": None if start is None else str(start),
            "end": str(end),
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def __call__(self, start, end) -> pd.DataFrame:
        if pd.Timestamp(end) > pd.Timestamp(DISCOVERY_END):
            raise HoldoutError(f"end {end} is after the discovery period ({DISCOVERY_END})")
        key = self.key(start, end)
        data, meta = self.directory / f"{key}.parquet", self.directory / f"{key}.json"
        if data.is_file() and meta.is_file():
            rows = pd.read_parquet(data)
            if json.loads(meta.read_text(encoding="utf-8")).get("digest") == rows_digest(rows):
                self.hits += 1
                return rows
        self.misses += 1
        rows = self.loader(start, end)
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = data.with_suffix(f".{os.getpid()}.partial")
        rows.to_parquet(partial)
        digest = rows_digest(rows)
        if rows_digest(pd.read_parquet(partial)) == digest:  # round-trips exactly
            os.replace(partial, data)
            meta.write_text(json.dumps({"digest": digest}) + "\n", encoding="utf-8")
        else:  # a dtype parquet can't keep: never serve it from the cache
            partial.unlink()
        return rows
