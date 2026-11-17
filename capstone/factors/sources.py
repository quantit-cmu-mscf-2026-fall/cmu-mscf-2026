"""Where papers come from: a hand-written file, or paperlog's queue.

A source gives plain paper dicts (at least `key` and `title`) and is told
which papers were fully processed. Only paperlog keeps that state: a paper
is marked ingested there after its hypotheses and factors are stored, never
before, so a failed run hands the same paper out again next time.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path


class FileSource:
    """Papers listed in a TOML file, one `[[papers]]` table each.

    Each needs `key` (for example "arxiv:2502.16789") and `title`; `abstract`,
    `url`, `citation` and anything else are passed through to the prompt.
    """

    def __init__(self, path: str | Path, keys: Iterable[str] | None = None):
        self.path = Path(path)
        self.keys = list(keys) if keys else None

    def papers(self) -> list[dict]:
        with open(self.path, "rb") as handle:
            papers = tomllib.load(handle).get("papers", [])
        for paper in papers:
            missing = [k for k in ("key", "title") if not str(paper.get(k, "")).strip()]
            if missing:
                raise ValueError(f"{self.path}: a paper is missing {', '.join(missing)}")
        if self.keys is None:
            return papers
        by_key = {paper["key"]: paper for paper in papers}
        unknown = [key for key in self.keys if key not in by_key]
        if unknown:
            raise KeyError(f"{self.path} has no paper {', '.join(unknown)}")
        return [by_key[key] for key in self.keys]

    def done(self, keys: Iterable[str]) -> None:
        """A file keeps no state; re-running it proposes again (duplicates are recorded)."""


class PaperlogSource:
    """Papers paperlog has not yet handed to a downstream consumer.

    Needs the `paperlog` package importable (it lives in research/paperlog or
    in a separate install). `db_path` defaults to paperlog's own setting,
    `PAPERLOG_DB`. With `keys`, exactly those papers are taken from the log,
    pending or not, which is how one paper is run on purpose.
    """

    def __init__(
        self,
        db_path: str | None = None,
        min_relevance: int = 0,
        limit: int = 20,
        keys: Iterable[str] | None = None,
    ):
        try:
            from paperlog import api
        except ImportError as exc:
            raise ImportError(
                "paperlog is not importable; install research/paperlog or use a file source"
            ) from exc
        self._api = api
        self.db_path = db_path
        self.min_relevance = min_relevance
        self.limit = limit
        self.keys = list(keys) if keys else None

    def papers(self) -> list[dict]:
        if self.keys is None:
            return self._api.pending(
                min_relevance=self.min_relevance, limit=self.limit, db_path=self.db_path
            )
        papers = [self._api.get(key, db_path=self.db_path) for key in self.keys]
        unknown = [key for key, paper in zip(self.keys, papers, strict=True) if paper is None]
        if unknown:
            raise KeyError(f"paperlog has no paper {', '.join(unknown)}; add it with paperlog add")
        return papers

    def done(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        if keys:
            self._api.mark_ingested(keys, db_path=self.db_path)
