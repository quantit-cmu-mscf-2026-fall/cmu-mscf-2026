"""Small pieces the search-rows cache shares with the rest of the pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

__all__ = ["REPO", "rows_digest", "sorted_rows"]

REPO = Path(__file__).resolve().parents[2]


def rows_digest(rows: pd.DataFrame) -> str:
    """12 hex digits of the SHA-256 of the rows as CSV: the same rows give the same
    digest in any process and on pandas 2 and 3 (whose string hashes differ)."""
    text = rows.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sorted_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(["date", "permno"], kind="stable").reset_index(drop=True)
