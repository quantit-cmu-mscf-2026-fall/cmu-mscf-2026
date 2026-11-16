"""Inspect archived discovery date bounds before materializing any observations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

NEXT_PERIOD = pd.Timestamp("2021-01-01")
POLICY = "discovery-parquet-footer-boundary-v1"


class ArchivedInputBlocked(ValueError):
    def __init__(self, reason, role):
        self.reason, self.role = reason, role
        super().__init__(f"{reason}:{role}")


def verify_discovery_dates(path, role):
    """Footer statistics are source metadata; no Arrow/Pandas row read occurs here."""
    file = pq.ParquetFile(Path(path))
    if role == "events":
        columns = ("event_date",)
    elif role == "fundamentals":
        columns = ("datadate", "published_at", "vintage_at")
    elif role in {
        "universe",
        "half_spread",
        "opens",
        "closes",
        "total_returns",
        "half_spread_bps",
        "sic",
        "company",
        "symbols",
    } or re.fullmatch(r"(?:field|availability):[a-z][a-z0-9_]*", role):
        metadata = file.schema_arrow.metadata or {}
        try:
            indexes = json.loads(metadata[b"pandas"])["index_columns"]
        except (KeyError, ValueError, TypeError) as error:
            raise ArchivedInputBlocked("archive_date_index_metadata_missing", role) from error
        if not isinstance(indexes, list) or len(indexes) != 1 or not isinstance(indexes[0], str):
            raise ArchivedInputBlocked("archive_date_index_unsupported", role)
        columns = (indexes[0],)
        if role.startswith("availability:"):
            columns += ("known_at",)
    else:
        raise ArchivedInputBlocked("archive_role_unsupported", role)
    columns += tuple(
        name
        for name in ("known_at", "published_at", "vintage_at")
        if name in file.schema_arrow.names and name not in columns
    )
    for name in columns:
        if name not in file.schema_arrow.names:
            raise ArchivedInputBlocked("archive_date_column_missing", role)
        dtype = file.schema_arrow.field(name).type
        if not file.metadata.num_rows and pa.types.is_null(dtype):
            continue
        if not (pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)) or (
            pa.types.is_timestamp(dtype) and dtype.tz is not None
        ):
            raise ArchivedInputBlocked("archive_date_type_unqualified", role)
        physical_names = file.schema.names
        if physical_names.count(name) != 1:
            raise ArchivedInputBlocked("archive_date_column_ambiguous", role)
        offset = physical_names.index(name)
        for group in range(file.metadata.num_row_groups):
            rows = file.metadata.row_group(group)
            column = rows.column(offset)
            if column.path_in_schema != name:
                raise ArchivedInputBlocked("archive_date_column_ambiguous", role)
            stats = column.statistics
            if not rows.num_rows:
                continue
            if stats is None:
                raise ArchivedInputBlocked("archive_date_bounds_unavailable", role)
            if stats.null_count == rows.num_rows:
                continue  # Missing availability/data is refused by its own qualification check.
            if not stats.has_min_max:
                raise ArchivedInputBlocked("archive_date_bounds_unavailable", role)
            last = pd.Timestamp(stats.max)
            if last.tz is not None:
                raise ArchivedInputBlocked("archive_date_type_unqualified", role)
            if pd.isna(last) or last >= NEXT_PERIOD:
                raise ArchivedInputBlocked("archive_crosses_discovery_boundary", role)
    return {"role": role, "policy": POLICY, "date_columns": list(columns)}


def discovery_filter(path, evidence):
    """Defence in depth after footer preflight; retain nulls for qualification."""
    schema = pq.ParquetFile(path).schema_arrow
    expression = None
    for name in evidence["date_columns"]:
        if pa.types.is_null(schema.field(name).type):
            continue
        column = ds.field(name)
        bound = NEXT_PERIOD.date() if pa.types.is_date(schema.field(name).type) else NEXT_PERIOD
        term = (column < bound) | column.is_null()
        expression = term if expression is None else expression & term
    return expression
