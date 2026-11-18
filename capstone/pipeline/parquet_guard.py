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


def _temporal_leaves(dtype, path, repeated=False):
    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        yield path, dtype, repeated
    elif pa.types.is_struct(dtype):
        for field in dtype:
            yield from _temporal_leaves(field.type, (*path, field.name), repeated)
    elif pa.types.is_dictionary(dtype):
        yield from _temporal_leaves(dtype.value_type, path, repeated)
    elif (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    ):
        yield from _temporal_leaves(dtype.value_type, path, True)
    elif pa.types.is_map(dtype):
        yield from _temporal_leaves(dtype.key_type, path, True)
        yield from _temporal_leaves(dtype.item_type, path, True)
    elif pa.types.is_nested(dtype):
        # Fail closed: a nested type this walker does not handle (list_view, unions, ...)
        # could hide a date, so it is treated as an unqualified repeated date.
        yield path, dtype, True


def verify_discovery_dates(path, role):
    """Check every date/timestamp-typed leaf using footers, without reading any observations.

    Scheduled events.pay_date is exempt; list/map dates have no qualified row
    filter and are refused. All-null dates are retained for caller qualification.
    """
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
    fields = []
    for name in columns:
        if name not in file.schema_arrow.names:
            raise ArchivedInputBlocked("archive_date_column_missing", role)
        dtype = file.schema_arrow.field(name).type
        fields.append(((name,), dtype))
    for field in file.schema_arrow:
        if role == "events" and field.name == "pay_date":
            continue
        for parts, dtype, repeated in _temporal_leaves(field.type, (field.name,)):
            if repeated:
                raise ArchivedInputBlocked("archive_date_type_unqualified", role)
            if parts not in [path for path, _ in fields]:
                fields.append((parts, dtype))
    physical_paths = [file.schema.column(i).path for i in range(len(file.schema))]
    for parts, dtype in fields:
        name = ".".join(parts)
        if not file.metadata.num_rows and pa.types.is_null(dtype):
            continue
        if not (pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)) or (
            pa.types.is_timestamp(dtype) and dtype.tz is not None
        ):
            raise ArchivedInputBlocked("archive_date_type_unqualified", role)
        if physical_paths.count(name) != 1:
            raise ArchivedInputBlocked("archive_date_column_ambiguous", role)
        offset = physical_paths.index(name)
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
                continue  # Nulls establish no future date; qualification remains the caller's job.
            if not stats.has_min_max:
                raise ArchivedInputBlocked("archive_date_bounds_unavailable", role)
            last = pd.Timestamp(stats.max)
            if last.tz is not None:
                raise ArchivedInputBlocked("archive_date_type_unqualified", role)
            if pd.isna(last) or last >= NEXT_PERIOD:
                raise ArchivedInputBlocked("archive_crosses_discovery_boundary", role)
    evidence = {
        "role": role,
        "policy": POLICY,
        "date_columns": [".".join(parts) for parts, _ in fields],
    }
    if any(len(parts) > 1 for parts, _ in fields):
        evidence["date_paths"] = [list(parts) for parts, _ in fields]
    return evidence


def discovery_filter(path, evidence):
    """Defence in depth after footer preflight; retain nulls for qualification."""
    schema = pq.ParquetFile(path).schema_arrow
    expression = None
    paths = evidence.get("date_paths", [[name] for name in evidence["date_columns"]])
    for parts in paths:
        dtype = schema.field(parts[0]).type
        for name in parts[1:]:
            dtype = dtype.field(name).type
        if pa.types.is_null(dtype):
            continue
        column = ds.field(tuple(parts))
        bound = NEXT_PERIOD.date() if pa.types.is_date(dtype) else NEXT_PERIOD
        term = (column < bound) | column.is_null()
        expression = term if expression is None else expression & term
    return expression
