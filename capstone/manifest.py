"""Content fingerprints for a data panel, so two machines can prove they match.

Sharing a data file gives you sameness by assumption. This module gives you
sameness you can check: a short digest per panel that is identical if and only
if the underlying numbers are, and that travels as text rather than as data.

The failure it exists to catch is quiet. Someone refreshes a pull after a
vendor revision, the refreshed copy reaches three of five laptops, and the
results stop agreeing six weeks later with no obvious cause. A digest compared
at the start of a run turns that into an error message on day one.

Hashing the cache file itself does NOT work, which is worth stating because it
is the obvious first instinct: parquet bytes vary with writer version,
compression and row order, so two byte-different files routinely hold identical
data. The digests here are taken over the derived panels -- dates x securities,
the shape research actually consumes -- which is the level at which "the same
data" is a meaningful claim.

The primitives take a plain mapping of name -> DataFrame and depend on nothing
else in this package, so they apply equally to a CRSP pull, a Ken French factor
table, or a synthetic control panel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from capstone.data import CACHE_DIR

# Bumped when the digest definition changes in a way that alters its output.
# A manifest written under a different revision is not comparable, and the
# verifier says so rather than reporting a spurious mismatch.
DIGEST_REVISION = 1

# The panels a CRSP pull yields. Order is fixed so a regenerated manifest
# diffs cleanly against its predecessor.
CRSP_PANELS = ("open", "high", "low", "close", "volume", "returns")

# Field separator inside the hashed label stream. A non-printing byte cannot
# occur in a ticker or a date, so labels cannot be spliced to collide.
_SEP = "\x1f"


def frame_digest(frame: pd.DataFrame) -> str:
    """SHA-256 over a frame's labels and values, normalised for portability.

    Normalisation matters more than the hash choice. Index and columns are
    sorted so two pulls differing only in row order agree; values are cast to a
    single concrete dtype so pandas' nullable extension types (`Float64` arrives
    from parquet on sparse columns) do not hash differently from the numpy
    floats holding the same numbers.
    """
    ordered = frame.sort_index().reindex(sorted(frame.columns, key=str), axis=1)

    digest = hashlib.sha256()
    digest.update(f"rev{DIGEST_REVISION}".encode())
    digest.update(_SEP.join(str(c) for c in ordered.columns).encode())
    digest.update(_SEP.join(str(i) for i in ordered.index).encode())

    if all(is_bool_dtype(dtype) for dtype in ordered.dtypes):
        values = ordered.to_numpy(dtype=np.bool_)
    else:
        # Float64 -> float64 round-trip also absorbs object columns holding pd.NA.
        values = ordered.astype("Float64").astype("float64").to_numpy()
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def panel_digests(panels: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """Digest each named panel. Keys are sorted so output is deterministic."""
    return {name: frame_digest(panels[name]) for name in sorted(panels)}


def panel_coverage(panels: Mapping[str, pd.DataFrame]) -> dict:
    """Serializable dimensions of a panel set, for human eyes in the manifest.

    Coverage is READ OFF the data, never declared by the caller. A hand-written
    date range is a second source of truth that drifts from the pull it claims
    to describe; this one cannot.
    """
    reference = panels[sorted(panels)[0]]
    index = pd.to_datetime(reference.index)
    return {
        "start": str(index.min()),
        "end": str(index.max()),
        "n_dates": int(len(reference)),
        "n_securities": int(reference.shape[1]),
        "panels": sorted(panels),
    }


def build_manifest(panels: Mapping[str, pd.DataFrame], *, source: str) -> dict:
    """Describe a panel set completely enough to detect any change in it."""
    if not panels:
        raise ValueError("cannot build a manifest from zero panels")
    return {
        "digest_revision": DIGEST_REVISION,
        "source": source,
        "coverage": panel_coverage(panels),
        "digests": panel_digests(panels),
        "library_versions": {"pandas": pd.__version__, "numpy": np.__version__},
    }


def write_manifest(panels: Mapping[str, pd.DataFrame], path: str | Path, *, source: str) -> dict:
    """Build a manifest and write it as sorted, diff-friendly JSON."""
    built = build_manifest(panels, source=source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return built


def read_manifest(path: str | Path) -> dict:
    """Load a manifest written by `write_manifest`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(expected: dict, actual: dict) -> list[str]:
    """Names of panels whose digests disagree, sorted.

    A digest computed under a different `DIGEST_REVISION` is not comparable, so
    that case raises rather than returning a mismatch list that would read as a
    data problem when it is a code-version problem. A panel present on one side
    only also counts as a disagreement -- a dropped panel is a real change.
    """
    if expected.get("digest_revision") != actual.get("digest_revision"):
        raise ValueError(
            f"manifest digest_revision {expected.get('digest_revision')!r} cannot be compared "
            f"with {actual.get('digest_revision')!r}; regenerate the manifest"
        )
    expected_digests = expected.get("digests", {})
    actual_digests = actual.get("digests", {})
    return sorted(
        name
        for name in set(expected_digests) | set(actual_digests)
        if expected_digests.get(name) != actual_digests.get(name)
    )


def verify(panels: Mapping[str, pd.DataFrame], expected: dict) -> tuple[bool, list[str]]:
    """Check panels against a manifest. Returns (matches, differing panel names)."""
    actual = build_manifest(panels, source=expected.get("source", ""))
    mismatched = compare(expected, actual)
    return (not mismatched), mismatched


def resolve_cache_path(name: str) -> Path:
    """Accept a bare cache name or a parquet path; fail listing what IS cached.

    An unhelpful "file not found" here is a real time sink: cache names are
    chosen by whoever ran the pull, so the reader's first guess is often a
    near-miss rather than a wrong idea.
    """
    candidate = Path(name)
    if candidate.suffix == ".parquet" and candidate.exists():
        return candidate
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return path
    available = sorted(p.stem for p in CACHE_DIR.glob("*.parquet")) if CACHE_DIR.exists() else []
    raise FileNotFoundError(
        f"no cached dataset {name!r} in {CACHE_DIR}; "
        f"available: {', '.join(available) if available else '(none)'}"
    )


def load_panels(path: str | Path, *, identifier: str = "permno") -> dict[str, pd.DataFrame]:
    """Build panels from a cache file of either shape.

    A tidy vendor extract (one row per date-security) needs pivoting; a file
    that is already dates x securities -- Ken French, sample prices -- is
    fingerprinted as the single panel it is.
    """
    path = Path(path)
    frame = pd.read_parquet(path)
    if "date" in frame.columns and identifier in frame.columns:
        return panels_from_crsp_daily(frame, identifier=identifier)
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    return {path.stem: frame.sort_index()}


def panels_from_crsp_daily(daily: pd.DataFrame, *, identifier: str = "permno") -> dict:
    """Pivot a tidy CRSP frame into the dates x securities panels to fingerprint.

    Keyed on PERMNO by default: CRSP reuses tickers across decades, so a panel
    keyed on ticker can silently collide, and two teammates comparing digests
    would be comparing differently-built objects.

    Prices are made positive -- a negative CRSP `prc` means there was no trade
    that day and the bid/ask midpoint was stored instead -- which matches what
    `wrds_loader.to_price_panel` does.
    """
    if identifier not in daily.columns:
        raise KeyError(f"identifier column {identifier!r} is absent")

    fields = {
        "open": "openprc",
        "high": "askhi",
        "low": "bidlo",
        "close": "prc",
        "volume": "vol",
        "returns": "ret",
    }
    available = {name: col for name, col in fields.items() if col in daily.columns}
    if not available:
        raise KeyError(f"CRSP frame has none of the expected columns: {sorted(fields.values())}")

    panels = {}
    for name, column in available.items():
        values = pd.to_numeric(daily[column], errors="coerce")
        if name in ("open", "high", "low", "close"):
            values = values.abs()
        panel = daily.assign(_v=values).pivot_table(
            index="date", columns=identifier, values="_v", aggfunc="last"
        )
        panel.index = pd.to_datetime(panel.index)
        panels[name] = panel.sort_index()
    return panels
