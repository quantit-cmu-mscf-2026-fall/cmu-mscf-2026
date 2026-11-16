"""Versioned private historical input archives; no broker state or licensed data in Git.

An archive is a folder of parquet frames plus `historical-inputs.json` (their digests, the
runtime versions and the discovery cutoff). It is written once and read back only through
`shared_data.load`: the folder stands in for the shared cache for the duration of the read,
so one audited loader (with its date cut applied inside the parquet read) serves both.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from capstone import shared_data as sd
from capstone.factors.evaluation import Panel
from capstone.pipeline.historical import CUTOFF, HistoricalInputs, validate_events
from capstone.pipeline.parquet_guard import verify_discovery_dates
from capstone.pipeline.study import file_digest, write_once
from capstone.release import runtime_versions
from capstone.strategy.research import digest

# The column each role's rows are cut on; every other role is indexed by `date`.
DATE_COLUMNS = {"events": "event_date", "fundamentals": "datadate"}
KNOWLEDGE_COLUMNS = {"fundamentals": ("published_at", "vintage_at")}


def save_inputs(path, inputs):
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("historical input archives cannot be overwritten")
    root.mkdir(parents=True, exist_ok=True)
    frames = {
        **{"field:" + n: frame for n, frame in inputs.panel.fields.items()},
        **{
            "availability:" + n: frame.to_frame("known_at")
            for n, frame in inputs.panel.availability.items()
        },
        "universe": inputs.panel.universe,
        "half_spread": inputs.panel.half_spread,
        "opens": inputs.opens,
        "closes": inputs.closes,
        "events": inputs.events,
        "total_returns": inputs.total_returns,
        "half_spread_bps": inputs.half_spread_bps,
    }
    for name in ("sic", "company", "fundamentals"):
        if getattr(inputs.panel, name) is not None:
            frames[name] = getattr(inputs.panel, name)
    if inputs.symbols is not None:
        frames["symbols"] = inputs.symbols
    manifest = {}
    for i, (label, frame) in enumerate(sorted(frames.items())):
        name = f"input-{i}.parquet"
        if label not in DATE_COLUMNS:  # the date index is stored as the `date` column
            frame = frame.rename_axis("date")
        frame.to_parquet(root / name)
        manifest[label] = {"file": name, "sha256": file_digest(root / name)}
    metadata = {
        "schema_version": 1,
        "runtime": runtime_versions(),
        "frames": manifest,
        "data_version": inputs.panel.data_version,
        "qualification": inputs.qualification,
        "calendar": [str(d.date()) for d in inputs.panel.calendar],
        "cutoff": str(CUTOFF.date()),
    }
    metadata["input_hash"] = digest(metadata)
    write_once(root / "historical-inputs.json", metadata)
    return root


@contextmanager
def _as_shared_cache(root):
    """Point `shared_data` at the archive folder, then restore the caller's cache."""
    saved = os.environ.get("CAPSTONE_CACHE_DIR")
    os.environ["CAPSTONE_CACHE_DIR"] = str(root)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("CAPSTONE_CACHE_DIR", None)
        else:
            os.environ["CAPSTONE_CACHE_DIR"] = saved


def _read(file, label):
    return sd.load(
        file.stem,
        end=str(CUTOFF.date()),
        auto_sync=False,
        date_column=DATE_COLUMNS.get(label, "date"),
        knowledge_columns=KNOWLEDGE_COLUMNS.get(label, ()),
    )


def load_inputs(path):
    root = Path(path).resolve()
    metadata = json.loads((root / "historical-inputs.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("cutoff") != "2020-12-31":
        raise ValueError("unsupported historical input archive")
    if digest({k: v for k, v in metadata.items() if k != "input_hash"}) != metadata["input_hash"]:
        raise ValueError("historical input metadata changed")
    if metadata["runtime"] != runtime_versions():
        raise ValueError("historical replay requires its pinned environment")
    verified = {}
    for label, item in metadata["frames"].items():
        file = (root / item["file"]).resolve()
        if file.parent != root or file_digest(file) != item["sha256"]:
            raise ValueError("historical input archive integrity failure")
        verified[label] = (file, verify_discovery_dates(file, label))
    # A later bad file must block before any earlier frame is materialized.
    frames = {}
    with _as_shared_cache(root):
        for label, (file, _) in verified.items():
            frames[label] = _read(file, label)
            if label not in {"fundamentals", "events"} and frames[label].index.max() > CUTOFF:
                raise ValueError("historical input archive crosses the sealed holdout")
    if any(
        file_digest(file) != metadata["frames"][label]["sha256"]
        for label, (file, _) in verified.items()
    ):
        raise ValueError("historical input archive changed while materializing")
    fundamentals = frames.get("fundamentals")
    if fundamentals is not None:
        for col in ("datadate", "published_at", "vintage_at"):
            if (pd.to_datetime(fundamentals[col]) > CUTOFF).any():
                raise ValueError("historical accounting archive crosses the sealed holdout")
    panel = Panel(
        {n.split(":", 1)[1]: f for n, f in frames.items() if n.startswith("field:")},
        frames["universe"],
        frames["half_spread"],
        frames.get("sic"),
        metadata["data_version"],
        pd.DatetimeIndex(metadata["calendar"]),
        {
            n.split(":", 1)[1]: f.iloc[:, 0]
            for n, f in frames.items()
            if n.startswith("availability:")
        },
        frames.get("company"),
        fundamentals,
    )
    if not panel.data_version.startswith("synthetic-"):
        calendar = panel.calendar
        if (
            calendar.has_duplicates
            or not calendar.is_monotonic_increasing
            or sum(calendar > CUTOFF) != 1
            or calendar[0].year != 1990
            or panel.universe.index.min() != calendar[0]
            or panel.universe.index.max() != calendar[calendar <= CUTOFF][-1]
        ):
            raise ValueError("historical archive needs the complete 1990–2020 source calendar")
    return HistoricalInputs(
        panel,
        frames["opens"],
        frames["closes"],
        validate_events(frames["events"]),
        frames["total_returns"],
        frames["half_spread_bps"],
        metadata["qualification"],
        symbols=frames.get("symbols"),
    )
