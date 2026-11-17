"""Target weights of a program strategy, as canonical JSON.

Loads a schema-6 `ProgramStrategySpec` file and a JSON panel (the format
`write_panel` writes), computes `target_weights` through the shared dispatch in
`capstone.strategy.contract`, and prints one line of canonical JSON (sorted
keys, weights fixed to `DIGEST_DECIMALS`), then its SHA-256. Run it in a fresh
process and compare the bytes with the same call in-process
(tests/test_program_fresh_process.py).

    python scripts/fresh_process_targets.py SPEC.toml PANEL.json AS_OF
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from capstone.factors.evaluation import Panel
from capstone.factors.research_contracts import canonical
from capstone.strategy.contract import load_spec, target_weights

# Weights are written to a fixed number of decimals so builds that differ in the last
# bits (~1e-17) give the same bytes.
DIGEST_DECIMALS = 12


def _fixed(x: float) -> str:
    """`x` to DIGEST_DECIMALS decimals, a negative zero written as zero."""
    return f"{round(float(x), DIGEST_DECIMALS) + 0.0:.{DIGEST_DECIMALS}f}"


def _dates(values):
    return [str(pd.Timestamp(day).date()) for day in values]


def _cells(values):
    return [
        None if isinstance(v, float) and math.isnan(v) else v for v in pd.Series(values).tolist()
    ]


def _encode(frame):
    if isinstance(frame, pd.Series):
        return {"index": _dates(frame.index), "values": _cells(frame)}
    return {
        "index": _dates(frame.index),
        "columns": [int(c) for c in frame.columns],
        "values": [_cells(row) for row in frame.to_numpy().tolist()],
    }


def _decode(data):
    index = pd.DatetimeIndex(data["index"])
    if "columns" not in data:
        return pd.Series(data["values"], index=index, dtype=float)
    columns = pd.Index(data["columns"], name="permno")
    frame = pd.DataFrame(data["values"], index=index, columns=columns)
    return frame if frame.dtypes.eq(bool).all() else frame.astype(float)


def write_panel(panel: Panel, path: Path) -> None:
    """The fields, universe, spreads, calendar and availability a program reads."""
    document = {
        "data_version": panel.data_version,
        "calendar": _dates(panel.calendar),
        "universe": _encode(panel.universe),
        "half_spread": _encode(panel.half_spread),
        "fields": {name: _encode(frame) for name, frame in panel.fields.items()},
        "availability": {
            name: {
                "index": _dates(s.index),
                "values": [None if pd.isna(v) else str(v.date()) for v in s],
            }
            for name, s in panel.availability.items()
        },
    }
    Path(path).write_text(canonical(document), encoding="utf-8")


def read_panel(path: Path) -> Panel:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return Panel(
        {name: _decode(frame) for name, frame in document["fields"].items()},
        _decode(document["universe"]),
        _decode(document["half_spread"]),
        data_version=document["data_version"],
        calendar=pd.DatetimeIndex(document["calendar"]),
        availability={
            name: pd.Series(pd.to_datetime(s["values"]), index=pd.DatetimeIndex(s["index"]))
            for name, s in document["availability"].items()
        },
    )


def render(spec, panel: Panel, as_of) -> str:
    """Canonical JSON of the book held after `as_of`'s close."""
    book = target_weights(spec, panel, as_of)
    return canonical(
        {
            "as_of": str(pd.Timestamp(as_of).date()),
            "spec_id": spec.id,
            "weights": {str(int(p)): _fixed(w) for p, w in book.items()},
        }
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("as_of")
    args = parser.parse_args(argv)
    text = render(load_spec(args.spec), read_panel(args.panel), args.as_of)
    print(text)
    print(hashlib.sha256(text.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
