"""One folder per analysis: data tables, headline numbers and numbered statements.

    experiments/analysis/<run>/<unit>/
        data/<name>.parquet   tidy tables (and a .csv copy for reading by eye)
        stats.json            every headline number: value, definition, inputs
        STATEMENTS.md         numbered statements, each citing its number in stats.json
        figures/<name>.png    optional

A statement must cite a key of `stats` (`source`), so every sentence that may end
up on a slide traces back to a number and the files it came from. A statement
without a source, or citing a key that isn't in `stats`, is refused before
anything is written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "analysis"


@dataclass(frozen=True)
class Stat:
    """A headline number: its value, what it means, and the files it was computed from."""

    value: object
    definition: str
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Statement:
    """One sentence for a slide. `source` is the key of the `Stat` it rests on."""

    text: str
    source: str
    caveat: str = ""


def output_dir(run: str, unit: str, root: Path | None = None) -> Path:
    return (ROOT if root is None else Path(root)) / run / unit


def _plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _checked(stats: Mapping[str, Stat], statements: Sequence[Statement]) -> None:
    for name, stat in stats.items():
        if not isinstance(stat, Stat) or not stat.definition.strip():
            raise ValueError(f"stat {name!r} needs a Stat with a definition")
    for i, s in enumerate(statements, 1):
        if not s.text.strip():
            raise ValueError(f"statement {i} is empty")
        if not s.source.strip():
            raise ValueError(f"statement {i} has no source: cite a key of stats")
        if s.source not in stats:
            raise ValueError(f"statement {i} cites {s.source!r}, which is not in stats")


def markdown(title: str, statements: Sequence[Statement]) -> str:
    lines = [f"# {title}", ""]
    for i, s in enumerate(statements, 1):
        lines.append(f"S{i}. {s.text} (`stats.json: {s.source}`)")
        if s.caveat:
            lines.append(f"    Caveat: {s.caveat}")
        lines.append("")
    return "\n".join(lines)


def write_outputs(
    directory: Path,
    *,
    title: str,
    tables: Mapping[str, pd.DataFrame] | None = None,
    stats: Mapping[str, Stat],
    statements: Sequence[Statement],
    figures: Mapping[str, object] | None = None,
) -> Path:
    """Write one analysis folder; refuses unsourced statements before writing anything."""
    _checked(stats, statements)
    directory = Path(directory)
    (directory / "data").mkdir(parents=True, exist_ok=True)
    for name, frame in (tables or {}).items():
        frame.to_parquet(directory / "data" / f"{name}.parquet")
        frame.to_csv(directory / "data" / f"{name}.csv")
    atomic_json(
        directory / "stats.json",
        {
            name: {
                "value": _plain(stat.value),
                "definition": stat.definition,
                "inputs": list(stat.inputs),
            }
            for name, stat in stats.items()
        },
    )
    (directory / "STATEMENTS.md").write_text(markdown(title, statements), encoding="utf-8")
    if figures:
        (directory / "figures").mkdir(exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(directory / "figures" / f"{name}.png", dpi=150)
    return directory
