"""`python -m capstone.pipeline.autopilot init --dir <folder>`: the starter set for a first run.

Writes the files of `docs/examples/starter/` into `--dir`: `autopilot.toml`, the example
config with safe defaults (`real_model_calls = false`, both spend ceilings 0). It never
overwrites a file: an existing one is kept and listed as kept. Each value a person must
choose carries a `TODO(person)` marker, which `preflight` reports as `missing` until it is
gone. Nothing else is read or written; nothing connects.
"""

from __future__ import annotations

import sys
from pathlib import Path

STARTER = Path(__file__).resolve().parents[3] / "docs" / "examples" / "starter"
FILES = ("autopilot.toml",)


class InitError(ValueError):
    """The starter set cannot be written as asked."""


def contents():
    """{file name: text} of the starter set."""
    try:
        return {name: (STARTER / name).read_text(encoding="utf-8") for name in FILES}
    except OSError as error:
        raise InitError(f"the starter set is not readable: {error}") from error


def init(folder):
    """Write the starter set; (written names, kept names). Never overwrites."""
    texts = contents()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written, kept = [], []
    for name in FILES:
        try:
            with (folder / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(texts[name])
        except FileExistsError:
            kept.append(name)
            continue
        written.append(name)
    return written, kept


def main(args):
    """The `init` subcommand (`__main__`)."""
    try:
        written, kept = init(args.dir)
    except (InitError, OSError) as error:
        print(f"autopilot init: {error}", file=sys.stderr)
        return 2
    folder = Path(args.dir).resolve()
    for name in written:
        print(f"wrote {folder / name}")
    for name in kept:
        print(f"kept  {folder / name} (exists; not overwritten)")
    print(
        "next: replace every TODO(person) marker, then run "
        f"`python -m capstone.pipeline.autopilot preflight --config {folder / 'autopilot.toml'}`"
    )
    return 0


def add_arguments(parser):
    parser.add_argument("--dir", required=True, help="the folder the starter set goes into")
