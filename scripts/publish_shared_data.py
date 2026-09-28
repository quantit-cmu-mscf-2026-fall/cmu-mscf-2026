"""Publish updated files to the team's shared Drive folder and refresh SHA256SUMS.

Usage::

    python scripts/publish_shared_data.py FILE [FILE ...]            # preview only
    python scripts/publish_shared_data.py FILE [FILE ...] --apply    # publish

Each FILE is copied into the synced `mscf-capstone-data` folder under its own
name, replacing any file of that name, and SHA256SUMS is rewritten to match.
Every copy is re-hashed after writing, so a bad copy fails here rather than on
a teammate's machine. Teammates' next `shared_data.sync()` picks the change up.

Without --apply nothing is written: it prints what would be added, replaced or
left alone, and the dataset version before and after.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from capstone import shared_data as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="actually publish")
    args = parser.parse_args()

    drive = sd.find_drive_folder()
    if drive is None:
        raise SystemExit("no synced 'mscf-capstone-data' folder found (or set CAPSTONE_DATA_DIR)")
    manifest = drive / sd.MANIFEST
    entries = sd._read_manifest(manifest)

    updates = {}
    for path in args.files:
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
        digest = sd._sha256(path)
        old = entries.get(path.name)
        state = "unchanged" if old == digest else ("replace" if old else "add")
        updates[path.name] = (path, digest, state)
        print(f"  {state:9s} {path.name}  {digest[:12]}")

    new_entries = {**entries, **{n: d for n, (_, d, _) in updates.items()}}
    with tempfile.TemporaryDirectory() as tmp:
        preview = Path(tmp) / sd.MANIFEST
        sd.write_manifest(preview, new_entries)
        after = sd.manifest_version(preview)
    print(f"version {sd.manifest_version(manifest)} -> {after}  ({drive})")

    changed = {n: u for n, u in updates.items() if u[2] != "unchanged"}
    if not changed:
        print("nothing to publish")
        return
    if not args.apply:
        print("preview only; re-run with --apply to publish")
        return

    for name, (path, digest, _) in changed.items():
        shutil.copyfile(path, drive / name)
        if sd._sha256(drive / name) != digest:
            raise SystemExit(f"copy of {name} did not verify; SHA256SUMS left unchanged")
    sd.write_manifest(manifest, new_entries)
    print(f"published {len(changed)} file(s); version is now {sd.manifest_version(manifest)}")


if __name__ == "__main__":
    main()
