"""Publish updated files to the team's shared Drive folder and refresh SHA256SUMS.

Usage::

    python scripts/publish_shared_data.py FILE [FILE ...]            # preview only
    python scripts/publish_shared_data.py FILE [FILE ...] --apply    # publish

Each FILE is copied into the synced `mscf-capstone-data` folder under its own
name, replacing any file of that name, and SHA256SUMS is rewritten to match.
Every copy is re-hashed before it replaces anything. A failure at any step,
including while files are being swapped in, puts the old files and SHA256SUMS
back, so Drive never holds a half-updated set. Teammates' next
`shared_data.sync()` picks the change up.

Without --apply nothing is written: it prints what would be added, replaced or
left alone, and the dataset version before and after.
"""

from __future__ import annotations

import argparse
import os
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

    # All or nothing: copy everything to temporary names and verify it, then
    # write the new manifest to a temporary name. Only then swap each file in,
    # moving the one it replaces aside to .<name>.bak, and the manifest last.
    # A failure at any step puts the originals back and removes the new files.
    staged = {name: drive / f".{name}.publishing" for name in changed}
    temp_manifest = drive / f".{sd.MANIFEST}.publishing"
    backups: dict[str, Path] = {}
    swapped: list[str] = []
    try:
        for name, (path, digest, _) in changed.items():
            shutil.copyfile(path, staged[name])
            if sd._sha256(staged[name]) != digest:
                raise SystemExit(f"copy of {name} did not verify; nothing was published")
        sd.write_manifest(temp_manifest, new_entries)
        for name, temp in staged.items():
            target = drive / name
            if target.exists():
                backups[name] = drive / f".{name}.bak"
                os.replace(target, backups[name])
            os.replace(temp, target)
            swapped.append(name)
        os.replace(temp_manifest, manifest)
    except BaseException:
        _roll_back(drive, staged, temp_manifest, backups, swapped)
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)
    print(f"published {len(changed)} file(s); version is now {sd.manifest_version(manifest)}")


def _roll_back(drive: Path, staged, temp_manifest, backups, swapped) -> None:
    """Undo a partial publish: drop new files, restore the originals, clean up."""
    stuck = []

    def attempt(action, *paths):
        try:
            action(*paths)
        except OSError as err:
            stuck.append(str(err))

    for name in swapped:
        if name not in backups:  # a file this publish added
            attempt(Path.unlink, drive / name)
    for name, backup in backups.items():
        attempt(os.replace, backup, drive / name)
    for temp in [*staged.values(), temp_manifest]:
        attempt(lambda path: path.unlink(missing_ok=True), temp)
    if stuck:
        print("rollback incomplete; fix these by hand (.bak files hold the originals):")
        for line in stuck:
            print(f"  {line}")


if __name__ == "__main__":
    main()
