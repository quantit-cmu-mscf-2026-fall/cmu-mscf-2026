"""Move cached datasets between `data_cache/` and a shared folder, verified.

    python scripts/sync_shared.py push --all
    python scripts/sync_shared.py pull crsp_sp500_2022_2023

The shared folder is any path this machine can see: set `CAPSTONE_SHARED_DIR`
or pass `--dest`. Google Drive for desktop mounts as an ordinary folder, so a
plain copy is all this needs -- no OAuth, no API keys, and nothing that could
leak a credential into a public repository.

A pull always ends in verification against the committed manifest, because the
copy that silently went stale is the failure this whole mechanism exists to
catch. `push` refuses to publish a dataset whose local copy does not match its
own manifest, so a bad copy cannot become everyone's copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from capstone import manifest as manifest_mod
from capstone.data import CACHE_DIR

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


def resolve_dest(explicit: Path | None) -> Path:
    """The shared folder, from --dest or the environment, checked to exist."""
    configured = os.environ.get("CAPSTONE_SHARED_DIR")
    dest = explicit or (Path(configured) if configured else None)
    if dest is None:
        raise SystemExit("set CAPSTONE_SHARED_DIR or pass --dest /path/to/shared/folder")
    if not dest.exists():
        raise SystemExit(f"shared folder does not exist: {dest}")
    return dest


def local_matches_manifest(name: str, *, identifier: str) -> bool:
    """True when the local cache agrees with its committed manifest."""
    manifest_path = MANIFEST_DIR / f"{name}.json"
    if not manifest_path.exists():
        return True  # nothing to check against; build_manifest.py has not run
    panels = manifest_mod.load_panels(manifest_mod.resolve_cache_path(name), identifier=identifier)
    matches, _ = manifest_mod.verify(panels, manifest_mod.read_manifest(manifest_path))
    return matches


def push(names: list[str], dest: Path, *, identifier: str) -> int:
    failures = 0
    for name in names:
        try:
            source = manifest_mod.resolve_cache_path(name)
        except FileNotFoundError as exc:
            print(f"MISSING   {name}: {exc}")
            failures += 1
            continue

        if not local_matches_manifest(name, identifier=identifier):
            print(f"REFUSED   {name}: local copy does not match its manifest; not publishing")
            failures += 1
            continue

        target = dest / source.name
        shutil.copy2(source, target)
        size_mb = target.stat().st_size / 1e6
        print(f"PUSHED    {name}  ({size_mb:,.1f} MB)  ->  {target}")
    return failures


def pull(names: list[str], dest: Path, *, identifier: str) -> int:
    failures = 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = dest / f"{name}.parquet"
        if not source.exists():
            print(f"MISSING   {name}: not in {dest}")
            failures += 1
            continue

        target = CACHE_DIR / source.name
        shutil.copy2(source, target)
        size_mb = target.stat().st_size / 1e6

        manifest_path = MANIFEST_DIR / f"{name}.json"
        if not manifest_path.exists():
            print(f"PULLED    {name}  ({size_mb:,.1f} MB)  [no manifest to verify against]")
            continue

        panels = manifest_mod.load_panels(target, identifier=identifier)
        try:
            matches, mismatched = manifest_mod.verify(
                panels, manifest_mod.read_manifest(manifest_path)
            )
        except ValueError as exc:
            print(f"STALE     {name}: {exc}")
            failures += 1
            continue

        if matches:
            print(f"PULLED    {name}  ({size_mb:,.1f} MB)  verified")
        else:
            print(f"DIVERGED  {name}: {', '.join(mismatched)} -- the shared copy is not the one")
            failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("push", "pull"))
    parser.add_argument("names", nargs="*", help="dataset names")
    parser.add_argument("--all", action="store_true", help="every dataset with a manifest")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--identifier", default="permno")
    args = parser.parse_args(argv)

    names = args.names
    if args.all:
        names = sorted(p.stem for p in MANIFEST_DIR.glob("*.json"))
    if not names:
        parser.error("name at least one dataset, or pass --all")

    dest = resolve_dest(args.dest)
    runner = push if args.action == "push" else pull
    failures = runner(names, dest, identifier=args.identifier)

    if failures:
        print(f"\n{failures} of {len(names)} dataset(s) failed.")
        return 1
    print(f"\nAll {len(names)} dataset(s) {args.action}ed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
