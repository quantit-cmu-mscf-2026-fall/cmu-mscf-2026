"""Check that your local cache matches the manifest the team committed.

    python scripts/verify_universe.py crsp_sp500_2022_2023
    python scripts/verify_universe.py --all

Exits 0 on MATCH and 1 on DIVERGED, so it can gate a pipeline run or sit in a
pre-run hook. A divergence names the panels that differ, which is what makes it
actionable: `returns` alone usually means a vendor revision, while every panel
differing usually means a different date range or universe filter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from capstone import manifest as manifest_mod

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


def verify_one(name: str, *, manifest_dir: Path, identifier: str) -> bool:
    """Print a verdict for one dataset. Returns True when it matches."""
    manifest_path = manifest_dir / f"{name}.json"
    if not manifest_path.exists():
        print(f"SKIP      {name}: no manifest at {manifest_path}")
        return True

    expected = manifest_mod.read_manifest(manifest_path)
    try:
        panels = manifest_mod.load_panels(
            manifest_mod.resolve_cache_path(name), identifier=identifier
        )
    except FileNotFoundError as exc:
        print(f"MISSING   {name}: {exc}")
        return False

    try:
        matches, mismatched = manifest_mod.verify(panels, expected)
    except ValueError as exc:
        print(f"STALE     {name}: {exc}")
        return False

    coverage = expected["coverage"]
    if matches:
        print(
            f"MATCH     {name}  "
            f"({coverage['n_dates']:,} dates x {coverage['n_securities']:,} securities)"
        )
        return True

    print(f"DIVERGED  {name}: {', '.join(mismatched)}")
    actual = manifest_mod.build_manifest(panels, source=name)
    for panel in mismatched:
        print(
            f"            {panel:<10} expected {str(expected['digests'].get(panel))[:16]}"
            f"  got {str(actual['digests'].get(panel))[:16]}"
        )
    for field in ("n_dates", "n_securities", "start", "end"):
        if coverage.get(field) != actual["coverage"].get(field):
            print(
                f"            {field:<14} expected {coverage.get(field)}"
                f"  got {actual['coverage'].get(field)}"
            )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="cache names to check")
    parser.add_argument("--all", action="store_true", help="check every committed manifest")
    parser.add_argument("--identifier", default="permno")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)

    names = args.names
    if args.all:
        names = sorted(p.stem for p in args.manifest_dir.glob("*.json"))
    if not names:
        parser.error("name at least one dataset, or pass --all")

    results = [
        verify_one(name, manifest_dir=args.manifest_dir, identifier=args.identifier)
        for name in names
    ]
    if not all(results):
        print(f"\n{results.count(False)} of {len(results)} dataset(s) diverged.")
        return 1
    print(f"\nAll {len(results)} dataset(s) match the committed manifests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
