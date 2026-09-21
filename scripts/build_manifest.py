"""Fingerprint a cached dataset so teammates can prove they hold the same one.

    python scripts/build_manifest.py crsp_sp500_2022_2023
    python scripts/build_manifest.py french_factors_daily

Writes `manifests/<name>.json`: small, text, and safe to commit, because it
carries digests, counts and a date range but never values. The cached parquet
it describes stays in `data_cache/`, which is gitignored.

Run this once after a pull. Teammates then run `verify_universe.py` against the
committed manifest to confirm their copy agrees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from capstone import manifest as manifest_mod

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="cache names in data_cache/, or parquet paths")
    parser.add_argument(
        "--identifier",
        default="permno",
        help="column to key tidy panels on (default: permno; tickers are reused over time)",
    )
    parser.add_argument("--out-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)

    for name in args.names:
        try:
            path = manifest_mod.resolve_cache_path(name)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        panels = manifest_mod.load_panels(path, identifier=args.identifier)
        out = args.out_dir / f"{path.stem}.json"
        built = manifest_mod.write_manifest(panels, out, source=path.stem)

        coverage = built["coverage"]
        print(f"{path.stem}  ->  {out}")
        print(
            f"  {coverage['n_dates']:,} dates x {coverage['n_securities']:,} securities"
            f"  [{coverage['start'][:10]} .. {coverage['end'][:10]}]"
        )
        for panel, digest in sorted(built["digests"].items()):
            print(f"    {panel:<10} {digest[:32]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
