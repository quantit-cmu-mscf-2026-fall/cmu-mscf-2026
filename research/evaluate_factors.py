"""Evaluate the stored factors on the CRSP discovery period (QUANTIT-81).

    python research/evaluate_factors.py --store D:/paperlog/factors.db [--limit N]
                                        [--record-variants]

Builds the 1990-2020 panel from the shared data (the 2021-2025 holdout is
never loaded) and evaluates every stored factor under every trading spec in
research/evaluate_factors.toml: each variant (factor, spec) a ledger trial,
logged as its returns exist. The store is opened read-only unless
--record-variants, which writes the variants to its `variants` table after
copying the store to <store>.bak-<timestamp>. Writes the
dates x factor_id matrix of net daily returns and a per-factor summary to
experiments/factor_evaluation/<data version>/ (gitignored: CRSP-derived, never
committed). The numbers belong in the PR description.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
import tomllib
from pathlib import Path

import pandas as pd

from capstone.factors import store
from capstone.factors.evaluation import evaluate_factors, grid, load_panel
from capstone.factors.tree import parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "factor_evaluation"
PARAMS = ROOT / "research" / "evaluate_factors.toml"


def stored_factors(con: sqlite3.Connection) -> dict:
    """factor_id -> tree, every stored factor."""
    rows = con.execute("SELECT id, expression FROM factors ORDER BY created_at, id").fetchall()
    return {fid: parse(expression) for fid, expression in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True, help="the factor store (read-only)")
    parser.add_argument("--limit", type=int, help="evaluate only the first N factors")
    parser.add_argument(
        "--record-variants",
        action="store_true",
        help="write variants to the store (backed up first)",
    )
    args = parser.parse_args()

    params = tomllib.loads(PARAMS.read_text(encoding="utf-8"))
    specs = grid(params["holds"], params["neutral"])
    if args.record_variants:
        backup = args.store.with_name(f"{args.store.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(args.store, backup)
        print(f"store backed up to {backup}")
        con = store.connect(args.store)
    else:
        con = sqlite3.connect(f"file:{args.store.as_posix()}?mode=ro", uri=True)
    trees = stored_factors(con)
    if args.limit:
        trees = dict(list(trees.items())[: args.limit])
    t0 = time.time()
    panel = load_panel()
    shape = panel.fields["returns"].shape
    print(
        f"panel {shape[0]} days x {shape[1]} stocks, data {panel.data_version}, "
        f"built in {time.time() - t0:.0f}s; {len(trees)} factors x {len(specs)} specs"
    )

    # A bug that makes every factor fail would log every one as a failed trial;
    # stop if the first three all fail, before the rest are spent.
    first = dict(list(trees.items())[:3])
    extra = {"store": args.store.name}
    variant_con = con if args.record_variants else None
    run = dict(specs=specs, con=variant_con, extra_params=extra)
    matrix, summary = evaluate_factors(first, panel, **run)
    if len(first) and summary.empty:
        raise SystemExit("the first factors all failed: check the code before spending more trials")
    rest = dict(list(trees.items())[3:])
    if rest:
        more_matrix, more_summary = evaluate_factors(rest, panel, **run)
        matrix = matrix.join(more_matrix, how="outer")
        summary = pd.concat([summary, more_summary]) if not more_summary.empty else summary
    elapsed = time.time() - t0

    out = OUT / panel.data_version
    out.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out / "matrix.parquet")
    summary.to_csv(out / "summary.csv")
    con.close()
    failed = len(trees) * len(specs) - len(summary)
    print(f"\n{len(summary)} evaluated, {failed} failed, {elapsed:.0f}s; wrote {out}")
    if not summary.empty:
        cols = ["hold", "neutral", "sharpe_gross", "sharpe_net", "sharpe_net_full_spread"]
        best = summary.sort_values("sharpe_net", ascending=False)
        print(best[[*cols, "expression"]].head(25).to_string(max_colwidth=50))


if __name__ == "__main__":
    main()
