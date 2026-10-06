"""Evaluate the stored factors on the CRSP discovery period (QUANTIT-81).

    python research/evaluate_factors.py --store D:/paperlog/factors.db [--limit N]

Reads the factor store read-only, builds the 1990-2020 panel from the shared
data (the 2021-2025 holdout is never loaded), and evaluates every stored
factor under every holding period in research/evaluate_factors.toml: each
(factor, holding period) a ledger trial, logged as its returns exist. Writes the
dates x factor_id matrix of net daily returns and a per-factor summary to
experiments/factor_evaluation/<data version>/ (gitignored: CRSP-derived, never
committed). The numbers belong in the PR description.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
import tomllib
from pathlib import Path

import pandas as pd

from capstone.factors.evaluation import evaluate_factors, load_panel
from capstone.factors.tree import parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "factor_evaluation"
PARAMS = ROOT / "research" / "evaluate_factors.toml"


def stored_factors(store: Path) -> dict:
    """factor_id -> tree, from a factor store opened read-only."""
    con = sqlite3.connect(f"file:{store.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, expression FROM factors ORDER BY created_at, id").fetchall()
    finally:
        con.close()
    return {fid: parse(expression) for fid, expression in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True, help="the factor store (read-only)")
    parser.add_argument("--limit", type=int, help="evaluate only the first N factors")
    args = parser.parse_args()

    holds = tuple(tomllib.loads(PARAMS.read_text(encoding="utf-8"))["holds"])
    trees = stored_factors(args.store)
    if args.limit:
        trees = dict(list(trees.items())[: args.limit])
    t0 = time.time()
    panel = load_panel()
    shape = panel.fields["returns"].shape
    print(
        f"panel {shape[0]} days x {shape[1]} stocks, data {panel.data_version}, "
        f"built in {time.time() - t0:.0f}s; evaluating {len(trees)} factors x holds {holds}"
    )

    # A bug that makes every factor fail would log every one as a failed trial;
    # stop if the first three all fail, before the rest are spent.
    first = dict(list(trees.items())[:3])
    extra = {"store": args.store.name}
    matrix, summary = evaluate_factors(first, panel, holds=holds, extra_params=extra)
    if len(first) and summary.empty:
        raise SystemExit("the first factors all failed: check the code before spending more trials")
    rest = dict(list(trees.items())[3:])
    if rest:
        more_matrix, more_summary = evaluate_factors(rest, panel, holds=holds, extra_params=extra)
        matrix = matrix.join(more_matrix, how="outer")
        summary = pd.concat([summary, more_summary]) if not more_summary.empty else summary
    elapsed = time.time() - t0

    out = OUT / panel.data_version
    out.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out / "matrix.parquet")
    summary.to_csv(out / "summary.csv")
    failed = len(trees) * len(holds) - len(summary)
    print(f"\n{len(summary)} evaluated, {failed} failed, {elapsed:.0f}s; wrote {out}")
    if not summary.empty:
        cols = ["hold", "expression", "sharpe_gross", "sharpe_net", "mean_turnover"]
        print(summary.sort_values("sharpe_net", ascending=False)[cols].to_string(max_colwidth=60))


if __name__ == "__main__":
    main()
