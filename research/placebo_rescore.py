"""Re-score a factor set on a placebo panel: the null for funnel-score comparisons.

    CAPSTONE_LEDGER_DIR=experiments/placebo python research/placebo_rescore.py \
        --summary <real summary.parquet> [--seed 0] [--out DIR]

Takes the unique factors (factor_id, expression, hypothesis_id) of a real
evaluation summary and evaluates each under every trading spec in
research/evaluate_factors.toml on the discovery-period panel whose stocks are
permuted within each date (`capstone.pipeline.placebo`), so no factor has real
predictive power. Trials go to the placebo ledger only: the script refuses to run
unless CAPSTONE_LEDGER_DIR names experiments/placebo, and refuses to write into
the real experiments/factor_evaluation directory. Writes matrix.parquet,
summary.parquet and info.json (gitignored: CRSP-derived, never committed). No
model calls; nothing after 2020-12-31 is read.
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import pandas as pd

from capstone import shared_data as sd
from capstone.factors.evaluation import (
    COLUMNS,
    DAILY_FILE,
    DISCOVERY_END,
    DISCOVERY_START,
    build_placebo_panel,
    evaluate_factors,
    grid,
)
from capstone.factors.tree import parse
from capstone.pipeline import placebo

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "research" / "evaluate_factors.toml"
REAL_OUT = ROOT / "experiments" / "factor_evaluation"
TAG = "e2e-run:crsp-v1:placebo"


def check_out(out: Path) -> Path:
    """`out`, unless it is (inside) the real factor_evaluation directory."""
    resolved = out.resolve()
    for real in {REAL_OUT.resolve(), (placebo.main_ledger_dir() / "factor_evaluation").resolve()}:
        if resolved == real or real in resolved.parents:
            raise placebo.PlaceboRefused(f"{out} is in the real factor_evaluation directory")
    return out


def factors_of(summary: pd.DataFrame) -> tuple[dict, dict]:
    """(factor_id -> tree, factor_id -> hypothesis_id) from a summary's unique factors."""
    unique = summary.drop_duplicates("factor_id").set_index("factor_id")
    trees = {fid: parse(expression) for fid, expression in unique["expression"].items()}
    return trees, unique["hypothesis_id"].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    placebo.check_ledger()  # before anything is read or logged
    out = check_out(args.out or ROOT / "experiments" / "placebo" / "crsp-v1" / f"seed{args.seed}")
    trees, families = factors_of(pd.read_parquet(args.summary))
    params = tomllib.loads(PARAMS.read_text(encoding="utf-8"))
    specs = grid(params["holds"], params["neutral"])
    print(f"{len(trees)} factors x {len(specs)} specs, seed {args.seed}", flush=True)

    t0 = time.time()
    daily = sd.load(DAILY_FILE, columns=COLUMNS, start=DISCOVERY_START, end=DISCOVERY_END)
    panel = build_placebo_panel(daily, args.seed, sd.data_version())
    matrix, summary = evaluate_factors(
        trees,
        panel,
        specs=specs,
        families=families,
        extra_params={"run": "crsp-v1", "placebo_seed": args.seed},
        tags=[TAG],
    )
    out.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out / "matrix.parquet")
    summary.rename_axis("variant_id").to_parquet(out / "summary.parquet")
    info = {
        "seed": args.seed,
        "placebo_version": panel.data_version,
        "data_version": sd.data_version(),
        "n_variants": matrix.shape[1],
        "dates": [str(matrix.index.min().date()), str(matrix.index.max().date())],
        "seconds": round(time.time() - t0),
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
