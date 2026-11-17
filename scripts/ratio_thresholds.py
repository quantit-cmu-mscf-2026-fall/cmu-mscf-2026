"""Ratio-mode thresholds at a run's shape (an opt-in proposal).

Simulates `ratio_thresholds.json`'s batch shape (a CRSP-sized shape: 15 families,
294 variants, 31 years of search, a 5-year holdout) with `capstone.ratio` and
prints, for each lambda (`miss_costs`) and pi1, the (q1, q4) of least expected
cost on the default grid, its expected false passes, misses and cost per batch,
and the same at the registered FDR thresholds (0.10, 0.30). One simulation
serves every lambda and pi1. `n_boot` is the smallest draw count resolving
q = 0.05 across the families.

The result is logged to the ledger as "validation-ratio" (tags "synthetic",
"ratio") before anything is printed.

    python scripts/ratio_thresholds.py            # the full run
    python scripts/ratio_thresholds.py --quick    # 4 worlds, a smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from capstone import ratio
from capstone.funnel import Thresholds
from capstone.power import PERIODS_PER_YEAR
from capstone.runlog import log_run

PARAMS = Path(__file__).with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="4 worlds, for a smoke test")
    parser.add_argument("--params", type=Path, default=PARAMS, help="parameter file")
    args = parser.parse_args()
    p = json.loads(args.params.read_text())
    spec = ratio.RatioSpec(
        miss_cost=1.0,  # replaced per lambda below; the simulation doesn't read it
        pi1=0.0,
        variants=p["variants"],
        within_corr=p["within_corr"],
        across_corr=p["across_corr"],
        search_periods=round(p["search_years"] * PERIODS_PER_YEAR),
        holdout_periods=round(p["holdout_years"] * PERIODS_PER_YEAR),
        n_boot=math.ceil(len(p["variants"]) / 0.05) - 1,
        alt_sharpe=p["alt_sharpe"],
        seed=p["seed"],
        n_sims=4 if args.quick else p["n_sims"],
    )
    start = time.perf_counter()
    table = ratio.simulate(spec)
    seconds = time.perf_counter() - start
    fdr = table.grid.index((Thresholds().q1, Thresholds().q4))

    rows = []
    for pi1 in p["pi1s"]:
        for lam in p["miss_costs"]:
            k = table.choose(lam, pi1)
            false, missed, cost = table.expected(lam, pi1, at=k)
            f_fdr, m_fdr, c_fdr = table.expected(lam, pi1, at=fdr)
            rows.append(
                {
                    "pi1": pi1,
                    "lambda": lam,
                    "q1": table.grid[k][0],
                    "q4": table.grid[k][1],
                    "false": false,
                    "missed": missed,
                    "cost": cost,
                    "fdr_false": f_fdr,
                    "fdr_missed": m_fdr,
                    "fdr_cost": c_fdr,
                }
            )
    log_run(
        "validation-ratio",
        params={
            **{k: v for k, v in p.items() if k != "description"},
            "n_sims": spec.n_sims,
            "n_boot": spec.n_boot,
            "grid": [list(g) for g in spec.grid],
            "params_file": args.params.name,
            "data_version": "synthetic",
        },
        metrics={
            "choices": rows,
            "false_r0": list(table.false_passes[0]),
            "false_r1": list(table.false_passes[1]),
            "misses_r1": list(table.misses[1]),
            "seconds": seconds,
        },
        seed=spec.seed,
        tags=["synthetic", "ratio"],
    )

    print(
        f"{spec.n_sims} worlds, seed {spec.seed}, {len(spec.variants)} families "
        f"({sum(spec.variants)} variants), alternative Sharpe {spec.alt_sharpe}, "
        f"n_boot {spec.n_boot}, {seconds:.0f} s\n"
    )
    print("grid point     false|r=0  false|r=1  missed|r=1")
    for k, (q1, q4) in enumerate(table.grid):
        print(
            f"({q1:.2f}, {q4:.2f})   {table.false_passes[0][k]:9.3f}  "
            f"{table.false_passes[1][k]:9.3f}  {table.misses[1][k]:10.3f}"
        )
    print("\n pi1  lambda   (q1, q4)      false  missed   cost | FDR: false  missed   cost")
    for r in rows:
        print(
            f"{r['pi1']:.2f}  {r['lambda']:6g}   ({r['q1']:.2f}, {r['q4']:.2f})  "
            f"{r['false']:6.3f}  {r['missed']:6.3f} {r['cost']:6.3f} |      "
            f"{r['fdr_false']:6.3f}  {r['fdr_missed']:6.3f} {r['fdr_cost']:6.3f}"
        )


if __name__ == "__main__":
    main()
