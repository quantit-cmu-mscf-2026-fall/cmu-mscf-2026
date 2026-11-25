"""Distributions of scores at each funnel stage (1-3), search period only.

    python research/funnel_distributions.py --matrix M --summary S --run NAME
        [--null-matrix NM --null-summary NS] [--classes classes.parquet]
        [--n-boot 1999 --seed 0 --q1 0.1] [--out DIR]

Report only: reproduces the gate's stage 1-3 block without the look and writes
tables, stats.json and STATEMENTS.md to experiments/analysis/<run>/funnel/
(gitignored). The ledger entry is written before anything is computed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from capstone.analysis.funnel_distributions import run_analysis
from capstone.analysis.outputs import output_dir, write_outputs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--null-matrix", type=Path)
    p.add_argument("--null-summary", type=Path)
    p.add_argument("--classes", type=Path)
    p.add_argument("--n-boot", type=int, default=1999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--q1", type=float, default=0.1)
    p.add_argument("--out", type=Path, help="output folder (default analysis/<run>/funnel)")
    a = p.parse_args(argv)
    tables, stats, statements = run_analysis(
        a.matrix,
        a.summary,
        run=a.run,
        null_matrix_path=a.null_matrix,
        null_summary_path=a.null_summary,
        classes_path=a.classes,
        n_boot=a.n_boot,
        seed=a.seed,
        q1=a.q1,
    )
    out = a.out or output_dir(a.run, "funnel")
    write_outputs(
        out,
        title=f"Funnel stage 1-3 distributions: {a.run}",
        tables=tables,
        stats=stats,
        statements=statements,
    )
    print(out)


if __name__ == "__main__":
    main()
