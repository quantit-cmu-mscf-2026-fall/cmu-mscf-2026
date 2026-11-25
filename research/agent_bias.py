"""Agent bias report for the crsp-v1 run: what the agents produced and how concentrated.

Logs the run first (input digests only), then reads the inputs and writes the
analysis folder (tables, stats.json, STATEMENTS.md). No model calls.

    python research/agent_bias.py [--summary ...] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from capstone import runlog
from capstone.analysis.agent_bias import analyse
from capstone.analysis.outputs import write_outputs

DATA = Path("G:/My Drive/mscf-capstone-data/factors")
EXP = Path("C:/Users/caleb/code/cmu-mscf-2026/experiments")
ATTR = EXP / "analysis" / "crsp-v1-reeval-359" / "fix-attribution"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run(
    summary: Path,
    run_summary: Path,
    new: Path,
    placebo: Path,
    attribution: Path,
    out: Path,
) -> Path:
    paths = {
        "summary": summary,
        "run_summary": run_summary,
        "new": new,
        "placebo": placebo,
        "attribution": attribution,
    }
    runlog.log_run(
        name="agent-bias",
        params={k: {"file": Path(v).name, "sha256_16": digest(v)} for k, v in paths.items()},
        tags=["synthetic", "methodology"],
        notes="report-only agent bias analysis of crsp-v1; no model calls, search period only",
    )
    tables, stats, statements = analyse(
        summary=pd.read_parquet(summary),
        run_summary=json.loads(Path(run_summary).read_text(encoding="utf-8")),
        new=pd.read_parquet(new),
        placebo=pd.read_parquet(placebo),
        attribution=pd.read_csv(attribution),
        inputs=tuple(Path(v).name for v in paths.values()),
    )
    return write_outputs(
        out, title="Agent bias: crsp-v1", tables=tables, stats=stats, statements=statements
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DATA / "crsp-v1_summary.parquet")
    parser.add_argument("--run-summary", type=Path, default=DATA / "crsp-v1_run_summary.json")
    parser.add_argument(
        "--new",
        type=Path,
        default=EXP / "factor_evaluation" / "crsp-v1-reeval-359" / "summary.parquet",
    )
    parser.add_argument(
        "--placebo", type=Path, default=EXP / "placebo" / "crsp-v1" / "seed0" / "summary.parquet"
    )
    parser.add_argument("--attribution", type=Path, default=ATTR / "rows.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.cwd() / "experiments" / "analysis" / "crsp-v1-reeval-359" / "bias",
    )
    a = parser.parse_args()
    print(run(a.summary, a.run_summary, a.new, a.placebo, a.attribution, a.out))


if __name__ == "__main__":
    main()
