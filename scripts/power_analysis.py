"""Is the registered bar reasonable? The funnel's power at a run's shape.

Simulates the registered funnel on null batches shaped like the run in
`power_analysis.json` (families, variants per family, within- and
across-family correlation, 31 years of search, a 5-year holdout) with one real
family planted at each Sharpe of the grid, and prints each stage's minimum
detectable Sharpe at 50% and 80% power and the end-to-end power curve
(`capstone.power`). The thresholds and the bootstrap draws come from the
registered `gate_config` in `scripts/gate_crsp_v1.json` (or `--config`), and
its hash goes in the logged params; the shape file holds only the shape.
Stage 3 is reported as no gate: the spanning test is not part of the simulated
worlds. The registered decision rule (the baseline) is not simulated here,
only the funnel it is reported beside.

The result is logged to the ledger as "validation-power" (tags "synthetic",
"power") before anything is printed. Simulated data only; paste the printed
table into the PR description rather than committing it.

    python scripts/power_analysis.py                 # the full run (about 30 minutes)
    python scripts/power_analysis.py --quick         # 10 worlds, a smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from capstone.gate import GateConfig
from capstone.power import PowerBudget, PowerSpec, power_analysis
from capstone.runlog import log_run

PARAMS = Path(__file__).with_suffix(".json")
CONFIG = Path(__file__).with_name("gate_crsp_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="10 worlds, for a smoke test")
    parser.add_argument("--params", type=Path, default=PARAMS, help="run shape file")
    parser.add_argument(
        "--config", type=Path, default=CONFIG, help="registered gate config (gate_config key)"
    )
    args = parser.parse_args()
    p = json.loads(args.params.read_text())
    stray = {"thresholds", "n_boot"} & p.keys()
    if stray:
        raise SystemExit(
            f"{args.params.name} sets {sorted(stray)}: they come from {args.config.name}"
        )
    config = GateConfig.from_dict(json.loads(args.config.read_text())["gate_config"])
    thresholds = config.thresholds
    spec = PowerSpec.from_years(
        search_years=p["search_years"],
        holdout_years=p["holdout_years"],
        variants=p["variants"],
        within_corr=p["within_corr"],
        across_corr=p["across_corr"],
        n_trials=p["n_trials"],
        n_boot=config.n_boot,
        thresholds=thresholds,
    )
    budget = PowerBudget(
        n_sims=10 if args.quick else p["n_sims"], sharpe_grid=p["sharpe_grid"], seed=p["seed"]
    )
    start = time.perf_counter()
    report = power_analysis(spec, budget)
    seconds = time.perf_counter() - start

    curves = {s.stage: s for s in (*report.stages, report.end_to_end_stage)}
    log_run(
        "validation-power",
        params={
            **{k: v for k, v in p.items() if k != "description"},
            "n_sims": budget.n_sims,
            "n_boot": spec.n_boot,
            "thresholds": asdict(thresholds),
            "config_hash": config.config_hash,
            "config_file": args.config.name,
            "params_file": args.params.name,
            "data_version": "synthetic",
        },
        metrics={
            **{f"{k}_mds50": s.mds50 for k, s in curves.items()},
            **{f"{k}_mds80": s.mds80 for k, s in curves.items()},
            **{f"{k}_null_rate": s.null_rate for k, s in curves.items()},
            "seconds": seconds,
        },
        seed=budget.seed,
        tags=["synthetic", "power"],
    )

    def fmt(x) -> str:
        return "  n/a" if x is None else f"{x:5.2f}"

    print(
        f"{budget.n_sims} worlds, seed {budget.seed}, {spec.n_families} families "
        f"({sum(spec.variants)} variants), within {spec.within_corr}, across "
        f"{spec.across_corr}, {spec.search_periods} search / {spec.holdout_periods} holdout "
        f"days, n_boot {spec.n_boot}, {seconds:.0f} s\n"
    )
    print("stage       gated  nominal  null rate  MDS 50%  MDS 80%")
    for name, s in curves.items():
        nominal = "    -" if name == "end_to_end" else f"{s.nominal:5.2f}"
        print(
            f"{name:<11} {s.gated!s:<6} {nominal:>7}  {s.null_rate:9.3f}  "
            f"{fmt(s.mds50):>7}  {fmt(s.mds80):>7}"
        )
    print("\npower by true Sharpe")
    print("sharpe " + " ".join(f"{k:>10}" for k in curves))
    for j, sharpe in enumerate(report.sharpe_grid):
        print(f"{sharpe:6.2f} " + " ".join(f"{s.power[j]:10.3f}" for s in curves.values()))


if __name__ == "__main__":
    main()
