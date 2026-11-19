"""Calibrate the funnel's thresholds for one generator's measured structure.

    python scripts/calibrate_generator.py scripts/calibrate_crsp_v1.json [--workers 4]

`docs/validation/framework.md` ("Calibration"): simulate candidate sets with
the structure measured from the real generator (`measure_generator.py`),
plant real families whose identity we know, run the funnel and the baseline on
each, and pick the thresholds with the most power whose worst-setting
false-discovery rate stays below the target with a margin. Nothing here reads
real returns; the parameter file carries the measured structure.

Worlds come from `capstone.power`: one null world per seed (every family's
statistics, bootstrap included), then only the planted families recomputed.
The real families planted in a seed are drawn at random, so every family size
gets its turn. Logged to the ledger (`synthetic`, `calibration`) before
anything is computed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.calibration import run_baseline
from capstone.funnel import Thresholds, run_funnel
from capstone.power import PowerSpec, null_side, planted_side
from capstone.runlog import log_run


def _spec(p: dict) -> PowerSpec:
    return PowerSpec.from_years(
        search_years=p["search_years"],
        holdout_years=p["holdout_years"],
        variants=p["variants"],
        within_corr=p["within_corr"],
        across_corr=p["across_corr"],
        ar1=p["ar1"],
        t_df=p["t_df"],
        garch=tuple(p["garch"]) if p["garch"] else None,
        n_boot=p["n_boot"],
        thresholds=Thresholds(stage1="bootstrap"),
    )


def _rules(p: dict) -> dict:
    """Every decision rule swept: funnel (q1, q4) and baseline q."""
    rules = {
        f"funnel q1={q1} q4={q4}": Thresholds(stage1="bootstrap", q1=q1, q4=q4)
        for q1, q4 in itertools.product(p["sweep"]["q1"], p["sweep"]["q4"])
    }
    rules |= {f"baseline q={q}": q for q in p["baseline_q"]}
    return rules


def one_seed(args: tuple[dict, int]) -> list[dict]:
    """Every setting and rule on one seed's world: discoveries, false and true."""
    p, seed = args
    side = null_side(_spec(p), seed)
    families = side.stats.index
    rng = np.random.default_rng(seed)
    rules = _rules(p)
    out = []
    for n_real, sharpe in itertools.product(p["n_real"], p["sharpe_real"]):
        if n_real == 0 and sharpe != p["sharpe_real"][0]:
            continue  # the null world has no Sharpe to vary
        targets = list(rng.choice(families, size=n_real, replace=False)) if n_real else []
        stats = planted_side(side, targets, sharpe) if targets else side.stats
        real = families.isin(targets)
        for name, rule in rules.items():
            if isinstance(rule, Thresholds):
                accepted = run_funnel(stats, rule)["stage4"]
            else:
                accepted = run_baseline(stats, q=rule, stage1="bootstrap")
            accepted = accepted.reindex(families, fill_value=False).to_numpy()
            out.append(
                {
                    "seed": seed,
                    "n_real": n_real,
                    "sharpe": sharpe if n_real else 0.0,
                    "rule": name,
                    "discoveries": int(accepted.sum()),
                    "false": int((accepted & ~real).sum()),
                    "true": int((accepted & real).sum()),
                }
            )
    return out


def summarise(rows: pd.DataFrame, target: float, margin: float) -> pd.DataFrame:
    """Per rule: worst-setting FDR, mean power, and the bad-run statistics."""
    rows = rows.assign(
        fdp=rows["false"] / rows["discoveries"].clip(lower=1),
        power=rows["true"] / rows["n_real"].where(rows["n_real"] > 0),
        any_false=rows["false"] > 0,
    )
    by = rows.groupby(["rule", "n_real", "sharpe"])
    per_setting = by.agg(
        fdr=("fdp", "mean"),
        power=("power", "mean"),
        p_any_false=("any_false", "mean"),
        worst_false=("false", "max"),
    )
    by_rule = per_setting.groupby("rule").agg(
        worst_fdr=("fdr", "max"),
        mean_power=("power", "mean"),
        null_any_false=("p_any_false", "first"),  # n_real = 0 sorts first
        worst_false=("worst_false", "max"),
    )
    by_rule["safe"] = by_rule["worst_fdr"] <= target - margin
    return by_rule.sort_values(["safe", "mean_power"], ascending=False), per_setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="write per-run rows here (CSV)")
    args = parser.parse_args()
    p = json.loads(args.params.read_text(encoding="utf-8"))
    seeds = range(*p["seeds"])
    _spec(p)  # refuse a configuration the funnel can't run before logging it

    log_run(
        "generator-calibration",
        params={k: v for k, v in p.items() if k != "seeds"} | {"seeds": list(p["seeds"])},
        tags=["synthetic", "calibration"],
        notes=f"{args.params.name}; capstone.power worlds, stage 1 bootstrap",
    )
    with ProcessPoolExecutor(args.workers) as pool:
        rows = pd.DataFrame(
            [r for batch in pool.map(one_seed, [(p, s) for s in seeds]) for r in batch]
        )
    if args.out:
        rows.to_csv(args.out, index=False)
    by_rule, per_setting = summarise(rows, p["target_fdr"], p["margin"])
    pd.set_option("display.width", 160)
    print(f"{len(seeds)} seeds; target FDR {p['target_fdr']} with margin {p['margin']}\n")
    print(by_rule.round(3).to_string())
    best = by_rule[by_rule["safe"]].index
    for name in [*best[:1], *[r for r in by_rule.index if r.startswith("baseline")]]:
        print(f"\n{name}\n{per_setting.loc[name].round(3).to_string()}")


if __name__ == "__main__":
    main()
