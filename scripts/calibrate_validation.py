"""Calibrate the validation pipeline on signals with known truth (#50, Q5).

Simulates hypothesis families like an agent's output (near-copy variants,
autocorrelation, fat tails, volatility clustering, 31 years of search and a
5-year holdout), plants real families at each share and strength in
`calibrate_validation.json`, and measures three ways of deciding:

- naive stack: every screening method as a gate at once, then the holdout;
- baseline: BH at 10% on each family's search-adjusted p-value;
- funnel: stage 1 screen, stage 2 confirmation on unused years, stage 3
  placeholder (not measured), stage 4 on the holdout; for every threshold
  set in the sweep, and with stage 2 on fresh data ("split") or not ("reuse").

It then picks the funnel thresholds with the most power whose average FDR is
at most the target in every simulated setting, including when nothing is
real, and prints:

1. the three approaches at those thresholds, setting by setting;
2. where real signals die and where noise leaks, stage by stage;
3. what each open design choice costs (stage-4 gate, its trial count, stage
   2 on fresh data or not, the stage-1 method);
4. the best few threshold sets.

Each simulated setting is logged to the ledger as "validation-calibration"
(tag "synthetic") as soon as its numbers exist, before anything is printed.

    python scripts/calibrate_validation.py            # the full grid
    python scripts/calibrate_validation.py --quick    # 3 seeds, for a smoke test
    python scripts/calibrate_validation.py --params scripts/calibrate_validation_extended.json

`alpha2 = 1.0` in a sweep means no stage-2 gate (every survivor passes).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.calibration import (
    Thresholds,
    family_statistics,
    gate_report,
    make_family_set,
    run_baseline,
    run_funnel,
    run_naive_stack,
    score,
)
from capstone.runlog import log_run

PARAMS = Path(__file__).with_suffix(".json")


def threshold_grid(sweep: dict) -> list[Thresholds]:
    finals = [
        {"final": "dsr", "d4": d, "final_trials": t}
        for d, t in itertools.product(sweep["dsr"]["d4"], sweep["dsr"]["final_trials"])
    ] + [{"final": "bh", "q4": q} for q in sweep["bh"]["q4"]]
    return [
        Thresholds(q1=q1, alpha2=a2, **f)
        for q1, a2, f in itertools.product(sweep["q1"], sweep["alpha2"], finals)
    ]


def label(th: Thresholds) -> str:
    final = f"dsr>={th.d4} ({th.final_trials})" if th.final == "dsr" else f"bh@{th.q4} (survivors)"
    return f"q1={th.q1} a2={th.alpha2} final={final}"


def settings(p: dict) -> list[tuple[float, float]]:
    out = [(0.0, p["sharpe_real"][0])] if 0.0 in p["share_real"] else []
    out += [(s, sr) for s in p["share_real"] if s > 0 for sr in p["sharpe_real"]]
    return out


def evaluate_setting(p, share, sharpe, seeds, grid, stage1, n_boot=0):
    """Every pipeline on every seed of one simulated setting."""
    rows, gates = [], []
    for seed in seeds:
        fs = make_family_set(
            p["n_families"],
            p["k"],
            share,
            sharpe,
            within_rho=p["within_rho"],
            n_search=p["search_years"] * 252,
            n_holdout=p["holdout_years"] * 252,
            seed=seed,
            **{k: tuple(v) if isinstance(v, list) else v for k, v in p["defects"].items()},
        )
        stats = family_statistics(fs, stage2_years=p["stage2_years"], n_boot=n_boot, seed=seed)
        truth = stats["reuse"]["truth"]
        base = {"share_real": share, "sharpe_real": sharpe, "seed": seed}
        rows.append(
            {
                **base,
                "arm": "baseline",
                "design": "-",
                "thresholds": f"q={p['baseline_q']}",
                **score(run_baseline(stats["reuse"], q=p["baseline_q"], stage1=stage1), truth),
            }
        )
        for th in grid:
            rows.append(
                {
                    **base,
                    "arm": "naive",
                    "design": "-",
                    "thresholds": label(th),
                    **score(run_naive_stack(stats["reuse"], th), truth),
                }
            )
            for design in ("split", "reuse"):
                stages = run_funnel(stats[design], th, stage1=stage1)
                rows.append(
                    {
                        **base,
                        "arm": "funnel",
                        "design": design,
                        "thresholds": label(th),
                        **score(stages["stage4"], truth),
                    }
                )
                if design == "split":
                    g = gate_report(stages, truth).reset_index()
                    gates.append(g.assign(**base, thresholds=label(th)))
    return pd.DataFrame(rows), pd.concat(gates, ignore_index=True)


def recommend(results: pd.DataFrame, target: float) -> pd.DataFrame:
    """Funnel threshold sets ranked by mean power, among those whose mean FDR
    is at most `target` in every setting (including nothing-real)."""
    funnel = results[(results["arm"] == "funnel") & (results["design"] == "split")]
    by_setting = funnel.groupby(["thresholds", "share_real", "sharpe_real"])[
        ["fdr", "power"]
    ].mean()
    worst_fdr = by_setting["fdr"].groupby("thresholds").max()
    mean_power = by_setting["power"].groupby("thresholds").mean()
    table = pd.DataFrame({"worst_fdr": worst_fdr, "mean_power": mean_power})
    ok = table[table["worst_fdr"] <= target]
    return ok.sort_values("mean_power", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="3 seeds, for a smoke test")
    parser.add_argument("--params", type=Path, default=PARAMS, help="parameter file")
    args = parser.parse_args()
    p = json.loads(args.params.read_text())
    seeds = range(3) if args.quick else range(*p["seeds"])
    grid = threshold_grid(p["sweep"])

    all_rows, all_gates = [], []
    for share, sharpe in settings(p):
        rows, gates = evaluate_setting(p, share, sharpe, seeds, grid, p["stage1"])
        log_run(
            "validation-calibration",
            params={
                **{
                    k: v
                    for k, v in p.items()
                    if k not in ("seeds", "share_real", "sharpe_real", "bootstrap_check")
                },
                "share_real": share,
                "sharpe_real": sharpe,
                "seed_range": [seeds.start, seeds.stop],
                "params_file": args.params.name,
            },
            metrics={
                f"{arm}_{m}": float(rows.loc[rows["arm"] == arm, m].mean())
                for arm in ("naive", "baseline", "funnel")
                for m in ("fdr", "power")
            },
            tags=["synthetic", "calibration"],
        )
        all_rows.append(rows)
        all_gates.append(gates)

    bc = p["bootstrap_check"]
    boot_seeds = range(2) if args.quick else range(*bc["seeds"])
    boot_rows = []
    for share in bc["share_real"]:
        for method, n_boot in (("rho", 0), ("bootstrap", bc["n_boot"])):
            r, _ = evaluate_setting(
                p, share, bc["sharpe_real"], boot_seeds, grid, method, n_boot=n_boot
            )
            boot_rows.append(r.assign(stage1=method))
    boot = pd.concat(boot_rows, ignore_index=True)
    log_run(
        "validation-calibration",
        params={"bootstrap_check": bc, "seed_range": [boot_seeds.start, boot_seeds.stop]},
        metrics={
            f"{m}_{a}_{k}": float(boot.loc[(boot["stage1"] == m) & (boot["arm"] == a), k].mean())
            for m in ("rho", "bootstrap")
            for a in ("baseline", "funnel")
            for k in ("fdr", "power")
        },
        tags=["synthetic", "calibration"],
    )

    results = pd.concat(all_rows, ignore_index=True)
    gates = pd.concat(all_gates, ignore_index=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    ranked = recommend(results, p["target_fdr"])
    print(
        f"{len(seeds)} seeds per setting; {p['n_families']} hypotheses x {p['k']} variants "
        f"(within-family correlation {p['within_rho']}), defects {p['defects']}, "
        f"{p['search_years']}y search + {p['holdout_years']}y holdout, stage 2 on the last "
        f"{p['stage2_years']}y of search. Target end-to-end FDR {p['target_fdr']}.\n"
    )
    if ranked.empty:
        print("No funnel threshold set meets the target in every setting.")
        chosen = None
    else:
        chosen = ranked.index[0]
        print(f"Recommended funnel thresholds: {chosen}\n")

    print("1. The three approaches, by setting (mean over seeds)")
    pick = results[
        (results["arm"] == "baseline")
        | (
            (results["arm"] == "funnel")
            & (results["design"] == "split")
            & (results["thresholds"] == chosen)
        )
        | ((results["arm"] == "naive") & (results["thresholds"] == chosen))
    ]
    print(
        pick.groupby(["share_real", "sharpe_real", "arm"])[["fdr", "power", "true", "discoveries"]]
        .mean()
        .round(3)
        .unstack("arm")
        .to_string()
    )

    print("\n2. Where real signals die and noise leaks (funnel, summed over seeds)")
    g = gates[gates["thresholds"] == chosen]
    for (share, sharpe), block in g.groupby(["share_real", "sharpe_real"]):
        if share in (0.0, 0.02) and sharpe in (p["sharpe_real"][0], 1.0):
            t = block.groupby("stage")[["real_in", "real_out", "null_in", "null_out"]].sum()
            t["pass_rate_real"] = (t["real_out"] / t["real_in"]).round(3)
            t["pass_rate_null"] = (t["null_out"] / t["null_in"]).round(3)
            t.loc["stage3", ["pass_rate_real", "pass_rate_null"]] = np.nan
            print(f"share {share}, Sharpe {sharpe} (stage3 not measured):")
            print(t.to_string(), "\n")

    print("3. What each design choice costs (funnel; best thresholds within each choice)")
    funnel = results[results["arm"] == "funnel"].copy()
    funnel["final"] = (
        funnel["thresholds"]
        .str.extract(r"final=(\S+ \(\w+\))")[0]
        .str.replace(r">=\S+|@\S+", "", regex=True)
    )
    for design in ("split", "reuse"):
        d = funnel[funnel["design"] == design]
        by = d.groupby(["final", "thresholds", "share_real", "sharpe_real"])[
            ["fdr", "power"]
        ].mean()
        summary = pd.DataFrame(
            {
                "worst_fdr": by["fdr"].groupby(["final", "thresholds"]).max(),
                "mean_power": by["power"].groupby(["final", "thresholds"]).mean(),
            }
        )
        best = (
            summary[summary["worst_fdr"] <= p["target_fdr"]]
            .sort_values("mean_power", ascending=False)
            .groupby("final")
            .head(1)
        )
        print(f"stage 2 {design}:")
        print(best.round(3).to_string() if not best.empty else "  nothing meets the target", "\n")
    print(f"stage-1 method (Sharpe {bc['sharpe_real']}, seeds {list(boot_seeds)}):")
    bsel = boot[
        (boot["arm"] == "baseline")
        | ((boot["arm"] == "funnel") & (boot["design"] == "split") & (boot["thresholds"] == chosen))
    ]
    cols = ["fdr", "power", "any_false"]
    print(bsel.groupby(["share_real", "stage1", "arm"])[cols].mean().round(3).to_string(), "\n")

    print("4. Best threshold sets (worst-setting FDR <= target)")
    print(ranked.head(8).round(3).to_string())
    n_logged = len(settings(p)) + 1
    print(f"\nlogged {n_logged} runs to the ledger as 'validation-calibration'")


if __name__ == "__main__":
    main()
