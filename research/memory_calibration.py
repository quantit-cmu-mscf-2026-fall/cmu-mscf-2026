"""Calibrate learned memory search's veto and pool cap.

Two questions, each answered directly rather than by running the deepen loop:

1. **Veto.** How often is an edit that is no worse than the others vetoed,
   and how quickly is a bad one, for each (kappa, tau_c)? Simulates only the
   memory's own statistics (`PairStats`, `vetoed`): children fail and score
   at given rates, residuals are standard normal (mean 0, or below 0 for a bad
   edit). The gate uses |mean| / sd, so the result doesn't depend on the
   scorer's units. No model is called and no data is read.
2. **Correlation cap.** How many near-duplicates (variants of one factor) and
   how many genuinely different factors does each cap block? Reads the daily
   net returns of an evaluation run already on the ledger (`summary.csv` and
   `matrix.parquet` in the folder named by `[correlation] run`): no new
   trials. With `run = ""` this table is skipped.

Parameters are in research/memory_calibration.toml. The run is logged to the
ledger as a synthetic calibration before its tables are printed.

    python research/memory_calibration.py
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.factors.llm import load_config
from capstone.factors.memory import PairStats, vetoed
from capstone.runlog import log_run

PARAMS = Path(__file__).with_suffix(".toml")
ROOT = Path(__file__).resolve().parents[1]


def veto_rate(config, *, attempts, p_fail, share_scored, residual, reps, seed) -> float:
    """Share of `reps` simulated pairs vetoed after `attempts` children."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        stats = PairStats()
        for _ in range(attempts):
            scored = rng.random() < share_scored
            status = "rejected" if rng.random() < p_fail else "admitted"
            stats.add(status, rng.normal(residual, 1.0) if scored else None)
        hits += vetoed(stats, config)
    return hits / reps


def veto_table(params: dict) -> pd.DataFrame:
    v, reps, seed = params["veto"], params["reps"], params["seed"]
    base = load_config()
    rows = []
    for kappa, tau_c in product(v["kappa"], v["confidence"]):
        config = replace(base, memory_kappa=kappa, memory_veto_confidence=tau_c)
        false = max(
            veto_rate(
                config, attempts=a, p_fail=f, share_scored=s, residual=0.0, reps=reps, seed=seed
            )
            for a, f, s in product(v["attempts"], v["null_failure"], v["null_scored"])
        )
        bad = {
            f"bad@{a}": veto_rate(
                config,
                attempts=a,
                p_fail=v["bad_failure"],
                share_scored=v["bad_scored"],
                residual=v["bad_residual"],
                reps=reps,
                seed=seed,
            )
            for a in v["attempts"]
        }
        never = min((a for a in range(1, 41) if vetoed(_all_rejected(a), config)), default=None)
        rows.append(
            {
                "kappa": kappa,
                "tau_c": tau_c,
                "worst_false_veto": false,
                **bad,
                "never_scored_at": never,
            }
        )
    table = pd.DataFrame(rows)
    table["meets_target"] = table["worst_false_veto"] <= v["target_false_veto"]
    return table


def _all_rejected(attempts: int) -> PairStats:
    stats = PairStats()
    for _ in range(attempts):
        stats.add("rejected", None)
    return stats


def correlation_table(params: dict, root: Path = ROOT) -> pd.DataFrame | None:
    c = params["correlation"]
    if not c["run"]:
        return None
    run = Path(root) / c["run"]
    summary = pd.read_csv(run / "summary.csv", index_col=0)
    matrix = pd.read_parquet(run / "matrix.parquet")
    corr = matrix.corr().abs()
    info = summary.loc[corr.index]
    upper = np.triu_indices(len(corr), 1)
    values = corr.to_numpy()[upper]
    fid = info["factor_id"].to_numpy()
    same = (fid[:, None] == fid[None, :])[upper]
    spec = (
        (info["hold"] == c["spec"]["hold"]) & (info["neutral"] == c["spec"]["neutral"])
    ).to_numpy()
    at_spec = (spec[:, None] & spec[None, :])[upper] & ~same
    rows = [
        {
            "cap": cap,
            "blocks_same_factor": float(np.mean(values[same] > cap)),
            "blocks_different_factors": float(np.mean(values[~same] > cap)),
            "blocks_different_at_spec": float(np.mean(values[at_spec] > cap)),
        }
        for cap in c["caps"]
    ]
    return pd.DataFrame(rows)


def main() -> None:
    params = tomllib.loads(PARAMS.read_text(encoding="utf-8"))
    veto, corr = veto_table(params), correlation_table(params)
    log_run(
        "memory-calibration",
        params=params,
        metrics={
            "veto": veto.to_dict("records"),
            "correlation": None if corr is None else corr.to_dict("records"),
        },
        seed=params["seed"],
        tags=["synthetic", "calibration", "learned-memory-search"],
    )
    with pd.option_context("display.width", 140, "display.precision", 3):
        print(veto.to_string(index=False))
        if corr is not None:
            print()
            print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
