"""Does hidden specification search inflate the FDR of what we report? (issue #41)

Each of `n_families` hypotheses tries K variants and reports only its best.
A share of the hypotheses are real. The validator sees one winner per
hypothesis, and selects with:

- naive BH on the winners' p-values, as if each were a single trial;
- BH on search-adjusted p-values, 1 - (1 - p)^K (`search_fdr`);
- the deflated Sharpe at `dsr_cutoff`, counting trials as reported
  (n_families) and as searched (n_families x K).

Realised FDR is the share of selected hypotheses with no real variant. If it
climbs with K for the methods that ignore the search, the search matters.

Parameters live in `search_fdr_experiment.json`. Each grid cell is logged to
the ledger under "search-fdr-hidden-search" (tag "synthetic") as soon as its
numbers exist, before anything is printed. Run:

    python scripts/search_fdr_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from capstone.evaluate import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    false_discovery_rate,
    power,
)
from capstone.runlog import log_run
from capstone.search_fdr import search_adjusted_pvalue, simulate_family_winners

PARAMS = Path(__file__).with_suffix(".json")


def _score(selected: pd.Series, truth: pd.Series) -> dict:
    return {
        "selected": int(selected.sum()),
        "selected_null": int((selected & ~truth).sum()),
        "fdr": false_discovery_rate(selected, truth) if selected.any() else 0.0,
        "power": power(selected, truth),
    }


def run_cell(p: dict, k: int, within_rho: float, seed: int) -> dict:
    d = simulate_family_winners(
        p["n_families"],
        k,
        p["share_real"],
        p["sharpe_real"],
        p["n_obs"],
        within_rho=within_rho,
        periods_per_year=p["periods_per_year"],
        seed=seed,
    )
    truth = d["truth"]

    def dsr_selected(n_trials: int) -> pd.Series:
        dsr = d["sharpe"].apply(
            lambda s: deflated_sharpe_ratio(
                s, n_trials, p["n_obs"], periods_per_year=p["periods_per_year"]
            )
        )
        return dsr >= p["dsr_cutoff"]

    results = {
        "bh_naive": benjamini_hochberg(d["pvalue"], p["alpha"]),
        "bh_search_adjusted": benjamini_hochberg(
            search_adjusted_pvalue(d["pvalue"], k), p["alpha"]
        ),
        "dsr_reported_trials": dsr_selected(p["n_families"]),
        "dsr_searched_trials": dsr_selected(p["n_families"] * k),
    }
    return {name: _score(sel, truth) for name, sel in results.items()}


def main() -> None:
    p = json.loads(PARAMS.read_text())
    seeds = range(*p["seeds"])
    rows = []
    for within_rho in p["within_rho_grid"]:
        for k in p["k_grid"]:
            cell = []
            for seed in seeds:
                for method, sc in run_cell(p, k, within_rho, seed).items():
                    cell.append({"within_rho": within_rho, "k": k, "method": method, **sc})
            # One ledger entry per configuration: the seeds are replications of
            # one simulated design, not separate hypotheses tested on data.
            means = pd.DataFrame(cell).groupby("method")[["fdr", "power"]].mean()
            log_run(
                "search-fdr-hidden-search",
                params={
                    **{x: v for x, v in p.items() if x != "seeds"},
                    "k": k,
                    "within_rho": within_rho,
                    "seed_range": p["seeds"],
                },
                metrics={
                    f"{m}_{s}": float(v) for m, row in means.iterrows() for s, v in row.items()
                },
                tags=["synthetic"],
            )
            rows.extend(cell)

    table = (
        pd.DataFrame(rows)
        .groupby(["within_rho", "method", "k"])[["selected", "selected_null", "fdr", "power"]]
        .mean()
        .round(3)
    )
    pd.set_option("display.width", 120)
    print(
        f"{len(seeds)} seeds per cell; {p['n_families']} hypotheses, "
        f"{p['share_real']:.0%} real, annualised Sharpe {p['sharpe_real']}, "
        f"{p['n_obs']} periods, alpha {p['alpha']}\n"
    )
    for within_rho, block in table.groupby(level="within_rho"):
        print(f"within-family correlation {within_rho}:")
        print(block.droplevel("within_rho").unstack("k").to_string())
        print()
    n_logged = len(p["within_rho_grid"]) * len(p["k_grid"])
    print(f"logged {n_logged} runs to the ledger as 'search-fdr-hidden-search'")


if __name__ == "__main__":
    main()
