"""Measure a generator's search structure from its performance matrix.

    python scripts/measure_generator.py --matrix <run>_matrix.parquet \
        --summary <run>_summary.parquet [--out structure.json]

The calibration (`docs/validation/framework.md`, "Calibration") simulates
candidate sets whose structure matches the real generator's output. This
measures that structure: how many variants each hypothesis family searched,
how correlated variants are within and across families, and how their returns
behave over time (autocorrelation, tails, volatility clustering).

It reads only the return matrix and each variant's family. It never keeps or
reports a mean, Sharpe ratio or p-value (returns are only centred), so nothing
it reports says whether any strategy works, and the thresholds calibrated from
it can't be steered by which real candidates look good. The result is logged to
the ledger (tags `synthetic` and `methodology`: bookkeeping, not a trial) before
it is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.runlog import log_run

MIN_OVERLAP = 252  # days two variants must share for their correlation to count


def _pair_means(corr: pd.DataFrame, family: pd.Series) -> tuple[float, float, float]:
    """Mean pairwise correlation within a family, across families, and within a factor."""
    c = corr.to_numpy()
    fam = family.reindex(corr.index).to_numpy()
    factor = np.array([v.split("-")[0] for v in corr.index])
    upper = np.triu(np.ones_like(c, dtype=bool), k=1) & np.isfinite(c)
    same_fam = fam[:, None] == fam[None, :]
    same_factor = factor[:, None] == factor[None, :]
    return (
        float(c[upper & same_fam].mean()),
        float(c[upper & ~same_fam].mean()),
        float(c[upper & same_factor].mean()),
    )


def _within_by_family(corr: pd.DataFrame, family: pd.Series) -> dict[str, float]:
    """Mean pairwise correlation within each family, so no one family's pairs dominate."""
    out = {}
    for name, members in family.groupby(family).groups.items():
        c = corr.loc[members, members].to_numpy()
        upper = np.triu(np.ones_like(c, dtype=bool), k=1) & np.isfinite(c)
        out[str(name)] = round(float(c[upper].mean()), 4)
    return out


def measure(returns: pd.DataFrame, family: pd.Series) -> dict:
    """Structure of a dates x variants matrix with one family label per variant."""
    returns = returns[family.index]
    corr = returns.corr(min_periods=MIN_OVERLAP)
    within, across, within_factor = _pair_means(corr, family)
    by_family = _within_by_family(corr, family)

    demeaned = returns - returns.mean()  # centred only; the mean itself is never kept
    ar1 = returns.apply(lambda s: s.dropna().autocorr(1))
    sq_ar1 = (demeaned**2).apply(lambda s: s.dropna().autocorr(1))
    kurt = returns.apply(lambda s: s.dropna().kurt())  # excess kurtosis
    days = returns.notna().sum()
    variants = family.value_counts()

    def q(s: pd.Series) -> dict:
        p25, p50, p75 = s.quantile([0.25, 0.5, 0.75])
        return {"p25": round(p25, 4), "median": round(p50, 4), "p75": round(p75, 4)}

    return {
        "n_families": int(variants.size),
        "variants_per_family": sorted(int(v) for v in variants),
        "n_variants": int(variants.sum()),
        "n_dates": int(len(returns)),
        "dates": [str(returns.index.min().date()), str(returns.index.max().date())],
        "days_per_variant": q(days),
        "within_family_corr": round(within, 4),
        "across_family_corr": round(across, 4),
        "within_factor_corr": round(within_factor, 4),
        "within_family_corr_by_family": by_family,
        "within_family_corr_family_median": round(float(np.median(list(by_family.values()))), 4),
        "ar1": q(ar1),
        "excess_kurtosis": q(kurt),
        "squared_ar1": q(sq_ar1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="the run's manifest, for its ids")
    parser.add_argument("--out", type=Path, help="also write the structure as JSON here")
    args = parser.parse_args()

    returns = pd.read_parquet(args.matrix)
    if "date" in returns.columns:
        returns = returns.set_index("date")
    returns.index = pd.DatetimeIndex(returns.index)
    # Only the family label: the summary's performance columns are never loaded.
    family = pd.read_parquet(args.summary, columns=["hypothesis_id"])["hypothesis_id"]

    params = {"matrix": args.matrix.name, "min_overlap": MIN_OVERLAP}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        params |= {k: manifest.get(k) for k in ("run_id", "data_version", "matrix_digest")}

    structure = measure(returns, family)
    log_run(
        "generator-structure",
        params=params,
        metrics=structure,
        tags=["synthetic", "methodology", "calibration"],
        notes="structure only: no mean, Sharpe or p-value is read",
    )
    if args.out:
        args.out.write_text(json.dumps(structure, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(structure, indent=2))


if __name__ == "__main__":
    main()
