"""Distributions of scores at each stage of the validation funnel (stages 1-3).

Report only: it reproduces the gate's stages 1-3 block (`gate.run_gate`) on a
run's search-period matrix without the look, and describes how variant Sharpe
ratios, variant and family p-values and the stage pass flags are distributed,
optionally against a placebo run scored the same way. Nothing here decides
anything. Dates past `search_end` are refused (`funnel.family_statistics`
without a `holdout_start`), so the holdout is never read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from capstone import funnel, grading, intake, runlog
from capstone.analysis.outputs import Stat, Statement
from capstone.evaluate import average_correlation, bh_adjusted

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
DEFAULT_SEARCH_END = "2020-12-31"
LEDGER_NAME = "funnel-distributions"
NOMINAL = 0.05
GRADED_COLUMNS = ["by", "holm", "lfdr", "sharpe", "psr", "min_trl_years", "haircut_sharpe"]


@dataclass(frozen=True)
class Scored:
    """One run's stage 1-3 scores."""

    variants: pd.DataFrame  # per variant
    families: pd.DataFrame  # per family
    pbo: float
    n_variants: int


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def default_thresholds(q1: float = 0.1) -> funnel.Thresholds:
    """The registered gate's stage 1-3 thresholds: bootstrap stage 1, no stage-2 gate."""
    return funnel.Thresholds(stage1="bootstrap", q1=q1, alpha2=1.0)


def variant_table(summary: pd.DataFrame, classes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-variant score columns from an evaluation summary, with class columns joined on factor."""
    out = pd.DataFrame(index=summary.index)
    for c in ("sharpe_net", "pvalue_screen", "mean_turnover"):
        out[c] = summary[c] if c in summary else np.nan
    out["family"] = summary["hypothesis_id"]
    factor = pd.Series(summary.index.astype(str), index=summary.index).str.split("-").str[0]
    out["factor_id"] = summary["factor_id"] if "factor_id" in summary else factor
    if classes is not None:
        cl = classes.set_index("factor_id") if "factor_id" in classes.columns else classes
        cl = cl.loc[~cl.index.duplicated()]
        out = out.join(cl.add_prefix("class_"), on="factor_id")
    return out


def score_run(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    search_end: str = DEFAULT_SEARCH_END,
    n_boot: int = 1999,
    seed: int = 0,
    thresholds: funnel.Thresholds | None = None,
    classes: pd.DataFrame | None = None,
) -> Scored:
    """Stages 1-3 for a matrix, as the gate computes them, without opening a holdout.

    Raises ValueError if the matrix has any date after `search_end`.
    """
    th = thresholds or default_thresholds()
    fi = intake.from_evaluation(matrix, summary)
    stats = funnel.family_statistics(
        fi.returns, fi.family, search_end=search_end, n_boot=n_boot, seed=seed, q=th.q1
    )["reuse"]
    stages = funnel.screen(stats, th)
    p1 = funnel.stage1_pvalues(stats, th.stage1)
    n_variants = fi.returns.shape[1]
    avg_corr = average_correlation(fi.returns) if n_variants > 1 else float("nan")
    graded = grading.graded_scores(fi.returns, stats["best"], p1, n_variants, avg_corr, seed)
    fam = pd.DataFrame(
        {
            "k": fi.family.value_counts().reindex(stats.index),
            "best": stats["best"],
            "p1_bootstrap": p1,
            "p1_bh_adjusted": bh_adjusted(p1),
            "stage1": stages["stage1"],
            "stage2": stages["stage2"],
            "stage3": stages["stage3"],
        }
    ).join(graded[GRADED_COLUMNS].add_prefix("graded_"))
    var = variant_table(summary.reindex(fi.returns.columns), classes)
    return Scored(var, fam, grading.pbo(fi.returns), n_variants)


def quantile_table(variants: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of variant Sharpe and screen p, for all variants and by family."""
    rows = []
    groups = [("ALL", variants)] + [(str(f), g) for f, g in variants.groupby("family")]
    for name, g in groups:
        for metric in ("sharpe_net", "pvalue_screen"):
            q = g[metric].quantile(list(QUANTILES))
            rows.append(
                {"family": name, "metric": metric, "n": int(g[metric].notna().sum())}
                | {f"q{int(round(p * 100)):03d}": v for p, v in zip(QUANTILES, q, strict=True)}
            )
    return pd.DataFrame(rows).set_index(["family", "metric"])


def ks_statistic(a: pd.Series, b: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max gap between the empirical CDFs)."""
    a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
    if not len(a) or not len(b):
        return float("nan")
    return float(sps.ks_2samp(a, b).statistic)


def comparison_table(real: Scored, null: Scored) -> pd.DataFrame:
    """Quantiles of variant Sharpe and family p1, real vs null, with the gap and KS statistic."""
    rows = []
    for label, r, n in (
        ("variant_sharpe_net", real.variants["sharpe_net"], null.variants["sharpe_net"]),
        ("family_p1_bootstrap", real.families["p1_bootstrap"], null.families["p1_bootstrap"]),
    ):
        ks = ks_statistic(r, n)
        for p in QUANTILES:
            rq, nq = float(r.quantile(p)), float(n.quantile(p))
            rows.append(
                {
                    "metric": label,
                    "quantile": p,
                    "real": rq,
                    "null": nq,
                    "gap": rq - nq,
                    "ks_statistic": ks,
                }
            )
    return pd.DataFrame(rows).set_index(["metric", "quantile"])


def _summary_stats(prefix: str, s: Scored, inputs: tuple[str, ...]) -> dict[str, Stat]:
    v, f = s.variants, s.families
    p = v["pvalue_screen"].dropna()
    sq = v["sharpe_net"].quantile([0.05, 0.5, 0.95])
    surv = {stage: sorted(map(str, f.index[f[stage]])) for stage in ("stage1", "stage2", "stage3")}
    return {
        f"{prefix}n_families": Stat(len(f), "number of hypothesis families scored", inputs),
        f"{prefix}n_variants": Stat(s.n_variants, "number of variants scored", inputs),
        f"{prefix}stage1_survivors": Stat(
            len(surv["stage1"]),
            "families passing stage 1: BH at q1 on the family-max bootstrap p-values",
            inputs,
        ),
        f"{prefix}stage2_survivors": Stat(
            len(surv["stage2"]), "stage-1 survivors passing the stage-2 p cut-off", inputs
        ),
        f"{prefix}stage3_survivors": Stat(
            len(surv["stage3"]),
            "stage-2 survivors passing stage 3 (spanning is not measured: passes stage 2 through)",
            inputs,
        ),
        f"{prefix}survivor_families": Stat(
            surv, "family ids passing each of stages 1, 2 and 3", inputs
        ),
        f"{prefix}share_p_below_5pct": Stat(
            float((p < NOMINAL).mean()) if len(p) else float("nan"),
            "share of variants whose one-sided screen p-value is below 0.05 "
            "(5% expected if every variant were null and independent)",
            inputs,
        ),
        f"{prefix}sharpe_quantiles": Stat(
            {"p05": sq.iloc[0], "median": sq.iloc[1], "p95": sq.iloc[2]},
            "5th, 50th and 95th percentiles of variants' net annualised Sharpe ratio",
            inputs,
        ),
        f"{prefix}pbo": Stat(
            s.pbo,
            "probability of backtest overfitting (CSCV, 16 blocks) over all variants; "
            "NaN if too few variants or rows",
            inputs,
        ),
    }


def build_outputs(
    real: Scored,
    *,
    null: Scored | None,
    inputs: tuple[str, ...],
    null_inputs: tuple[str, ...] = (),
) -> tuple[dict[str, pd.DataFrame], dict[str, Stat], list[Statement]]:
    """The tables, stats and statements for `outputs.write_outputs`."""
    tables = {
        "variants": real.variants,
        "families": real.families,
        "quantiles": quantile_table(real.variants),
    }
    stats = _summary_stats("", real, inputs)
    sq = stats["sharpe_quantiles"].value
    share = stats["share_p_below_5pct"].value
    base = "Search period only; family-max bootstrap p-values; one run, one seed."
    statements = [
        Statement(
            f"{stats['n_variants'].value} variants in {stats['n_families'].value} families "
            "were scored.",
            "n_variants",
        ),
        Statement(
            f"{stats['stage1_survivors'].value} families pass stage 1, "
            f"{stats['stage2_survivors'].value} pass stage 2 and "
            f"{stats['stage3_survivors'].value} pass stage 3.",
            "stage1_survivors",
            base,
        ),
        Statement(
            f"{share:.1%} of variants have a one-sided screen p-value below 0.05, "
            "against 5% if all were null.",
            "share_p_below_5pct",
            "Variants within a family are highly correlated, so the share is not "
            "an independent-test rate.",
        ),
        Statement(
            f"Variant net Sharpe has 5th percentile {sq['p05']:.2f}, median {sq['median']:.2f} "
            f"and 95th percentile {sq['p95']:.2f}.",
            "sharpe_quantiles",
            "Net of costs, search period only.",
        ),
        Statement(
            f"Probability of backtest overfitting is {stats['pbo'].value:.2f}.",
            "pbo",
            "Over all variants of the run as one batch; NaN means too few to compute.",
        ),
        Statement(
            f"Stage-1 survivors: {stats['survivor_families'].value['stage1']}.",
            "survivor_families",
            base,
        ),
    ]
    if null is not None:
        tables["null_variants"] = null.variants
        tables["null_families"] = null.families
        tables["null_quantiles"] = quantile_table(null.variants)
        cmp = comparison_table(real, null)
        tables["comparison"] = cmp
        stats |= _summary_stats("null_", null, null_inputs)
        gap = cmp.loc["variant_sharpe_net", "gap"]
        gaps = {f"q{int(round(p * 100)):03d}": float(gap.loc[p]) for p in QUANTILES}
        both = inputs + null_inputs
        stats["sharpe_quantile_gaps"] = Stat(
            gaps,
            "real minus null quantile of variants' net Sharpe, at 0, 5, 25, 50, 75, 95, 100%",
            both,
        )
        stats["sharpe_ks"] = Stat(
            float(cmp.loc["variant_sharpe_net", "ks_statistic"].iloc[0]),
            "two-sample KS statistic between real and null variant Sharpe",
            both,
        )
        stats["stage1_survivors_real_vs_null"] = Stat(
            {"real": stats["stage1_survivors"].value, "null": stats["null_stage1_survivors"].value},
            "stage-1 survivor families, real run vs null run",
            both,
        )
        statements += [
            Statement(
                f"The null run has {stats['null_stage1_survivors'].value} stage-1 survivors "
                f"against {stats['stage1_survivors'].value} in the real run.",
                "stage1_survivors_real_vs_null",
                "A null run should have about none; one null run is one draw. " + base,
            ),
            Statement(
                f"Real variants' median net Sharpe sits {gaps['q050']:+.2f} from the null's, "
                f"and the 95th percentile {gaps['q095']:+.2f} (KS statistic "
                f"{stats['sharpe_ks'].value:.2f}).",
                "sharpe_quantile_gaps",
                "Only as comparable as the placebo's construction is to the real run.",
            ),
        ]
    return tables, stats, statements


def run_analysis(
    matrix_path: Path,
    summary_path: Path,
    *,
    run: str,
    null_matrix_path: Path | None = None,
    null_summary_path: Path | None = None,
    classes_path: Path | None = None,
    n_boot: int = 1999,
    seed: int = 0,
    q1: float = 0.1,
    search_end: str = DEFAULT_SEARCH_END,
) -> tuple[dict[str, pd.DataFrame], dict[str, Stat], list[Statement]]:
    """Log the ledger entry, then score and describe. The entry precedes every result."""
    th = default_thresholds(q1)
    if null_matrix_path is not None and null_summary_path is None:
        raise ValueError("a null matrix needs a null summary")
    params = {
        "run": run,
        "matrix": Path(matrix_path).name,
        "matrix_digest": file_digest(matrix_path),
        "null_matrix_digest": file_digest(null_matrix_path) if null_matrix_path else None,
        "n_boot": n_boot,
        "seed": seed,
        "search_end": search_end,
        "thresholds": {"stage1": th.stage1, "q1": th.q1, "alpha2": th.alpha2},
    }
    runlog.log_run(
        LEDGER_NAME,
        params=params,
        seed=seed,
        tags=["synthetic", "methodology"],
        notes="report-only distributions of funnel stage 1-3 scores; no holdout, no decision",
    )
    classes = pd.read_parquet(classes_path) if classes_path else None

    def load(m: Path, s: Path) -> Scored:
        matrix = pd.read_parquet(m)
        if "date" in matrix.columns:
            matrix = matrix.set_index("date")
        matrix = matrix.set_axis(pd.DatetimeIndex(matrix.index))
        return score_run(
            matrix,
            pd.read_parquet(s),
            search_end=search_end,
            n_boot=n_boot,
            seed=seed,
            thresholds=th,
            classes=classes,
        )

    real = load(matrix_path, summary_path)
    null = load(null_matrix_path, null_summary_path) if null_matrix_path is not None else None
    ins = (Path(matrix_path).name, Path(summary_path).name)
    null_ins = (Path(null_matrix_path).name, Path(null_summary_path).name) if null else ()
    return build_outputs(real, null=null, inputs=ins, null_inputs=null_ins)
