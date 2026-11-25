"""Distributions of scores at each stage of the validation funnel (stages 1-3).

Report only: it reproduces the gate's stages 1-3 block (`gate.run_gate`) on a
run's search-period matrix without the look, and describes how variant Sharpe
ratios, variant and family p-values and the stage pass flags are distributed,
optionally against a placebo run scored the same way. Nothing here decides
anything. Dates past `search_end` are refused (`funnel.family_statistics`
without a `holdout_start`), so the holdout is never read.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from capstone import funnel, grading, intake, runlog, search_fdr
from capstone.analysis.outputs import Stat, Statement
from capstone.evaluate import average_correlation, bh_adjusted

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
DEFAULT_SEARCH_END = "2020-12-31"
LEDGER_NAME = "funnel-distributions"
NOMINAL = 0.05
MIN_BLOCKS = 10  # search_fdr.bootstrap_family_pvalues' default min_blocks
GRADED_COLUMNS = ["by", "holm", "lfdr", "sharpe", "psr", "min_trl_years", "haircut_sharpe"]


@dataclass(frozen=True)
class Scored:
    """One run's stage 1-3 scores."""

    variants: pd.DataFrame  # per variant
    families: pd.DataFrame  # per family
    pbo: float
    n_variants: int
    dropped: tuple[str, ...] = ()  # too few returns for the family bootstrap


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
    for c in ("sharpe_net", "sharpe_gross", "pvalue_screen", "mean_turnover"):
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
    # The family bootstrap needs every column to span `MIN_BLOCKS` blocks (search_fdr's
    # rule); a variant that rarely trades is dropped and counted, not allowed to crash.
    rows = fi.returns.dropna(how="all")
    need = max(3, MIN_BLOCKS * search_fdr.block_length_rule(len(rows)))
    short = rows.notna().sum() < need
    dropped = tuple(map(str, rows.columns[short]))
    if short.any():
        keep = rows.columns[~short]
        fi = dataclasses.replace(fi, returns=fi.returns[keep], family=fi.family[keep])
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
    return Scored(var, fam, grading.pbo(fi.returns), n_variants, dropped)


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
        ("variant_sharpe_gross", real.variants["sharpe_gross"], null.variants["sharpe_gross"]),
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


BAND_QUANTILES = (0.05, 0.5, 0.95)


def null_band_table(real: Scored, nulls: list[Scored]) -> pd.DataFrame:
    """Real value against the min / median / max over null seeds, per metric.

    `n_null_below_real` counts the seeds whose value the real one exceeds.
    """

    def row(metric: str, quantile: float, rv: float, nv: list[float]) -> dict:
        a = np.asarray(nv, dtype=float)
        return {
            "metric": metric,
            "quantile": quantile,
            "real": rv,
            "null_min": float(np.nanmin(a)),
            "null_median": float(np.nanmedian(a)),
            "null_max": float(np.nanmax(a)),
            "n_null": len(a),
            "n_null_below_real": int((rv > a).sum()),
            "share_null_below_real": float((rv > a).mean()),
        }

    rows = []
    for label, get in (
        ("variant_sharpe_net", lambda s: s.variants["sharpe_net"]),
        ("variant_sharpe_gross", lambda s: s.variants["sharpe_gross"]),
        ("family_p1_bootstrap", lambda s: s.families["p1_bootstrap"]),
    ):
        for p in BAND_QUANTILES:
            rows.append(
                row(
                    label,
                    p,
                    float(get(real).quantile(p)),
                    [float(get(n).quantile(p)) for n in nulls],
                )
            )
    rows.append(
        row(
            "stage1_survivors",
            -1.0,
            float(real.families["stage1"].sum()),
            [float(n.families["stage1"].sum()) for n in nulls],
        )
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
        f"{prefix}dropped_variants": Stat(
            list(s.dropped),
            f"variants left out: fewer returns than the family bootstrap needs "
            f"({MIN_BLOCKS} blocks)",
            inputs,
        ),
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
    null: Scored | None = None,
    inputs: tuple[str, ...],
    null_inputs: tuple[str, ...] = (),
    nulls: list[Scored] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Stat], list[Statement]]:
    """The tables, stats and statements for `outputs.write_outputs`.

    `nulls` (several placebo runs) gives the band statements; one null, given as
    `null` or as a single-item `nulls`, keeps the single-draw comparison.
    """
    if nulls is not None and len(nulls) == 1 and null is None:
        null, nulls = nulls[0], None
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
        *(
            [
                Statement(
                    f"{len(real.dropped)} variants were left out for too few returns.",
                    "dropped_variants",
                )
            ]
            if real.dropped
            else []
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
        gross = cmp.loc["variant_sharpe_gross", "gap"]
        gross_gaps = {f"q{int(round(p * 100)):03d}": float(gross.loc[p]) for p in QUANTILES}
        stats["gross_sharpe_quantile_gaps"] = Stat(
            gross_gaps,
            "real minus null quantile of variants' gross Sharpe (before costs), "
            "at 0, 5, 25, 50, 75, 95, 100%",
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
            Statement(
                f"Before costs, real variants' median Sharpe sits {gross_gaps['q050']:+.2f} "
                f"from the null's, and the 95th percentile {gross_gaps['q095']:+.2f}.",
                "gross_sharpe_quantile_gaps",
                "A cross-section permutation trades far more than a real signal, so net "
                "Sharpe overstates the gap; gross is the fairer comparison.",
            ),
        ]
        if null.dropped:
            statements.append(
                Statement(
                    f"{len(null.dropped)} null variants were left out for too few returns.",
                    "null_dropped_variants",
                )
            )
    if nulls:
        _add_band(tables, stats, statements, real, nulls, inputs + null_inputs, base)
    return tables, stats, statements


def _add_band(
    tables: dict[str, pd.DataFrame],
    stats: dict[str, Stat],
    statements: list[Statement],
    real: Scored,
    nulls: list[Scored],
    both: tuple[str, ...],
    base: str,
) -> None:
    n = len(nulls)
    band = null_band_table(real, nulls)
    tables["null_band"] = band
    tables["null_seeds"] = pd.DataFrame(
        {
            "n_variants": [s.n_variants for s in nulls],
            "n_dropped": [len(s.dropped) for s in nulls],
            "stage1_survivors": [int(s.families["stage1"].sum()) for s in nulls],
            "sharpe_gross_q050": [s.variants["sharpe_gross"].quantile(0.5) for s in nulls],
            "sharpe_gross_q095": [s.variants["sharpe_gross"].quantile(0.95) for s in nulls],
            "sharpe_net_q050": [s.variants["sharpe_net"].quantile(0.5) for s in nulls],
        }
    ).rename_axis("null_seed")
    surv = band.loc[("stage1_survivors", -1.0)]
    sv = {
        "real": int(surv["real"]),
        "null_min": int(surv["null_min"]),
        "null_median": float(surv["null_median"]),
        "null_max": int(surv["null_max"]),
        "n_null": n,
    }
    stats["n_null_runs"] = Stat(n, "number of placebo runs in the band", both)
    stats["null_band_stage1_survivors"] = Stat(
        sv, "stage-1 survivor families: real run vs min / median / max over null runs", both
    )
    gross = {}
    for q, name in ((0.5, "median"), (0.95, "p95")):
        r = band.loc[("variant_sharpe_gross", q)]
        real_q = real.variants["sharpe_gross"].quantile(q)
        gaps = [real_q - s.variants["sharpe_gross"].quantile(q) for s in nulls]
        gross[name] = {
            "k": int(r["n_null_below_real"]),
            "n": n,
            "gap_min": float(min(gaps)),
            "gap_max": float(max(gaps)),
        }
    stats["null_band_gross_sharpe"] = Stat(
        gross,
        "variants' gross Sharpe (before costs): seeds where the real median / 95th percentile "
        "exceeds the null's, with the range of real-minus-null gaps",
        both,
    )
    net = band.loc[("variant_sharpe_net", 0.5)]
    stats["null_band_net_sharpe_median"] = Stat(
        {
            "real": float(net["real"]),
            "null_min": float(net["null_min"]),
            "null_max": float(net["null_max"]),
            "k": int(net["n_null_below_real"]),
            "n": n,
        },
        "variants' median net Sharpe: real vs range over null runs",
        both,
    )
    statements += [
        Statement(
            f"The real run has {sv['real']} stage-1 survivors; the {n} null runs have "
            f"{sv['null_min']} to {sv['null_max']} (median {sv['null_median']:g}).",
            "null_band_stage1_survivors",
            f"A null should have about none; {n} draws bound the range only roughly. " + base,
        ),
        Statement(
            f"Real median gross Sharpe exceeds the null median in {gross['median']['k']} of "
            f"{n} seeds, with real-minus-null gaps from {gross['median']['gap_min']:+.2f} to "
            f"{gross['median']['gap_max']:+.2f}.",
            "null_band_gross_sharpe",
            "Search period only; the placebo is a cross-section permutation, so it is only as "
            "comparable as its construction is to the real run.",
        ),
        Statement(
            f"Real 95th-percentile gross Sharpe exceeds the null's in {gross['p95']['k']} of "
            f"{n} seeds, with gaps from {gross['p95']['gap_min']:+.2f} to "
            f"{gross['p95']['gap_max']:+.2f}.",
            "null_band_gross_sharpe",
            "The upper tail of many correlated variants is driven by few families.",
        ),
        Statement(
            f"Real median net Sharpe is {net['real']:.2f} against a null range of "
            f"{net['null_min']:.2f} to {net['null_max']:.2f}; net comparisons are dominated "
            "by the placebo's much higher turnover costs, so gross is the fair comparison.",
            "null_band_net_sharpe_median",
            "A permuted signal trades far more than a real one.",
        ),
    ]
    dropped = sum(len(s.dropped) for s in nulls)
    if dropped:
        stats["null_dropped_total"] = Stat(
            dropped, "null variants left out for too few returns, summed over null runs", both
        )
        statements.append(
            Statement(
                f"{dropped} null variants were left out for too few returns across "
                f"the {n} null runs.",
                "null_dropped_total",
            )
        )


def run_analysis(
    matrix_path: Path,
    summary_path: Path,
    *,
    run: str,
    null_matrix_path: Path | None = None,
    null_summary_path: Path | None = None,
    null_dirs: list[Path] | None = None,
    null_n_boot: int | None = None,
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
        "null_n_boot": null_n_boot if null_n_boot is not None else n_boot,
        "null_dirs": {
            str(d): {
                "matrix_digest": file_digest(Path(d) / "matrix.parquet"),
                "summary_digest": file_digest(Path(d) / "summary.parquet"),
            }
            for d in null_dirs or ()
        },
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

    nb = null_n_boot if null_n_boot is not None else n_boot

    def load(m: Path, s: Path, n_boot: int = n_boot) -> Scored:
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
    ins = (Path(matrix_path).name, Path(summary_path).name)
    dirs = [Path(d) for d in null_dirs or ()]
    if dirs:
        if null_matrix_path is not None:
            raise ValueError("give null dirs or a null matrix, not both")
        nulls = [load(d / "matrix.parquet", d / "summary.parquet", nb) for d in dirs]
        null_ins = tuple(
            f"{d.name}/{f}" for d in dirs for f in ("matrix.parquet", "summary.parquet")
        )
        return build_outputs(real, nulls=nulls, inputs=ins, null_inputs=null_ins)
    null = load(null_matrix_path, null_summary_path, nb) if null_matrix_path is not None else None
    null_ins = (Path(null_matrix_path).name, Path(null_summary_path).name) if null else ()
    return build_outputs(real, null=null, inputs=ins, null_inputs=null_ins)
