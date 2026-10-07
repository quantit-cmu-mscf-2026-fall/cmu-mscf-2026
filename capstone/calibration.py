"""Calibrating the validation pipeline on signals with known truth.

`docs/validation/framework.md` gives each method one role and each stage one
gate. This module measures whether that works: it feeds the pipeline simulated
hypothesis families where we know which are real, and reports, for every gate
and for the whole pipeline,

- the **false-pass rate** on signal-free candidates (is the gate calibrated?),
- the **pass rate on real signals** (is it too strict?),
- the **false-discovery rate** of what comes out (the target is 10%, #50 Q5),
- the **true discoveries**, and the share of accepted candidates that are real.

It compares three ways of deciding (#50, Q5):

- **naive stack:** every screening method as a gate at once, on the whole
  search period, then the final holdout test;
- **baseline:** BH at q on each family's search-adjusted p-value over the
  search period, the rule that decides until the funnel beats it;
- **funnel:** stage 1 screens families, stage 2 checks survivors (by default
  on the same search period, with no gate: the registered design in
  `docs/validation/framework.md`; "split" confirms on years stage 1 didn't
  use instead), stage 3 (spanning) is a placeholder that passes everything
  and is reported as not measured, stage 4 tests the survivors on the
  holdout.

Each simulated data set is summarised once per hypothesis (`family_statistics`);
every pipeline and threshold setting is then a cheap filter over that summary,
so threshold sweeps cost almost nothing. `scripts/calibrate_validation.py`
runs the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone.evaluate import (
    average_correlation,
    benjamini_hochberg,
    bh_adjusted,
    by_adjusted,
    deflated_sharpe_ratio,
    holm_adjusted,
    sharpe_test,
)
from capstone.search_fdr import bootstrap_family_pvalues, search_adjusted_pvalue
from capstone.synth import make_return_matrix

STAGE1_METHODS = ("independent", "rho", "bootstrap")

# ---------------------------------------------------------------------------
# Test signals with known truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySet:
    """Simulated hypothesis families, split into periods by position.

    returns: dates x variants. family: hypothesis label per variant column.
    truth: per hypothesis, True when its variants carry a real signal.
    n_search / n_holdout: the first n_search rows are the search period
    (agents and stages 1-3), the last n_holdout the holdout (stage 4 only).
    """

    returns: pd.DataFrame
    family: pd.Series
    truth: pd.Series
    n_search: int
    n_holdout: int

    @property
    def search(self) -> pd.DataFrame:
        return self.returns.iloc[: self.n_search]

    @property
    def holdout(self) -> pd.DataFrame:
        return self.returns.iloc[self.n_search : self.n_search + self.n_holdout]


def make_family_set(
    n_families: int,
    k: int,
    share_real: float,
    sharpe_real: float,
    *,
    within_rho: float = 0.9,
    n_search: int = 31 * 252,
    n_holdout: int = 5 * 252,
    seed: int = 0,
    **defects,
) -> FamilySet:
    """Families of `k` variants of one idea, with realistic defects.

    Each variant is sqrt(within_rho) x its family's series + sqrt(1 - within_rho)
    x its own, both from `synth.make_return_matrix` with the same `defects`
    (`rho`, `ar1`, `t_df`, `garch`, `vol`), so variants of one idea are
    near-copies and every defect carries through. A real family's series is
    scaled so that each of its *variants* has annualised Sharpe `sharpe_real`.
    The default periods follow #43: 31 years of search, 5 of holdout.
    """
    if not 0.0 < within_rho <= 1.0:
        raise ValueError("within_rho must be in (0, 1]")
    n_obs = n_search + n_holdout
    n_real = int(round(share_real * n_families))
    base = make_return_matrix(
        n_obs=n_obs,
        n_candidates=n_families,
        n_real=n_real,
        sharpe_real=sharpe_real / np.sqrt(within_rho),
        seed=seed,
        **defects,
    )
    own = make_return_matrix(
        n_obs=n_obs, n_candidates=n_families * k, n_real=0, seed=seed + 1_000_003, **defects
    )
    b = base.returns.to_numpy()
    e = own.returns.to_numpy().reshape(n_obs, n_families, k)
    x = np.sqrt(within_rho) * b[:, :, None] + np.sqrt(1.0 - within_rho) * e
    hypotheses = list(base.returns.columns)
    columns = [f"{h}_v{j}" for h in hypotheses for j in range(k)]
    returns = pd.DataFrame(
        x.reshape(n_obs, n_families * k), index=base.returns.index, columns=columns
    )
    family = pd.Series(np.repeat(hypotheses, k), index=columns, name="hypothesis")
    return FamilySet(returns, family, base.truth.rename("truth"), n_search, n_holdout)


# ---------------------------------------------------------------------------
# One summary per hypothesis, computed once per data set
# ---------------------------------------------------------------------------


def _one_sided_p(returns: pd.DataFrame) -> pd.Series:
    return pd.Series({c: sharpe_test(returns[c]).pvalue_greater for c in returns.columns})


def _stage1(
    window: pd.DataFrame, p: pd.Series, family: pd.Series, n_boot: int, seed: int | None
) -> pd.DataFrame:
    """Each family's best variant on `window` and its family p-value, three ways.

    `p` is each variant's one-sided HAC p-value on `window`. `n_boot=0` skips
    the bootstrap (its column is then NaN), which is most of the cost.
    """
    sharpe = (window.mean() / window.std()).dropna()
    # Group only variants with a Sharpe: pandas 3 raises on an all-NaN group.
    best = sharpe.groupby(family[sharpe.index]).idxmax().reindex(family.unique())
    if best.isna().any():
        empty = best[best.isna()].index[0]
        raise ValueError(f"hypothesis {empty!r} has no variant with data in this window")
    k = family.value_counts()
    rho = pd.Series(
        {h: max(average_correlation(window[cols.index]), 0.0) for h, cols in family.groupby(family)}
    )
    best_p = p.groupby(family).min()
    return pd.DataFrame(
        {
            "best": best,
            "p1_independent": search_adjusted_pvalue(best_p, k),
            "p1_rho": search_adjusted_pvalue(best_p, k, rho=rho),
            "p1_bootstrap": (
                bootstrap_family_pvalues(window, family, n_boot=n_boot, seed=seed)["pvalue"]
                if n_boot
                else np.nan
            ),
        }
    )


def family_statistics(
    fs: FamilySet, *, stage2_years: float = 7.0, n_boot: int = 1999, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Everything the pipelines need, per hypothesis, for both stage-2 designs.

    Returns {"reuse": ..., "split": ...}. In "reuse", the registered design
    (#50, "Stage 2"), both stages see the whole search period and stage 2 is
    not a gate. In "split", stage 1 sees the search period minus its last
    `stage2_years`, and stage 2 confirms on those years; calibration found it
    costs power. The baseline and the naive stack always use the whole search
    period, so they read "reuse".

    Columns: `best` (the variant carried forward), `p1_<method>` for each
    stage-1 method, `p2` (one-sided HAC p of `best` at stage 2), the holdout
    `ho_*` fields stage 4 needs, `naive_pass` (some variant passed every
    stacked screening gate) and `truth`.
    """
    n_b = int(round(stage2_years * 252))
    if not 0 < n_b < fs.n_search:
        raise ValueError("stage2_years must leave data for both stages")
    search = fs.search
    first, last = search.iloc[: fs.n_search - n_b], search.iloc[fs.n_search - n_b :]
    # Every variant is tested in each period the pipelines use; name the first
    # one that can't be, rather than failing deep inside a Sharpe test.
    for name, window in (("search", first), ("stage 2", last), ("holdout", fs.holdout)):
        thin = window.columns[window.notna().sum() < 3]
        if len(thin):
            raise ValueError(
                f"hypothesis {fs.family[thin[0]]!r} has no variant with data in the "
                f"{name} period ({thin[0]!r} has fewer than 3 values)"
            )

    # Holdout statistics for every variant a pipeline might carry forward.
    holdout_tests = {c: sharpe_test(fs.holdout[c]) for c in fs.returns.columns}

    # Naive stack: every screening method as a gate, at once, on every variant
    # over the whole search period.
    full_tests = {c: sharpe_test(search[c]) for c in search.columns}
    p_all = pd.Series({c: t.pvalue_greater for c, t in full_tests.items()})
    n_variants = len(p_all)
    dsr_all = pd.Series(
        {
            c: deflated_sharpe_ratio(
                t.sharpe, n_variants, t.n_obs, t.skew, t.kurtosis, variance=t.variance
            )
            for c, t in full_tests.items()
        }
    )
    psr_all = pd.Series({c: t.psr for c, t in full_tests.items()})
    stacked = (
        (holm_adjusted(p_all) <= 0.05)
        & (by_adjusted(p_all) <= 0.05)
        & (bh_adjusted(p_all) <= 0.05)
        & (psr_all >= 0.95)
        & (dsr_all >= 0.95)
    )
    # The variant the naive stack carries forward: its best that passed.
    full_sharpe = search.mean() / search.std()
    # Group only the variants that passed (pandas 3 raises on an all-NaN
    # group); families with no pass come back NaN from the reindex.
    passed = full_sharpe[stacked].dropna()
    naive_best = passed.groupby(fs.family[passed.index]).idxmax().reindex(fs.truth.index)

    out = {}
    windows = {"split": (first, last, _one_sided_p(first)), "reuse": (search, search, p_all)}
    for design, (window1, window2, p1) in windows.items():
        s = _stage1(window1, p1, fs.family, n_boot, seed)
        s["p2"] = (
            p_all.reindex(s["best"]).to_numpy()
            if design == "reuse"
            else [sharpe_test(window2[b]).pvalue_greater for b in s["best"]]
        )
        nb = naive_best.reindex(s.index)
        for field in ("sharpe", "n_obs", "skew", "kurtosis", "variance"):
            s[f"ho_{field}"] = [getattr(holdout_tests[b], field) for b in s["best"]]
            if field == "sharpe":
                s["ho_p"] = [holdout_tests[b].pvalue_greater for b in s["best"]]
                s["nv_p"] = [
                    holdout_tests[b].pvalue_greater if isinstance(b, str) else np.nan for b in nb
                ]
            s[f"nv_{field}"] = [
                getattr(holdout_tests[b], field) if isinstance(b, str) else np.nan for b in nb
            ]
        s["naive_pass"] = nb.notna()
        s["truth"] = fs.truth.reindex(s.index)
        s.attrs["n_variants"] = n_variants
        out[design] = s
    return out


# ---------------------------------------------------------------------------
# The three pipelines, as filters over the summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """One threshold setting for the funnel.

    The defaults are the set registered in `docs/validation/framework.md`
    ("Results"): q1 = 0.10, no stage-2 gate, BH at q4 = 0.30 on the holdout.

    q1: BH level across families at stage 1. alpha2: one-sided p cut-off at
    stage 2; 1.0 means no gate. final: the stage-4 gate, "bh" (BH at q4
    across the survivors' one-sided holdout p-values; registered) or "dsr"
    (deflated Sharpe >= d4). final_trials: for "dsr", the trial count it
    charges, "survivors" (only the candidates taken to the holdout, which is
    independent data) or "searched" (every variant tried; calibration found
    it cuts power to about a sixth).
    """

    q1: float = 0.10
    alpha2: float = 1.0
    final: str = "bh"
    d4: float = 0.95
    q4: float = 0.30
    final_trials: str = "survivors"


def _final_gate(
    stats: pd.DataFrame, alive: pd.Series, th: Thresholds, prefix: str = "ho_"
) -> pd.Series:
    """Stage 4 on the holdout, for the candidates still alive.

    `prefix` picks whose holdout statistics to read: "ho_" for the variant the
    funnel carried forward, "nv_" for the naive stack's.
    """
    if th.final not in ("dsr", "bh"):
        raise ValueError("final must be 'dsr' or 'bh'")
    if th.final_trials not in ("searched", "survivors"):
        raise ValueError("final_trials must be 'searched' or 'survivors'")
    n_alive = int(alive.sum())
    if n_alive == 0:
        return alive.copy()
    if th.final == "bh":
        passed = benjamini_hochberg(stats.loc[alive, f"{prefix}p"], th.q4)
        return passed.reindex(stats.index, fill_value=False)
    n_trials = stats.attrs["n_variants"] if th.final_trials == "searched" else n_alive
    d4 = th.d4
    passed = pd.Series(False, index=stats.index)
    for h in alive[alive].index:
        r = stats.loc[h]
        dsr = deflated_sharpe_ratio(
            r[f"{prefix}sharpe"],
            n_trials,
            int(r[f"{prefix}n_obs"]),
            r[f"{prefix}skew"],
            r[f"{prefix}kurtosis"],
            variance=r[f"{prefix}variance"],
        )
        passed[h] = dsr >= d4
    return passed


def run_funnel(stats: pd.DataFrame, th: Thresholds, *, stage1: str = "rho") -> pd.DataFrame:
    """The staged funnel. Returns per-hypothesis pass flags for each stage.

    `stage3` is a placeholder: it passes everything and is reported as not
    measured until the spanning test exists, so it can never look good by
    default.
    """
    if stage1 not in STAGE1_METHODS:
        raise ValueError(f"stage1 must be one of {STAGE1_METHODS}")
    s1 = benjamini_hochberg(stats[f"p1_{stage1}"], th.q1)
    s2 = s1 & (stats["p2"] <= th.alpha2)
    s3 = s2.copy()
    s4 = s3 & _final_gate(stats, s3, th)
    return pd.DataFrame({"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4})


def run_baseline(stats: pd.DataFrame, *, q: float = 0.10, stage1: str = "rho") -> pd.Series:
    """BH at `q` on each family's search-adjusted p-value (#50, Q5).

    Pass "reuse" statistics, so the p-values cover the whole search period.
    """
    return benjamini_hochberg(stats[f"p1_{stage1}"], q).rename("accepted")


def run_naive_stack(stats: pd.DataFrame, th: Thresholds) -> pd.Series:
    """Every screening gate at once, then the same final holdout test."""
    alive = stats["naive_pass"].astype(bool)
    final = _final_gate(stats, alive, th, prefix="nv_")
    return (alive & final).rename("accepted")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(accepted: pd.Series, truth: pd.Series) -> dict:
    """Outcome of one pipeline on one data set.

    `fdr` is false discoveries / max(discoveries, 1), so a run that accepts
    nothing scores 0, and its average over runs is the FDR proper.
    `power` is NaN when nothing is real.
    """
    accepted = accepted.reindex(truth.index).fillna(False).astype(bool)
    truth = truth.astype(bool)
    n = int(accepted.sum())
    true = int((accepted & truth).sum())
    false = n - true
    return {
        "discoveries": n,
        "true": true,
        "false": false,
        "any_false": false > 0,
        "fdr": false / max(n, 1),
        "power": true / truth.sum() if truth.any() else np.nan,
        "precision": true / n if n else np.nan,
    }


def gate_report(stages: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    """Where real signals die and where noise leaks, stage by stage.

    For each stage: how many real and null hypotheses enter and pass it, and
    the pass rates among those that entered (in-pipeline power and false-pass
    rate). Stage 3 is flagged as not measured, and its pass rates are NaN.
    """
    truth = truth.reindex(stages.index).astype(bool)
    entering = pd.Series(True, index=stages.index)
    rows = []
    for stage in stages.columns:
        passed = stages[stage].astype(bool)
        # Stage 3 passes everything until the spanning test exists; a 1.0
        # pass rate there would read as a measurement.
        measured = stage != "stage3"
        real_in, null_in = int((entering & truth).sum()), int((entering & ~truth).sum())
        real_out, null_out = int((passed & truth).sum()), int((passed & ~truth).sum())
        rows.append(
            {
                "stage": stage,
                "real_in": real_in,
                "real_out": real_out,
                "null_in": null_in,
                "null_out": null_out,
                "pass_rate_real": real_out / real_in if real_in and measured else np.nan,
                "pass_rate_null": null_out / null_in if null_in and measured else np.nan,
                "measured": measured,
            }
        )
        entering = passed
    return pd.DataFrame(rows).set_index("stage")


def gates_alone(stats: pd.DataFrame, th: Thresholds, *, stage1: str = "rho") -> pd.DataFrame:
    """Each gate applied by itself to every hypothesis, for comparison with
    its pass rates inside the pipeline: a gate that looks fine alone can still
    repeat a penalty another gate already charged."""
    everyone = pd.Series(True, index=stats.index)
    alone = pd.DataFrame(
        {
            "stage1": benjamini_hochberg(stats[f"p1_{stage1}"], th.q1),
            "stage2": stats["p2"] <= th.alpha2,
            "stage4": _final_gate(stats, everyone, th),
        }
    )
    truth = stats["truth"].astype(bool)
    return pd.DataFrame(
        {
            "pass_rate_real": alone[truth].mean() if truth.any() else np.nan,
            "pass_rate_null": alone[~truth].mean(),
        }
    )
