"""The gate's graded scores and batch diagnostics: reported, never decisive.

The acceptance gate (`capstone.gate`) calls these on the search
period before the holdout look (so a failure here can't spend it) and, for
`stage4_stat`, after it. Nothing here is read back into a decision.

- `graded_scores`: per family, BY / Holm / local FDR of the stage-1 p-value,
  Sharpe, PSR, MinTRL, and the haircut Sharpe with the ledger trial count;
- `search_fdr_estimate`, `pbo`: batch diagnostics for the header;
- `stage4_stat`: the statistic stage 4's gate compared with its threshold;
- `expected_mean`: the per-period mean incubation should expect (see the
  gate module's docstring for why it is the smallest of three).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from capstone.evaluate import (
    MIN_LOCAL_FDR_CANDIDATES,
    _pfdr,
    bh_adjusted,
    by_adjusted,
    deflated_sharpe_ratio,
    estimate_pi0,
    haircut_sharpe,
    holm_adjusted,
    local_fdr,
    min_track_record_length,
    sharpe_test,
)
from capstone.funnel import Thresholds
from capstone.pbo import cscv

__all__ = [
    "expected_mean",
    "family_lfdr",
    "graded_scores",
    "pbo",
    "search_fdr_estimate",
    "stage4_stat",
]

PERIODS_PER_YEAR = 252


def family_lfdr(pvalues: pd.Series) -> pd.Series:
    """Local FDR of each family's p-value; NaN with fewer than 200 families."""
    if len(pvalues.dropna()) < MIN_LOCAL_FDR_CANDIDATES:
        return pd.Series(np.nan, index=pvalues.index)
    return local_fdr(pvalues)


def _haircut(
    sharpe: pd.Series, n_obs: pd.Series, n_trials: int, avg_corr: float, seed: int
) -> pd.Series:
    """The most conservative of the authors' three adjusted Sharpes, per family.

    NaN where undefined (Sharpe not positive, fewer than 2 trials). One call
    per distinct `n_obs`: with a fixed seed every call simulates the same
    family of other trials, so batching changes cost, not results.
    """
    out = pd.Series(np.nan, index=sharpe.index)
    ok = sharpe > 0
    if n_trials < 2 or not ok.any():
        return out
    rho = min(max(avg_corr, 0.0), 0.99) if math.isfinite(avg_corr) else 0.2
    for n, group in sharpe[ok].groupby(n_obs[ok]):
        frame = haircut_sharpe(
            group, int(n), n_trials=n_trials, frequency="daily", avg_correlation=rho, seed=seed
        )
        adjusted = frame[["sharpe_bonferroni", "sharpe_holm", "sharpe_bhy"]].min(axis=1)
        out[group.index] = adjusted.reindex(group.index).to_numpy()  # frame is sorted
    return out


def graded_scores(
    search: pd.DataFrame,
    best: pd.Series,
    p1: pd.Series,
    n_trials: int,
    avg_corr: float,
    seed: int,
) -> pd.DataFrame:
    """Every graded score, per family. Reported only; no decision reads it."""
    tests = {fam: sharpe_test(search[b]) for fam, b in best.items()}
    sharpe = pd.Series({f: t.sharpe for f, t in tests.items()}, dtype=float)
    n_obs = pd.Series({f: t.n_obs for f, t in tests.items()})
    min_trl = {
        f: min_track_record_length(t.sharpe, skew=t.skew, kurtosis=t.kurtosis, variance=t.variance)
        / PERIODS_PER_YEAR
        for f, t in tests.items()
    }
    return pd.DataFrame(
        {
            "p1": p1,
            "by": by_adjusted(p1),
            "holm": holm_adjusted(p1),
            "lfdr": family_lfdr(p1),
            "sharpe": sharpe,
            "psr": pd.Series({f: t.psr for f, t in tests.items()}, dtype=float),
            "min_trl_years": pd.Series(min_trl, dtype=float),
            "haircut_sharpe": _haircut(sharpe, n_obs, n_trials, avg_corr, seed),
        },
        index=p1.index,
    )


def search_fdr_estimate(p1: pd.Series, passed: pd.Series) -> float:
    """Storey's positive FDR of the stage-1 rejections, capped at 1.

    The rejection region is [0, t], with t the largest passing p-value, and the
    estimate is pi0 * t / ((R / m) * (1 - (1 - t)^m)) for R p-values at or below
    t among m (`evaluate._pfdr`, Storey 2002 Algorithm 1(c)). The last factor
    is the chance of making any rejection at all. Without it the estimate is
    pi0 * m * t / R, which at BH's own cutoff cannot exceed pi0 * q, so it
    could never say "stage 1 is passing noise". With it, a lone pass among
    all-null candidates reads near 1. NaN when nothing passed.

    Like `evaluate.estimate_pi0`, it assumes independent candidates: when a
    shared factor lifts many candidates at once (`rho` near 0.5), the batch
    passes together and the estimate reads low, so treat a low value under
    strong correlation as unverified.
    """
    p = p1.dropna()
    if int(passed.reindex(p.index, fill_value=False).sum()) == 0:
        return float("nan")
    threshold = float(p[passed.reindex(p.index, fill_value=False)].max())
    values = p.to_numpy(dtype=float)
    if threshold <= 0.0:  # the limit as t -> 0 is pi0 / R (and 1 - (1 - t)^m would be 0)
        rejected = max(int(np.sum(values <= 0.0)), 1)
        return float(min(1.0, estimate_pi0(p, 0.5) / rejected))
    return _pfdr(values, 0.5, threshold)


def pbo(search: pd.DataFrame) -> float:
    try:
        return float(cscv(search)["pbo"])
    except ValueError:  # too few variants or rows for 16 blocks
        return float("nan")


def stage4_stat(stats: pd.DataFrame, alive: pd.Series, th: Thresholds) -> pd.Series:
    """The statistic stage 4's gate compares with its threshold (NaN if not reached)."""
    out = pd.Series(np.nan, index=stats.index)
    if not alive.any():
        return out
    if th.final == "bh":
        # A survivor with no holdout p-value is in the family at p = 1, as the
        # gate decides it (`funnel._final_gate`); dropping it would show a
        # statistic under q4 beside a failed stage.
        out[alive] = bh_adjusted(stats.loc[alive, "ho_p"].fillna(1.0))
        return out
    n_trials = stats.attrs["n_variants"] if th.final_trials == "searched" else int(alive.sum())
    for h in alive[alive].index:
        r = stats.loc[h]
        if not r["ho_n_obs"] > 0:  # never tested: no statistic
            continue
        out[h] = deflated_sharpe_ratio(
            r["ho_sharpe"],
            n_trials,
            int(r["ho_n_obs"]),
            r["ho_skew"],
            r["ho_kurtosis"],
            variance=r["ho_variance"],
        )
    return out


def expected_mean(card: pd.DataFrame) -> pd.Series:
    """Per-period mean incubation should expect: at most the raw in-sample mean.

    min(search mean x max(haircut Sharpe / Sharpe, 0), holdout mean, search
    mean), skipping NaN terms (a family with no haircut keeps the other two).
    The raw search mean itself is returned only when the other two terms are
    both missing. See the module docstring for why.
    """
    ratio = (card["haircut_sharpe"] / card["sharpe"]).clip(lower=0.0)
    haircut_mean = card["is_mean"] * ratio
    return pd.concat([haircut_mean, card["ho_mean"], card["is_mean"]], axis=1).min(axis=1)
