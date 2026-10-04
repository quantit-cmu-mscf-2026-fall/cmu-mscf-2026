"""Search-adjusted false-discovery rates (López de Prado & Fabozzi 2026).

A reported candidate is usually the best of several variants tried for the
same idea: lookbacks, normalisations, universes, every revision an agent made
after seeing a backtest. Its statistic is then a *maximum*, not a single draw,
and every correction that treats it as a single draw understates how often it
is false. López de Prado & Fabozzi (2026, SSRN 6450418) show that from the
reported statistics alone the false-discovery rate is not even identified:
very different mixes of real ideas, effect sizes and search intensity K give
nearly the same reported significance profile (their Theorem 1, Table 1).

The way out is to know K. Our pipeline logs every trial, so for us K per
hypothesis is a count, not a guess, and this module uses it three ways:

- `search_adjusted_pvalue`: the p-value of a hypothesis's best variant,
  adjusted for its K variants, ready for BH/BY across hypotheses.
- `familywise_errors` / `search_adjusted_fdr`: the paper's Eqs. 9, 10, 17
  and 18, the FDR implied by a threshold, a base rate, a power and K. This is
  the arithmetic behind calibrating how strict a stage needs to be.
- `fit_max_of_mixture` / `fdr_by_search_intensity`: the paper's Section 6
  estimator, for a batch whose search we did *not* see (published factors,
  a teammate's shortlist), reported over a grid of assumed K.

The false discovery throughout is family-level, as in the paper: a reported
hypothesis is false when *none* of its variants is real (its Eq. 10). That is
the conservative definition; counting a null winner of a family that also
holds a real variant as false would only raise the FDR.

`tests/test_search_fdr.py` reproduces the paper's printed numbers and checks
null calibration and power on data with known truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

# ---------------------------------------------------------------------------
# Adjusting one hypothesis's best variant for its search
# ---------------------------------------------------------------------------


def search_adjusted_pvalue(pvalues: pd.Series, k: int | pd.Series) -> pd.Series:
    """p-value of the best of `k` independent variants: 1 - (1 - p)^k.

    `pvalues` holds, per hypothesis, the smallest one-sided p-value among its
    variants; `k` is how many variants were tried (a scalar, or a Series
    aligned with `pvalues`). The result is uniform under the null that no
    variant is real, so it can go straight into `benjamini_hochberg`,
    `bh_adjusted` or `evidence_profile` across hypotheses. This is the paper's
    familywise Type I error, Eq. 17, applied to a p-value (Šidák's correction).

    Variants of one idea are usually positively correlated, and then this is
    conservative: K correlated variants search less than K independent ones.
    The paper's K is an *effective* number of independent trials for that
    reason (its footnote 7). For a sharper adjustment that keeps the
    correlation, bootstrap the maximum across the family's return series.

    NaN stays NaN. Raises on k < 1.
    """
    k = pd.Series(k, index=pvalues.index) if np.isscalar(k) else k.reindex(pvalues.index)
    if (k.dropna() < 1).any():
        raise ValueError("k must be at least 1")
    p = pvalues.astype(float).clip(0.0, 1.0)
    # -expm1(k * log1p(-p)) is 1 - (1 - p)^k without cancellation for small p.
    with np.errstate(divide="ignore"):
        adjusted = -np.expm1(k.astype(float) * np.log1p(-p))
    return adjusted.where(p.notna())


def search_intensity(hypothesis_ids: Iterable) -> pd.Series:
    """K per hypothesis: how many trials were logged under each hypothesis id.

    Pass one id per trial, every trial whose result anyone or any agent saw,
    including discarded variants (`docs/validation.md`, "Trial counts"). Trials
    with a missing id are not counted toward any hypothesis.
    """
    ids = pd.Series(list(hypothesis_ids), dtype=object).dropna()
    return ids.value_counts().sort_index().rename("k")


# ---------------------------------------------------------------------------
# The FDR implied by a threshold under search (paper Section 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilywiseErrors:
    """Error probabilities of the reported (best-of-K) statistic.

    alpha_k: chance the best of K null variants clears the threshold (Eq. 17).
    beta_k: chance the best variant misses, given at least one is real (Eq. 18).
    pi0_k: chance that none of the K variants is real (Eq. 9).
    """

    alpha_k: float
    beta_k: float
    pi0_k: float


def familywise_errors(alpha: float, beta: float, pi0: float, k: int) -> FamilywiseErrors:
    """The paper's Eqs. 9, 17 and 18: single-trial errors to best-of-K errors.

    Args:
        alpha: chance one null variant clears the threshold.
        beta: chance one real variant misses it (1 - power).
        pi0: chance one variant is null (the trial-level null prevalence).
        k: variants tried per reported hypothesis, independent or effective.
    """
    for name, value in (("alpha", alpha), ("beta", beta), ("pi0", pi0)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if k < 1:
        raise ValueError("k must be at least 1")
    alpha_k = 1.0 - (1.0 - alpha) ** k
    pi0_k = pi0**k
    if pi0_k >= 1.0:
        beta_k = float("nan")  # no family holds a real variant
    else:
        beta_k = ((pi0 * (1.0 - alpha) + (1.0 - pi0) * beta) ** k - (pi0 * (1.0 - alpha)) ** k) / (
            1.0 - pi0_k
        )
    return FamilywiseErrors(alpha_k=alpha_k, beta_k=beta_k, pi0_k=pi0_k)


def fdr(alpha: float, beta: float, pi0: float) -> float:
    """Share of rejections that are null: alpha·pi0 / (alpha·pi0 + (1 - beta)(1 - pi0)).

    The paper's Eq. 10 with K = 1. NaN when nothing is ever rejected.
    """
    false = alpha * pi0
    true = 0.0 if pi0 >= 1.0 else (1.0 - beta) * (1.0 - pi0)
    total = false + true
    return float("nan") if total == 0 else false / total


def search_adjusted_fdr(alpha: float, beta: float, pi0: float, k: int) -> float:
    """FDR of reported best-of-`k` hypotheses at a fixed threshold (Eq. 10).

    The arithmetic for a calibration question: with this threshold, this base
    rate, this power and this much search per hypothesis, what share of the
    hypotheses we report have no real variant at all?
    """
    e = familywise_errors(alpha, beta, pi0, k)
    return fdr(e.alpha_k, e.beta_k, e.pi0_k)


def sharpe_threshold(n_obs: int, rho: float = 0.0, alpha: float = 0.05) -> float:
    """Two-sided per-period Sharpe threshold under AR(1) returns (paper Eq. 30).

    z_{1-alpha/2} · sqrt((1 + rho) / ((1 - rho) · n_obs)). Used to reproduce the
    paper's calibration; for testing our own candidates, `evaluate.sharpe_test`
    is the better variance (it also allows for fat tails and longer memory).
    """
    if n_obs < 1:
        raise ValueError("n_obs must be at least 1")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(z * np.sqrt((1.0 + rho) / ((1.0 - rho) * n_obs)))


# ---------------------------------------------------------------------------
# Estimating the FDR when the search was not seen (paper Section 6)
# ---------------------------------------------------------------------------


def max_of_mixture_cdf(
    x, k: int, pi0: float, delta1: float, sigma0: float = 1.0, sigma1: float = 1.0
) -> np.ndarray:
    """CDF of the best of `k` draws from the trial-level mixture (Eq. 34).

    The mixture is pi0·N(0, sigma0²) + (1 - pi0)·N(delta1, sigma1²).
    """
    x = np.asarray(x, dtype=float)
    mix = pi0 * special.ndtr(x / sigma0) + (1.0 - pi0) * special.ndtr((x - delta1) / sigma1)
    return mix**k


def max_of_mixture_logpdf(
    x, k: int, pi0: float, delta1: float, sigma0: float = 1.0, sigma1: float = 1.0
) -> np.ndarray:
    """Log density of the best of `k` mixture draws (Eq. 36), computed in logs.

    k · F^(k-1) · f underflows in the left tail for large k; this doesn't.
    """
    x = np.asarray(x, dtype=float)
    z0 = x / sigma0
    z1 = (x - delta1) / sigma1
    log_w0, log_w1 = np.log(pi0), np.log1p(-pi0)
    log_cdf = np.logaddexp(log_w0 + special.log_ndtr(z0), log_w1 + special.log_ndtr(z1))
    log_pdf = np.logaddexp(
        log_w0 + stats.norm.logpdf(z0) - np.log(sigma0),
        log_w1 + stats.norm.logpdf(z1) - np.log(sigma1),
    )
    return np.log(k) + (k - 1) * log_cdf + log_pdf


@dataclass(frozen=True)
class MaxMixtureFit:
    """Trial-level parameters fitted to reported best-of-K statistics (Eq. 37)."""

    k: int
    pi0: float
    delta1: float
    sigma0: float
    sigma1: float
    loglik: float


def _unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
    # pi0 in (0, 1); delta1 > 0; sigma0 > 0; sigma1 = sigma0 + extra >= sigma0.
    a, b, g, h = theta
    return special.expit(a), np.exp(b), np.exp(g), np.exp(g) + np.exp(h)


def fit_max_of_mixture(
    x, k: int, *, delta1_min: float = 0.1, seed: int | None = 0
) -> MaxMixtureFit:
    """Maximum-likelihood fit of the max-of-mixture model for a fixed `k` (Eq. 37).

    Each observation is read as the best of `k` latent variants, each drawn
    from pi0·N(0, sigma0²) + (1 - pi0)·N(delta1, sigma1²). As in the paper,
    the null mean is fixed at zero, sigma1 >= sigma0, and delta1 >= `delta1_min`
    keeps the "real" component from being a wider copy of the null; set it in
    the units of `x` (the paper uses 0.1 for monthly Sharpe ratios).

    The likelihood has local optima, so this starts a bounded local search
    from a grid of points and from a seeded global search, and keeps the best.

    `k` is assumed, not estimated: the paper's Theorem 1 is that the data
    cannot pin it down. Report fits over a grid of `k` (`fdr_by_search_intensity`)
    and treat the likelihood across `k` as a fit diagnostic only.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        raise ValueError("need at least 5 finite observations")
    if k < 1:
        raise ValueError("k must be at least 1")
    if delta1_min <= 0:
        raise ValueError("delta1_min must be positive")

    scale = float(np.std(x, ddof=1)) or 1.0
    span = float(np.max(np.abs(x))) + scale
    bounds = [
        (-10.0, 10.0),
        (np.log(delta1_min), np.log(max(10.0 * span, 2.0 * delta1_min))),
        # A floor on sigma0: without one, a near-empty null component can
        # collapse onto a single observation and the likelihood diverges.
        # The paper's code floors it at 0.02 in monthly Sharpe units, about
        # a fifth of its cross-section's spread.
        (np.log(0.2 * scale), np.log(10.0 * scale)),
        (np.log(1e-3 * scale), np.log(10.0 * scale)),
    ]

    def objective(theta: np.ndarray) -> float:
        pi0, delta1, sigma0, sigma1 = _unpack(theta)
        value = -np.sum(max_of_mixture_logpdf(x, k, pi0, delta1, sigma0, sigma1))
        return value if np.isfinite(value) else 1e100

    starts = [
        np.array([special.logit(p0), np.log(d), np.log(s0 * scale), np.log(e * scale)])
        for p0 in (0.05, 0.5, 0.9, 0.99)
        for d in sorted({delta1_min, 2.0 * delta1_min, span / 2.0})
        for s0 in (0.3, 1.0)
        for e in (0.01, 0.5)
    ]
    starts = [np.clip(s, [lo for lo, _ in bounds], [hi for _, hi in bounds]) for s in starts]
    best = None
    for theta0 in starts:
        res = optimize.minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    de = optimize.differential_evolution(
        objective, bounds, seed=seed, maxiter=60, popsize=15, polish=False
    )
    res = optimize.minimize(objective, de.x, method="L-BFGS-B", bounds=bounds)
    if res.fun < best.fun:
        best = res

    pi0, delta1, sigma0, sigma1 = _unpack(best.x)
    return MaxMixtureFit(
        k=k, pi0=pi0, delta1=delta1, sigma0=sigma0, sigma1=sigma1, loglik=-float(best.fun)
    )


def fdr_by_search_intensity(
    x, thresholds, ks: Iterable[int], *, delta1_min: float = 0.1, seed: int | None = 0
) -> pd.DataFrame:
    """The paper's Table 3: fitted parameters and implied FDR for each assumed K.

    For each `k`, fit `fit_max_of_mixture`, then take each observation's
    rejection threshold (`thresholds`, a scalar or one per observation, in the
    units of `x`), compute its single-trial errors from the fitted components
    (Eq. 38) and their best-of-K versions (Eq. 39), average across
    observations, and combine them into the FDR (Eq. 40).

    The FDR column is conditional on each assumed `k`; it is not an estimate
    of `k`. Read the whole column, not its likelihood-maximising row alone.
    """
    x = np.asarray(x, dtype=float)
    c = np.broadcast_to(np.asarray(thresholds, dtype=float), x.shape)
    keep = np.isfinite(x) & np.isfinite(c)
    x, c = x[keep], c[keep]
    rows = []
    for k in ks:
        f = fit_max_of_mixture(x, k, delta1_min=delta1_min, seed=seed)
        alpha_n = 1.0 - special.ndtr(c / f.sigma0)
        beta_n = special.ndtr((c - f.delta1) / f.sigma1)
        alpha_k = 1.0 - (1.0 - alpha_n) ** k
        pi0_k = f.pi0**k
        beta_k = (
            (f.pi0 * (1.0 - alpha_n) + (1.0 - f.pi0) * beta_n) ** k - (f.pi0 * (1.0 - alpha_n)) ** k
        ) / (1.0 - pi0_k)
        rows.append(
            {
                "k": k,
                "pi0": f.pi0,
                "delta1": f.delta1,
                "sigma0": f.sigma0,
                "sigma1": f.sigma1,
                "pi0_k": pi0_k,
                "alpha_k": float(alpha_k.mean()),
                "beta_k": float(beta_k.mean()),
                "loglik": f.loglik,
                "fdr": fdr(float(alpha_k.mean()), float(beta_k.mean()), pi0_k),
            }
        )
    return pd.DataFrame(rows).set_index("k")


# ---------------------------------------------------------------------------
# Test data with known truth
# ---------------------------------------------------------------------------


def simulate_family_winners(
    n_families: int,
    k: int,
    share_real: float,
    sharpe_real: float,
    n_obs: int,
    *,
    within_rho: float = 0.0,
    periods_per_year: int = 252,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Hypothesis families that each tried `k` variants and report only the best.

    Simulates Sharpe estimates directly, sharpe_hat ~ N(SR, 1 / n_obs) per
    period, so thousands of families cost nothing. A real family's variants
    all carry the annualised Sharpe `sharpe_real`; a null family's carry
    zero. `within_rho` correlates the estimation noise of a family's variants,
    as variants of one idea are correlated.

    Returns one row per family: `sharpe` (annualised, best variant), `pvalue`
    (its one-sided p-value, unadjusted), `k`, and `truth` (True = real).
    """
    if not 0.0 <= within_rho < 1.0:
        raise ValueError("within_rho must be in [0, 1)")
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_families, dtype=bool)
    truth[: int(round(share_real * n_families))] = True
    rng.shuffle(truth)
    mean = np.where(truth, sharpe_real / np.sqrt(periods_per_year), 0.0)
    common = rng.standard_normal((n_families, 1))
    own = rng.standard_normal((n_families, k))
    noise = np.sqrt(within_rho) * common + np.sqrt(1.0 - within_rho) * own
    per_period = mean[:, None] + noise / np.sqrt(n_obs)
    best = per_period.max(axis=1)
    index = pd.Index([f"hyp_{i:04d}" for i in range(n_families)], name="hypothesis")
    return pd.DataFrame(
        {
            "sharpe": best * np.sqrt(periods_per_year),
            "pvalue": stats.norm.sf(best * np.sqrt(n_obs)),
            "k": k,
            "truth": truth,
        },
        index=index,
    )
