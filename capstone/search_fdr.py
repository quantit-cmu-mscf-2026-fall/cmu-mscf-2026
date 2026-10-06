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
holds a real variant as false would only raise the FDR. One consequence: the
family-level FDR need not rise with the per-variant threshold alpha, because a
looser threshold also lets more real families through.

`fdr_by_search_intensity` is a sensitivity table, not a gate. On data with no
real hypotheses it can still report a low FDR (#51 review: 0.000 to 0.999
across seeds), because the paper's identification problem doesn't go away by
assuming a K. Decisions use `search_adjusted_pvalue` and BH.

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


def _aligned(name: str, value, index: pd.Index) -> pd.Series:
    """A scalar broadcast to `index`, or a Series that must cover all of it."""
    if np.isscalar(value):
        return pd.Series(float(value), index=index)
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a scalar or a pandas Series indexed by hypothesis")
    missing = index.difference(value.dropna().index)
    if len(missing):
        # A missing value would turn the hypothesis's p-value into NaN, which BH
        # drops from the family: the hypothesis could never be selected, and
        # the family would shrink, making BH too lenient for the rest.
        raise ValueError(f"{name} is missing for {len(missing)} hypotheses, e.g. {missing[0]!r}")
    return value.reindex(index).astype(float)


# Grid over the shared factor U ~ N(0, 1) for the equicorrelated adjustment.
_U = np.linspace(-9.0, 9.0, 3601)
_U_WEIGHTS = stats.norm.pdf(_U) * (_U[1] - _U[0])


def search_adjusted_pvalue(
    pvalues: pd.Series, k: int | pd.Series, *, rho: float | pd.Series = 0.0
) -> pd.Series:
    """p-value of a hypothesis's best variant, adjusted for its `k` variants.

    `pvalues` holds, per hypothesis, the smallest one-sided p-value among its
    variants; `k` is how many variants were tried (a scalar, or a Series
    covering every hypothesis). The result is uniform under the null that no
    variant is real, so it can go straight into `benjamini_hochberg`,
    `bh_adjusted` or `evidence_profile` across hypotheses.

    `rho` is the correlation between a hypothesis's variants (scalar or
    Series). At `rho=0`, the default, this is the paper's Eq. 17 applied to a
    p-value, 1 - (1 - p)^k (Šidák): exact for **independent** variants. An
    agent's variants of one idea are usually near-copies, and then the
    independent formula over-corrects badly: at k = 20 and rho = 0.9 a null
    family's chance of being called real is about 0.011, not 0.05, and power
    falls with it (#51 review). With `rho` > 0 the adjustment is exact for
    equicorrelated normal statistics: the chance that the best of `k`
    variants sharing correlation `rho` reaches this p-value. Estimate `rho`
    from the family's own return series with `evaluate.average_correlation`.
    When the correlation is far from even across variants, bootstrap the
    family's maximum instead.

    Use the **actual** number of variants tried for `k`. Don't pass an
    "effective" count such as `evaluate.implied_independent_trials`: it is
    anti-conservative here (a null family's rejection rate is 0.058-0.071 at a
    nominal 0.05), for the same reason the ledger count must not be shrunk.
    Allow for correlation through `rho` instead.

    NaN p-values stay NaN. Raises on p outside [0, 1], on a missing,
    non-finite or < 1 `k`, and on `rho` outside [0, 1].
    """
    if not isinstance(pvalues, pd.Series):
        raise TypeError("pvalues must be a pandas Series indexed by hypothesis")
    p = pvalues.astype(float)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be in [0, 1]")
    k = _aligned("k", k, p.index)
    if not np.isfinite(k).all() or (k < 1).any():
        raise ValueError("k must be finite and at least 1")
    rho = _aligned("rho", rho, p.index)
    if ((rho < 0) | (rho > 1)).any():
        raise ValueError("rho must be in [0, 1]")

    # Independent variants: -expm1(k * log1p(-p)) is 1 - (1 - p)^k without
    # cancellation for small p.
    with np.errstate(divide="ignore"):
        adjusted = -np.expm1(k * np.log1p(-p))

    # Equicorrelated variants: z_j = sqrt(rho) U + sqrt(1 - rho) e_j. Given U,
    # the variants are independent, so P(max >= z) = E_U[1 - Phi(a(U))^k] with
    # a(u) = (z - sqrt(rho) u) / sqrt(1 - rho). Each term is computed as
    # -expm1(k log Phi(a)), positive and free of cancellation, then averaged
    # over a fine grid of U.
    corr = (rho > 0) & (rho < 1) & p.notna()
    if corr.any():
        z = stats.norm.isf(p[corr].to_numpy())[:, None]
        r = rho[corr].to_numpy()[:, None]
        a = (z - np.sqrt(r) * _U) / np.sqrt(1.0 - r)
        tail = -np.expm1(k[corr].to_numpy()[:, None] * special.log_ndtr(a))
        adjusted[corr] = np.clip(tail @ _U_WEIGHTS, p[corr].to_numpy(), 1.0)
    # Perfectly correlated variants are one variant searched k times.
    adjusted = adjusted.where(rho < 1, p)
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
# Bootstrapping a family's best variant (adapted from Carl Cui's
# `carl/validation1-hypothesis-fdr`, paper_validation/.../bootstrap.py)
# ---------------------------------------------------------------------------


def block_length_rule(n_obs: int) -> int:
    """Default circular block length, ceil(2 · n_obs^(1/3)): 16 at 500, 28 at 2,520.

    Carl's calibration at n_obs = 500 (P(p <= 0.01) against a target of 0.010,
    AR(1) coefficient phi):

        block:        8        16        25        50
        phi=0.0    0.0121    0.0132    0.0159    0.0227
        phi=0.2    0.0141    0.0144    0.0169    0.0234
        phi=0.5    0.0198    0.0172    0.0189    0.0245

    Short blocks cut serial dependence at their edges; long blocks leave too
    few distinct blocks. The minimum sits near 16, which this rule gives. A
    1.3-1.7x excess at the 1% level remains at every length: a finite-sample
    property of the studentised block bootstrap.
    """
    return max(1, int(np.ceil(2.0 * n_obs ** (1.0 / 3.0))))


def bootstrap_family_pvalues(
    returns: pd.DataFrame,
    family: pd.Series,
    *,
    n_boot: int = 1999,
    block_length: int | None = None,
    seed: int | None = 0,
    batch_size: int = 64,
) -> pd.DataFrame:
    """p-value of each hypothesis's best variant, by a block bootstrap of the maximum.

    The stage-1 gate's input (`docs/validation/framework.md`, Q2): for each
    hypothesis, how often the best of *its* variants would look this good in
    a world where none of them is real, with the variants' actual correlation,
    autocorrelation and tails. Unlike `search_adjusted_pvalue`, it needs no
    assumption about how the variants are correlated, at the cost of
    resampling.

    Three rules make it correct (Carl's design, kept as he wrote it):

    1. **One resampling for everyone.** Each replication draws one set of
       circular blocks of dates and applies it to every column, so the
       correlation between variants and between hypotheses survives.
    2. **The null comes from centring.** The bootstrap statistic is
       sqrt(T)(mean* - mean) / sd*, so the resampled world has no real variant
       while keeping everything else about the data.
    3. **The search is redone inside each replication.** The family maximum
       is retaken every time, so the reference is the best of a search of this
       size and this correlation.

    The statistic is the one-sided t of the mean, sqrt(T) · mean / sd, which
    ranks variants exactly as their Sharpe ratios do. Rows with a missing value
    in any column are dropped first, so every variant is scored on the same
    dates.

    **Resolution.** A bootstrap p-value can't go below 1 / (n_boot + 1). BH
    across H hypotheses at level q tests its most extreme rank at q / H, so
    n_boot + 1 must exceed H / q, or the strongest hypotheses can't be told
    apart and power collapses silently (Carl measured exactly that). This
    raises if it doesn't.

    Args:
        returns: performance matrix, dates x candidates, higher is better.
        family: hypothesis label per column of `returns` (a Series indexed by
            column name). Families may differ in size.
        n_boot: bootstrap replications.
        block_length: circular block length; `block_length_rule` if omitted.
        seed: seed for the resampling.
        batch_size: replications per vectorised pass; memory only, never the
            result.

    Returns:
        DataFrame indexed by hypothesis: `pvalue`, `k` (variants), `best`
        (the column with the largest t) and `t` (its t-statistic).
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(family, pd.Series):
        raise TypeError("returns must be a DataFrame and family a Series")
    missing = returns.columns.difference(family.dropna().index)
    if len(missing):
        raise ValueError(f"family is missing for {len(missing)} columns, e.g. {missing[0]!r}")
    data = returns.dropna(how="any")
    n_obs = len(data)
    block = block_length if block_length is not None else block_length_rule(n_obs)
    if not 1 <= block <= n_obs:
        raise ValueError("block_length must be between 1 and the number of complete rows")
    if n_obs < 3:
        raise ValueError("need at least 3 rows with no missing value")

    # Lay the columns out family by family, so each family is one slice.
    labels = family.reindex(data.columns)
    order = np.argsort(labels.to_numpy().astype(str), kind="stable")
    data = data.iloc[:, order]
    labels = labels.iloc[order]
    hypotheses, starts_at = np.unique(labels.to_numpy().astype(str), return_index=True)
    n_hyp = len(hypotheses)
    if n_boot + 1 <= n_hyp / 0.05:
        raise ValueError(
            f"n_boot={n_boot} can't resolve BH at 0.05 across {n_hyp} hypotheses; "
            f"use n_boot >= {int(np.ceil(n_hyp / 0.05))}"
        )

    x = data.to_numpy(dtype=np.float64)
    mean = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0, ddof=1), 1e-300)
    t_obs = np.sqrt(n_obs) * mean / sd
    best_obs = np.maximum.reduceat(t_obs, starts_at)

    # A circular block sum is a difference of prefix sums over the doubled
    # series, so each replication moves n_blocks x N numbers, not T x N.
    doubled = np.concatenate([x, x], axis=0)
    p1 = np.zeros((2 * n_obs + 1, x.shape[1]))
    p2 = np.zeros_like(p1)
    np.cumsum(doubled, axis=0, out=p1[1:])
    np.cumsum(doubled**2, axis=0, out=p2[1:])
    n_full, remainder = divmod(n_obs, block)

    rng = np.random.default_rng(seed)
    exceed = np.zeros(n_hyp, dtype=np.int64)
    done = 0
    while done < n_boot:
        size = min(batch_size, n_boot - done)
        starts = rng.integers(0, n_obs, size=(size, n_full))  # rule 1: shared draw
        s1 = (p1[starts + block] - p1[starts]).sum(axis=1)
        s2 = (p2[starts + block] - p2[starts]).sum(axis=1)
        if remainder:
            tail = rng.integers(0, n_obs, size=(size, 1))
            s1 += (p1[tail + remainder] - p1[tail]).sum(axis=1)
            s2 += (p2[tail + remainder] - p2[tail]).sum(axis=1)
        mean_star = s1 / n_obs
        var_star = np.maximum((s2 - n_obs * mean_star**2) / (n_obs - 1), 1e-300)
        t_star = np.sqrt(n_obs) * (mean_star - mean) / np.sqrt(var_star)  # rule 2
        best_star = np.maximum.reduceat(t_star, starts_at, axis=1)  # rule 3
        exceed += (best_star >= best_obs).sum(axis=0)
        done += size

    best_col = [
        data.columns[i + int(np.argmax(t_obs[i:j]))]
        for i, j in zip(starts_at, [*starts_at[1:], len(t_obs)], strict=True)
    ]
    return pd.DataFrame(
        {
            "pvalue": (1.0 + exceed) / (n_boot + 1.0),
            "k": np.diff([*starts_at, len(t_obs)]),
            "best": best_col,
            "t": best_obs,
        },
        index=pd.Index(hypotheses, name="hypothesis"),
    )


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
    at_bound: bool = False  # an estimate sits on a fitting bound: read with care


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
    # A misspecified K often parks delta1 on its floor; say so rather than
    # report the floor as an estimate.
    at_bound = any(
        np.isclose(t, lo, atol=1e-4) or np.isclose(t, hi, atol=1e-4)
        for t, (lo, hi) in zip(best.x, bounds, strict=True)
    )
    return MaxMixtureFit(
        k=k,
        pi0=pi0,
        delta1=delta1,
        sigma0=sigma0,
        sigma1=sigma1,
        loglik=-float(best.fun),
        at_bound=bool(at_bound),
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

    A sensitivity table, not a gate: it can report a low FDR on data with no
    real hypotheses. `at_bound` marks rows whose fit sits on a bound, usually
    a sign that the assumed `k` doesn't fit.

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
                "at_bound": f.at_bound,
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
