"""Multiple-testing corrections and Sharpe-ratio inference.

When candidates are generated at machine scale, the number of hypotheses tried
is the input every correction depends on — and it is the quantity that is
easiest to lose track of. Deciding early how trials will be counted is cheaper
than reconstructing it in week 12.

Each function here is a claim about error control. `tests/test_evaluate.py`
checks those claims on data where the truth is known; if you modify anything in
this module, those tests are what tell you whether the guarantee still holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_pvalue(sharpe: float, n_obs: int, periods_per_year: int = 252) -> float:
    """Two-sided p-value for an annualised Sharpe ratio against H0: SR = 0.

    Uses the standard large-sample result that an IID Sharpe estimate has
    standard error ~ sqrt(1/n) per period. The annualised figure is converted
    back to a per-period quantity before testing, so `periods_per_year` must
    match the annualisation used to produce `sharpe`.

    Args:
        sharpe: annualised Sharpe ratio.
        n_obs: number of periods the estimate is based on.
        periods_per_year: annualisation factor used for `sharpe`.

    Returns:
        Two-sided p-value in [0, 1]; NaN if `sharpe` is NaN.

    This ignores the skew and kurtosis of the return distribution, which makes
    it optimistic for realistic strategies. `deflated_sharpe_ratio` accounts for
    both, and for the number of trials.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if not np.isfinite(sharpe):
        return float("nan")

    per_period = sharpe / np.sqrt(periods_per_year)
    statistic = per_period * np.sqrt(n_obs)
    return float(2.0 * stats.norm.sf(abs(statistic)))


def expected_max_sharpe(
    n_trials: int,
    n_obs: int,
    periods_per_year: int = 252,
    *,
    trials_sharpe_variance: float | None = None,
) -> float:
    """Annualised Sharpe you should expect from the BEST of `n_trials` nulls.

    This is the benchmark a candidate must clear to be interesting. Searching
    harder raises it, which is the entire point: the best of 1000 coin-flip
    strategies looks good, and this says how good.

    Uses the extreme-value approximation for the maximum of `n_trials`
    independent normals (Bailey & Lopez de Prado 2014, Eq. 1), scaled by the
    spread of the trials' Sharpe ratios. The paper uses the variance measured
    across the trials; by default this uses `periods_per_year / n_obs`, the
    variance of an annualised Sharpe estimate when every trial is an IID normal
    null. Pass `trials_sharpe_variance` (the variance of the trials' annualised
    Sharpe ratios) when the trials are not that: real trials usually spread
    wider, which raises the bar.

    `n_trials` should count independent trials; see `implied_independent_trials`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if trials_sharpe_variance is not None and trials_sharpe_variance < 0:
        raise ValueError("trials_sharpe_variance cannot be negative")
    if n_trials == 1:
        return 0.0

    euler_mascheroni = 0.5772156649015329
    quantile = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials) + (
        euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    if trials_sharpe_variance is None:
        trials_sharpe_variance = periods_per_year / n_obs
    return float(quantile * np.sqrt(trials_sharpe_variance))


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
    *,
    trials_sharpe_variance: float | None = None,
    variance: float | None = None,
) -> float:
    """Probability that an observed Sharpe reflects genuine skill.

    The deflated Sharpe ratio (Bailey & Lopez de Prado 2014, Eq. 2) is the
    probabilistic Sharpe ratio measured against `expected_max_sharpe` rather
    than against zero: does this Sharpe exceed what the best of `n_trials` null
    strategies would have produced?

    Args:
        sharpe: observed annualised Sharpe ratio.
        n_trials: how many candidates were tried to find this one. Counting
            this honestly is the hard part, and it is not our call to make.
        n_obs: number of periods in the track record.
        skew: skewness of the strategy's returns. Negative skew makes a given
            Sharpe less impressive.
        kurtosis: kurtosis of the strategy's returns (3.0 = normal). Fat tails
            make a given Sharpe less impressive.
        periods_per_year: annualisation factor used for `sharpe`.
        trials_sharpe_variance: variance of the trials' annualised Sharpe
            ratios across trials, passed to `expected_max_sharpe`; it sets the
            threshold. Defaults to `periods_per_year / n_obs`, the spread of
            IID normal nulls.
        variance: asymptotic variance V of this candidate's per-period Sharpe
            estimate, from `sharpe_variance`; it sets the standard error of
            the test against the threshold. Defaults to the skew/kurtosis
            expression, which assumes no autocorrelation; see `sharpe_variance`.

    The two arguments correct different things, so pass both for
    autocorrelated trials. `variance` alone fixes the test statistic but leaves
    the threshold assuming IID trials, which autocorrelation widens: on 50
    AR(1) nulls (coefficient 0.3, T = 1000) the best null passed DSR > 0.95 in
    10% of sets by default, 3.7% with `variance` from `sharpe_variance(...,
    "hac")` alone, and under 1% with both (300 simulated sets). A good
    `trials_sharpe_variance` is the variance of the trials' annualised Sharpe
    ratios, or `periods_per_year * V / n_obs` using a typical trial's HAC V.

    Returns:
        Probability in [0, 1]. Conventionally a candidate is retained at > 0.95.
        Returns NaN if `sharpe` is NaN.

    A value near 1.0 does not mean the strategy works — it means this Sharpe is
    unlikely to have arisen from this many trials on null data. That is a much
    narrower claim, and it is the honest one.
    """
    if not np.isfinite(sharpe):
        return float("nan")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")

    threshold = expected_max_sharpe(
        n_trials, n_obs, periods_per_year, trials_sharpe_variance=trials_sharpe_variance
    )
    return probabilistic_sharpe_ratio(
        sharpe,
        n_obs,
        benchmark=threshold,
        skew=skew,
        kurtosis=kurtosis,
        variance=variance,
        periods_per_year=periods_per_year,
    )


def _nonnormal_variance(sr: float, skew: float, kurtosis: float) -> float:
    # Mertens (2002), as written in Bailey & Lopez de Prado (2012, Eqs. 8 and 11).
    return 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2


def newey_west_lags(n_obs: int) -> int:
    """Default truncation lag for `sharpe_variance`: floor(4 (T/100)^(2/9)).

    The conventional Newey-West rule of thumb: it grows with the sample, slowly
    enough that the lag stays a vanishing fraction of it, which is the
    consistency condition in Lo (2002, App. A). Pass `lags` explicitly when the
    return process has longer memory than this allows for.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def sharpe_variance(returns: pd.Series, method: str = "hac", lags: int | None = None) -> float:
    """Asymptotic variance V of the per-period Sharpe: sqrt(T)(SR^ - SR) -> N(0, V).

    Three methods, each dropping an assumption the one before makes:

    - "normal": IID normal returns, V = 1 + SR^2 / 2 (Lo 2002, Eq. 8).
    - "nonnormal": IID returns of any distribution, which adds skew and
      kurtosis, V = 1 - skew SR + (kurtosis - 1)/4 SR^2 (Mertens 2002). This is
      the variance inside the probabilistic and deflated Sharpe ratios.
    - "hac": stationary returns, which adds autocorrelation and volatility
      clustering. Lo's GMM estimator (2002, App. A, Eqs. A8-A15): the moment
      conditions (R - mu, (R - mu)^2 - sigma^2) with a Newey-West (Bartlett)
      long-run covariance, mapped through the delta method. With `lags=0` it
      equals "nonnormal" exactly.

    Use "hac" unless you know returns are serially uncorrelated. Mertens'
    expression is sometimes said to hold for any stationary returns (Bailey &
    Lopez de Prado 2012, citing Opdyke 2007); it has no autocorrelation term and
    does not: on AR(1) nulls with coefficient 0.3 it rejects about 15% of the
    time at a nominal 5% (`tests/test_evaluate.py`). "hac" is much closer but
    still somewhat liberal in finite samples, as Ledoit & Wolf (2008) also find.

    Args:
        returns: per-period returns; NaNs are dropped.
        method: "normal", "nonnormal" or "hac".
        lags: truncation lag for "hac"; default `newey_west_lags(T)`.

    Returns:
        V, in per-period Sharpe units; the standard error of the per-period
        Sharpe is sqrt(V / T). NaN when returns have zero variance.
    """
    if method not in ("normal", "nonnormal", "hac"):
        raise ValueError(f"unknown method {method!r}; options: normal, nonnormal, hac")
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 non-NaN returns")
    if np.ptp(x) == 0:
        return float("nan")
    mu = x.mean()
    centred = x - mu
    var = float(np.mean(centred**2))
    sr = mu / np.sqrt(var)

    if method == "normal":
        return float(1.0 + 0.5 * sr**2)
    if method == "nonnormal":
        lags = 0
    elif lags is None:
        lags = newey_west_lags(n)
    elif lags < 0:
        raise ValueError("lags cannot be negative")

    moments = np.column_stack([centred, centred**2 - var])
    omega = moments.T @ moments / n
    for j in range(1, min(lags, n - 1) + 1):
        gamma = moments[j:].T @ moments[:-j] / n
        omega += (1.0 - j / (lags + 1.0)) * (gamma + gamma.T)
    gradient = np.array([1.0 / np.sqrt(var), -mu / (2.0 * var**1.5)])
    return float(gradient @ omega @ gradient)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    variance: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Probability that the true Sharpe exceeds `benchmark`.

    Bailey & Lopez de Prado (2012), Eq. 11.

    Args:
        sharpe: observed annualised Sharpe ratio.
        n_obs: number of periods in the track record.
        benchmark: annualised Sharpe to beat; 0 asks "is there any skill?".
        skew, kurtosis: of the returns (kurtosis 3.0 = normal). Ignored when
            `variance` is given.
        variance: asymptotic variance from `sharpe_variance`, to use in place of
            the skew/kurtosis expression, e.g. the "hac" one for autocorrelated
            returns.
        periods_per_year: annualisation factor used for `sharpe` and `benchmark`.

    Returns:
        Probability in [0, 1]; NaN if `sharpe` is NaN.
    """
    if not np.isfinite(sharpe):
        return float("nan")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    sr = sharpe / np.sqrt(periods_per_year)
    sr_star = benchmark / np.sqrt(periods_per_year)
    if variance is None:
        variance = _nonnormal_variance(sr, skew, kurtosis)
    if not variance > 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(variance)))


def min_track_record_length(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    variance: float | None = None,
    alpha: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """Observations needed before `sharpe` is significantly above `benchmark`.

    Bailey & Lopez de Prado (2012), Eq. 13. Arguments as for
    `probabilistic_sharpe_ratio`; `alpha` is the one-sided significance level.
    Returns a count of periods, not years. Infinite when `sharpe` does not
    exceed `benchmark`: no track record is long enough. NaN when `sharpe` is
    NaN or the variance is not positive, which the skew/kurtosis expression
    can be for extreme skew; `probabilistic_sharpe_ratio` does the same.

    The formula is asymptotic; the paper warns that the moments going into it
    should come from a longer series than a short MinTRL suggests.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if np.isnan(sharpe):
        return float("nan")
    sr = sharpe / np.sqrt(periods_per_year)
    sr_star = benchmark / np.sqrt(periods_per_year)
    if not sr > sr_star:
        return float("inf")
    if variance is None:
        variance = _nonnormal_variance(sr, skew, kurtosis)
    if not variance > 0:
        return float("nan")
    z = stats.norm.ppf(1.0 - alpha)
    return float(1.0 + variance * (z / (sr - sr_star)) ** 2)


@dataclass(frozen=True)
class SharpeTest:
    """Result of `sharpe_test`. `sharpe` and `se` are annualised."""

    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float
    method: str
    variance: float
    se: float
    pvalue: float
    pvalue_greater: float
    psr: float


def sharpe_test(
    returns: pd.Series,
    *,
    method: str = "hac",
    benchmark: float = 0.0,
    lags: int | None = None,
    periods_per_year: int = 252,
) -> SharpeTest:
    """Test H0: SR = `benchmark` on a return series, with a variance that fits the data.

    Measures skew and kurtosis instead of assuming them and, with the default
    "hac" method, allows for autocorrelation. `pvalue` is two-sided, like
    `sharpe_pvalue`; `pvalue_greater` is one-sided, against SR > `benchmark`;
    `psr` is the probability that SR > `benchmark`.

    Screen candidates on `pvalue_greater`. Only positive Sharpe ratios are
    discoveries, and one-sided statistics that are positively correlated, as
    candidates from one search usually are, are the case in which
    Benjamini-Hochberg is proven to control the FDR (Benjamini & Yekutieli
    2001, Theorem 1.2 and Section 3.1, Case 1). Two-sided p-values under
    correlation are not covered by that theorem.
    """
    clean = returns.dropna()
    variance = sharpe_variance(clean, method=method, lags=lags)
    x = clean.to_numpy(dtype=float)
    n = len(x)
    centred = x - x.mean()
    m2, m3, m4 = (float(np.mean(centred**k)) for k in (2, 3, 4))
    if m2 > 0:
        sharpe = float(x.mean() / np.sqrt(m2) * np.sqrt(periods_per_year))
        skew, kurtosis = m3 / m2**1.5, m4 / m2**2
    else:
        sharpe = skew = kurtosis = float("nan")
    se = float(np.sqrt(variance / n * periods_per_year))
    z = (sharpe - benchmark) / se
    return SharpeTest(
        sharpe=sharpe,
        n_obs=n,
        skew=skew,
        kurtosis=kurtosis,
        method=method,
        variance=variance,
        se=se,
        pvalue=float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
        pvalue_greater=float(stats.norm.sf(z)) if np.isfinite(z) else float("nan"),
        psr=probabilistic_sharpe_ratio(
            sharpe, n, benchmark=benchmark, variance=variance, periods_per_year=periods_per_year
        ),
    )


def average_correlation(returns: pd.DataFrame) -> float:
    """Equal-weighted average off-diagonal correlation of the trials.

    Bailey & Lopez de Prado (2014), Eq. 8.
    """
    m = returns.shape[1]
    if m < 2:
        raise ValueError("need at least 2 trials")
    corr = returns.corr().to_numpy()
    return float((np.nansum(corr) - np.trace(corr)) / (m * (m - 1)))


def implied_independent_trials(n_trials: int, avg_correlation: float) -> float:
    """Independent trials implied by `n_trials` correlated ones.

    Bailey & Lopez de Prado (2014), Eq. 9: N = rho + (1 - rho) M, interpolating
    between one trial (all perfectly correlated) and M (uncorrelated). A rough
    correction, as the paper says. Use it for `expected_max_sharpe`, never to
    shrink the trial count a multiple-testing correction on p-values uses.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if not -1.0 <= avg_correlation <= 1.0:
        raise ValueError("avg_correlation must be in [-1, 1]")
    rho = max(avg_correlation, 0.0)
    return float(rho + (1.0 - rho) * n_trials)


def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg step-up procedure controlling the false-discovery rate.

    Controls the expected PROPORTION of rejections that are false, at `alpha`.
    This is a weaker and usually more useful guarantee than Bonferroni's, which
    bounds the probability of even one false rejection.

    Args:
        pvalues: p-values indexed by candidate name. NaNs are never rejected.
        alpha: target false-discovery rate.

    Returns:
        Boolean Series on the same index, True where the null is rejected.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    rejected = pd.Series(False, index=pvalues.index)
    usable = pvalues.dropna()
    if usable.empty:
        return rejected

    ordered = usable.sort_values()
    m = len(ordered)
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = ordered.to_numpy() <= thresholds

    if not passing.any():
        return rejected

    # Step-up: reject everything at or below the LARGEST passing rank, not only
    # the individually-passing ones.
    cutoff_rank = int(np.max(np.flatnonzero(passing)))
    rejected.loc[ordered.index[: cutoff_rank + 1]] = True
    return rejected


def bonferroni(pvalues: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Bonferroni correction controlling the family-wise error rate.

    Bounds the probability of ANY false rejection at `alpha`. Conservative by
    design: with many candidates it will reject almost nothing, which is the
    correct behaviour when the cost of a single false discovery is high.

    NaN p-values are counted in the family size — they were trials too.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if len(pvalues) == 0:
        return pd.Series(False, index=pvalues.index, dtype=bool)

    return (pvalues <= alpha / len(pvalues)).fillna(False)


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def holm(pvalues: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Holm's step-down procedure controlling the family-wise error rate.

    Sort the p-values; reject the i-th smallest while p_(i) <= alpha / (m - i + 1)
    and stop at the first that fails. Same guarantee as Bonferroni (no false
    rejection with probability 1 - alpha, under any dependence), never fewer
    rejections. Use it wherever Bonferroni would be used.

    NaN p-values are never rejected and count toward the family size m, as in
    `bonferroni`: they were trials too.
    """
    _check_alpha(alpha)
    rejected = pd.Series(False, index=pvalues.index)
    m = len(pvalues)
    ordered = pvalues.dropna().sort_values()
    if ordered.empty:
        return rejected
    thresholds = alpha / (m - np.arange(len(ordered)))
    failing = ordered.to_numpy() > thresholds
    n_rejected = int(np.argmax(failing)) if failing.any() else len(ordered)
    rejected.loc[ordered.index[:n_rejected]] = True
    return rejected


def benjamini_yekutieli(pvalues: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Yekutieli: the BH step-up procedure run at alpha / sum(1/i).

    Controls the false-discovery rate at `alpha` under ANY dependence between
    the tests (Benjamini & Yekutieli 2001, Theorem 1.3). Plain
    `benjamini_hochberg` already does under positive dependence (PRDS,
    their Theorem 1.2), which covers one-sided tests on positively correlated
    candidates; use this one when that cannot be assumed: two-sided p-values,
    candidates that include mirror images or hedges of each other, or any
    negative correlation. The price is power: at m = 1000 the level is divided
    by about 7.5.

    NaN p-values are never rejected and count toward m, as in `bonferroni`.
    """
    _check_alpha(alpha)
    m = len(pvalues)
    if m == 0:
        return pd.Series(False, index=pvalues.index, dtype=bool)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    rejected = pd.Series(False, index=pvalues.index)
    ordered = pvalues.dropna().sort_values()
    if ordered.empty:
        return rejected
    thresholds = alpha / harmonic * np.arange(1, len(ordered) + 1) / m
    passing = ordered.to_numpy() <= thresholds
    if passing.any():
        cutoff_rank = int(np.max(np.flatnonzero(passing)))
        rejected.loc[ordered.index[: cutoff_rank + 1]] = True
    return rejected


_LAMBDA_GRID = np.round(np.arange(0.0, 0.951, 0.05), 2)


def estimate_pi0(
    pvalues: pd.Series,
    lambda_: float | str = 0.5,
    *,
    n_boot: int = 200,
    gamma: float = 0.05,
    seed: int | None = 0,
) -> float:
    """Estimated share of true nulls among the candidates (Storey 2002).

    pi0(lambda) = #{p > lambda} / ((1 - lambda) m): p-values above lambda come
    almost only from nulls, which are uniform, so their count scales up to the
    number of nulls (Algorithm 1(b)). Conservative, biased upward, for any
    lambda. `lambda_=0` gives 1; larger lambda has less bias and more variance.
    `lambda_="bootstrap"` picks lambda from {0, 0.05, ..., 0.95} by bootstrap
    mean-squared error of the pFDR estimate at rejection region [0, `gamma`]
    (Algorithm 3).

    The paper assumes independent p-values throughout, and correlation breaks
    the estimate, not just its precision: when a shared factor moves every
    candidate the same way, few p-values land above lambda and pi0 collapses.
    On all-null sets of 100 candidates with pairwise correlation 0.5 it
    averaged 0.73 rather than 1, fell below 0.8 in 41% of sets and reached 0
    (`tests/test_evaluate.py`). Treat it as a description of independent
    candidates only.

    NaN p-values are dropped.
    """
    p = pvalues.dropna().to_numpy(dtype=float)
    if len(p) == 0:
        raise ValueError("no usable p-values")
    if lambda_ == "bootstrap":
        lambda_ = _bootstrap_lambda(p, gamma=gamma, n_boot=n_boot, seed=seed)
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError("lambda_ must be in [0, 1) or 'bootstrap'")
    return float(min(1.0, _pi0(p, lambda_)))


def _pi0(p: np.ndarray, lambda_: float) -> float:
    return float(np.sum(p > lambda_) / ((1.0 - lambda_) * len(p)))


def _pfdr(p: np.ndarray, lambda_: float, gamma: float) -> float:
    # Storey (2002), Algorithm 1(c), capped at 1 as Section 3 recommends.
    m = len(p)
    rejected_share = max(int(np.sum(p <= gamma)), 1) / m
    value = _pi0(p, lambda_) * gamma / (rejected_share * (1.0 - (1.0 - gamma) ** m))
    return min(value, 1.0)


def _bootstrap_lambda(p: np.ndarray, *, gamma: float, n_boot: int, seed: int | None) -> float:
    # Storey (2002), Algorithm 3: the plug-in target is the smallest pFDR
    # estimate over the grid; pick the lambda whose bootstrap estimates sit
    # closest to it in mean square.
    rng = np.random.default_rng(seed)
    estimates = np.array([_pfdr(p, lam, gamma) for lam in _LAMBDA_GRID])
    target = estimates.min()
    samples = [p[rng.integers(0, len(p), size=len(p))] for _ in range(n_boot)]
    mse = [np.mean([(_pfdr(s, lam, gamma) - target) ** 2 for s in samples]) for lam in _LAMBDA_GRID]
    return float(_LAMBDA_GRID[int(np.argmin(mse))])


def storey_qvalues(
    pvalues: pd.Series, lambda_: float | str = 0.5, *, pfdr: bool = False
) -> pd.Series:
    """q-values: the smallest false-discovery rate at which each candidate is called.

    Storey (2002), Algorithm 2: q(p_(m)) = FDR(p_(m)), and moving down the
    sorted p-values, q(p_(i)) = min(FDR(p_(i)), q(p_(i+1))), with the FDR
    estimate pi0 * p * m / #{p_j <= p}. That is BH's adjusted p-value times the
    estimated share of nulls pi0, so it is never weaker than BH and gains when
    many candidates are real, because BH implicitly assumes pi0 = 1. At
    `lambda_=0`, pi0 = 1 and the q-values are exactly `bh_adjusted`.

    `pfdr=True` uses Storey's positive-FDR estimate instead, as in the paper.
    It divides by the chance of making any rejection, so as p goes to 0 it
    tends to pi0 over the number of discoveries (Eq. 22): with ten real
    candidates among a hundred no q-value falls much below 0.1, however strong
    the signal. That is the paper's intent, but it makes pFDR q-values a poor
    score when real candidates are rare, which is the usual case here.

    Only for independent candidates. Because pi0 collapses under correlation
    (see `estimate_pi0`), q-values of correlated candidates are too small: on
    all-null sets with pairwise correlation 0.5, some q fell below 0.05 in 12%
    of sets, against 4.5% for BH. Never screen or decide on them there; use
    `bh_adjusted` (positive dependence) or `by_adjusted` (any dependence).

    NaN p-values get NaN q-values and do not count toward m.
    """
    usable = pvalues.dropna()
    q = pd.Series(np.nan, index=pvalues.index)
    if usable.empty:
        return q
    ordered = usable.sort_values()
    p = ordered.to_numpy(dtype=float)
    m = len(p)
    pi0 = estimate_pi0(usable, lambda_)
    # R(p): p-values at or below each one, so tied p-values share a q-value.
    counts = np.searchsorted(p, p, side="right")
    raw = pi0 * p * m / counts
    if pfdr:
        # Pr(R > 0) under the null; at p = 0 the ratio tends to pi0 / R
        # (Storey 2002, Eq. 22), which is what the limit gives here.
        at_least_one = 1.0 - (1.0 - p) ** m
        raw = np.where(
            at_least_one > 0, raw / np.where(at_least_one > 0, at_least_one, 1.0), pi0 / counts
        )
    raw = np.minimum(raw, 1.0)
    q.loc[ordered.index] = np.minimum.accumulate(raw[::-1])[::-1]
    return q


def _adjusted(pvalues: pd.Series, family_size: int, scale, *, step_up: bool) -> pd.Series:
    # Adjusted p-value of the i-th smallest p: the smallest level at which the
    # procedure would reject it. Step-down procedures take a running maximum
    # from the smallest p upward, step-up ones a running minimum from the top.
    adjusted = pd.Series(np.nan, index=pvalues.index)
    ordered = pvalues.dropna().sort_values()
    if ordered.empty:
        return adjusted
    ranks = np.arange(1, len(ordered) + 1)
    raw = np.minimum(scale(ordered.to_numpy(dtype=float), ranks, family_size), 1.0)
    if step_up:
        values = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        values = np.maximum.accumulate(raw)
    adjusted.loc[ordered.index] = values
    return adjusted


def holm_adjusted(pvalues: pd.Series) -> pd.Series:
    """Holm-adjusted p-values: `holm(pvalues, alpha)` rejects exactly those <= alpha.

    A graded score in place of a yes/no: the family-wise error rate at which
    this candidate would first be called real. NaN stays NaN and counts toward m.
    """
    return _adjusted(pvalues, len(pvalues), lambda p, i, m: (m - i + 1) * p, step_up=False)


def bh_adjusted(pvalues: pd.Series) -> pd.Series:
    """BH-adjusted p-values: `benjamini_hochberg(pvalues, alpha)` rejects those <= alpha.

    The false-discovery rate at which this candidate would first be selected.
    NaN stays NaN and, as in `benjamini_hochberg`, does not count toward m.
    Equal to `storey_qvalues(pvalues, lambda_=0, pfdr=False)`.
    """
    usable = int(pvalues.notna().sum())
    return _adjusted(pvalues, usable, lambda p, i, m: m * p / i, step_up=True)


def by_adjusted(pvalues: pd.Series) -> pd.Series:
    """BY-adjusted p-values: `benjamini_yekutieli(pvalues, alpha)` rejects those <= alpha.

    BH-adjusted p-values scaled by sum(1/i), m counting NaN, as in
    `benjamini_yekutieli`: the FDR level valid under any dependence.
    """
    m = len(pvalues)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1))) if m else 1.0
    return _adjusted(pvalues, m, lambda p, i, n: harmonic * n * p / i, step_up=True)


def evidence_profile(pvalues: pd.Series, *, lambda_: float | str = 0.5) -> pd.DataFrame:
    """Every correction's graded score for each candidate, side by side.

    One row per candidate, sorted by p-value, with the raw p-value and the level
    at which each procedure would first call it real: `holm` (family-wise
    error, any dependence), `by` (FDR, any dependence), `bh` (FDR, positive
    dependence), and Storey's `q` (estimated pFDR, independence). Lower is
    stronger in every column; a candidate that is strong only in the right-hand
    columns is plausible, not established.

    Describing a candidate set is what this is for. A decision still takes one
    column and one threshold, fixed before the data is scored; choosing the
    column afterwards is itself a multiple-testing problem. Never decide on
    `q` for correlated candidates; see `storey_qvalues`.
    """
    profile = pd.DataFrame(
        {
            "p": pvalues,
            "holm": holm_adjusted(pvalues),
            "by": by_adjusted(pvalues),
            "bh": bh_adjusted(pvalues),
            "q": storey_qvalues(pvalues, lambda_=lambda_),
        }
    )
    return profile.sort_values("p", na_position="last")


def false_discovery_rate(rejected: pd.Series, truth: pd.Series) -> float:
    """Realised proportion of rejections that were false.

    Args:
        rejected: boolean Series, True where a candidate was selected.
        truth: boolean Series, True where the candidate is genuinely predictive.

    Returns:
        False discoveries / total discoveries. NaN when nothing was rejected —
        the rate is undefined with an empty denominator, and reporting 0.0 there
        would read as perfect precision from a procedure that made no calls.
    """
    aligned_truth = truth.reindex(rejected.index)
    if aligned_truth.isna().any():
        missing = aligned_truth.index[aligned_truth.isna()].tolist()
        raise KeyError(f"no ground truth for candidates: {missing[:5]}")

    n_rejected = int(rejected.sum())
    if n_rejected == 0:
        return float("nan")

    false_positives = int((rejected & ~aligned_truth.astype(bool)).sum())
    return false_positives / n_rejected


def power(rejected: pd.Series, truth: pd.Series) -> float:
    """Share of genuinely predictive candidates that were found.

    The companion to `false_discovery_rate`: a procedure that rejects nothing
    has a perfect false-discovery record and zero power. Both belong in any
    honest report of a selection rule.

    Returns NaN when there are no true signals to find.
    """
    aligned_truth = truth.reindex(rejected.index).astype(bool)
    n_true = int(aligned_truth.sum())
    if n_true == 0:
        return float("nan")
    return int((rejected & aligned_truth).sum()) / n_true


# --------------------------------------------------------------------------- #
# Haircut Sharpe: Harvey & Liu (2015) -- QUANTIT-51, QUANTIT-52
#
# This section follows the authors' own MATLAB reference implementation,
# `Haircut_SR.m` and `sample_random_multests.m`
# (https://people.duke.edu/~charvey/backtesting/), not a reading of the paper.
# `scripts/haircut_reference.py` is a literal transcription of those two files
# and is the oracle the tests compare this code against.
#
# IMPORTANT: Harvey & Liu's time conventions are NOT this repository's. They
# convert every frequency to MONTHS and treat a daily year as 360 days; every
# other method in this module annualises with `periods_per_year`, default 252.
# Those conventions are confined to this section, reached only through
# `haircut_sharpe`, and must not be propagated outward.
# --------------------------------------------------------------------------- #

# Harvey, Liu & Zhu (2014) parameters, verbatim from `Haircut_SR.m`:
# rows are average correlation levels, columns [rho, m_tot, p_0, lambda].
_HLZ_PARAMETERS = np.array(
    [
        [0.0, 1295, 3.9660 * 0.1, 5.4995 * 0.001],
        [0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001],
        [0.4, 1476, 4.8604 * 0.1, 5.5413 * 0.001],
        [0.6, 1773, 5.9902 * 0.1, 5.5512 * 0.001],
        [0.8, 3109, 8.3901 * 0.1, 5.5956 * 0.001],
    ]
)

# The authors' `sm_fre` frequencies and their periods per year. 360, not 365 or
# 252, is what `Haircut_SR.m` divides by for daily data.
_HL_PERIODS_PER_YEAR = {
    "daily": 360,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
}

# `sample_random_multests.m`: the simulated trials are monthly returns with this
# assumed volatility over this many periods.
_HLZ_MONTHLY_VOL = 0.15 / np.sqrt(12)
_HLZ_N_OBS = 240

_HAIRCUT_METHODS = ("bonferroni", "holm", "bhy")


def _hlz_parameters(avg_correlation: float) -> np.ndarray:
    """`para_inter`: the HLZ row interpolated on the assumed average correlation.

    Reproduces the authors' branch structure exactly, including the fact that
    their `RHO >= 0.8` case reuses the `[0.6, 0.8)` formula and therefore
    extrapolates (at 0.9 it gives `-0.5*row4 + 1.5*row5`), and that a RHO
    outside `[0, 1)` silently falls back to the 0.2 row.
    """
    rho = float(avg_correlation)
    if not 0.0 <= rho < 1.0:
        return _HLZ_PARAMETERS[1]
    index = min(int(rho / 0.2), 3)
    weight = (rho - _HLZ_PARAMETERS[index, 0]) / 0.2
    return (1.0 - weight) * _HLZ_PARAMETERS[index] + weight * _HLZ_PARAMETERS[index + 1]


def _simulate_hlz_tstats(
    n_other: int,
    *,
    rho: float,
    p_0: float,
    lambda_: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """`sample_random_multests.m`: a panel of |t| ratios for `n_other` trials.

    One row per repetition. A share `1 - p_0` of trials are given a non-zero
    mean drawn from `Exponential(lambda_)`; all of them get an equicorrelated
    normal shock.

    The authors build the shock with `mvnrnd` on a full `m_tot x m_tot`
    equicorrelation matrix and then use only its first M columns. An
    equicorrelation matrix has the one-factor representation
    `sqrt(rho) * common + sqrt(1 - rho) * own`, so drawing M columns that way is
    distributionally identical to drawing `m_tot` and slicing, at a fraction of
    the cost. `scripts/haircut_reference.py` keeps the literal `mvnrnd`, and
    `TestHaircutReferenceComparison` checks the two agree.
    """
    scale = _HLZ_MONTHLY_VOL / np.sqrt(_HLZ_N_OBS)
    common = rng.standard_normal((n_simulations, 1))
    own = rng.standard_normal((n_simulations, n_other))
    shock = (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * own) * scale

    is_real = rng.uniform(size=(n_simulations, n_other)) > p_0
    means = rng.exponential(lambda_, size=(n_simulations, n_other))
    return np.abs(is_real * means + shock) / scale


def _hl_adjusted_pvalues(
    pvalue: float, simulated: np.ndarray, family_size: int, harmonic: float
) -> tuple[float, float]:
    """Holm and BHY adjusted p-values for one observed p among simulated trials.

    `simulated` is the panel of |t| ratios, one row per repetition. For each row
    the observed p-value joins the row's p-values, the family of `family_size`
    is adjusted, and the observed strategy's own adjusted value is read back out.
    The authors take the MEDIAN across repetitions, which is what makes the
    result stable without being any one draw.

    Vectorised over repetitions. Holm's running maximum and BHY's running
    minimum are the same recursions the reference transcription writes as loops.
    """
    p_simulated = 2.0 * stats.norm.sf(simulated)
    n_reps = p_simulated.shape[0]
    ordered = np.sort(np.column_stack([p_simulated, np.full(n_reps, pvalue)]), axis=1)
    # Leftmost position of the observed p-value in each sorted row, matching the
    # authors' `p_val_order == p_val` followed by taking the first match.
    position = (p_simulated < pvalue).sum(axis=1)
    rows = np.arange(n_reps)

    ranks = np.arange(family_size)
    holm = np.minimum(np.maximum.accumulate((family_size - ranks) * ordered, axis=1), 1.0)

    # BHY: scale rank k by family_size * c(family_size) / k, then take a running
    # minimum from the largest p-value down. The largest is left unscaled, as in
    # the authors' `if kk == (M+1)` branch.
    scaled = (family_size * harmonic / (ranks + 1.0)) * ordered
    scaled[:, -1] = ordered[:, -1]
    bhy = np.minimum.accumulate(scaled[:, ::-1], axis=1)[:, ::-1]

    return float(np.median(holm[rows, position])), float(np.median(bhy[rows, position]))


def haircut_sharpe(
    sharpe: pd.Series | float,
    n_obs: int,
    *,
    n_trials: int | None = None,
    experiment: str | None = None,
    frequency: str = "monthly",
    annualized: bool = True,
    autocorrelation: float = 0.0,
    avg_correlation: float = 0.2,
    n_simulations: int = 2000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """The Sharpe that survives the size of the search (Harvey & Liu 2015).

    Turn the reported Sharpe into a p-value, adjust that p-value for every trial
    the search ran, and read the adjusted p-value back out as a Sharpe. The
    difference is the haircut. All three of the authors' adjustments are
    reported, plus their average, because their program reports all four and
    choosing between them after seeing the numbers is itself multiple testing.

    A **graded score, not a gate** (`docs/validation/framework.md`, stage 4):
    the stage-4 gate is the deflated Sharpe, and charging both charges the same
    search twice.

    **This follows the authors' reference code**, `Haircut_SR.m` and
    `sample_random_multests.m`, which differs from a naive reading of the paper
    in ways that matter:

    - p-values are **two-sided**, from a **t distribution with N-1 degrees of
      freedom** for the reported Sharpe and from a **normal** for the simulated
      trials. There is no one-sided option: the authors do not offer one, and a
      one-sided variant would not be comparable to any published figure.
    - everything is converted to **monthly** units, and a daily year is **360
      days**. This contradicts the `periods_per_year=252` convention every other
      function here uses; it is deliberate, confined to this function, and must
      not be copied outward. Pass `n_obs` in the units named by `frequency`.
    - the trials that were run but are not reported are **simulated**, not
      assumed away, from the Harvey, Liu & Zhu (2014) empirical p-value
      distribution. Holm and BHY are computed on each simulated family and the
      **median** adjusted p-value over `n_simulations` repetitions is taken, so
      the result is stochastic. Fix `seed` to make it reproducible.

    **The trial count, and the M / M+1 distinction.** `n_trials` is the ledger's
    total, which *includes* the strategy being haircut. The authors' `num_test`
    (their `M`) is the number of OTHER trials, so `num_test = n_trials - 1`. Get
    this wrong by one and every adjusted p-value is wrong. The split is not
    cosmetic, because the authors' own code is inconsistent about it:

    - **Bonferroni** uses `min(num_test * p, 1)` -- `M`, not `M + 1`.
    - **Holm and BHY** use a family of `num_test + 1 = n_trials`, the simulated
      trials plus the reported one, and BHY's harmonic constant is
      `sum(1/i)` over that same `n_trials`.

    Both are reproduced as the authors wrote them. `.attrs` records `n_trials`
    and `num_test` separately so the convention in force is never in doubt.

    Leave `n_trials` as None and it comes from `runlog.trial_count(experiment)`.
    `LookupError` is left to propagate: a haircut against no trials is no
    haircut.

    Args:
        sharpe: reported Sharpe per candidate, indexed by candidate name; a bare
            float is accepted and reported under the name "strategy". Annualised
            unless `annualized=False`.
        n_obs: observations behind each Sharpe, in the units of `frequency`.
        n_trials: ledger total, *including* this strategy. None reads the ledger.
        experiment: ledger experiment name to count; ignored if `n_trials` given.
        frequency: "daily", "weekly", "monthly", "quarterly" or "annual".
        annualized: whether `sharpe` is already annualised (the authors'
            `ind_an`). False multiplies by sqrt(periods per year).
        autocorrelation: first-order autocorrelation of the returns at
            `frequency` (their `rho`). Non-zero applies their autocorrelation
            correction to the annualised Sharpe before testing; it does nothing
            at `frequency="annual"`, as in their code.
        avg_correlation: assumed average correlation across trials (their
            `RHO`), which selects the simulated p-value distribution. Only
            `[0, 1)` is meaningful; outside it the authors fall back to 0.2.
        n_simulations: repetitions of the simulated family (their `WW`).
        seed: RNG seed for the simulation. None draws fresh entropy.

    Returns:
        DataFrame indexed by candidate, with `sharpe` as given,
        `sharpe_annualized` after annualisation and any autocorrelation
        correction, `pvalue`, and then `pvalue_*`, `sharpe_*` and `haircut_*`
        for each of `bonferroni`, `holm`, `bhy` and `average`. `haircut` is the
        fraction given up, `(sr - sr_adjusted) / sr`.

        A candidate whose Sharpe is not positive gets NaN across the board: the
        authors' `p_val = 2*(1 - tcdf(T, N-1))` uses `T`, not `|T|`, and exceeds
        1 for `T < 0`, so their procedure is defined only for a positive
        reported Sharpe.

        `.attrs` carries `n_trials`, `num_test`, `n_monthly_obs`, `frequency`,
        `n_simulations`, `seed` and `avg_correlation`.

    Raises:
        ValueError: unknown `frequency`; `n_obs` below 1; `n_trials` below 2
            (the search must contain at least this strategy and one other);
            `n_simulations` below 1; `autocorrelation` outside (-1, 1); or fewer
            than 2 monthly observations after conversion, which leaves the t
            distribution without a degree of freedom.
        LookupError: from `runlog.trial_count` when the ledger has no match.
    """
    if frequency not in _HL_PERIODS_PER_YEAR:
        raise ValueError(
            f"unknown frequency {frequency!r}; options: {', '.join(_HL_PERIODS_PER_YEAR)}"
        )
    if n_obs < 1:
        raise ValueError("n_obs must be at least 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")
    if not -1.0 < autocorrelation < 1.0:
        raise ValueError("autocorrelation must be in (-1, 1)")

    reported = pd.Series(sharpe, dtype=float)
    if reported.index.equals(pd.RangeIndex(len(reported))) and len(reported) == 1:
        reported.index = pd.Index(["strategy"])

    if n_trials is None:
        # Imported here so `evaluate` stays importable where the ledger is not
        # configured; the ledger is consulted only when the count is not given.
        from capstone.runlog import trial_count

        n_trials = trial_count(experiment)
    if n_trials < 2:
        raise ValueError(
            f"n_trials ({n_trials}) must be at least 2: it is the ledger's total "
            "including this strategy, and the authors' num_test = n_trials - 1 "
            "must leave at least one other trial"
        )

    num_test = int(n_trials) - 1  # the authors' M
    family_size = int(n_trials)  # the authors' M + 1, for Holm and BHY
    harmonic = float(np.sum(1.0 / np.arange(1, family_size + 1)))

    periods = _HL_PERIODS_PER_YEAR[frequency]
    n_monthly = int(np.floor(n_obs * 12 / periods))
    if n_monthly < 2:
        raise ValueError(
            f"{n_obs} {frequency} observations is {n_monthly} months; the "
            "authors' t test needs at least 2"
        )

    annualised = reported if annualized else reported * np.sqrt(periods)
    if autocorrelation and frequency != "annual":
        rho = autocorrelation
        annualised = (
            annualised
            * (
                1.0
                + (2.0 * rho / (1.0 - rho))
                * (1.0 - ((1.0 - rho**periods) / (periods * (1.0 - rho))))
            )
            ** -0.5
        )

    statistic = annualised / np.sqrt(12) * np.sqrt(n_monthly)
    pvalue = pd.Series(2.0 * (1.0 - stats.t.cdf(statistic, n_monthly - 1)), index=reported.index)
    testable = annualised.notna() & (annualised > 0) & pvalue.le(1.0)

    rng = np.random.default_rng(seed)
    params = _hlz_parameters(avg_correlation)
    simulated = _simulate_hlz_tstats(
        num_test,
        rho=float(params[0]),
        p_0=float(params[2]),
        lambda_=float(params[3]),
        n_simulations=n_simulations,
        rng=rng,
    )

    columns = {name: pd.Series(np.nan, index=reported.index) for name in _HAIRCUT_METHODS}
    for name in reported.index[testable]:
        p = float(pvalue[name])
        holm_p, bhy_p = _hl_adjusted_pvalues(p, simulated, family_size, harmonic)
        columns["bonferroni"][name] = min(num_test * p, 1.0)
        columns["holm"][name] = holm_p
        columns["bhy"][name] = bhy_p

    adjusted = pd.DataFrame(columns)
    adjusted["average"] = adjusted.mean(axis=1)

    out = pd.DataFrame(
        {
            "sharpe": reported,
            "sharpe_annualized": annualised.where(testable),
            "pvalue": pvalue.where(testable),
        }
    )
    for name in (*_HAIRCUT_METHODS, "average"):
        # Every adjusted p-value is clipped at 1, so the inverse is never
        # negative and the haircut never exceeds 100%.
        survived = (
            stats.t.ppf(1.0 - adjusted[name] / 2.0, n_monthly - 1)
            / np.sqrt(n_monthly)
            * np.sqrt(12)
        )
        out[f"pvalue_{name}"] = adjusted[name]
        out[f"sharpe_{name}"] = survived
        out[f"haircut_{name}"] = 1.0 - survived / out["sharpe_annualized"]

    out.attrs.update(
        {
            "n_trials": int(n_trials),
            "num_test": num_test,
            "n_monthly_obs": n_monthly,
            "frequency": frequency,
            "n_simulations": int(n_simulations),
            "seed": seed,
            "avg_correlation": float(avg_correlation),
        }
    )
    return out.sort_values("pvalue")
