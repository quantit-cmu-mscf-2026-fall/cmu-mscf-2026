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


def expected_max_sharpe(n_trials: int, n_obs: int, periods_per_year: int = 252) -> float:
    """Annualised Sharpe you should expect from the BEST of `n_trials` nulls.

    This is the benchmark a candidate must clear to be interesting. Searching
    harder raises it, which is the entire point: the best of 1000 coin-flip
    strategies looks good, and this says how good.

    Uses the standard extreme-value approximation for the maximum of `n_trials`
    independent standard normals (Bailey & Lopez de Prado 2014).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if n_trials == 1:
        return 0.0

    euler_mascheroni = 0.5772156649015329
    quantile = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials) + (
        euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    per_period = quantile / np.sqrt(n_obs)
    return float(per_period * np.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> float:
    """Probability that an observed Sharpe reflects genuine skill.

    The deflated Sharpe ratio (Bailey & Lopez de Prado 2014) asks whether an
    observed Sharpe exceeds what the best of `n_trials` null strategies would
    have produced, correcting for the non-normality of the return series.

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

    benchmark = expected_max_sharpe(n_trials, n_obs, periods_per_year)

    sr = sharpe / np.sqrt(periods_per_year)
    sr0 = benchmark / np.sqrt(periods_per_year)

    # Variance of the Sharpe estimator under non-normal returns.
    variance = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_obs - 1)
    if variance <= 0:
        return float("nan")

    return float(stats.norm.cdf((sr - sr0) / np.sqrt(variance)))


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


def compare_rejection_rules(
    summary: pd.DataFrame,
    truth: pd.Series,
    *,
    alpha: float = 0.05,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compare Benjamini-Hochberg and Bonferroni on the same candidate set.

    Args:
        summary: per-candidate backtest summary, as produced by `capstone.backtest.sweep`.
        truth: boolean Series indexed by candidate name, True where the candidate is real.
        alpha: target family-wise or false-discovery level.
        periods_per_year: annualisation factor to match the Sharpe used in the summary.

    Returns:
        DataFrame indexed by method with columns `rejections`, `power`, and `fdr`.
    """
    aligned_truth = truth.reindex(summary.index).astype(bool)
    if aligned_truth.isna().any():
        missing = aligned_truth.index[aligned_truth.isna()].tolist()
        raise KeyError(f"no ground truth for candidates: {missing[:5]}")

    pvalues = pd.Series(
        [
            sharpe_pvalue(
                float(row["sharpe"]), int(row["n_obs"]), periods_per_year=periods_per_year
            )
            for _, row in summary.iterrows()
        ],
        index=summary.index,
    )

    rows = []
    for name, selector in [("benjamini_hochberg", benjamini_hochberg), ("bonferroni", bonferroni)]:
        rejected = selector(pvalues, alpha=alpha)
        fdr = false_discovery_rate(rejected, aligned_truth)
        rows.append(
            {
                "method": name,
                "rejections": int(rejected.sum()),
                "power": float(power(rejected, aligned_truth)),
                "fdr": float(fdr) if not np.isnan(fdr) else float("nan"),
            }
        )

    return pd.DataFrame(rows).set_index("method")["rejections power fdr".split()]
