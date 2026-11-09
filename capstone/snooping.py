"""White's Reality Check for data snooping (White 2000).

Picking the best of many candidates makes the winner look good even when none
of them is. The Reality Check asks whether the best candidate's mean performance
over a benchmark is larger than the best of the same candidates would be by luck
alone, and answers with one p-value for the whole set: the null is that no
candidate beats the benchmark.

The null distribution is empirical. Each candidate's relative performance is
centred on its own mean, so every candidate is exactly null in the resampled
world, and the matrix is then resampled with the stationary bootstrap on the
same dates for every column. Taking the maximum in each replicate carries the
number of candidates and the correlation between them into the distribution, so
no separate correction for either is needed. The matrix must hold every
candidate tried: one left out is a trial the maximum does not pay for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from capstone.bootstrap import stationary_bootstrap_indices


# eq=False: the generated __eq__ would compare the Series and array fields, and
# their element-wise results have no single truth value.
@dataclass(frozen=True, eq=False)
class RealityCheckResult:
    """Outcome of White's Reality Check.

    Attributes:
        statistic: V, the largest sqrt(n) * mean relative performance.
        pvalue: bootstrap p-value of V; small means the best candidate's edge
            is hard to explain by luck across this many candidates.
        best_candidate: column name of the candidate that attains V.
        mean_relative_performance: mean performance over the benchmark, per
            period, indexed by candidate.
        bootstrap_statistics: the n_boot replicates V*, the empirical null.
    """

    statistic: float
    pvalue: float
    best_candidate: object
    mean_relative_performance: pd.Series
    bootstrap_statistics: np.ndarray


def _relative_performance(performance: pd.DataFrame, benchmark: pd.Series | float) -> np.ndarray:
    """Validate the inputs and return performance minus benchmark as a float array."""
    if not isinstance(performance, pd.DataFrame):
        raise ValueError("performance must be a DataFrame of dates x candidates")
    if performance.shape[1] < 1 or performance.empty:
        raise ValueError("performance must have at least 1 candidate and be non-empty")
    if len(performance) < 2:
        raise ValueError("performance must have at least 2 observations")

    values = performance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("performance must not contain NaN or infinite values")

    if isinstance(benchmark, pd.Series):
        if not benchmark.index.equals(performance.index):
            raise ValueError("benchmark must have the same index as performance")
        base = benchmark.to_numpy(dtype=float)[:, np.newaxis]
        if not np.isfinite(base).all():
            raise ValueError("benchmark must not contain NaN or infinite values")
    elif isinstance(benchmark, bool) or not isinstance(benchmark, Real):
        raise ValueError(f"benchmark must be a Series or a number, got {benchmark!r}")
    elif not math.isfinite(benchmark):
        raise ValueError(f"benchmark must be finite, got {benchmark!r}")
    else:
        base = float(benchmark)
    return values - base


def white_reality_check(
    performance: pd.DataFrame,
    *,
    benchmark: pd.Series | float = 0.0,
    mean_block: float = 10.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> RealityCheckResult:
    """Test whether the best candidate beats the benchmark beyond data snooping.

    Args:
        performance: per-period performance, dates x candidates, every candidate
            tried. No NaN or infinite values.
        benchmark: per-period benchmark to beat, a number or a Series on exactly
            the index of `performance`.
        mean_block: expected stationary-bootstrap block length, in dates (>= 1).
        n_boot: number of bootstrap replicates (>= 1).
        seed: RNG seed for the bootstrap.

    Returns:
        RealityCheckResult with V, its p-value and the bootstrap distribution.
    """
    relative = _relative_performance(performance, benchmark)
    n_obs = len(relative)
    idx = stationary_bootstrap_indices(n_obs, mean_block, n_boot, seed=seed)

    means = relative.mean(axis=0)
    centred = relative - means
    root_n = math.sqrt(n_obs)
    statistic = root_n * means.max()
    # One replicate at a time: the full n_boot x n_obs x n_candidates resample
    # would not fit in memory for a large sweep.
    bootstrap_statistics = root_n * np.array([centred[row].mean(axis=0).max() for row in idx])

    return RealityCheckResult(
        statistic=float(statistic),
        pvalue=float((1 + np.count_nonzero(bootstrap_statistics >= statistic)) / (n_boot + 1)),
        best_candidate=performance.columns[int(means.argmax())],
        mean_relative_performance=pd.Series(means, index=performance.columns),
        bootstrap_statistics=bootstrap_statistics,
    )
