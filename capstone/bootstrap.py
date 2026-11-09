"""Stationary bootstrap of a performance matrix (Politis & Romano 1994).

Resampling single days destroys what makes candidate returns hard to test:
autocorrelation and volatility clustering (`synth.make_return_matrix`'s `ar1`
and `garch`). The stationary bootstrap resamples blocks of consecutive dates
instead. Each block starts at a uniformly random date and runs for a geometric
number of dates with mean `mean_block`, wrapping from the last date back to the
first. Random block lengths make the resampled series stationary; the
wrap-around gives every date, including the first and last, the same chance of
being drawn.

This module only draws the dates. A replicate is one row of row indices, and a
caller applies the same row to every column of the matrix, so the correlation
between candidates survives along with each series' own dependence. Callers
compute the statistic they need from `values[idx[b]]` one replicate (or one
batch) at a time; the full n_boot x n_obs x n_candidates resample is never
built here, and should not be built by callers either.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def _check_int(name: str, value, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")


def stationary_bootstrap_indices(
    n_obs: int,
    mean_block: float,
    n_boot: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Row indices for `n_boot` stationary-bootstrap replicates of `n_obs` dates.

    Each replicate starts at a uniformly random date. At every later date it
    starts a new block, at a uniformly random date, with probability
    1 / `mean_block`; otherwise it moves to the next date, wrapping from
    `n_obs - 1` back to 0.

    Args:
        n_obs: number of dates in the sample (>= 2).
        mean_block: expected block length, in dates (>= 1). 1 is the IID
            bootstrap; larger values keep longer-range dependence.
        n_boot: number of replicates (>= 1).
        seed: RNG seed. Pass a value to make the draw reproducible.

    Returns:
        Integer array of shape (n_boot, n_obs) with values in [0, n_obs). Row b
        is replicate b: `values[idx[b]]` resamples every column of a
        dates x candidates array on the same dates.
    """
    _check_int("n_obs", n_obs, 2)
    _check_int("n_boot", n_boot, 1)
    if (
        isinstance(mean_block, bool)
        or not isinstance(mean_block, Real)
        or not math.isfinite(mean_block)
        or mean_block < 1
    ):
        # Below 1 the restart probability would exceed 1; infinite blocks make
        # every replicate a rotation of the sample, which is not a bootstrap.
        raise ValueError(f"mean_block must be a finite number >= 1, got {mean_block!r}")

    rng = np.random.default_rng(seed)
    shape = (n_boot, n_obs)
    new_block = rng.random(shape) < 1.0 / mean_block
    new_block[:, 0] = True
    block_starts = rng.integers(0, n_obs, size=shape)

    # For each date, the position where its block began; the index is that
    # block's random start date plus the steps taken since, wrapped.
    steps = np.arange(n_obs)
    began = np.maximum.accumulate(np.where(new_block, steps, 0), axis=1)
    start = np.take_along_axis(block_starts, began, axis=1)
    return (start + (steps - began)) % n_obs
