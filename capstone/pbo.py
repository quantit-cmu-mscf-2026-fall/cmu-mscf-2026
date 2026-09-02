"""Probability of backtest overfitting, via combinatorially symmetric cross-validation.

Implements CSCV from Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of
Backtest Overfitting" (Journal of Computational Finance, 2016).

The question this answers is not "is this strategy good?" but "does picking the
best of these N strategies on one half of the sample tell me anything about the
other half?" CSCV answers it by brute force: chop the sample into `n_blocks`
contiguous blocks, and for every way of choosing half the blocks as in-sample,
pick the IS winner and look up where it lands in the out-of-sample ranking. PBO
is the share of splits where the winner comes in below the OOS median.

Read the number against the right baseline. PBO near 0.5 is what *independent*
no-skill candidates produce, not a clean bill of health: when each candidate is
its own draw, its IS and OOS halves are independent, so the winner's OOS rank is
uniform and half the splits land below the median by symmetry. That is the case
`synth.make_panel(n_real=0)` generates, and `tests/test_pbo.py` pins it there
rather than high -- see that file for the measured range.

PBO climbs toward 1 only when the candidates are *coupled*: variants of one
search over one sample, whose full-sample performances therefore cluster
together while their IS/OOS splits do not. Then an exceptional IS half is
borrowed from its OOS half, selecting on IS systematically selects that
borrowing, and the winner reverts. That is the signature of an overfit parameter
sweep, and it is the regime the statistic was built to expose.

PBO near 0 says IS selection genuinely predicts OOS. Confirm it is a real edge
and not a leaking split before believing it -- a planted-signal panel and an
IS/OOS split that share rows both score 0 here.

Input is a T x N matrix of per-period strategy returns, one column per trial.
`backtest.sweep` produces per-candidate *summaries* rather than this matrix, so
build it by running each candidate through `backtest.run_backtest` and
concatenating the resulting series column-wise.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy import stats

# Splits are processed in batches so peak memory stays proportional to this
# times the strategy count, not to C(n_blocks, n_blocks/2) times it.
_COMBO_CHUNK = 4096

# C(24, 12) is 2.7 million splits; past that the run stops being interactive and
# the extra resolution buys nothing. The paper's own examples use 8 to 16.
_MAX_BLOCKS = 24


def _chunked_masks(n_blocks: int, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield in-sample block indicator matrices, `chunk_size` splits at a time.

    Each row is one split: a 0/1 vector over blocks, with exactly half set. The
    combinations are generated lazily rather than materialised, which is what
    keeps a large `n_blocks` from allocating the whole split list up front.
    """
    half = n_blocks // 2
    combinations = itertools.combinations(range(n_blocks), half)

    while True:
        batch = list(itertools.islice(combinations, chunk_size))
        if not batch:
            return
        mask = np.zeros((len(batch), n_blocks))
        chosen = np.array(batch, dtype=int)
        mask[np.repeat(np.arange(len(batch)), half), chosen.ravel()] = 1.0
        yield mask


def _sharpe_from_moments(
    total: np.ndarray,
    total_sq: np.ndarray,
    n_obs: int,
    periods_per_year: int,
) -> np.ndarray:
    """Annualised Sharpe from summed returns and summed squared returns.

    Working from block-level moments is what makes the full combinatorial sweep
    affordable: a split's statistics are the sum of its blocks' moments, so each
    block is visited once no matter how many splits reuse it.

    Returns NaN where the variance is not positive -- a flat strategy has no
    Sharpe, and inventing one would let it win a split.
    """
    mean = total / n_obs
    variance = (total_sq - n_obs * mean**2) / (n_obs - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(variance > 0, mean / np.sqrt(variance), np.nan)
    return sharpe * np.sqrt(periods_per_year)


def cscv(
    returns_matrix: pd.DataFrame,
    n_blocks: int = 16,
    *,
    periods_per_year: int = 252,
) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric CV.

    Args:
        returns_matrix: T x N per-period strategy returns, one column per trial.
            Rows with any missing value are dropped, so that every strategy is
            scored on identical periods -- blocks must line up across columns
            for the IS/OOS comparison to mean anything.
        n_blocks: number of contiguous blocks. Must be even, since every split
            takes exactly half as in-sample. All C(n_blocks, n_blocks/2) splits
            are evaluated, complements included; that symmetry is what makes the
            IS and OOS distributions directly comparable.
        periods_per_year: annualisation factor. Affects only the reported Sharpe
            levels: ranks are invariant to it, and the degradation slope is a
            ratio of equally-scaled quantities.

    Returns:
        dict with:
            pbo: share of splits whose IS winner landed below the OOS median.
            degradation_slope: OLS slope of the winner's OOS Sharpe on its IS
                Sharpe across splits. At or below zero means IS advantage is
                not merely uninformative about OOS but actively inverted.
            degradation_intercept: intercept of that same fit.
            oos_ranks: Series of the winner's relative OOS rank per split, in
                (0, 1), where 0.5 is the median. This is the distribution PBO
                summarises; its shape says more than the scalar does.
            winner_is_sharpe / winner_oos_sharpe: the paired Sharpes behind the
                regression, one entry per split.
            winner_columns: which strategy won each split, by column name.
            n_splits, n_strategies, n_blocks, block_size, n_obs_used: the shape
                actually evaluated, after dropping incomplete rows and trimming.

    Raises:
        ValueError: if `n_blocks` is odd, below 2, above 24, exceeds the usable
            row count, or leaves fewer than 2 observations per side; or if fewer
            than 2 strategies are supplied, since a lone strategy always ranks
            first against itself and PBO would be vacuously 0.
    """
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even and at least 2, got {n_blocks}")
    if n_blocks > _MAX_BLOCKS:
        raise ValueError(
            f"n_blocks above {_MAX_BLOCKS} means more than C({_MAX_BLOCKS}, {_MAX_BLOCKS // 2}) "
            "splits; use fewer blocks"
        )

    clean = returns_matrix.dropna(how="any")
    n_obs_total, n_strategies = clean.shape
    if n_strategies < 2:
        raise ValueError("CSCV needs at least 2 strategies; a single column always ranks first")
    if n_obs_total < n_blocks:
        raise ValueError(f"only {n_obs_total} usable rows for {n_blocks} blocks")

    block_size = n_obs_total // n_blocks
    half = n_blocks // 2
    n_side = block_size * half
    if n_side < 2:
        raise ValueError(
            f"{block_size} rows per block leaves {n_side} observations per side; "
            "a variance needs at least 2"
        )

    # Trim the tail remainder so blocks are equal length and each split's IS and
    # OOS halves are the same size -- otherwise Sharpes carry different
    # estimation error on each side and the ranking compares unlike things.
    values = clean.to_numpy(dtype=float)[: block_size * n_blocks]
    blocks = values.reshape(n_blocks, block_size, n_strategies)
    block_total = blocks.sum(axis=1)
    block_total_sq = np.square(blocks).sum(axis=1)
    grand_total = block_total.sum(axis=0)
    grand_total_sq = block_total_sq.sum(axis=0)

    ranks: list[np.ndarray] = []
    is_sharpes: list[np.ndarray] = []
    oos_sharpes: list[np.ndarray] = []
    winners: list[np.ndarray] = []

    for is_mask in _chunked_masks(n_blocks, _COMBO_CHUNK):
        is_total = is_mask @ block_total
        is_total_sq = is_mask @ block_total_sq
        # The complement is the rest of the sample, so it comes from a
        # subtraction rather than a second matrix product.
        oos_total = grand_total - is_total
        oos_total_sq = grand_total_sq - is_total_sq

        is_sharpe = _sharpe_from_moments(is_total, is_total_sq, n_side, periods_per_year)
        oos_sharpe = _sharpe_from_moments(oos_total, oos_total_sq, n_side, periods_per_year)

        # Rank ascending, so the best OOS Sharpe takes the highest rank. A
        # strategy with no variance sorts to the bottom instead of poisoning the
        # comparison with a NaN.
        rankable = np.where(np.isnan(oos_sharpe), -np.inf, oos_sharpe)
        ranked = stats.rankdata(rankable, axis=1)

        selectable = np.where(np.isnan(is_sharpe), -np.inf, is_sharpe)
        winner = np.argmax(selectable, axis=1)
        rows = np.arange(len(winner))

        ranks.append(ranked[rows, winner])
        is_sharpes.append(is_sharpe[rows, winner])
        oos_sharpes.append(oos_sharpe[rows, winner])
        winners.append(winner)

    winner_rank = np.concatenate(ranks)
    winner_is = np.concatenate(is_sharpes)
    winner_oos = np.concatenate(oos_sharpes)
    winner_index = np.concatenate(winners)

    # Divide by N + 1 so the median rank maps exactly onto 0.5 and no split can
    # sit on the boundary by construction.
    relative_rank = winner_rank / (n_strategies + 1)
    pbo = float(np.mean(relative_rank < 0.5))

    slope, intercept = _degradation_fit(winner_is, winner_oos)

    return {
        "pbo": pbo,
        "degradation_slope": slope,
        "degradation_intercept": intercept,
        "oos_ranks": pd.Series(relative_rank, name="relative_oos_rank"),
        "winner_is_sharpe": winner_is,
        "winner_oos_sharpe": winner_oos,
        "winner_columns": clean.columns[winner_index],
        "n_splits": int(len(relative_rank)),
        "n_strategies": int(n_strategies),
        "n_blocks": int(n_blocks),
        "block_size": int(block_size),
        "n_obs_used": int(block_size * n_blocks),
    }


def _degradation_fit(winner_is: np.ndarray, winner_oos: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of the winner's OOS Sharpe on its IS Sharpe.

    Splits where either side had no Sharpe are dropped rather than zero-filled;
    a flat strategy carries no information about degradation and a zero would be
    read as one. Returns NaNs when too few splits survive, or when every winner
    posted the same IS Sharpe and the slope is undefined.
    """
    usable = np.isfinite(winner_is) & np.isfinite(winner_oos)
    if usable.sum() < 2:
        return float("nan"), float("nan")

    x = winner_is[usable]
    y = winner_oos[usable]
    if np.ptp(x) == 0:
        return float("nan"), float("nan")

    fit = stats.linregress(x, y)
    return float(fit.slope), float(fit.intercept)
