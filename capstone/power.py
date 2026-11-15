"""Power analysis of the acceptance gate: simulated null and planted worlds.

For registered `Thresholds` and a batch's shape (families, variants per
family, their correlation, search and holdout lengths, the bootstrap's size)
the power analysis asks how strong a family must be before the gate accepts
it. This module builds the worlds that question is answered on. It adds no
statistics: every number comes from the funnel's own code
(`funnel.family_statistics`) on `calibration.make_family_set` data.

**A world** is a null batch from the family generator (variants of one family
correlate `within_corr`, of different families `across_corr`). A target
family's variants are then given a true Sharpe on the search period and the
holdout (`plant`, the generator's own rule), and only the target's statistics
are recomputed: a family's statistics, bootstrap included, depend only on its
own columns and the seed (tested). One null world serves every Sharpe.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from capstone import funnel
from capstone.calibration import FamilySet, make_family_set
from capstone.funnel import Thresholds
from capstone.search_fdr import block_length_rule

__all__ = [
    "NullSide",
    "PowerSpec",
    "null_side",
    "null_world",
    "plant",
    "planted_side",
]

PERIODS_PER_YEAR = 252
VOL = 0.01  # `synth.make_return_matrix`'s per-period volatility
FIRST_DATE = "1990-01-01"


@dataclass(frozen=True, kw_only=True)
class PowerSpec:
    """The configuration whose power is measured.

    variants: each family's variant count, or one count for `n_families`.
    within_corr / across_corr: average pairwise correlation of two variants of
    one family / of different families. Lengths are trading days
    (`from_years`). n_boot: stage 1's bootstrap draws. n_trials: the ledger's
    count, recorded only: no registered decision reads it.
    """

    variants: Sequence[int] | int  # stored as a tuple
    within_corr: float
    search_periods: int
    holdout_periods: int
    n_boot: int
    thresholds: Thresholds = field(default_factory=Thresholds)
    across_corr: float = 0.0
    n_families: int | None = None
    n_trials: int | None = None

    def __post_init__(self) -> None:
        v = self.variants
        if np.ndim(v) == 0:
            if self.n_families is None:
                raise ValueError("one variant count for all families needs n_families")
            v = (v,) * int(self.n_families)
        v = tuple(int(x) for x in v)
        if not v or min(v) < 1:
            raise ValueError("every family needs at least one variant")
        if self.n_families is not None and self.n_families != len(v):
            raise ValueError(f"n_families={self.n_families} but {len(v)} variant counts")
        object.__setattr__(self, "variants", v)
        object.__setattr__(self, "n_families", len(v))
        if not 0.0 <= self.across_corr < 1.0:
            raise ValueError("across_corr must be in [0, 1)")
        if not self.across_corr <= self.within_corr <= 1.0:
            raise ValueError("within_corr must be in [across_corr, 1]")
        # The bootstrap's floor: every column needs 10 blocks.
        floor = 10 * block_length_rule(self.search_periods)
        if self.search_periods < floor or self.holdout_periods < 2:
            raise ValueError(f"need >= {floor} search periods and >= 2 holdout periods")
        if self.n_boot < 0 or (
            self.thresholds.stage1 == "bootstrap" and self.n_boot + 1 < 20 * len(v)
        ):
            raise ValueError(f"n_boot + 1 must be >= 20 x {len(v)} families for the bootstrap")
        if self.n_trials is not None and self.n_trials < sum(v):
            raise ValueError(f"n_trials={self.n_trials} is fewer than the {sum(v)} variants")

    @classmethod
    def from_years(cls, *, search_years: float, holdout_years: float, **kwargs) -> PowerSpec:
        return cls(
            search_periods=round(search_years * PERIODS_PER_YEAR),
            holdout_periods=round(holdout_years * PERIODS_PER_YEAR),
            **kwargs,
        )

    @property
    def within_rho(self) -> float:
        """`make_family_set`'s family-factor weight w: its variants correlate
        w + (1 - w) x across_corr within a family and across_corr between."""
        if self.within_corr >= 1.0:
            return 1.0
        return max((self.within_corr - self.across_corr) / (1.0 - self.across_corr), 1e-12)


def null_world(spec: PowerSpec, seed: int) -> FamilySet:
    """A batch with no real family, on business days from 1990."""
    fs = make_family_set(
        spec.n_families,
        spec.variants,
        0.0,
        0.0,
        within_rho=spec.within_rho,
        rho=spec.across_corr,
        n_search=spec.search_periods,
        n_holdout=spec.holdout_periods,
        seed=seed,
    )
    dates = _dates(len(fs.returns))
    return FamilySet(fs.returns.set_axis(dates), fs.family, fs.truth, fs.n_search, fs.n_holdout)


@functools.cache
def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(FIRST_DATE, periods=n)


def plant(returns: pd.DataFrame, family: pd.Series, target, sharpe: float) -> pd.DataFrame:
    """`returns` with `target`'s variants given annualised Sharpe `sharpe`.

    `make_family_set`'s planting: a real family's variants earn a mean of
    sharpe / sqrt(252) x their volatility on top of the same noise.
    """
    out = returns.copy()
    cols = family.index[family == target]
    out[cols] = out[cols] + sharpe / np.sqrt(PERIODS_PER_YEAR) * VOL
    return out


class NullSide(NamedTuple):
    """A null world's funnel inputs: its batch, the bootstrap/period arguments
    and every family's statistics."""

    fs: FamilySet
    kwargs: dict
    stats: pd.DataFrame


def null_side(spec: PowerSpec, seed: int) -> NullSide:
    """The funnel's statistics on `null_world(spec, seed)`, thresholds aside."""
    fs = null_world(spec, seed)
    dates = fs.returns.index
    ends = dict(search_end=dates[fs.n_search - 1], holdout_start=dates[fs.n_search])
    kwargs = dict(**ends, n_boot=spec.n_boot, seed=seed)
    null = funnel.family_statistics(fs.returns, fs.family, **kwargs)["reuse"]
    return NullSide(fs, kwargs, null)


def planted_side(side: NullSide, targets: Sequence, sharpe: float) -> pd.DataFrame:
    """The statistics of `side` with the `targets` families' variants planted
    at `sharpe`; only their rows are recomputed."""
    fs, kwargs, null = side
    targets = list(targets)
    cols = fs.family.index[fs.family.isin(targets)]
    planted = fs.returns[cols]
    for target in targets:
        planted = plant(planted, fs.family[cols], target, sharpe)
    rows = funnel.family_statistics(planted, fs.family[cols], **kwargs)["reuse"]
    stats = pd.concat([null.drop(index=targets), rows]).reindex(null.index)
    stats.attrs["n_variants"] = null.attrs["n_variants"]
    return stats
