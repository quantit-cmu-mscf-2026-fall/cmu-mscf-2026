"""Power analysis of the acceptance gate: simulated null and planted worlds.

For registered `Thresholds` and a batch's shape (families, variants per
family, their correlation, search and holdout lengths, the bootstrap's size)
the power analysis asks how strong a family must be before the gate accepts
it. This module builds the worlds that question is answered on. It adds no
statistics: every number comes from the funnel's own code
(`funnel.family_statistics`) on `calibration.make_family_set` data.

**A world** is a null batch from the family generator (variants of one family
correlate `within_corr`, of different families `across_corr`; optionally
autocorrelated, fat-tailed and volatility-clustered, see `PowerSpec`). A target
family's variants are then given a true Sharpe on the search period and the
holdout (`plant`, the generator's own rule), and only the target's statistics
are recomputed. That is valid for the columns the gate's stages read (`best`,
`p1_*`, `p2`, `ho_*`): those, bootstrap included, depend only on the family's
own columns and the seed (tested). One null world serves every Sharpe. The
naive stack's columns (`naive_pass`, `nv_*`) are not: its Holm, BY and BH
corrections and the DSR's trial count run over every variant in the batch, so
`planted_side` leaves them out.
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
    (`from_years`). n_boot: bootstrap draws (0 skips the bootstrap; allowed
    only when stage 1 is not the bootstrap). n_trials: the ledger's
    count, recorded only: no registered decision reads it.

    Synthetic defects (`synth.make_return_matrix`'s, same meaning and ranges):
    ar1 is every series' AR(1) coefficient, in (-1, 1); t_df gives Student-t
    innovations with that many degrees of freedom (> 2; None is normal); garch
    is a GARCH(1,1) `(alpha, beta)` variance process (alpha, beta >= 0,
    alpha + beta < 1; None is constant variance). Each keeps every series'
    mean and standard deviation, so a null world stays null and `plant` shifts
    the mean by the stated Sharpe whatever is switched on. The defaults (none)
    are the cross-correlated Gaussian world. Power measured only on that world
    flatters the gate: iid-over-time data is where HAC and bootstrap p-values
    look best.
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
    ar1: float = 0.0
    t_df: float | None = None
    garch: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        v = self.variants
        if np.ndim(v) == 0:
            if self.n_families is None:
                raise ValueError("one variant count for all families needs n_families")
            v = (v,) * int(self.n_families)
        v = tuple(int(x) for x in v)
        if not v or min(v) < 2:
            raise ValueError(f"every family needs at least 2 variants (trials), got {v}")
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
        # BH at level q across H families needs n_boot + 1 >= H / q. The funnel
        # runs the bootstrap whenever n_boot > 0, whatever the stage-1 method.
        bootstrap = self.thresholds.stage1 == "bootstrap" or self.n_boot > 0
        if self.n_boot < 0 or (bootstrap and self.n_boot + 1 < len(v) / self.bootstrap_q):
            raise ValueError(
                f"n_boot + 1 must be >= {len(v)} families / q = {self.bootstrap_q} "
                "for the bootstrap"
            )
        if self.n_trials is not None and self.n_trials < sum(v):
            raise ValueError(f"n_trials={self.n_trials} is fewer than the {sum(v)} variants")
        # The generator's own ranges, refused here rather than at the first world.
        if not -1.0 < self.ar1 < 1.0:
            raise ValueError(f"ar1 must be in (-1, 1), got {self.ar1}")
        if self.t_df is not None and not self.t_df > 2:
            raise ValueError(f"t_df must exceed 2, or the variance is infinite, got {self.t_df}")
        if self.garch is not None:
            garch = tuple(float(x) for x in self.garch)
            if len(garch) != 2 or min(garch) < 0 or sum(garch) >= 1:
                raise ValueError(f"garch needs (alpha, beta) >= 0 with alpha + beta < 1: {garch}")
            object.__setattr__(self, "garch", garch)

    @classmethod
    def from_years(cls, *, search_years: float, holdout_years: float, **kwargs) -> PowerSpec:
        return cls(
            search_periods=round(search_years * PERIODS_PER_YEAR),
            holdout_periods=round(holdout_years * PERIODS_PER_YEAR),
            **kwargs,
        )

    @property
    def bootstrap_q(self) -> float:
        """The BH level the bootstrap must resolve: the registered q1 for a
        bootstrap stage 1 (never looser than the funnel's default 0.05)."""
        if self.thresholds.stage1 == "bootstrap":
            return min(0.05, self.thresholds.q1)
        return 0.05

    @property
    def defects(self) -> dict:
        """The synthetic defects as `make_family_set` / `make_return_matrix`
        keyword arguments (cross-correlation, AR(1), fat tails, GARCH)."""
        return dict(rho=self.across_corr, ar1=self.ar1, t_df=self.t_df, garch=self.garch)

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
        **spec.defects,
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
    sharpe / sqrt(252) x their volatility on top of the same noise. A target
    that names no family is refused: it would plant nothing.
    """
    out = returns.copy()
    cols = family.index[family == target]
    if len(cols) == 0:
        raise ValueError(f"unknown plant target {target!r}: not a family of this batch")
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
    kwargs = dict(**ends, n_boot=spec.n_boot, seed=seed, q=spec.bootstrap_q)
    null = funnel.family_statistics(fs.returns, fs.family, **kwargs)["reuse"]
    return NullSide(fs, kwargs, null)


def planted_side(side: NullSide, targets: Sequence, sharpe: float) -> pd.DataFrame:
    """The statistics of `side` with the `targets` families' variants planted
    at `sharpe`; only their rows are recomputed.

    The naive stack's columns (`naive_pass`, `nv_*`) are dropped: they depend
    on the whole batch, so the targets' rows alone would get them wrong. For
    them, run `funnel.family_statistics` on the whole planted batch.
    """
    fs, kwargs, null = side
    targets = list(targets)
    duplicated = sorted({t for t in targets if targets.count(t) > 1}, key=str)
    if duplicated:
        # Planting twice would add the shift twice (Sharpe 1 x 2 = one plant at 2).
        raise ValueError(f"duplicate plant targets {duplicated}: list each family once")
    unknown = [t for t in targets if t not in null.index]
    if unknown:
        raise ValueError(f"unknown plant targets {unknown}: not families of this batch")
    cols = fs.family.index[fs.family.isin(targets)]
    planted = fs.returns[cols]
    for target in targets:
        planted = plant(planted, fs.family[cols], target, sharpe)
    rows = funnel.family_statistics(planted, fs.family[cols], **kwargs)["reuse"]
    stats = pd.concat([null.drop(index=targets), rows]).reindex(null.index)
    stats = stats.drop(columns=[c for c in stats if c == "naive_pass" or c.startswith("nv_")])
    stats.attrs["n_variants"] = null.attrs["n_variants"]
    return stats
