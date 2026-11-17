"""Power analysis of the acceptance gate: simulated null and planted worlds.

For registered `Thresholds` and a batch's shape (families, variants per
family, their correlation, search and holdout lengths, the bootstrap's size)
the power analysis asks how strong a family must be before the gate accepts
it. This module builds the worlds that question is answered on, then reports
each stage's pass probability and **minimum detectable Sharpe** (annualised, at
50% and 80% power) and the funnel's end-to-end power curve. It adds no
statistics: every number comes from the funnel's own code
(`funnel.family_statistics`, `funnel.screen`, the stage-4 gate) on
`calibration.make_family_set` data.

**A world** is a null batch from the family generator (variants of one family
correlate `within_corr`, of different families `across_corr`; optionally
autocorrelated, fat-tailed and volatility-clustered, see `PowerSpec`). A target
family's variants are then given a true Sharpe on the search period and the
holdout (`plant`, the generator's own rule), and only the target's statistics
are recomputed. That is valid for the columns the gate's stages read (`best`,
`p1_*`, `p2`, `ho_*`): those, bootstrap included, depend only on the family's
own columns and the seed (tested). One null world serves every Sharpe, so the
curves are smooth. The naive stack's columns (`naive_pass`, `nv_*`) are not:
its Holm, BY and BH corrections and the DSR's trial count run over every
variant in the batch, so `planted_side` leaves them out.

**Each stage's curve** is the chance the target passes that stage's gate when
tested there: stage 1, BH at `q1` across all the families; stage 2, its
one-sided p <= `alpha2` (1.0, registered, is no gate); stage 3, not measured
yet (the funnel passes stage 2's survivors through; reported as no gate); stage
4, the stage-4 gate across the target and the null families that passed stages
1-3 in that world. Stages 2-4 test the target whatever happened before, so this
is the pass rate given that it got there. **End to end** is the chance the
target is accepted, every stage in order, as `funnel.run_funnel` does.

**At Sharpe 0** every family is null and each stage's `null_rate` is in the
unit of its level (`nominal`): stage 1, the chance any family passes (BH's
familywise rate under the complete null, q1); stage 2, the chance a family
passes; stage 4, the chance a lone null survivor passes (q4).
`end_to_end_null` is the chance anything is accepted.

**Minimum detectable Sharpe**: the first grid Sharpe where the curve's running
maximum reaches the power, interpolated linearly between grid points; None
when the grid never reaches it or the stage is no gate.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from capstone import funnel
from capstone.calibration import FamilySet, make_family_set
from capstone.funnel import Thresholds
from capstone.search_fdr import block_length_rule

__all__ = [
    "DEFAULT_GRID",
    "STAGES",
    "MinDetectableSharpe",
    "NullSide",
    "PowerBudget",
    "PowerReport",
    "PowerSpec",
    "StagePower",
    "accepted",
    "null_side",
    "null_world",
    "plant",
    "planted_side",
    "planted_sides",
    "power_analysis",
]

Stage = Literal["stage1", "stage2", "stage3", "stage4"]
STAGES: tuple[Stage, ...] = ("stage1", "stage2", "stage3", "stage4")
END_TO_END = "end_to_end"

PERIODS_PER_YEAR = 252
VOL = 0.01  # `synth.make_return_matrix`'s per-period volatility
FIRST_DATE = "1990-01-01"
DEFAULT_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(21))  # 0.0 to 2.0
_CHUNK_ELEMENTS = 2**19  # values per `planted_sides` funnel call


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


def _checked_targets(targets: Sequence, families: pd.Index) -> list:
    targets = list(targets)
    duplicated = sorted({t for t in targets if targets.count(t) > 1}, key=str)
    if duplicated:
        # Planting twice would add the shift twice (Sharpe 1 x 2 = one plant at 2).
        raise ValueError(f"duplicate plant targets {duplicated}: list each family once")
    unknown = [t for t in targets if t not in families]
    if unknown:
        raise ValueError(f"unknown plant targets {unknown}: not families of this batch")
    return targets


def _drop_naive(stats: pd.DataFrame) -> pd.DataFrame:
    return stats.drop(columns=[c for c in stats if c == "naive_pass" or c.startswith("nv_")])


def planted_side(side: NullSide, targets: Sequence, sharpe: float) -> pd.DataFrame:
    """The statistics of `side` with the `targets` families' variants planted
    at `sharpe`; only their rows are recomputed.

    The naive stack's columns (`naive_pass`, `nv_*`) are dropped: they depend
    on the whole batch, so the targets' rows alone would get them wrong. For
    them, run `funnel.family_statistics` on the whole planted batch.
    """
    fs, kwargs, null = side
    targets = _checked_targets(targets, null.index)
    cols = fs.family.index[fs.family.isin(targets)]
    planted = fs.returns[cols]
    for target in targets:
        planted = plant(planted, fs.family[cols], target, sharpe)
    rows = funnel.family_statistics(planted, fs.family[cols], **kwargs)["reuse"]
    stats = pd.concat([null.drop(index=targets), rows]).reindex(null.index)
    stats = _drop_naive(stats)
    stats.attrs["n_variants"] = null.attrs["n_variants"]
    return stats


def planted_sides(side: NullSide, targets: Sequence, sharpes: Sequence[float]) -> list:
    """`[planted_side(side, targets, s) for s in sharpes]` in a few funnel calls.

    Each Sharpe's planted copy of the targets enters one frame as its own
    families. A family's statistics, bootstrap included, depend only on its own
    columns and the seed, so every column the stages read equals
    `planted_side`'s (tested). At most `_CHUNK_ELEMENTS` values per call; the
    bootstrap's resolution check stays per Sharpe (m Sharpes in a call pass q x m).
    """
    fs, kwargs, null = side
    targets = _checked_targets(targets, null.index)
    sharpes = list(sharpes)
    cols = fs.family.index[fs.family.isin(targets)]
    which = pd.Index(targets).get_indexer(fs.family[cols])  # target position per column
    base, n_t, q = fs.returns[cols].to_numpy(), len(targets), kwargs["q"]
    per = max(1, min(_CHUNK_ELEMENTS // base.size, int(np.ceil(1.0 / q)) - 1))
    out = []
    for lo in range(0, len(sharpes), per):
        chunk = sharpes[lo : lo + per]
        # `plant`'s shift, column for column: one addition per planted value.
        values = np.hstack([base + s / np.sqrt(PERIODS_PER_YEAR) * VOL for s in chunk])
        family = pd.Series(np.concatenate([j * n_t + which for j in range(len(chunk))]))
        frame = pd.DataFrame(values, index=fs.returns.index)
        rows = funnel.family_statistics(frame, family, **kwargs | {"q": q * len(chunk)})["reuse"]
        for j in range(len(chunk)):
            mine = rows.loc[j * n_t + np.arange(n_t)].set_axis(pd.Index(targets))
            # `best` is a position in the tiled frame: back to the variant's own column.
            mine["best"] = [cols[int(b) % len(cols)] if pd.notna(b) else b for b in mine["best"]]
            stats = _drop_naive(pd.concat([null.drop(index=targets), mine]).reindex(null.index))
            stats.attrs["n_variants"] = null.attrs["n_variants"]
            out.append(stats)
    return out


@dataclass(frozen=True, kw_only=True)
class PowerBudget:
    """How much to simulate: `n_sims` worlds, each scored at every Sharpe of
    `sharpe_grid` (annualised, increasing, >= 0), from `seed`."""

    n_sims: int = 200
    sharpe_grid: tuple[float, ...] = DEFAULT_GRID
    seed: int = 0

    def __post_init__(self) -> None:
        grid = tuple(float(s) for s in self.sharpe_grid)
        object.__setattr__(self, "sharpe_grid", grid)
        if self.n_sims < 1:
            raise ValueError("n_sims must be at least 1")
        if not grid or not np.isfinite(grid).all() or grid[0] < 0 or (np.diff(grid) <= 0).any():
            raise ValueError("sharpe_grid must be non-empty, finite, >= 0 and increasing")


@dataclass(frozen=True)
class MinDetectableSharpe:
    """A stage's minimum detectable Sharpe (annualised) at 50% and 80% power.

    `gated` False: the stage passes everything (both None). A gated stage's
    None: the power isn't reached on the simulated grid.
    """

    stage: str
    gated: bool
    power_50: float | None
    power_80: float | None


@dataclass(frozen=True)
class StagePower:
    """One stage: its level, its error rate at Sharpe 0 and its power curve."""

    stage: str
    gated: bool
    nominal: float
    null_rate: float
    power: tuple[float, ...]
    mds50: float | None
    mds80: float | None


@dataclass(frozen=True)
class PowerReport:
    spec: PowerSpec
    budget: PowerBudget
    stages: tuple[StagePower, ...]
    end_to_end: tuple[float, ...]
    end_to_end_null: float
    end_to_end_mds50: float | None
    end_to_end_mds80: float | None

    @property
    def sharpe_grid(self) -> tuple[float, ...]:
        return self.budget.sharpe_grid

    @property
    def end_to_end_stage(self) -> StagePower:
        """End to end as a `StagePower` (no single nominal level: NaN)."""
        e = (self.end_to_end_null, self.end_to_end, self.end_to_end_mds50, self.end_to_end_mds80)
        return StagePower(END_TO_END, True, float("nan"), *e)

    def min_detectable_sharpe(self) -> tuple[MinDetectableSharpe, ...]:
        return tuple(
            MinDetectableSharpe(s.stage, s.gated, s.mds50, s.mds80)
            for s in (*self.stages, self.end_to_end_stage)
        )


def accepted(stats: pd.DataFrame, th: Thresholds) -> pd.Series:
    """Per family, accepted end to end: the funnel's own `run_funnel` stage 4."""
    return funnel.run_funnel(stats, th)["stage4"]


def _world(spec: PowerSpec, grid: tuple[float, ...], i: int, seed: int):
    """One world: (null rate per stage, end-to-end null, power per stage x
    grid, end-to-end per grid point). The target cycles through the families
    with `i`, so the curve averages over their sizes."""
    th = spec.thresholds
    side = null_side(spec, seed)
    null = side.stats
    stages0 = funnel.screen(null, th)
    lone = [
        bool(funnel._final_gate(null, pd.Series(null.index == f, index=null.index), th)[f])
        for f in null.index
    ]
    null_rates = [stages0["stage1"].any(), (null["p2"] <= th.alpha2).mean(), 1.0, np.mean(lone)]
    end_null = float(accepted(null, th).any())

    target = null.index[i % len(null.index)]
    passes = np.zeros((len(STAGES), len(grid)))
    hits = np.zeros(len(grid))
    for j, stats in enumerate(planted_sides(side, [target], grid)):
        stages = funnel.screen(stats, th)
        forced = stages["stage3"].copy()
        forced[target] = True  # stage 4 tested whatever happened before
        passes[:, j] = [
            stages.loc[target, "stage1"],
            stats.loc[target, "p2"] <= th.alpha2,
            True,
            funnel._final_gate(stats, forced, th)[target],
        ]
        hits[j] = accepted(stats, th)[target]
    return null_rates, end_null, passes, hits


def _crossing(grid: tuple[float, ...], curve: np.ndarray, level: float) -> float | None:
    """First Sharpe where the curve's running maximum reaches `level`."""
    env = np.maximum.accumulate(curve)
    hit = np.flatnonzero(env >= level)
    if not len(hit):
        return None
    j = int(hit[0])
    if j == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[j - 1], grid[j], env[j - 1], env[j]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def power_analysis(spec: PowerSpec, budget: PowerBudget | None = None) -> PowerReport:
    """Simulate `budget.n_sims` worlds of `spec`; deterministic for `budget.seed`."""
    budget = PowerBudget() if budget is None else budget
    grid = budget.sharpe_grid
    seeds = np.random.default_rng(budget.seed).integers(0, 2**31 - 1, size=budget.n_sims)
    worlds = [_world(spec, grid, i, int(s)) for i, s in enumerate(seeds)]
    null_rates = np.mean([np.asarray(w[0], dtype=float) for w in worlds], axis=0)
    passes = np.mean([w[2] for w in worlds], axis=0)
    hits = np.mean([w[3] for w in worlds], axis=0)
    th = spec.thresholds
    nominal = (th.q1, th.alpha2, 1.0, th.q4 if th.final == "bh" else 1.0 - th.d4)
    gated = (True, th.alpha2 < 1.0, False, True)  # stage 3: not measured yet
    stages = tuple(
        StagePower(
            stage=name,
            gated=gated[s],
            nominal=float(nominal[s]),
            null_rate=float(null_rates[s]),
            power=tuple(float(x) for x in passes[s]),
            mds50=_crossing(grid, passes[s], 0.5) if gated[s] else None,
            mds80=_crossing(grid, passes[s], 0.8) if gated[s] else None,
        )
        for s, name in enumerate(STAGES)
    )
    return PowerReport(
        spec=spec,
        budget=budget,
        stages=stages,
        end_to_end=tuple(float(x) for x in hits),
        end_to_end_null=float(np.mean([w[1] for w in worlds])),
        end_to_end_mds50=_crossing(grid, hits, 0.5),
        end_to_end_mds80=_crossing(grid, hits, 0.8),
    )
