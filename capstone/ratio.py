"""Ratio-based thresholds (Harvey & Liu): choose q1 and q4 by expected cost.

An opt-in alternative to the funnel's flat false-discovery target. Off by default:
`Thresholds.mode` is "fdr" and nothing registered changes. In "ratio" mode stage 1's
`q1` and stage 4's `q4` are the point of a registered grid that minimises

    expected cost per batch = E[false passes] x 1 + E[missed real families] x lambda

where lambda (`miss_cost`) is what missing a real family costs relative to
passing a false one (Harvey & Liu's ratio of type II to type I error costs)
and pi1 is the prior share of real families: each of the batch's N families
is real with probability pi1, independently, and a real family's variants
earn the **alternative Sharpe**. Then, with R ~ Binomial(N, pi1) real
families,

    E[false passes] = sum_r P(R = r) F_r(q1, q4), E[missed] = sum_r P(R = r) M_r(q1, q4),

F_r and M_r the mean number of null families accepted and of real families
missed in a batch with r real ones. They are simulated by `power`'s worlds,
i.e. the calibration's family generator and the funnel's own code
(`family_statistics`, `screen`, the stage-4 gate): no new statistics. In each
world every family is planted once, and the batch with r real families takes
the planted rows of r of them (a rotation starting at the world's target, so
r = 1 is exactly `power`'s world) and the null rows of the rest; a family's
statistics depend only on its own columns. One simulation serves every grid
point, r, lambda and pi1: the worlds are summarised once, each point is a
filter over them, and pi1 and lambda only weight the results. (Counting false
passes in all-null batches alone would understate them: BH lets more nulls
through when real families are present.)

**The grid is a chain**: (q1, q4) pairs from strict to loose, each at least
the previous in both coordinates. Then, as lambda grows, the chosen point
never moves back down the chain, so q1 and q4 are both non-decreasing in
lambda (and expected false passes non-decreasing, misses non-increasing).
This is exact, not up to Monte Carlo noise, because the choice reads the
expectations' envelopes along the chain (false passes' running maximum,
misses' running minimum: a looser point is credited with at least what a
stricter one passes) and ties go to the strictest point. With pi1 = 0 the
strictest point is chosen. A full (q1, q4) product grid would let them trade
off freely but loses that guarantee; it is left as an open question.

**Defaults.** `DEFAULT_GRID`: a 10-point staircase from (0.01, 0.05) to
(0.30, 0.70) through the registered FDR point (0.10, 0.30), loosening q1 and
q4 in turn. `DEFAULT_ALT_SHARPE` = 0.75: the end-to-end minimum detectable
Sharpe at 80% power for the CRSP shape is about 0.73, so 0.75 is the weakest
signal the registered funnel already finds about 4 times in 5; a stronger
alternative makes misses rare and the choice collapses to the strictest
point, a much weaker one (below 0.3) is one the 5-year holdout can't
confirm whatever the thresholds. `DEFAULT_N_SIMS` = 400 worlds (Monte Carlo
SE <= 0.025 on M_1), seed 0. lambda and pi1 have no default: they are the
team's call.

**Registration.** `calibrate(spec)` returns the `Thresholds` to register
(mode "ratio", the spec, the chosen q1 and q4); the spec holds lambda, pi1,
the seed, the grid, the alternative Sharpe, the simulation size and the
batch shape it was simulated for, and the gate config's hash covers all of
it. `verify` is the check to run before the look: the batch must have the
registered shape (family sizes, search and holdout days, n_boot) and
recomputing the calibration must give the registered q1 and q4, or it raises
`RatioMismatch`. Recomputing costs one simulation at the batch's size
(results are cached per process). Stage 3 (spanning) is not measured yet:
the funnel passes stage 2's survivors through it.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

from capstone import funnel, power
from capstone.funnel import Thresholds
from capstone.ratio_spec import (
    DEFAULT_ALT_SHARPE,
    DEFAULT_GRID,
    DEFAULT_N_SIMS,
    RatioSpec,
    check_thresholds,
)

__all__ = [
    "DEFAULT_ALT_SHARPE",
    "DEFAULT_GRID",
    "DEFAULT_N_SIMS",
    "RatioMismatch",
    "RatioSpec",
    "RatioTable",
    "calibrate",
    "check_thresholds",
    "simulate",
    "verify",
]


class RatioMismatch(ValueError):
    """A ratio-mode config doesn't match its batch or its recomputed calibration."""


@dataclass(frozen=True)
class RatioTable:
    """Simulated means per number of real families r = 0..N (rows) and grid
    point (columns): `false_passes`, null families accepted; `misses`, real
    families not accepted. Lambda and pi1 are applied by `expected`/`choose`."""

    grid: tuple[tuple[float, float], ...]
    false_passes: tuple[tuple[float, ...], ...]
    misses: tuple[tuple[float, ...], ...]
    n_families: int
    n_sims: int

    def expected(self, miss_cost: float, pi1: float, at: int | None = None):
        """(false passes, missed families, cost) per batch under the prior,
        from the envelopes along the chain; arrays over the grid, or scalars
        at index `at`."""
        weights = binom.pmf(np.arange(self.n_families + 1), self.n_families, pi1)
        false = np.maximum.accumulate(weights @ np.asarray(self.false_passes, dtype=float))
        missed = np.minimum.accumulate(weights @ np.asarray(self.misses, dtype=float))
        cost = false + miss_cost * missed
        if at is None:
            return false, missed, cost
        return float(false[at]), float(missed[at]), float(cost[at])

    def choose(self, miss_cost: float, pi1: float) -> int:
        """Index of the cheapest grid point; ties go to the strictest."""
        return int(np.argmin(self.expected(miss_cost, pi1)[2]))


def _base(th: Thresholds) -> Thresholds:
    """`th`'s other settings in FDR mode, with q1 and q4 left to the grid."""
    return dataclasses.replace(th, mode="fdr", ratio=None, q1=0.5, q4=0.5)


def simulate(spec: RatioSpec, base: Thresholds | None = None) -> RatioTable:
    """The rates at every grid point of `spec`, for `base`'s other settings
    (stage-1 method, alpha2, alpha3; `Thresholds()` by default). Deterministic
    for the spec; lambda and pi1 don't enter, so they share one simulation."""
    sim = dataclasses.replace(spec, miss_cost=1.0, pi1=0.0)
    return _simulate(sim, _base(Thresholds() if base is None else base))


@functools.cache
def _simulate(spec: RatioSpec, base: Thresholds) -> RatioTable:
    points = [dataclasses.replace(base, q1=a, q4=b) for a, b in spec.grid]
    pspec = spec.power_spec(base)
    n = len(spec.variants)
    seeds = np.random.default_rng(spec.seed).integers(0, 2**31 - 1, size=spec.n_sims)
    false = np.zeros((n + 1, len(points)))
    missed = np.zeros((n + 1, len(points)))
    for i, seed in enumerate(seeds):
        side = power.null_side(pspec, int(seed))
        families = side.stats.index
        order = np.roll(families, -(i % n))  # starts at the world's target
        planted = power.planted_side(side, families, spec.alt_sharpe)
        for r in range(n + 1):
            real = families.isin(order[:r])
            stats = pd.concat([side.stats[~real], planted[real]]).reindex(families)
            stats.attrs["n_variants"] = side.stats.attrs["n_variants"]
            for k, th in enumerate(points):
                accepted = funnel.run_funnel(stats, th)["stage4"].to_numpy(dtype=bool)
                false[r, k] += (accepted & ~real).sum()
                missed[r, k] += (~accepted & real).sum()
    return RatioTable(
        grid=spec.grid,
        false_passes=tuple(tuple(float(x) for x in row) for row in false / spec.n_sims),
        misses=tuple(tuple(float(x) for x in row) for row in missed / spec.n_sims),
        n_families=n,
        n_sims=spec.n_sims,
    )


def calibrate(spec: RatioSpec, base: Thresholds | None = None) -> Thresholds:
    """The ratio-mode `Thresholds` to register: `base`'s other settings, the
    spec, and the (q1, q4) grid point of least expected cost."""
    base = Thresholds() if base is None else base
    q1, q4 = spec.grid[simulate(spec, base).choose(spec.miss_cost, spec.pi1)]
    return dataclasses.replace(base, mode="ratio", ratio=spec, q1=q1, q4=q4)


def verify(
    thresholds: Thresholds,
    *,
    variants,
    search_periods: int,
    holdout_periods: int,
    n_boot: int,
) -> None:
    """The check before a look: nothing in FDR mode; in ratio mode the batch has the
    registered shape and recomputing gives the registered q1 and q4.

    `variants` is the batch's family sizes (any order). Correlations are not compared:
    the registered ones are what the calibration simulated.
    """
    if thresholds.mode == "fdr":
        return
    spec = thresholds.ratio
    actual = {
        "variants": tuple(sorted(int(n) for n in variants)),
        "search_periods": int(search_periods),
        "holdout_periods": int(holdout_periods),
        "n_boot": int(n_boot),
    }
    registered = {name: getattr(spec, name) for name in actual}
    registered["variants"] = tuple(sorted(spec.variants))
    differ = {k: (registered[k], actual[k]) for k in actual if registered[k] != actual[k]}
    if differ:
        raise RatioMismatch(
            f"the ratio thresholds were calibrated for another batch: {differ} "
            "(registered, this batch); calibrate and register for this one"
        )
    again = calibrate(spec, thresholds)
    if (again.q1, again.q4) != (thresholds.q1, thresholds.q4):
        raise RatioMismatch(
            f"registered (q1, q4) = ({thresholds.q1}, {thresholds.q4}) but the recomputed "
            f"calibration gives ({again.q1}, {again.q4}); register calibrate(spec)'s thresholds"
        )
