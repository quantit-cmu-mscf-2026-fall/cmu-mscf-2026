"""The ratio mode's registered spec, kept free of imports.

An opt-in alternative to the funnel's flat false-discovery target. The default,
`Thresholds(mode="fdr")`, is unchanged: stage 1 and stage 4 run at the registered
q1 and q4 and no ratio is read. In `mode="ratio"` the pair (q1, q4) is a point on a
registered grid, chosen by expected cost: false passes x 1 + missed real families x
`miss_cost`. Whether the team adopts it, and with what `miss_cost` and prior `pi1`,
is its decision; this module only defines and validates what would be registered.

`funnel.Thresholds` holds a `RatioSpec` and checks it here, at construction. This
module imports nothing from the package at load time so that `funnel` can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capstone.funnel import Thresholds

__all__ = ["DEFAULT_ALT_SHARPE", "DEFAULT_GRID", "DEFAULT_N_SIMS", "RatioSpec", "check_thresholds"]

DEFAULT_GRID: tuple[tuple[float, float], ...] = (
    (0.01, 0.05),
    (0.02, 0.10),
    (0.05, 0.10),
    (0.05, 0.20),
    (0.10, 0.20),
    (0.10, 0.30),  # the registered FDR thresholds
    (0.20, 0.30),
    (0.20, 0.50),
    (0.30, 0.50),
    (0.30, 0.70),
)
DEFAULT_ALT_SHARPE = 0.75
DEFAULT_N_SIMS = 400


@dataclass(frozen=True, kw_only=True)
class RatioSpec:
    """Everything the ratio calibration reads, registered with the thresholds.

    miss_cost: lambda, the cost of missing a real family relative to passing a
    false one (> 0). pi1: prior share of real families, in [0, 1).
    alt_sharpe: annualised true Sharpe of a real family's variants (net, after
    factors), the alternative misses are counted at. seed / n_sims: the
    simulation's seed and world count. grid: the candidate (q1, q4) chain.
    The batch shape is `power.PowerSpec`'s: variants per family, average
    correlation within and across families, search and holdout days, the
    bootstrap's n_boot.
    """

    miss_cost: float
    pi1: float
    variants: tuple[int, ...]
    within_corr: float
    across_corr: float
    search_periods: int
    holdout_periods: int
    n_boot: int
    alt_sharpe: float = DEFAULT_ALT_SHARPE
    seed: int = 0
    n_sims: int = DEFAULT_N_SIMS
    grid: tuple[tuple[float, float], ...] = DEFAULT_GRID

    def __post_init__(self) -> None:
        grid = tuple((float(a), float(b)) for a, b in self.grid)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "variants", tuple(int(v) for v in self.variants))
        if not self.miss_cost > 0.0:
            raise ValueError("miss_cost (lambda) must be > 0")
        if not 0.0 <= self.pi1 < 1.0:
            raise ValueError("pi1 must be in [0, 1)")
        if not self.alt_sharpe > 0.0:
            raise ValueError("alt_sharpe must be > 0")
        if self.n_sims < 1:
            raise ValueError("n_sims must be at least 1")
        if not grid:
            raise ValueError("grid must have at least one (q1, q4) pair")
        if not all(0.0 < q < 1.0 for pair in grid for q in pair):
            raise ValueError("every grid q1 and q4 must be in (0, 1)")
        for a, b in zip(grid, grid[1:], strict=False):
            if not (a[0] <= b[0] and a[1] <= b[1] and a != b):
                raise ValueError(
                    f"grid must be a chain from strict to loose: {a} then {b} "
                    "(each pair at least the previous in q1 and q4, and different)"
                )
        self.power_spec(None)  # the shape's own checks

    def power_spec(self, thresholds: Thresholds | None):
        """The shape as a `power.PowerSpec` with `thresholds` (default ones if None)."""
        from capstone.funnel import Thresholds  # imports this module
        from capstone.power import PowerSpec

        return PowerSpec(
            thresholds=Thresholds() if thresholds is None else thresholds,
            variants=self.variants,
            within_corr=self.within_corr,
            across_corr=self.across_corr,
            search_periods=self.search_periods,
            holdout_periods=self.holdout_periods,
            n_boot=self.n_boot,
        )


def check_thresholds(th) -> None:
    """`Thresholds.__post_init__`'s ratio checks; reads a dict spec back in."""
    if isinstance(th.ratio, dict):
        object.__setattr__(th, "ratio", RatioSpec(**th.ratio))
    if th.mode != "ratio":
        raise ValueError(f'a ratio spec is only read in mode "ratio", not "{th.mode}"')
    if not isinstance(th.ratio, RatioSpec):
        raise ValueError('mode "ratio" needs ratio=RatioSpec(...)')
    if th.final != "bh":
        raise ValueError('mode "ratio" calibrates q4, so stage 4 must be final="bh"')
    if (th.q1, th.q4) not in th.ratio.grid:
        raise ValueError(f"(q1, q4) = ({th.q1}, {th.q4}) is not on the ratio grid")
