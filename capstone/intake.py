"""Validation's intake: what any candidate generator hands the funnel.

A generator (factor trees, Alpha-GPT, a hand-written strategy grid) produces
net returns, dates x candidates, and a hypothesis family per candidate.
`FunnelInput` is that hand-off, checked once at the boundary:

- the index is tz-naive, normalised, strictly increasing dates
  (`funnel.check_dates`, the contract the funnel applies);
- every candidate has a family label, and every label names a candidate;
- values are finite or missing, and every candidate has some returns;
- given the period dates, every candidate has enough returns in each period
  (`check_periods`, the same rule the funnel tests with), so a frame the
  funnel would refuse is refused here, naming the candidates;
- rows where no candidate has a return (an untraded first day, a warm-up
  every candidate shares) are dropped by `from_returns`, the same way for
  every generator.

`stage_inputs` then computes the funnel's per-family statistics
(`funnel.family_statistics`) with periods cut by date: stages 1-3 see
dates on or before `search_end` only, and the holdout, from `holdout_start`,
is read for stage 4's statistics alone.

This module takes returns, never strategies: it imports no generator code,
so validation never calls strategy code (`CLAUDE.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone import funnel


@dataclass(frozen=True, eq=False)
class FunnelInput:
    """Candidates' returns and families, as the funnel takes them.

    returns: dates x candidate, net returns (NaN where a candidate holds
        nothing). family: hypothesis label per candidate, in `returns`' column
        order. generator: what produced the candidates ("factor-trees",
        "alpha-gpt", ...). source: provenance (data version, candidate ids,
        rows dropped) for the ledger and the scorecard.
    """

    returns: pd.DataFrame
    family: pd.Series
    generator: str
    source: dict

    def __post_init__(self) -> None:
        _check(self.returns, self.family, self.generator)


def _check(returns: pd.DataFrame, family: pd.Series, generator: str) -> None:
    if not isinstance(generator, str) or not generator.strip():
        raise ValueError("generator must name what produced the candidates")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame indexed by date")
    funnel.check_dates(returns.index)
    if not isinstance(family, pd.Series):
        raise TypeError("family must be a Series: candidate -> hypothesis")
    if returns.columns.empty or returns.empty:
        raise ValueError("returns has no candidates or no dates")
    if not returns.columns.is_unique:
        raise ValueError("candidate ids (returns' columns) must be unique")
    if not family.index.is_unique:
        raise ValueError("family has a candidate more than once")
    blank = [c for c, h in family.items() if isinstance(h, str) and not h.strip()]
    if blank:
        raise ValueError(f"blank family label for candidates {blank!r}")
    unlabelled = returns.columns.difference(family.dropna().index)
    if len(unlabelled):
        raise ValueError(f"no family for {len(unlabelled)} candidates: {list(unlabelled)!r}")
    extra = family.index.difference(returns.columns)
    if len(extra):
        raise ValueError(f"family labels {len(extra)} unknown candidates: {list(extra)!r}")
    if np.isinf(returns.to_numpy(dtype=np.float64)).any():
        raise ValueError("returns must be finite (NaN for missing); found +/-inf")
    empty = returns.columns[returns.isna().all()]
    if len(empty):
        raise ValueError(f"no returns at all for {list(empty)!r}")


def check_periods(returns: pd.DataFrame, *, search_end, holdout_start=None) -> None:
    """Refuse candidates with too few returns in a period the funnel tests.

    The search period is the dates on or before `search_end`; the holdout,
    when `holdout_start` is given, the dates from then on. Every candidate
    needs `funnel.MIN_PERIOD_OBS` non-NaN returns in each (a candidate that
    starts after the search, or stops before the holdout, cannot be scored),
    and the error names them. The cuts' own order and bounds stay with
    `funnel.family_statistics`.
    """
    funnel.check_dates(returns.index)
    periods = {"search period": returns.loc[returns.index <= pd.Timestamp(search_end)]}
    if holdout_start is not None:
        periods["holdout"] = returns.loc[returns.index >= pd.Timestamp(holdout_start)]
    short = funnel.period_shortfalls(periods)
    if short:
        detail = "; ".join(f"{name}: {sorted(counts, key=str)!r}" for name, counts in short.items())
        raise ValueError(
            f"fewer than {funnel.MIN_PERIOD_OBS} non-NaN returns for candidates in the {detail}"
        )


def from_returns(
    returns: pd.DataFrame,
    family: pd.Series,
    *,
    generator: str,
    source: dict | None = None,
    search_end=None,
    holdout_start=None,
) -> FunnelInput:
    """A `FunnelInput` from any generator's returns and family labels.

    Rows where every candidate is missing are dropped; `family` is put in
    `returns`' column order. `source` is extra provenance (e.g. the
    `sd.data_version()` the returns were built on), kept alongside the
    candidate ids and the number of rows dropped. With `search_end` (and
    `holdout_start`) the periods are checked too (`check_periods`);
    `stage_inputs` checks them regardless.
    """
    _check(returns, family, generator)
    if search_end is not None:
        check_periods(returns, search_end=search_end, holdout_start=holdout_start)
    traded = returns.notna().any(axis=1)
    kept = returns.loc[traded]
    labels = family.reindex(kept.columns)
    return FunnelInput(
        returns=kept,
        family=labels,
        generator=generator,
        source={
            **(source or {}),
            "variant_ids": list(kept.columns),
            "n_dropped_rows": int((~traded).sum()),
            "start": str(kept.index[0].date()),
            "end": str(kept.index[-1].date()),
        },
    )


def stage_inputs(
    fi: FunnelInput,
    *,
    search_end,
    holdout_start=None,
    n_boot: int,
    seed: int,
    q: float,
) -> pd.DataFrame:
    """The funnel's per-family statistics for `fi`, periods cut by date.

    Stages 1-3 (`best`, `p1_*`, `p2`) see dates on or before `search_end`
    only; the `ho_*` / `nv_*` columns come from dates on or after
    `holdout_start`. Without `holdout_start`, returns reaching past
    `search_end` raise, so the holdout is never read by accident. `q` is the
    registered stage-1 BH level: `n_boot` too small to resolve it raises
    rather than leaving stage 1 without power. Returns the "reuse" design,
    the frame the funnel's stages filter. A candidate with too few returns in
    a period raises, naming it (`check_periods`).
    """
    check_periods(fi.returns, search_end=search_end, holdout_start=holdout_start)
    return funnel.family_statistics(
        fi.returns,
        fi.family,
        search_end=search_end,
        holdout_start=holdout_start,
        n_boot=n_boot,
        seed=seed,
        q=q,
    )["reuse"]
