"""Validation's intake: what any candidate generator hands the funnel.

A generator (factor trees, Alpha-GPT, a hand-written strategy grid) produces
net returns, dates x candidates, and a hypothesis family per candidate.
`FunnelInput` is that hand-off, checked once at the boundary:

- the index is increasing, unique dates;
- every candidate has a family label, and every label names a candidate;
- values are finite or missing, and every candidate has some returns;
- rows where no candidate has a return (an untraded first day, a warm-up
  every candidate shares) are dropped, the same way for every generator.

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
    if not isinstance(returns, pd.DataFrame) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a DataFrame indexed by date")
    if not isinstance(family, pd.Series):
        raise TypeError("family must be a Series: candidate -> hypothesis")
    if not (returns.index.is_monotonic_increasing and returns.index.is_unique):
        raise ValueError("returns' dates must be strictly increasing")
    if returns.columns.empty or returns.empty:
        raise ValueError("returns has no candidates or no dates")
    if not returns.columns.is_unique:
        raise ValueError("candidate ids (returns' columns) must be unique")
    if not family.index.is_unique:
        raise ValueError("family has a candidate more than once")
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


def from_returns(
    returns: pd.DataFrame, family: pd.Series, *, generator: str, source: dict | None = None
) -> FunnelInput:
    """A `FunnelInput` from any generator's returns and family labels.

    Rows where every candidate is missing are dropped; `family` is put in
    `returns`' column order. `source` is extra provenance (e.g. the
    `sd.data_version()` the returns were built on), kept alongside the
    candidate ids and the number of rows dropped.
    """
    _check(returns, family, generator)
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
) -> pd.DataFrame:
    """The funnel's per-family statistics for `fi`, periods cut by date.

    Stages 1-3 (`best`, `p1_*`, `p2`) see dates on or before `search_end`
    only; the `ho_*` / `nv_*` columns come from dates on or after
    `holdout_start`. Without `holdout_start`, returns reaching past
    `search_end` raise, so the holdout is never read by accident. Returns
    the "reuse" design, the frame the funnel's stages filter.
    """
    return funnel.family_statistics(
        fi.returns,
        fi.family,
        search_end=search_end,
        holdout_start=holdout_start,
        n_boot=n_boot,
        seed=seed,
    )["reuse"]
