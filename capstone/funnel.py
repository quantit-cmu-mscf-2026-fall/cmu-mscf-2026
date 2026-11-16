"""The validation funnel's per-family statistics, with periods cut by date.

`docs/validation/framework.md` gives each stage one gate. Stages 1-3 read the
search period; stage 4 reads the holdout, once. `capstone.calibration` cut
those periods by row position on simulated families. This module cuts them
by date on any performance matrix (dates x variants, one hypothesis label per
variant), so the acceptance gate can run the same code on real candidates:

- `family_statistics` takes the last search date and, separately, the first
  holdout date. A frame that reaches past `search_end` without an explicit
  `holdout_start` is refused, so the holdout can't be read by accident, and
  stages 1-3 never see a holdout date.
- `holdout_statistics` is stage 4's input on its own: the holdout statistics
  of each family's chosen variant, reading only those columns.

`capstone.calibration.family_statistics` maps its simulated periods to dates
and calls this, so calibration and the gate share one code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capstone.evaluate import (
    average_correlation,
    bh_adjusted,
    by_adjusted,
    deflated_sharpe_ratio,
    holm_adjusted,
    sharpe_test,
)
from capstone.search_fdr import bootstrap_family_pvalues, search_adjusted_pvalue

# The holdout statistics stage 4 reads, per carried-forward variant.
_HOLDOUT_FIELDS = ("p", "sharpe", "n_obs", "skew", "kurtosis", "variance")

# The fewest non-NaN returns `sharpe_test` can work with; every period a
# variant is tested on needs this many (see `period_shortfalls`).
MIN_PERIOD_OBS = 2


def check_dates(index: pd.Index) -> None:
    """Refuse a date index the funnel's date cuts cannot read.

    The periods are cut by calendar date, so the index must be a
    DatetimeIndex that is timezone-naive, normalised (midnight: a 16:00
    stamp on `search_end` would fall in the gap between the periods) and
    strictly increasing (a repeated date would count one day twice).
    `family_statistics` applies this; the validation intake applies the same
    check so it never accepts a frame the funnel then refuses.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("returns must be indexed by date (a DatetimeIndex)")
    if index.tz is not None:
        raise ValueError("returns' dates must be timezone-naive")
    if not index.equals(index.normalize()):
        raise ValueError("returns' dates must be normalised to midnight")
    if not index.is_monotonic_increasing:
        raise ValueError("returns' dates must be increasing")
    if not index.is_unique:
        raise ValueError(
            f"returns' dates must be unique (strictly increasing), "
            f"e.g. {index[index.duplicated()][0].date()} repeats"
        )


def period_shortfalls(periods: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Per period, the columns with too few non-NaN returns to be tested.

    `periods` maps a period's name to its frame. Returns {name: {column:
    non-NaN count}}, only for columns under `MIN_PERIOD_OBS`.
    """
    out = {}
    for name, frame in periods.items():
        counts = frame.count()
        short = counts[counts < MIN_PERIOD_OBS]
        if len(short):
            out[name] = short.to_dict()
    return out


def _tested(returns: pd.Series, period: str):
    """`sharpe_test`, refusing (by name) a variant with too few returns."""
    n = int(returns.count())
    if n < MIN_PERIOD_OBS:
        raise ValueError(
            f"variant {returns.name!r} has {n} non-NaN returns in the {period}; "
            f"need at least {MIN_PERIOD_OBS}"
        )
    return sharpe_test(returns)


def best_by_family(values: pd.Series, family: pd.Series) -> pd.Series:
    """Per family, the label of its variant with the largest value.

    Indexed by every family in `family`, in order of first appearance; a
    family whose values are all missing gets NaN. (Grouped idxmax over an
    all-NaN group warns on pandas 2.2 and raises on pandas 3, #68, so only
    present values are grouped.)
    """
    labels = family.reindex(values.index)
    present = values.dropna()
    best = present.groupby(labels.reindex(present.index)).idxmax()
    return best.reindex(pd.unique(family.dropna()))


def _one_sided_p(returns: pd.DataFrame, period: str) -> pd.Series:
    return pd.Series({c: _tested(returns[c], period).pvalue_greater for c in returns.columns})


def _stage1(
    window: pd.DataFrame,
    p: pd.Series,
    family: pd.Series,
    n_boot: int,
    seed: int | None,
    q: float = 0.05,
) -> pd.DataFrame:
    """Each family's best variant on `window` and its family p-value, three ways.

    `p` is each variant's one-sided HAC p-value on `window`. `n_boot=0` skips
    the bootstrap (its column is then NaN), which is most of the cost.
    """
    best = best_by_family(window.mean() / window.std(), family)
    k = family.value_counts()
    rho = pd.Series(
        {h: max(average_correlation(window[cols.index]), 0.0) for h, cols in family.groupby(family)}
    )
    best_p = p.groupby(family).min()
    return pd.DataFrame(
        {
            "best": best,
            "p1_independent": search_adjusted_pvalue(best_p, k),
            "p1_rho": search_adjusted_pvalue(best_p, k, rho=rho),
            "p1_bootstrap": (
                bootstrap_family_pvalues(window, family, n_boot=n_boot, seed=seed, q=q)["pvalue"]
                if n_boot
                else np.nan
            ),
        }
    )


def _holdout_fields(tests: dict, chosen: pd.Series, prefix: str) -> dict[str, list]:
    """Holdout statistics of each family's chosen variant; NaN where none."""

    def value(b, field: str) -> float:
        if pd.isna(b) or b not in tests:
            return np.nan
        return tests[b].pvalue_greater if field == "p" else getattr(tests[b], field)

    return {f"{prefix}{field}": [value(b, field) for b in chosen] for field in _HOLDOUT_FIELDS}


def holdout_statistics(holdout: pd.DataFrame, chosen: pd.Series) -> pd.DataFrame:
    """Stage 4's input: the holdout statistics of each family's chosen variant.

    Reads only the columns `chosen` names (NaN where a family has none), so a
    caller can hand it the holdout without exposing anything else. Columns
    are the `ho_*` fields `family_statistics` returns, indexed like `chosen`.
    """
    picked = chosen.dropna().unique()
    tests = {b: _tested(holdout[b], "holdout") for b in picked}
    return pd.DataFrame(_holdout_fields(tests, chosen, "ho_"), index=chosen.index)


def family_statistics(
    returns: pd.DataFrame,
    family: pd.Series,
    *,
    search_end,
    holdout_start=None,
    stage2_start=None,
    n_boot: int = 1999,
    seed: int = 0,
    q: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Everything the funnel needs, per hypothesis, with periods cut by date.

    Args:
        returns: dates x variants (a DatetimeIndex, increasing).
        family: hypothesis label per column of `returns`; every column needs
            one.
        search_end: last date of the search period (agents and stages 1-3).
        holdout_start: first holdout date (stage 4 only). Required when
            `returns` has any date after `search_end`; without it there are
            no holdout statistics (the `ho_*` and `nv_*` columns are NaN).
        stage2_start: if given, also return the "split" design: stage 1 sees
            the search period before this date and stage 2 confirms on the
            rest of it.
        n_boot: bootstrap replications for `p1_bootstrap` (0 skips it).
        seed: bootstrap seed.
        q: the smallest stage-1 BH level the caller will use (it sets how
            many bootstrap draws are enough).

    Returns {"reuse": ...} and, with `stage2_start`, {"split": ...}. In
    "reuse" stages 1 and 2 both see the whole search period. Columns: `best`
    (the variant carried forward), `p1_<method>` for each stage-1 method,
    `p2` (one-sided HAC p of `best` at stage 2), the holdout `ho_*` fields
    stage 4 needs, the naive stack's `nv_*` fields and `naive_pass` (some
    variant passed every stacked screening gate on the whole search period).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame indexed by date")
    check_dates(returns.index)
    # Every variant counts toward its family's k, so each needs a label. The
    # bootstrap checks this too, but n_boot=0 skips it.
    missing = returns.columns.difference(family.dropna().index)
    if len(missing):
        raise ValueError(f"family is missing for {len(missing)} columns, e.g. {missing[0]!r}")
    search_end = pd.Timestamp(search_end)
    last_date = returns.index.max()
    if holdout_start is None:
        if last_date > search_end:
            raise ValueError(
                f"returns run to {last_date.date()}, past search_end {search_end.date()}: "
                "pass holdout_start explicitly to use the later dates as the holdout"
            )
        holdout = None
    else:
        holdout_start = pd.Timestamp(holdout_start)
        if holdout_start <= search_end:
            raise ValueError("holdout_start must be after search_end")
        holdout = returns.loc[returns.index >= holdout_start]
        if holdout.empty:
            raise ValueError(f"no dates on or after holdout_start {holdout_start.date()}")
    search = returns.loc[returns.index <= search_end]
    if search.empty:
        raise ValueError(f"no dates on or before search_end {search_end.date()}")

    # Naive stack: every screening method as a gate, at once, on every variant
    # over the whole search period.
    full_tests = {c: _tested(search[c], "search period") for c in search.columns}
    p_all = pd.Series({c: t.pvalue_greater for c, t in full_tests.items()})
    n_variants = len(p_all)
    dsr_all = pd.Series(
        {
            c: deflated_sharpe_ratio(
                t.sharpe, n_variants, t.n_obs, t.skew, t.kurtosis, variance=t.variance
            )
            for c, t in full_tests.items()
        }
    )
    psr_all = pd.Series({c: t.psr for c, t in full_tests.items()})
    stacked = (
        (holm_adjusted(p_all) <= 0.05)
        & (by_adjusted(p_all) <= 0.05)
        & (bh_adjusted(p_all) <= 0.05)
        & (psr_all >= 0.95)
        & (dsr_all >= 0.95)
    )
    # The variant the naive stack carries forward: its best that passed.
    full_sharpe = search.mean() / search.std()
    naive_best = best_by_family(full_sharpe[stacked], family)

    windows = {"reuse": (search, search, p_all)}
    if stage2_start is not None:
        stage2_start = pd.Timestamp(stage2_start)
        first = search.loc[search.index < stage2_start]
        second = search.loc[search.index >= stage2_start]
        if first.empty or second.empty:
            raise ValueError("stage2_start must leave search dates for both stages")
        windows["split"] = (first, second, _one_sided_p(first, "search period before stage2_start"))

    out = {}
    for design, (window1, window2, p1) in windows.items():
        s = _stage1(window1, p1, family, n_boot, seed, q)
        s["p2"] = (
            p_all.reindex(s["best"]).to_numpy()
            if design == "reuse"
            else [
                _tested(window2[b], "search period from stage2_start").pvalue_greater
                if pd.notna(b)
                else np.nan
                for b in s["best"]
            ]
        )
        nb = naive_best.reindex(s.index)
        # Stage 4 reads only the variants carried forward; an unchosen
        # variant with no holdout data is not tested (nor does it block).
        carried = pd.unique(pd.concat([s["best"], nb]).dropna())
        holdout_tests = (
            {b: _tested(holdout[b], "holdout") for b in carried} if holdout is not None else {}
        )
        for column, values in {
            **_holdout_fields(holdout_tests, s["best"], "ho_"),
            **_holdout_fields(holdout_tests, nb, "nv_"),
        }.items():
            s[column] = values
        s["naive_pass"] = nb.notna()
        s.attrs["n_variants"] = n_variants
        out[design] = s
    return out
