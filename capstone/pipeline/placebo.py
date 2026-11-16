"""The placebo search: the whole pipeline on rows with no real predictability.

`[run] placebo = "cross_section_permutation"` plus `[run] placebo_seed = <int>` makes
`run_search` replace its search rows, after the panel loader read them (on CRSP:
`crsp_rows` through `capstone.shared_data`, unchanged), with a placebo panel. Every
model call, deepen round, trial and gate stage then runs as usual, so what the gate
lets through measures the end-to-end false-discovery rate, model included.

**Method `cross_section_permutation`.** On each date the whole rows (every field
together: returns, prices, volume, spreads, membership, delisting flags, SIC code)
are shuffled across the stock ids (`permno`) present that day, with a seeded RNG.

Why it is a valid null for cross-sectional predictability: each date keeps exactly
the same set of rows (`date_rows` multisets are unchanged), so the cross-sectional
distribution of every field, the market (equal- or cap-weighted) return, the universe
size and the spreads are those of the real data, and each row stays internally
consistent. But which id a row lands on is drawn afresh every date, independently of
everything before it, so a stock id's past says nothing about its future: any signal
built from an id's history (momentum, reversal, volatility, size trends) has an
expected rank IC of zero against the next return. Whatever the search keeps is noise.

Its limits: stock identity is meaningless, so per-stock time-series properties are
destroyed (return autocorrelation, volatility persistence, a delisting followed by
more rows for the same id, a price path that jumps between unrelated stocks). Signals
that use only the same day's cross-section (a row's own spread or size at t, with no
history) keep their contemporaneous relations, but they are only rewarded if they
predict the next day, which goes to a fresh random id. The placebo therefore tests
the cross-sectional search, not time-series signals; and costs and turnover are those
of a market where every stock is replaced daily, so the cost hurdle is harsher than real.

**It never touches the main trial count.** A placebo search needs `CAPSTONE_LEDGER_DIR`
set to a folder other than the repository's `experiments/` (`check_ledger`, used by
`real_run` and `run_search`); its run tag is `e2e-run:<run_id>:placebo`; and its data
version is `placebo:<method>:<seed>:<the real data version>`, so it can never be
confused with a real one. A placebo search has no holdout (`run_holdout` refuses it).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "METHODS",
    "PlaceboRefused",
    "apply",
    "check_ledger",
    "guard",
    "permute_cross_section",
    "placebo_rows",
    "placebo_version",
]

METHODS = ("cross_section_permutation",)
ID_COLUMN = "permno"
PREFIX = "placebo"


class PlaceboRefused(ValueError):
    """A placebo search was asked for where it could reach the main trial count."""


def main_ledger_dir() -> Path:
    """The repository's own ledger folder (`runlog`'s default): never a placebo's."""
    from capstone import runlog

    return Path(runlog.__file__).resolve().parent.parent / "experiments"


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def check_ledger(environ=None) -> Path:
    """The placebo's ledger folder; raises `PlaceboRefused` unless `CAPSTONE_LEDGER_DIR`
    is set and is not the repository's `experiments/` folder."""
    env = (os.environ if environ is None else environ).get("CAPSTONE_LEDGER_DIR")
    if not env:
        raise PlaceboRefused("a placebo search needs CAPSTONE_LEDGER_DIR set to its own ledger")
    path = Path(env)
    if _same(path, main_ledger_dir()):
        raise PlaceboRefused(
            f"CAPSTONE_LEDGER_DIR={path} is the repository's experiments/ ledger: "
            "a placebo search must never reach the main trial count"
        )
    return path


def placebo_version(method: str, seed: int, real_version: str) -> str:
    """The data version a placebo search registers: `placebo:<method>:<seed>:<real>`."""
    _check(method, seed)
    return f"{PREFIX}:{method}:{int(seed)}:{real_version}"


def _check(method: str, seed) -> None:
    if method not in METHODS:
        raise ValueError(f"placebo method must be one of {METHODS}, got {method!r}")
    if seed is None or isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError(f"placebo_seed must be a non-negative integer, got {seed!r}")


def permute_cross_section(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """`rows` with each date's whole rows shuffled across that date's stock ids.

    Deterministic in (`rows` content, `seed`): the rows are put in (date, permno)
    order first, one RNG draws a key per row, and within each date the ids are
    reassigned in key order. Returns the rows in (date, permno) order, index reset.
    """
    _check(METHODS[0], seed)
    ordered = rows.sort_values(["date", ID_COLUMN], kind="stable").reset_index(drop=True)
    if ordered.duplicated(["date", ID_COLUMN]).any():
        raise ValueError("placebo rows need one row per (date, permno)")
    ids = ordered[ID_COLUMN].to_numpy()
    dates = pd.factorize(ordered["date"], sort=True)[0]
    keys = np.random.default_rng(int(seed)).random(len(ordered))
    order = np.lexsort((keys, dates))  # within each date block: a random permutation of it
    shuffled = np.empty_like(ids)
    shuffled[order] = ids  # the block's k-th id goes to the block's k-th random row
    out = ordered.copy()
    out[ID_COLUMN] = shuffled
    return out.sort_values(["date", ID_COLUMN], kind="stable").reset_index(drop=True)


def placebo_rows(rows: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    """The placebo panel of `rows` by `method` (only `cross_section_permutation`)."""
    _check(method, seed)
    return permute_cross_section(rows, seed)


def guard(config, *, holdout: bool = False) -> None:
    """`run_search`'s and `run_holdout`'s check, before anything is read: nothing for a
    real config; for a placebo, `check_ledger`, and a refusal of any holdout."""
    if config.placebo is None:
        return
    if holdout:
        raise ValueError("a placebo search has no holdout: its rows are not the market's")
    check_ledger()


def apply(config, rows: pd.DataFrame, data_version: str) -> tuple[pd.DataFrame, str]:
    """The search rows and data version a config searches: `rows` and `data_version`
    unchanged for a real config; for a placebo, the placebo panel of the rows (made
    after the loader's read rules ran on the real rows) and its `placebo_version`."""
    if config.placebo is None:
        return rows, data_version
    method, seed = config.placebo, config.placebo_seed
    return placebo_rows(rows, method, seed), placebo_version(method, seed, data_version)
