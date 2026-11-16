"""Placebo-panel transformation and isolated-ledger checks; no search driver yet.

Not yet wired (cut b): a search driver will read config.placebo/placebo_seed,
resolve the shared-data return/spread rules BEFORE permutation, call guard/apply,
and tag placebo runs separately. It must refuse holdout execution and prevent a
real run from reusing the placebo ledger. No current caller enforces that wiring;
these helpers alone do not measure the pipeline's end-to-end false-discovery rate.

**Method `cross_section_permutation`.** On each date the whole rows (every field
together: resolved returns, prices, volume, spreads, membership, delisting flags)
are shuffled across the stock identities present that day, with a seeded RNG.
All recognized identifier columns travel together as the TARGET identity, including
company/security ids and SIC/NAICS classifications. A classification can itself
identify a stock history, so it cannot ride with the donor's economic fields.
Declare any additional identifier columns through identity_columns; do not retain
unrecognized donor identities in the panel.

Why it is a valid null for cross-sectional predictability: each date keeps exactly
the same set of economic rows, so the cross-sectional distribution of every field,
the market (equal-weighted or same-day cap-weighted) return, the universe
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

**Separate ledger convention.** guard/apply require CAPSTONE_LEDGER_DIR to name
experiments/placebo/ in this checkout or its primary checkout. Directory and file
aliases of either main ledger are refused. The folder is gitignored and included
in the private archive sync. Versions are placebo:<method>:<seed>:<real version>.
See docs/telemetry.md. These checks do not authorize a real-data placebo rerun.
"""

from __future__ import annotations

import os
import subprocess
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
IDENTITY_COLUMNS = (
    "permno",
    "permco",
    "cusip",
    "ncusip",
    "ticker",
    "gvkey",
    "lpermno",
    "lpermco",
    "siccd",
    "sic",
    "naics",
)
PREFIX = "placebo"


class PlaceboRefused(ValueError):
    """A placebo search was asked for where it could reach the main trial count."""


def main_ledger_dir() -> Path:
    """The repository's own ledger folder (`runlog`'s default): never a placebo's."""
    from capstone import runlog

    return Path(runlog.__file__).resolve().parent.parent / "experiments"


def _same(a: Path, b: Path) -> bool:
    try:
        if os.path.samefile(a, b):
            return True
    except FileNotFoundError:
        pass  # New ledger paths still need the canonical-path comparison.
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def _main_ledger_dirs() -> list[Path]:
    current = main_ledger_dir()
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=current.parent / "capstone",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        if (current.parent / ".git").is_file():
            raise PlaceboRefused(
                "cannot establish the primary checkout's ledger isolation"
            ) from None
        return [current]  # Installed/exported code has no primary Git checkout.
    return [current, Path(common).parent / "experiments"] if common else [current]


def check_ledger(environ=None) -> Path:
    """The placebo's ledger folder; raises `PlaceboRefused` unless `CAPSTONE_LEDGER_DIR`
    names an experiments/placebo directory, isolated from both main ledger files."""
    env = (os.environ if environ is None else environ).get("CAPSTONE_LEDGER_DIR")
    if not env:
        raise PlaceboRefused("a placebo search needs CAPSTONE_LEDGER_DIR set to its own ledger")
    path = Path(env)
    mains = _main_ledger_dirs()
    for main in mains:
        if _same(path, main) or _same(path / "runs.jsonl", main / "runs.jsonl"):
            raise PlaceboRefused(
                f"CAPSTONE_LEDGER_DIR={path} aliases an experiments/ ledger: "
                "a placebo search must never reach the main trial count"
            )
    if not any(_same(path, main / "placebo") for main in mains):
        raise PlaceboRefused(
            "CAPSTONE_LEDGER_DIR must name the gitignored experiments/placebo ledger"
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


def permute_cross_section(rows: pd.DataFrame, seed: int, *, identity_columns=()) -> pd.DataFrame:
    """`rows` with each date's whole rows shuffled across that date's stock ids.

    Deterministic in (`rows` content, `seed`): the rows are put in (date, permno)
    order first, one RNG draws a key per row, and within each date the ids are
    reassigned in key order. Known identifiers/classifications and explicitly
    declared identity_columns move as a tuple; other fields move together.
    Returns the rows in (date, permno) order, index reset; no columns are dropped.
    """
    _check(METHODS[0], seed)
    if isinstance(identity_columns, str):
        raise ValueError("identity_columns must be a collection of column names")
    identity_columns = tuple(identity_columns)
    missing = set(identity_columns) - set(rows.columns)
    if missing:
        raise ValueError(f"identity_columns not in rows: {sorted(missing)}")
    identities = [c for c in rows if c in (*IDENTITY_COLUMNS, *identity_columns)]
    if rows[["date", *identities]].isna().any().any():
        raise ValueError("placebo rows cannot have missing dates or identifiers")
    ordered = rows.sort_values(["date", ID_COLUMN], kind="stable").reset_index(drop=True)
    if ordered.duplicated(["date", ID_COLUMN]).any():
        raise ValueError("placebo rows need one row per (date, permno)")
    dates = pd.factorize(ordered["date"], sort=True)[0]
    keys = np.random.default_rng(int(seed)).random(len(ordered))
    order = np.lexsort((keys, dates))  # within each date block: a random permutation of it
    shuffled = np.empty(len(ordered), dtype=int)
    shuffled[order] = np.arange(len(ordered))
    out = ordered.copy()
    for column in identities:
        out[column] = ordered[column].iloc[shuffled].reset_index(drop=True)
    return out.sort_values(["date", ID_COLUMN], kind="stable").reset_index(drop=True)


def placebo_rows(
    rows: pd.DataFrame, method: str, seed: int, *, identity_columns=()
) -> pd.DataFrame:
    """The placebo panel of `rows` by `method` (only `cross_section_permutation`)."""
    _check(method, seed)
    return permute_cross_section(rows, seed, identity_columns=identity_columns)


def guard(config, *, holdout: bool = False) -> None:
    """Pre-read check for future drivers: real configs pass, placebo holdouts refuse.

    This helper currently has no production search/holdout caller.
    """
    if config.placebo is None:
        return
    if holdout:
        raise ValueError("a placebo search has no holdout: its rows are not the market's")
    check_ledger()


def apply(
    config, rows: pd.DataFrame, data_version: str, *, identity_columns=()
) -> tuple[pd.DataFrame, str]:
    """The search rows and data version a config searches: `rows` and `data_version`
    unchanged for a real config; for a placebo, the placebo panel of the rows (made
    after the loader's read rules ran on the real rows) and its `placebo_version`.

    The caller MUST resolve sd.daily_returns and sd.quoted_spread on original
    identities before calling this helper, and must not rerun those rules on
    permuted rows: their per-stock history rules would invent losses/spreads.
    No current driver calls this helper or enforces that ordering yet.
    """
    if config.placebo is None:
        return rows, data_version
    check_ledger()
    method, seed = config.placebo, config.placebo_seed
    return (
        placebo_rows(rows, method, seed, identity_columns=identity_columns),
        placebo_version(method, seed, data_version),
    )
