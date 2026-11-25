"""The real-data search: load the search-period rows, search, register, stop.

`run_search` is the driver's first entry point. It never opens the holdout: the
loader is asked for (`search_start`, `search_end`) only, `search_end` must not
pass `evaluation.DISCOVERY_END`, and rows are read only through
`pipeline.search_cache.CachedRows` (so `shared_data`) unless the caller supplies
a loader of its own (synthetic rows in tests).

Refusals, in order, all before any model call:

1. `placebo.guard` (a placebo needs its own ledger), then any placebo config:
   `placebo.apply` needs returns and spreads resolved on original identities,
   and `evaluation.build_panel` computes them from the rows it is given, so a
   placebo search is refused until the panel can take pre-resolved ones;
2. a `search_end` after `DISCOVERY_END`;
3. `e2e.check_thresholds` (bare default thresholds on real data);
4. a workdir that already holds a search (`manifest.json` or `factors.db`);
5. rows that are empty, outside [`search_start`, `search_end`], or undated.

Then it builds the panel, runs `e2e.search` through a `BudgetedClient` and calls
`manifest.register_search`, which checks the logged trials, registers the gate
config once and writes the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone import shared_data as sd
from capstone.factors.evaluation import (
    COLUMNS,
    DAILY_FILE,
    DISCOVERY_END,
    HoldoutError,
    build_panel,
)
from capstone.holdout_authority import require_discovery_query
from capstone.pipeline import e2e, manifest, placebo
from capstone.pipeline.budget import BudgetedClient
from capstone.pipeline.config import E2EConfig
from capstone.pipeline.rows import rows_digest, sorted_rows
from capstone.pipeline.search_cache import CachedRows

__all__ = ["SearchRun", "run_search"]


@dataclass(frozen=True, eq=False)
class SearchRun:
    manifest: manifest.RunManifest
    path: Path  # the manifest file
    found: e2e.Search


def _refuse_existing(workdir: Path) -> None:
    for name in (manifest.MANIFEST_FILE, "factors.db"):
        if (workdir / name).exists():
            raise FileExistsError(f"{workdir / name} exists: this workdir already holds a search")


def _search_rows(config: E2EConfig, loader) -> pd.DataFrame:
    """The loader's rows for [search_start, search_end], checked."""
    rows = sorted_rows(loader(config.search_start, config.search_end))
    if rows.empty:
        raise ValueError("the loader returned no rows")
    dates = pd.to_datetime(rows["date"], errors="raise")
    if dates.max() > pd.Timestamp(config.search_end):
        raise ValueError(f"the loader returned rows past search_end {config.search_end}")
    if config.search_start and dates.min() < pd.Timestamp(config.search_start):
        raise ValueError(f"the loader returned rows before {config.search_start}")
    return rows


def run_search(
    config: E2EConfig,
    *,
    client,
    workdir: str | Path,
    editor=None,
    loader=None,
    source_version: str | None = None,
    allow_default_thresholds: bool = False,
) -> SearchRun:
    """Search and register; never open the holdout. See the module docstring.

    `loader(start, end)` returns CRSP-shaped daily rows (`evaluation.COLUMNS`); the
    default is `CachedRows` on the shared daily file, whose `sd.data_version()` joins
    the registered data version (a custom loader names its own `source_version`).
    `client` must have `max_retries == 0`; it gets at most `config.max_model_calls`
    calls. `editor` writes the deepen edits (None: a model editor on `client`).
    """
    placebo.guard(config)
    if config.placebo is not None:
        raise placebo.PlaceboRefused(
            "a placebo search is not supported: build_panel recomputes returns and spreads "
            "on the permuted rows"
        )
    if pd.Timestamp(config.search_end) > pd.Timestamp(DISCOVERY_END):
        raise HoldoutError(f"search_end {config.search_end} is after {DISCOVERY_END}")
    require_discovery_query(config.search_end)
    e2e.check_thresholds(config, allow_default_thresholds=allow_default_thresholds)
    workdir = Path(workdir)
    _refuse_existing(workdir)

    default = loader is None
    rows = _search_rows(config, CachedRows(DAILY_FILE, COLUMNS) if default else loader)
    if default:
        source_version = sd.data_version()
    label = f"{config.data_label}:{source_version}" if source_version else config.data_label
    data_version = f"{label}:{rows_digest(rows)}"
    panel = build_panel(rows, data_version)

    workdir.mkdir(parents=True, exist_ok=True)
    budget = BudgetedClient(client, config.max_model_calls)
    found = e2e.search(config, panel, budget, editor, workdir)
    record = manifest.register_search(
        config,
        found,
        data_version=data_version,
        model_calls=budget.calls,
        workdir=workdir,
        allow_default_thresholds=allow_default_thresholds,
    )
    return SearchRun(record, workdir / manifest.MANIFEST_FILE, found)
