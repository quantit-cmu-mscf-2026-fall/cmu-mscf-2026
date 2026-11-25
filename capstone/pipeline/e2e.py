"""The search steps of the end-to-end driver: papers, deepen and evaluation on one panel.

`search` runs the first three steps on the search-period panel it is given:

1. **Papers -> hypotheses -> factors** (`factors.pipeline.run`, or
   `factors.paper_pool.run_isolated` with `[run] paper_mode = "isolated"`): the
   config's paper source through the model client into the factor store.
2. **Deepen** (`factors.deepen.Deepener`): `rounds` deepen rounds from the stored
   factors, each child scored by `factors.scoring.EvaluationScorer` and logged
   before the pool sees it. With `[deepen] selector = "agent"` a `DeepenAgent` on
   the run's client picks each round's parent and edit among the memory's
   candidates. `[deepen] max_children_per_family` caps the children per family.
3. **Evaluation** (`factors.evaluation.evaluate_factors`): every stored factor
   traded by every spec of the grid, on `pipeline.panel_pool.PanelPool` (the same
   bytes at one worker and at many); each variant is logged as a trial before
   anything uses it.

Every deepen score and every variant carries the run's tag (`e2e-run:<run_id>`)
and the data tag, so the trial count of a later gate registration is exactly this
search's. All model calls go through one counter (`pipeline.budget`).
The panel is the caller's: this module reads no market data.

`gate_config` builds the gate registration for a finished search (the draws sized
for its families, the trial scope of its run and data tags). Stage 3 is not
measured by the gate yet, so it registers `factors_digest=None`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone import gate
from capstone.factors import store, tree
from capstone.factors.deepen import (
    LEDGER_TAGS,
    BudgetExhausted,
    ChildOutcome,
    DeepenAgent,
    Deepener,
    FamiliesCapped,
    ModelEditor,
)
from capstone.factors.evaluation import Panel, evaluate_factors, hypothesis_of
from capstone.factors.paper_pool import run_isolated
from capstone.factors.pipeline import run as run_papers
from capstone.factors.scoring import EvaluationScorer
from capstone.factors.sources import FileSource
from capstone.pipeline import panel_pool
from capstone.pipeline.budget import BudgetedClient
from capstone.pipeline.config import E2EConfig
from capstone.scorecard import TrialScope

__all__ = [
    "N_BOOT_STEP",
    "Search",
    "ThresholdsNotRegistered",
    "check_thresholds",
    "gate_config",
    "generator_name",
    "report_path",
    "search",
    "sized_n_boot",
]

N_BOOT_STEP = 500  # n_boot + 1 is rounded up to a multiple of this (`sized_n_boot`)


class ThresholdsNotRegistered(ValueError):
    """A real-data run whose thresholds are the bare code defaults."""


@dataclass(frozen=True, eq=False)
class Search:
    """The result of `search`."""

    factor_ids: tuple[str, ...]  # stored from the papers, before deepening
    deepen: tuple[ChildOutcome, ...]
    trees: dict  # factor id -> tree, every stored factor
    matrix: pd.DataFrame  # search-period net returns, dates x traded variant
    summary: pd.DataFrame  # one row per variant (`evaluate_factors`)
    deepen_rounds: int = 0  # rounds finished
    deepen_stopped: str = ""  # why deepen stopped early ("": it ran every round)


def generator_name(config: E2EConfig) -> str:
    """What produced the candidates, registered with the gate's thresholds."""
    name = f"{config.generator}/{config.factors.model}/{config.factors.prompt_version}"
    return f"{name}/agent-deepen" if config.deepen_selector == "agent" else name


def sized_n_boot(configured: int, n_families: int, q: float = 0.05) -> int:
    """The gate's bootstrap draws: at least `configured` and `gate.required_n_boot`
    for the families produced, with n_boot + 1 rounded up to a multiple of
    `N_BOOT_STEP` (499, 999, ..., 3999 for 200 families, 4499 for 201)."""
    need = max(int(configured), gate.required_n_boot(n_families, q)) + 1
    return -(-need // N_BOOT_STEP) * N_BOOT_STEP - 1


def check_thresholds(config: E2EConfig, *, allow_default_thresholds: bool = False) -> None:
    """Refuse a real-data run whose thresholds are the bare code defaults.

    `funnel.Thresholds()` is a starting point from synthetic runs, not a registered
    set, and the repository holds no registered-thresholds file to fill an empty
    `[gate.thresholds]` from. So a non-synthetic config must state its thresholds
    (`config.thresholds_source == "config"`), unless the caller explicitly allows the
    defaults. Call it before any model call."""
    if config.synthetic or config.thresholds_source != "code-defaults":
        return
    if not allow_default_thresholds:
        raise ThresholdsNotRegistered(
            f"{config.run_id}: [gate.thresholds] is empty on real data ({config.data_label}), "
            "so the gate would register the code defaults. State the registered thresholds "
            "in the config, or pass allow_default_thresholds=True to run on the defaults"
        )


def gate_config(
    config: E2EConfig,
    data_version: str,
    n_families: int,
    *,
    allow_default_thresholds: bool = False,
) -> gate.GateConfig:
    """The run's gate registration: `sized_n_boot` draws, the trial scope of every run
    on this data and search period (this one included, so a rerun under a new run id
    keeps the earlier runs' trials in the charge). Stage 3 is unmeasured by the gate
    (`factors_digest=None`)."""
    check_thresholds(config, allow_default_thresholds=allow_default_thresholds)
    return gate.GateConfig(
        search_end=config.search_end,
        holdout_start=config.holdout_start,
        holdout_end=config.holdout_end,
        data_version=data_version,
        generator=generator_name(config),
        thresholds=config.thresholds,
        n_boot=sized_n_boot(config.n_boot, n_families, config.thresholds.q1),
        seed=config.seed,
        trial_scope=TrialScope(include_tags=(config.run_tag, config.data_tag)),
        factors_digest=None,
    )


def report_path(workdir: Path, registered: gate.GateConfig) -> Path:
    return Path(workdir) / "reports" / f"gate-{registered.config_hash[:12]}.json"


def _stored_in_order(con) -> list[str]:
    """Stored factor ids in the order they were stored (not by timestamp)."""
    return list(dict.fromkeys(p["factor_id"] for p in store.proposals(con, "stored")))


def _deepen(
    con, config: E2EConfig, panel, client, editor, seeds
) -> tuple[tuple[ChildOutcome, ...], int, str]:
    """The deepen rounds: their children, the rounds finished and why they stopped early
    ("" if they didn't). With `deepen_on_budget = "stop"`, a used-up budget ends the
    rounds; the round it interrupts keeps what it stored and logged. So does every
    family reaching `deepen_max_children_per_family`."""
    usage = Counter()  # deepen's own cap (`deepen_max_calls`), editor and agent together
    if editor is None:
        editor = ModelEditor(client, config.factors, max_calls=config.deepen_max_calls, usage=usage)
    agent = None
    if config.deepen_selector == "agent":
        agent = DeepenAgent(client, config.factors, max_calls=config.deepen_max_calls, usage=usage)
    deepener = Deepener(
        con,
        config.factors,
        EvaluationScorer(panel, config.scorer),
        editor,
        run_id=config.run_id,
        seed=config.seed,
        client=client,
        ledger_tags=[*LEDGER_TAGS, config.run_tag, config.data_tag],
        agent=agent,
        max_children_per_family=config.deepen_max_children_per_family,
    )
    if not any([deepener.add_parent(fid) for fid in seeds]):  # every seed is scored
        return (), 0, ""
    children: list[ChildOutcome] = []
    for done in range(config.deepen_rounds):
        try:
            children.extend(deepener.step())
        except BudgetExhausted as exc:
            if config.deepen_on_budget == "raise":
                raise
            return tuple(children), done, str(exc)
        except FamiliesCapped as exc:  # nothing left to deepen under the cap
            return tuple(children), done, str(exc)
    return tuple(children), config.deepen_rounds, ""


def search(
    config: E2EConfig, panel: Panel, client: BudgetedClient, editor, workdir: str | Path
) -> Search:
    """Papers, deepen and evaluation on `panel` (the search period), every trial logged
    with the run's tag. `editor` writes the deepen edits (None: a `ModelEditor` on
    `client`). The factor store is `<workdir>/factors.db`; evaluation runs on a
    `PanelPool` sized by `[resources]` and `CAPSTONE_MAX_WORKERS`."""
    workdir = Path(workdir)
    con = store.connect(workdir / "factors.db")
    try:
        source = FileSource(config.papers)
        if config.paper_mode == "isolated":
            papers = run_isolated(con, source, client, config.factors, config.concurrency)
        else:
            papers = run_papers(con, source, client, config.factors)
        client.check()
        if papers.papers_failed:
            raise RuntimeError(f"papers failed: {papers.papers_failed}")
        seeds = tuple(_stored_in_order(con))
        deepened, rounds, stopped = _deepen(con, config, panel, client, editor, seeds)
        ids = _stored_in_order(con)
        expressions = {f["id"]: f["expression"] for f in store.factors(con)}
        trees = {
            fid: tree.parse(expressions[fid], regime_operators=config.regime_operators)
            for fid in ids
        }
        size = panel_pool.workers(len(trees), panel, config.resources)
        with panel_pool.PanelPool(panel, size, workdir) as pool:
            matrix, summary = evaluate_factors(
                trees,
                panel,
                specs=config.specs,
                con=con,
                families={fid: hypothesis_of(con, fid) for fid in ids},
                extra_params={"run_id": config.run_id},
                tags=[config.run_tag, config.data_tag],
                outcomes=pool.outcomes(config.specs),
            )
    finally:
        con.close()
    return Search(seeds, deepened, trees, matrix, summary, rounds, stopped)
