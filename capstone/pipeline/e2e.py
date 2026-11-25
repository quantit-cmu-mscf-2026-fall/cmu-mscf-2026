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
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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

__all__ = ["Search", "generator_name", "search"]


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
