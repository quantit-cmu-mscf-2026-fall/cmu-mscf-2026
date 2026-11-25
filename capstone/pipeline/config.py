"""The end-to-end driver's parameters (`config/e2e_synthetic.toml`).

One frozen `E2EConfig` per run: the run id (its ledger tag) and model-call
budget, the periods, the paper source, the factor pipeline's settings, the
deepen rounds, selector and scorer spec, the evaluation grid, the gate's
generator and thresholds, stage 3's factor frame, the trading book and the
incubation monitor. Unknown sections and keys raise, so a typo is not
silently ignored. `config/e2e_crsp.toml` is the real-data run's
(`capstone.pipeline.search`, a follow-up).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, get_args

import pandas as pd

from capstone.factors.evaluation import TradingSpec, grid
from capstone.factors.llm import FactorConfig
from capstone.funnel import Thresholds
from capstone.pipeline.executor import AUTO, Resources
from capstone.pipeline.placebo import METHODS as PLACEBO_METHODS
from capstone.pipeline.rows import REPO

DEFAULT_CONFIG = REPO / "config" / "e2e_synthetic.toml"
DEFAULT_MAX_MODEL_CALLS = 300  # the team default; a run states its own in `[run] max_model_calls`


LiveEdge = Literal["expected", "none"]
# Who picks each deepen round's parent and edit: learned memory search alone, or
# the model as agent (`factors.deepen.DeepenAgent`), advised by the memory.
Selector = Literal["memory", "agent"]
# The run's call budget used up during deepen: raise (the run stops), or stop
# deepening and go on to evaluation and registration with the children so far.
OnBudget = Literal["raise", "stop"]
PaperMode = Literal["sequential", "isolated"]
# Stage 3's factor frame: none (stage 3 passes through, unmeasured), a seeded
# synthetic frame on the search dates (tests), or Ken French's FF5 + momentum.
Stage3Factors = Literal["none", "synthetic", "ff5_momentum"]


@dataclass(frozen=True)
class IncubationSettings:
    """The monitor's alpha and horizon, and the synthetic live continuation:
    re-centred on the gate's expected mean ("expected") or on zero ("none"),
    bootstrapped in circular blocks of `block` periods.

    An incubation-only (incubate) family also runs the confirming e-process
    (`incubation.confirm`) at `confirm_alpha`, and is promoted or
    retired within `confirm_horizon` observed periods (its decay monitor runs
    to that horizon too). `confirm_accepted` does the same for accepted
    families; off by default, they keep the decay-only monitor at `horizon`."""

    alpha: float = 0.05
    horizon: int = 252
    live_edge: LiveEdge = "expected"
    block: int = 20
    confirm_alpha: float = 0.05
    confirm_horizon: int = 1260
    confirm_accepted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.confirm_accepted, bool):
            raise ValueError("confirm_accepted must be true or false")
        if not 0 < self.confirm_alpha < 1:
            raise ValueError("confirm_alpha must be in (0, 1)")
        if not (isinstance(self.confirm_horizon, int) and self.confirm_horizon >= 1):
            raise ValueError("confirm_horizon must be a positive integer")
        if self.live_edge not in get_args(LiveEdge):
            raise ValueError(f"live_edge must be one of {get_args(LiveEdge)}")
        if self.block < 1:
            raise ValueError("block must be at least 1")


@dataclass(frozen=True)
class Stage3Settings:
    """Stage 3's factor frame (`capstone.pipeline.stage3`); `seed` is the synthetic one's."""

    factors: Stage3Factors = "none"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.factors not in get_args(Stage3Factors):
            raise ValueError(f"stage3 factors must be one of {get_args(Stage3Factors)}")


@dataclass(frozen=True)
class E2EConfig:
    """Everything a run decides with; `load_config` reads it from a TOML file."""

    run_id: str
    seed: int
    data_label: str
    search_end: str
    holdout_start: str
    holdout_end: str
    papers: str  # a TOML file whose [[papers]] are the run's paper source
    factors: FactorConfig
    deepen_rounds: int
    deepen_max_calls: int
    scorer: TradingSpec
    specs: tuple[TradingSpec, ...]
    generator: str
    thresholds: Thresholds
    n_boot: int
    gross: float
    min_trade: float
    incubation: IncubationSettings
    max_model_calls: int = (
        DEFAULT_MAX_MODEL_CALLS  # every model call of the run, through the driver's wrapper
    )
    deepen_selector: Selector = "memory"
    deepen_on_budget: OnBudget = "raise"
    # `[deepen] max_children_per_family`: children deepen may propose per family (a
    # parent's hypothesis). None: unlimited. Enters the digest only when set.
    deepen_max_children_per_family: int | None = None
    stage3: Stage3Settings = Stage3Settings()
    search_start: str | None = None  # None: the market rows' first date
    # Papers against the store as the paper phase found it, side by side (paper_pool;
    # "sequential" is pipeline.run). Enters the digest only when set.
    paper_mode: PaperMode = "sequential"
    # Papers in flight at once in "isolated" mode; AUTO sizes it from the machine. A
    # runtime knob, never in the digest: the results are the same at every value.
    concurrency: int = AUTO
    # A placebo search (`capstone.pipeline.placebo`): the method and its seed. Both enter
    # the digest only when set; a placebo run's ledger tag ends in ":placebo".
    placebo: str | None = None
    placebo_seed: int | None = None
    # `[factors] regime_operators`: admits the grammar's market-regime operators
    # (`tree.REGIME_FUNCS`). Off by default; enters the digest only when set.
    regime_operators: bool = False
    # `[resources]` caps on the evaluation pool (`pipeline.panel_pool`), with
    # `CAPSTONE_MAX_WORKERS`. Runtime only, never in the digest.
    resources: Resources = Resources()
    # Where `thresholds` came from: "config" (the file's [gate.thresholds]) or
    # "code-defaults" (the file left them empty: `funnel.Thresholds()`, a starting point,
    # not registered values). Runtime only, never in the digest; a real-data run on
    # "code-defaults" is refused at registration (`e2e.check_thresholds`).
    thresholds_source: str = "config"

    def __post_init__(self) -> None:
        if not isinstance(self.regime_operators, bool):
            raise ValueError("[factors] regime_operators must be true or false")
        names = ("search_end", "holdout_start", "holdout_end")
        for name in (*names, "search_start") if self.search_start else names:
            object.__setattr__(self, name, pd.Timestamp(getattr(self, name)).date().isoformat())
        if not self.search_end < self.holdout_start <= self.holdout_end:
            raise ValueError("need search_end < holdout_start <= holdout_end")
        if self.search_start and not self.search_start < self.search_end:
            raise ValueError("need search_start < search_end")
        if self.deepen_selector not in get_args(Selector):
            raise ValueError(f"deepen selector must be one of {get_args(Selector)}")
        if self.deepen_on_budget not in get_args(OnBudget):
            raise ValueError(f"deepen on_budget must be one of {get_args(OnBudget)}")
        cap = self.deepen_max_children_per_family
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError("[deepen] max_children_per_family must be at least 1")
        if self.paper_mode not in get_args(PaperMode):
            raise ValueError(f"paper_mode must be one of {get_args(PaperMode)}")
        if self.concurrency < AUTO:
            raise ValueError(f"concurrency must be >= {AUTO}")
        if self.max_model_calls < 1:
            raise ValueError("max_model_calls must be at least 1")
        if not self.run_id or not self.data_label or not self.specs:
            raise ValueError("run_id, data_label and at least one spec are required")
        object.__setattr__(self, "specs", tuple(self.specs))
        if self.placebo is not None or self.placebo_seed is not None:
            if self.placebo not in PLACEBO_METHODS:
                raise ValueError(
                    f"placebo must be one of {PLACEBO_METHODS} when placebo_seed is set"
                )
            seed = self.placebo_seed
            if seed is None or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("a placebo search needs placebo_seed, a non-negative integer")

    @property
    def synthetic(self) -> bool:
        return self.data_label.startswith("synthetic")

    @property
    def run_tag(self) -> str:
        tag = f"e2e-run:{self.run_id}"
        return f"{tag}:placebo" if self.placebo is not None else tag

    @property
    def data_tag(self) -> str:
        """Tag on every trial of every run over this data label and search period, so the
        multiple-testing charge accumulates across reruns (a new run id does not reset it)."""
        tag = f"e2e-data:{self.data_label}:{self.search_start or 'start'}:{self.search_end}"
        return f"{tag}:placebo" if self.placebo is not None else tag


_SECTIONS = {
    "run": {
        "run_id",
        "seed",
        "data_label",
        "max_model_calls",
        "papers",
        "paper_mode",
        "concurrency",
        "placebo",
        "placebo_seed",
    },
    "periods": {"search_start", "search_end", "holdout_start", "holdout_end"},
    "factors": {f.name for f in fields(FactorConfig)} | {"regime_operators"},
    "deepen": {
        "rounds",
        "max_calls",
        "hold",
        "neutral",
        "selector",
        "on_budget",
        "max_children_per_family",
    },
    "evaluation": {"holds", "neutral"},
    "gate": {"generator", "n_boot", "thresholds"},
    "trading": {"gross", "min_trade"},
    "incubation": {f.name for f in fields(IncubationSettings)},
    "stage3": {f.name for f in fields(Stage3Settings)},
    "papers": None,  # FileSource checks its own entries
    "resources": {"max_workers", "memory_budget_mb"},
}


def load_config(path: str | Path | None = None) -> E2EConfig:
    """Read a run's config. Unknown sections and keys raise.

    The papers are the file's own `[[papers]]`, or the file `[run] papers`
    names (relative to the repository root).
    """
    path = Path(path or DEFAULT_CONFIG)
    with open(path, "rb") as handle:
        values = tomllib.load(handle)
    for section in values:
        if section not in _SECTIONS:
            raise ValueError(f"unknown section [{section}] in {path}")
        allowed = _SECTIONS[section]
        unknown = set(values[section]) - allowed if allowed is not None else set()
        if unknown:
            raise ValueError(f"unknown keys in [{section}] of {path}: {sorted(unknown)}")
    run, deepen, ev = values["run"], values["deepen"], values["evaluation"]
    gate_values, trading = values["gate"], values["trading"]
    papers = REPO / run["papers"] if "papers" in run else path
    explicit = gate_values.get("thresholds", {})
    return E2EConfig(
        thresholds_source="config" if explicit else "code-defaults",
        run_id=run["run_id"],
        seed=int(run.get("seed", 0)),
        data_label=run["data_label"],
        max_model_calls=int(run.get("max_model_calls", DEFAULT_MAX_MODEL_CALLS)),
        paper_mode=run.get("paper_mode", "sequential"),
        concurrency=int(run.get("concurrency", AUTO)),
        placebo=run.get("placebo"),
        placebo_seed=run.get("placebo_seed"),
        resources=Resources.from_section(values.get("resources")),
        **values["periods"],
        papers=str(papers),
        factors=FactorConfig(
            **{k: v for k, v in values.get("factors", {}).items() if k != "regime_operators"}
        ),
        regime_operators=values.get("factors", {}).get("regime_operators", False),
        deepen_rounds=int(deepen["rounds"]),
        deepen_max_calls=int(deepen.get("max_calls", 10)),
        deepen_selector=deepen.get("selector", "memory"),
        deepen_on_budget=deepen.get("on_budget", "raise"),
        deepen_max_children_per_family=deepen.get("max_children_per_family"),
        scorer=TradingSpec(deepen.get("hold", 1), deepen.get("neutral", "none")),
        specs=tuple(grid(ev["holds"], ev["neutral"])),
        generator=gate_values["generator"],
        thresholds=Thresholds(**gate_values.get("thresholds", {})),
        n_boot=int(gate_values.get("n_boot", 1999)),
        gross=float(trading["gross"]),
        min_trade=float(trading["min_trade"]),
        incubation=IncubationSettings(**values.get("incubation", {})),
        stage3=Stage3Settings(**values.get("stage3", {})),
    )
