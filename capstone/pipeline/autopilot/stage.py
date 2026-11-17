"""The Stage protocol, its result and context, and dependency ordering."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capstone.pipeline.autopilot.reasons import ReasonSpec, Stop

STATES = ("pending", "running", "done", "blocked", "stopped")


@dataclass(frozen=True)
class StageResult:
    """What `run` returns. `artifacts` are immutable files under the run root (the runner
    records their sha256 and STOPs if one later changes); `data` is small JSON that
    downstream stages read with `ctx.data(<stage>)`."""

    status: str = "done"  # done | blocked | stopped
    code: str | None = None  # required unless done; declared in Stage.reasons
    detail: str = ""
    artifacts: tuple = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("done", "blocked", "stopped"):
            raise ValueError(f"unknown stage status {self.status!r}")
        if (self.status == "done") != (self.code is None):
            raise ValueError("a reason code is given exactly when the stage is not done")


@dataclass(frozen=True)
class Stage:
    """One node of the DAG. `run(ctx) -> StageResult` must be idempotent: a rerun after a
    kill at any point, in a fresh process, converges to the same outputs. It may also
    raise `Blocked`/`Stop`. `schema` maps each key of config section `[<section>]` to a
    `config.Key`. A `recurring` stage is run again on every tick after it is done.

    `spend(ctx)` marks a stage that freezes or makes model calls: it returns the call
    ceiling of `spend_scope` (default: the stage name), or None for "no calls this
    time". It is a hook only: the runner (a later change) passes it to a spend guard the
    caller supplies before every `run`, so a stage cannot forget the check."""

    name: str
    unit: str
    run: Callable[[Context], StageResult]
    depends_on: tuple[str, ...] = ()
    section: str | None = None  # default: the stage name
    schema: Mapping[str, Any] = field(default_factory=dict)
    reasons: Mapping[str, ReasonSpec] = field(default_factory=dict)
    recurring: bool = False
    spend: Callable[[Context], int | None] | None = None
    spend_scope: str | None = None
    # {dependency: blocked codes} under which this stage still runs in the same tick,
    # because the dependency raises them only after everything it completed (and this
    # stage reads from the run root) is complete, e.g. a daily stage's `data_stale`.
    after_blocked: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if self.section is None:
            object.__setattr__(self, "section", self.name)


@dataclass
class Context:
    """What a stage (or an ops hook, with `stage=None`) sees."""

    config: Any  # config.Config
    journal: Any  # journal.Journal
    stage: str | None = None
    services: Mapping[str, Any] = field(default_factory=dict)

    @property
    def run_root(self) -> Path:
        return self.config.run_root

    @property
    def section(self) -> Mapping[str, Any]:
        """This stage's validated config section."""
        return self.config.sections[self.config.section_of[self.stage]]

    @property
    def reviewer(self) -> str:
        """The audit identity of every rule-driven action."""
        return f"autopilot:{self.config.core['policy_id']}"

    def path(self, *parts) -> Path:
        return self.run_root.joinpath(*parts)

    def record(self, stage) -> dict:
        """Another stage's journal record (a copy)."""
        return self.journal.stage(stage)

    def data(self, stage) -> dict:
        return self.record(stage)["data"]

    def event(self, kind, **fields):
        """Append one audit event to the run's event log."""
        return self.journal.event(kind, stage=self.stage, **fields)

    def now(self) -> str:
        return self.journal.clock()

    def model_client(self):
        """The client for every pipeline call that may reach a model: the injected
        `services["model_client"]` (tests: always a fake), else None (the pipeline's real
        client) only when a person set `[autopilot] real_model_calls` and the process is
        not a pytest run; otherwise a STOP. A stage never passes `client=None` itself."""
        client = self.services.get("model_client")
        if client is not None:
            return client
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise Stop("model_client_not_authorized", "a test must inject a fake model client")
        if not self.config.core["real_model_calls"]:
            raise Stop("model_client_not_authorized", "[autopilot] real_model_calls is false")
        return None


def order(stages):
    """Stages in dependency order (stable: declaration order among the ready)."""
    by_name = {}
    for stage in stages:
        if stage.name in by_name:
            raise ValueError(f"stage {stage.name!r} is declared twice")
        by_name[stage.name] = stage
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in by_name:
                raise ValueError(f"stage {stage.name!r} depends on unknown stage {dep!r}")
    ordered, placed = [], set()
    while len(ordered) < len(by_name):
        ready = [
            s for s in stages if s.name not in placed and all(d in placed for d in s.depends_on)
        ]
        if not ready:
            raise ValueError(f"stage dependency cycle among {sorted(set(by_name) - placed)}")
        ordered.append(ready[0])
        placed.add(ready[0].name)
    return ordered
