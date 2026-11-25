"""A finished search's registration and manifest.

`register_search` takes the `e2e.Search` a driver produced and does the one-way part:
it builds the gate config (`e2e.gate_config`), checks the ledger holds a trial for
every variant in the gate's scope, registers the config (`gate.register`, once per
hash), saves the matrix and summary as parquet and writes `<workdir>/manifest.json`.
It refuses a workdir that already holds a manifest.

The manifest records what a later stage needs to re-check: the config digest, the
registered gate config and its hash, the trial count, and the SHA-256 of both
frames. `read_manifest` refuses unknown keys and a gate config that does not match
its hash; `read_frames` refuses a frame whose bytes changed.

Stage 3 hook: the gate does not measure stage 3 yet, so `factors_digest` is None and
`stage3` says "unmeasured". The PR that connects the spanning test fills both.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

from capstone import gate, intake, runlog
from capstone.pipeline import e2e
from capstone.pipeline.config import E2EConfig
from capstone.pipeline.search_identity import config_digest
from capstone.scorecard import TrialScope

__all__ = [
    "MANIFEST_FILE",
    "MATRIX_FILE",
    "SUMMARY_FILE",
    "RunManifest",
    "read_frames",
    "read_manifest",
    "register_search",
    "trials_in_scope",
]

MANIFEST_FILE = "manifest.json"
MATRIX_FILE = "search/matrix.parquet"
SUMMARY_FILE = "search/summary.parquet"
STAGE3_UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class RunManifest:
    """What a search did and registered."""

    run_id: str
    run_tag: str
    config_digest: str  # `config_digest` of the E2EConfig the search ran with
    data_version: str  # of the search rows
    gate_config: dict  # `GateConfig.to_dict()`, as registered
    gate_config_hash: str
    factors_digest: str | None
    stage3: str  # STAGE3_UNMEASURED until the gate measures it
    thresholds_source: str  # "config" or "code-defaults" (`E2EConfig.thresholds_source`)
    n_boot: int
    trial_count: int  # ledger trials in the gate's scope after the search
    run_trial_count: int  # this run's own trials (`run_tag` alone)
    model_calls: int
    families: tuple[str, ...]
    n_variants: int
    factor_ids: tuple[str, ...]  # stored from the papers, before deepening
    matrix_file: str  # relative to the workdir
    matrix_digest: str  # SHA-256 of the file's bytes
    summary_file: str
    summary_digest: str
    deepen_rounds: int = 0  # deepen rounds finished
    deepen_stopped: str = ""  # why deepen stopped early ("": it ran every round)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["families"], out["factor_ids"] = list(self.families), list(self.factor_ids)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> RunManifest:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown manifest keys: {sorted(unknown)}")
        data = dict(data)
        data["families"], data["factor_ids"] = tuple(data["families"]), tuple(data["factor_ids"])
        return cls(**data)


def read_manifest(path: str | Path) -> RunManifest:
    """The manifest at `path`; refuses unknown keys and an edited gate config."""
    manifest = RunManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    registered = gate.GateConfig.from_dict(manifest.gate_config)
    if registered.config_hash != manifest.gate_config_hash:
        raise ValueError("the manifest's gate config does not match its hash: it was edited")
    return manifest


def trials_in_scope(scope: TrialScope) -> int:
    """The trial count the gate charges for `scope`: `runlog.trial_count` with the
    gate's own entries left out (what `gate.run_gate` computes)."""
    try:
        return runlog.trial_count(
            scope.name,
            include_tags=None if scope.include_tags is None else list(scope.include_tags),
            exclude_tags=[*scope.exclude_tags, gate.GATE_TAG],
        )
    except LookupError:  # the ledger has no trial in scope
        return 0


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frame(workdir: Path, name: str, digest: str) -> pd.DataFrame:
    path = workdir / name
    if _file_digest(path) != digest:
        raise ValueError(f"{name} changed since the search (its digest differs)")
    return pd.read_parquet(path)


def read_frames(workdir: str | Path, manifest: RunManifest) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The saved (matrix, summary), refused if either file's bytes changed."""
    workdir = Path(workdir)
    return (
        _read_frame(workdir, manifest.matrix_file, manifest.matrix_digest),
        _read_frame(workdir, manifest.summary_file, manifest.summary_digest),
    )


def register_search(
    config: E2EConfig,
    found: e2e.Search,
    *,
    data_version: str,
    model_calls: int,
    workdir: str | Path,
    allow_default_thresholds: bool = False,
) -> RunManifest:
    """Register the search's gate config, save its frames and write its manifest.

    Refuses, before registering anything: a workdir that already has a manifest,
    thresholds that are the bare defaults on real data, and fewer ledger trials in
    scope than variants (some were not logged)."""
    workdir = Path(workdir)
    path = workdir / MANIFEST_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists: this workdir already holds a registered search")
    funnel_input = intake.from_evaluation(
        found.matrix, found.summary, generator=e2e.generator_name(config), data_version=data_version
    )
    families = tuple(sorted(str(f) for f in funnel_input.family.unique()))
    registered = e2e.gate_config(
        config, data_version, len(families), allow_default_thresholds=allow_default_thresholds
    )
    trials = trials_in_scope(registered.trial_scope)
    if trials < found.matrix.shape[1]:
        raise ValueError(f"{trials} trials in scope for {found.matrix.shape[1]} variants: log them")
    gate.register(registered)

    (workdir / MATRIX_FILE).parent.mkdir(parents=True, exist_ok=True)
    found.matrix.to_parquet(workdir / MATRIX_FILE)
    found.summary.to_parquet(workdir / SUMMARY_FILE)
    manifest = RunManifest(
        run_id=config.run_id,
        run_tag=config.run_tag,
        config_digest=config_digest(config),
        data_version=data_version,
        gate_config=registered.to_dict(),
        gate_config_hash=registered.config_hash,
        factors_digest=registered.factors_digest,
        stage3=STAGE3_UNMEASURED,
        thresholds_source=config.thresholds_source,
        n_boot=registered.n_boot,
        trial_count=trials,
        run_trial_count=trials_in_scope(TrialScope(include_tags=(config.run_tag,))),
        model_calls=model_calls,
        families=families,
        n_variants=int(found.matrix.shape[1]),
        factor_ids=found.factor_ids,
        matrix_file=MATRIX_FILE,
        matrix_digest=_file_digest(workdir / MATRIX_FILE),
        summary_file=SUMMARY_FILE,
        summary_digest=_file_digest(workdir / SUMMARY_FILE),
        deepen_rounds=found.deepen_rounds,
        deepen_stopped=found.deepen_stopped,
    )
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return manifest
