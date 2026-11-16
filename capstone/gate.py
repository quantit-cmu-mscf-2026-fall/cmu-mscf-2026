"""The acceptance gate's registration: fix the configuration before scoring.

`docs/validation/framework.md` gives the funnel its stages (`capstone.funnel`)
and the scorecard its shape (`capstone.scorecard`). For the gate's decision
to be trusted, everything it decides with is fixed before any stage runs:

- **`GateConfig`** holds the thresholds, the bootstrap draws and seed, the
  search and holdout dates, the generator, the data version, the ledger
  scope of the trial count, and the digest of stage 3's factor frame (None
  until the spanning test exists). Its hash is SHA-256 of its canonical
  JSON content, so equal content hashes equally in every process.
- **`register`** logs the config and its hash to the ledger
  (`runlog.log_run`, tags "gate" and "synthetic") before anything is scored.
  Registering the same content twice logs once and returns the first entry,
  so a repeat leaves no second record. A changed config has a different
  hash, so it has no registration until it is registered itself.
- **Input checks** that the run will make before it opens anything:
  returns indexed by increasing dates, a family label for every column, no
  date after `search_end`, and enough bootstrap draws for the stage-1
  bootstrap to resolve BH at `q1` across the families (`required_n_boot`).

The gate's own ledger entries carry the tag "gate" and are never counted as
trials (`_trial_count`). They also carry "synthetic", the runlog convention
for methodology entries, so `runlog.trial_count(exclude_tags=["synthetic"])`
leaves them out too; `python -m capstone.runlog stats` still lists them under
their own name.

This module only fixes and records the configuration. It does not yet refuse
to score an unregistered or mismatched config, and it does not record a second
registration of the same content: the gate run, a later change, is what
enforces that a run matches a registration and looks at the holdout once.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capstone import runlog
from capstone.funnel import Thresholds, thresholds_dict
from capstone.scorecard import TrialScope

__all__ = [
    "GATE_TAG",
    "REGISTRATION_NAME",
    "SYNTHETIC_TAG",
    "GateConfig",
    "TrialScope",
    "register",
    "required_n_boot",
]

REGISTRATION_NAME = "gate-registration"
GATE_TAG = "gate"
# runlog's convention for entries that are not trials of a strategy.
SYNTHETIC_TAG = "synthetic"


def _canonical_scope(scope: TrialScope) -> TrialScope:
    """`scope` with its tags sorted and de-duplicated, so equal scopes are equal."""
    if not isinstance(scope, TrialScope):
        raise TypeError(f"trial_scope must be a TrialScope, not {type(scope).__name__}")
    if scope.name is not None and not isinstance(scope.name, str):
        raise TypeError("trial_scope.name must be a string or None")
    tags = {}
    for field_name in ("include_tags", "exclude_tags"):
        value = getattr(scope, field_name)
        if value is not None and not all(isinstance(t, str) for t in value):
            raise TypeError(f"trial_scope.{field_name} must hold strings")
        tags[field_name] = None if value is None else tuple(sorted(set(value)))
    return TrialScope(name=scope.name, **tags)


@dataclass(frozen=True, kw_only=True)
class GateConfig:
    """Everything the gate decides with, fixed before it runs.

    Dates may be given as anything `pd.Timestamp` reads; they are stored as
    ISO dates, so equal dates hash equally. `data_version` is recorded with
    the registration, as every ledger entry records its data (`CLAUDE.md`).
    `generator` names what produced the candidates (agent, model, prompt
    set): thresholds belong to a generator. `factors_digest` is the SHA-256
    hex digest of stage 3's factor frame, or None for no stage-3 test.

    `seed` and `n_boot` are integers (NumPy integers are stored as ints; a
    float or bool is refused, so a fractional value can't hash like its
    truncation). The trial scope's tags are stored sorted and without
    duplicates, so equal scopes hash equally.
    """

    search_end: str
    holdout_start: str
    holdout_end: str
    data_version: str
    generator: str
    thresholds: Thresholds = field(default_factory=Thresholds)
    n_boot: int = 1999
    seed: int = 0
    trial_scope: TrialScope = field(default_factory=TrialScope)
    factors_digest: str | None = None

    def __post_init__(self) -> None:
        dates = {}
        for name in ("search_end", "holdout_start", "holdout_end"):
            value = getattr(self, name)
            stamp = None if value is None else pd.Timestamp(value)
            if stamp is None or pd.isna(stamp):
                raise ValueError(f"{name} is required (got {value!r})")
            dates[name] = stamp.date()
            object.__setattr__(self, name, stamp.date().isoformat())
        # Compare the dates themselves: as strings a missing date ("NaT") sorts last.
        if not dates["search_end"] < dates["holdout_start"] <= dates["holdout_end"]:
            raise ValueError("need search_end < holdout_start <= holdout_end")
        for name in ("n_boot", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer, not {type(value).__name__}: {value!r}")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "trial_scope", _canonical_scope(self.trial_scope))
        if self.n_boot < 0 or (self.thresholds.stage1 == "bootstrap" and self.n_boot == 0):
            raise ValueError("n_boot must be positive for the bootstrap stage-1 gate")
        if not self.data_version:
            raise ValueError("data_version is required")
        if not isinstance(self.generator, str) or not self.generator.strip():
            raise ValueError("generator is required: thresholds belong to a generator")
        digest = self.factors_digest
        if digest is not None and not re.fullmatch("[0-9a-f]{64}", str(digest)):
            raise ValueError("factors_digest must be a SHA-256 hex digest or None")

    def to_dict(self) -> dict:
        """Plain JSON content: what is registered and hashed."""
        scope = self.trial_scope
        return {
            "search_end": self.search_end,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "data_version": self.data_version,
            "generator": self.generator,
            "thresholds": thresholds_dict(self.thresholds),
            "n_boot": self.n_boot,
            "seed": self.seed,
            "trial_scope": {
                "name": scope.name,
                "include_tags": None if scope.include_tags is None else list(scope.include_tags),
                "exclude_tags": list(scope.exclude_tags),
            },
            "factors_digest": self.factors_digest,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GateConfig:
        data = dict(data)
        data["thresholds"] = Thresholds(**data["thresholds"])
        data["trial_scope"] = TrialScope(**data["trial_scope"])
        return cls(**data)

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON of `to_dict`: stable across processes."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entries_for(name: str, config_hash: str) -> list[dict]:
    return [
        e
        for e in runlog.read_entries()
        if e.get("name") == name and (e.get("params") or {}).get("config_hash") == config_hash
    ]


def register(config: GateConfig) -> dict:
    """Pre-register `config` on the ledger; returns its registration entry.

    Registering the same content twice logs once and returns the first entry
    (the repeat is not recorded). The entry is tagged "gate" and "synthetic":
    it is bookkeeping, not a trial of a strategy.
    """
    existing = _entries_for(REGISTRATION_NAME, config.config_hash)
    if existing:
        return existing[0]
    return runlog.log_run(
        REGISTRATION_NAME,
        params={
            "config_hash": config.config_hash,
            "config": config.to_dict(),
            "data_version": config.data_version,
        },
        seed=config.seed,
        tags=[GATE_TAG, SYNTHETIC_TAG],
        notes="pre-registered gate config: fixed before scoring, not a trial",
    )


def _trial_count(scope: TrialScope) -> int:
    """The ledger's trial count for `scope`, never counting the gate's own entries."""
    return runlog.trial_count(
        scope.name,
        include_tags=None if scope.include_tags is None else list(scope.include_tags),
        exclude_tags=[*scope.exclude_tags, GATE_TAG],
    )


def required_n_boot(n_families: int, q: float = 0.05) -> int:
    """Smallest draw count resolving BH at `q` across `n_families` families.

    The bootstrap p-value's smallest value is 1 / (n_boot + 1), and BH needs
    it to reach q / n_families (`search_fdr.bootstrap_family_pvalues`).
    """
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be finite and in (0, 1)")
    return int(np.ceil(n_families / q)) - 1


def _checked_family(returns: pd.DataFrame, family: pd.Series) -> pd.Series:
    """`family` aligned to `returns`' columns; every column needs a label."""
    if not isinstance(returns, pd.DataFrame) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a DataFrame indexed by date")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns' dates must be increasing")
    labels = family.reindex(returns.columns)
    if labels.isna().any():
        missing = labels.index[labels.isna()]
        raise ValueError(f"family is missing for {len(missing)} columns, e.g. {missing[0]!r}")
    return labels


def _checked_search(returns: pd.DataFrame, family: pd.Series, config: GateConfig) -> pd.Series:
    """The checks a run makes on its search-period input before it scores anything."""
    labels = _checked_family(returns, family)
    if returns.empty:
        raise ValueError("returns have no dates")
    last = returns.index.max()
    if last > pd.Timestamp(config.search_end):
        raise ValueError(
            f"returns run to {last.date()}, past search_end {config.search_end}: the gate "
            "takes the search period only"
        )
    th = config.thresholds
    n_families = labels.nunique()
    if th.stage1 == "bootstrap" and config.n_boot < required_n_boot(n_families, th.q1):
        raise ValueError(
            f"n_boot={config.n_boot} is too small for {n_families} families: the bootstrap "
            f"stage 1 needs n_boot + 1 >= families / q1 (n_boot >= "
            f"{required_n_boot(n_families, th.q1)}); register a config with more draws"
        )
    return labels
