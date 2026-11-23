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

`run_gate` is the run itself:

1. **Registered, once.** It refuses a config whose hash has no registration
   entry (`GateNotRegistered`; a config changed after registration has a
   different hash, so it is refused too) and a dataset whose holdout was
   already looked at (`HoldoutAlreadyOpened`). The look is a
   `gate-holdout-look` ledger entry written before the holdout provider is
   called, and it is written even when no family reaches stage 4: otherwise a
   screen that passed nothing could be repeated on another candidate set
   under the same registration until something did. Looks are read from the
   raw ledger, so a ledger correction cannot undo one.

   The budget is per **dataset**, not per config: see `GateConfig.look_key`.
   Re-registering changes a threshold; it does not buy a look.
2. **Authorised.** The caller names who authorises the look (`authorizer`, a
   non-blank string). It is checked before anything is read and recorded in
   the look entry. There is no default and no inferred authorizer: who may
   authorise the real holdout is a team decision (DECISIONS D-5).
3. **Blind.** Stages 1-3 get the search period only (a date after
   `search_end` is refused). The holdout comes from the caller's `holdout`
   provider, asked once, after the look is logged, for the stage-3 survivors'
   chosen variants and the registered holdout dates. Checks that can fail
   (inputs, bootstrap size, trial count, stages 1-3, graded scores) run
   before the look, so a failure there does not spend it.
   The provider's answer is refused unless it is exactly those columns on
   dates in [`holdout_start`, `holdout_end`], finite (the look stays spent).
   A survivor with no holdout returns fails stage 4.
4. **Graded scores never decide.** `grading` computes them from the search
   period; nothing reads them back.
5. **The decision rule decides `accepted`.** The config carries a `DecisionRule`
   (a hashed field; `load_registered` reads the registered one): the baseline
   is BH at its q on the stage-1 bootstrap p-values, the funnel is every stage
   passing. The funnel's verdict stays on the card as `funnel_pass` either way.
6. **Stage 3 is not measured yet:** it passes stage 2's survivors through and
   the scorecard says so (`s3_measured` False). A config that registers a
   factor frame is refused until the spanning test is connected.

The gate's own ledger entries carry the tags "gate" and "synthetic" and are
never counted as trials (`_trial_count`).
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import numbers
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from capstone import funnel, grading, holdout_authority, runlog
from capstone.evaluate import average_correlation, bh_adjusted
from capstone.funnel import Thresholds, thresholds_dict
from capstone.scorecard import (
    SCORECARD_COLUMNS,
    DecisionRule,
    GateHeader,
    GateResult,
    TrialScope,
)

__all__ = [
    "DEFAULT_MIN_HOLDOUT_COVERAGE",
    "GATE_TAG",
    "HOLDOUT_OPENS",
    "LOOK_NAME",
    "REGISTRATION_NAME",
    "SYNTHETIC_DATA_VERSION",
    "SYNTHETIC_TAG",
    "DecisionRule",
    "GateConfig",
    "GateNotRegistered",
    "HoldoutAlreadyOpened",
    "HoldoutProvider",
    "TrialScope",
    "load_registered",
    "register",
    "holdout_looks",
    "required_holdout_obs",
    "required_n_boot",
    "run_gate",
]

REGISTRATION_NAME = "gate-registration"
LOOK_NAME = "gate-holdout-look"
GATE_TAG = "gate"
# runlog's convention for entries that are not trials of a strategy.
SYNTHETIC_TAG = "synthetic"
# The only `data_version` allowed to search past HOLDOUT_OPENS: simulated data.
SYNTHETIC_DATA_VERSION = "synthetic"
# The first day of the real holdout (`docs/validation.md`, "Holdout | 2021-01-01
# to 2025-12-31 | The final stage only"). Real data's search period must end
# before it. One constant: the docs name it, and a test keeps them equal.
HOLDOUT_OPENS = datetime.date(2021, 1, 1)
# Stage 4's minimum holdout coverage per survivor (`GateConfig.min_holdout_coverage`).
DEFAULT_MIN_HOLDOUT_COVERAGE = 0.5


# Stage 4's source of holdout returns: (variant ids, holdout_start, holdout_end)
# -> dates x those variants, on holdout dates only. The gate ships no provider:
# the caller supplies one, and it is called only after the look is logged.
HoldoutProvider = Callable[[Sequence[str], pd.Timestamp, pd.Timestamp], pd.DataFrame]


class GateNotRegistered(RuntimeError):
    """`run_gate` was given a config whose hash is not on the ledger."""


class HoldoutAlreadyOpened(RuntimeError):
    """This dataset's holdout has been looked at already (`GateConfig.look_key`)."""


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

    Unless `data_version` is "synthetic", `search_end` must be before
    `HOLDOUT_OPENS` (2021-01-01): the real holdout is for the final stage only,
    so a config that searches into it is refused when it is built, before it
    can be registered or run.

    `min_holdout_coverage` is the share of the registered holdout's business
    days (`holdout_start` to `holdout_end`) on which a survivor's chosen
    variant must have a return, or it fails stage 4 untested, staying in the
    stage-4 family at p = 1. The default is one half, for three reasons. The
    thresholds were calibrated on a full-length holdout, and a variant seen
    on a fraction of it is a shorter, different test: the Sharpe's standard
    error grows as 1 / sqrt(n), so half the days is already 41% wider, and
    below that the p-value is mostly noise (a survivor seen on 3 of 756
    days reported a Sharpe of 18 and p = 0.02). A strategy can be out of
    the market on many days (NaN is a missing return, never a zero), so
    demanding every day would fail honest, selective strategies. And
    business days overstate a trading calendar by about 3.5% (holidays), so a
    bar at half leaves room for that. It is a guard, not a calibrated value.
    It is in the hashed config, so changing it re-registers the gate. At
    least 2 returns are always needed (`funnel.MIN_PERIOD_OBS`).

    `decision_rule` is which column decides `accepted` (the baseline, BH at
    its q on the bootstrap family p-values, or the funnel). It is in the
    hashed config, so changing the rule or its q re-registers the gate.

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
    thresholds: Thresholds = field(default_factory=lambda: Thresholds(stage1="bootstrap"))
    n_boot: int = 1999
    seed: int = 0
    trial_scope: TrialScope = field(default_factory=TrialScope)
    factors_digest: str | None = None
    min_holdout_coverage: float = DEFAULT_MIN_HOLDOUT_COVERAGE
    decision_rule: DecisionRule = field(default_factory=lambda: DecisionRule("funnel"))
    holdout_round: int = 0

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
        if self.data_version != SYNTHETIC_DATA_VERSION and dates["search_end"] >= HOLDOUT_OPENS:
            raise ValueError(
                f"search_end {dates['search_end']} is not before {HOLDOUT_OPENS}, where the real "
                f"holdout begins: stages 1-3 would score it. Only data_version="
                f"{SYNTHETIC_DATA_VERSION!r} may search that late (docs/validation.md)"
            )
        for name in ("n_boot", "seed", "holdout_round"):
            value = getattr(self, name)
            if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer, not {type(value).__name__}: {value!r}")
            object.__setattr__(self, name, int(value))
        if self.holdout_round < 0:
            raise ValueError(f"holdout_round must be 0 or more, not {self.holdout_round}")
        coverage = self.min_holdout_coverage
        if isinstance(coverage, bool | np.bool_) or not isinstance(coverage, numbers.Real):
            raise TypeError(f"min_holdout_coverage must be a real number, not {coverage!r}")
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError(f"min_holdout_coverage must be in (0, 1], got {coverage!r}")
        object.__setattr__(self, "min_holdout_coverage", float(coverage))
        if not isinstance(self.decision_rule, DecisionRule):
            raise TypeError(
                f"decision_rule must be a DecisionRule, not {type(self.decision_rule).__name__}"
            )
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
        out = {
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
            "min_holdout_coverage": self.min_holdout_coverage,
            "decision_rule": {
                "rule": self.decision_rule.rule,
                "q": self.decision_rule.q,
                "method": self.decision_rule.method,
            },
        }
        # A field added after a config was registered is written only when it
        # differs from its default, so an earlier registration keeps its
        # content and therefore its hash (`funnel.thresholds_dict` does the
        # same). crsp-v1's registered 5f590f13... must not move.
        if self.holdout_round:
            out["holdout_round"] = self.holdout_round
        return out

    @classmethod
    def from_dict(cls, data: dict) -> GateConfig:
        data = dict(data)
        data["thresholds"] = Thresholds(**data["thresholds"])
        data["trial_scope"] = TrialScope(**data["trial_scope"])
        if "decision_rule" in data:
            data["decision_rule"] = DecisionRule.from_dict(data["decision_rule"])
        return cls(**data)

    @property
    def dataset(self) -> str:
        """The holdout this config reads, in `holdout_authority`'s namespace.

        `REAL_DATASET` for real data, `"synthetic:<id>"` for simulated. Derived
        from `data_version` rather than stored, so adding it moves no hash.

        **Every real `data_version` maps to the one real dataset.** There is a
        single real holdout, 2021-2025, so it gets a single budget.
        `sd.data_version()` hashes the whole shared manifest, so re-pulling a
        1995 row or adding a Compustat column mints a new version while not one
        holdout row changes; keying on it would reopen the holdout for a routine
        refresh (@CPH200 on #360).
        """
        version = self.data_version
        if version == SYNTHETIC_DATA_VERSION:
            return f"{SYNTHETIC_DATA_VERSION}:default"
        if version.startswith(f"{SYNTHETIC_DATA_VERSION}:"):
            return version
        return holdout_authority.REAL_DATASET

    @property
    def look_key(self) -> str:
        """What a holdout look is spent on: `"<dataset>#<holdout_round>"`.

        A look is spent on the **evidence**, so the budget is counted per
        holdout, not per configuration and not per label.

        Within one key, a window already spent refuses any window overlapping
        it (`_overlapping`), so re-slicing 2021-2025 or nudging `holdout_start`
        by a day buys nothing, while a genuinely later window -- 2026 rows that
        did not exist before -- opens a look. The dates are known before any
        candidate is read, which is what lets the check run before the holdout
        is touched.

        Two earlier keys were wrong, both measured on signal-free data:

        * **`config_hash`** (`daccc5d`) made every hashed field a way to buy a
          look. Four looks from nothing but `seed` 0, 1, 2, 3; two accepted a
          family. The seed moves the bootstrap draws and nothing else.
        * **`data_version`** (`c7a569f`, the first attempt at this fix) moved
          the hole rather than closing it: the field is free text that nothing
          verified, so `"synthetic "`, `"Synthetic"` and `"synthetic-recheck"`
          each bought a look on identical data (@CPH200 on #360). Real
          `data_version`s now map to one dataset, and `run_gate` checks a real
          config's version against `sd.data_version()` so the recorded
          provenance is true.

        Only two things open a look:

        * **a later holdout window** under the same dataset -- genuinely new
          dates are genuinely new evidence;
        * **`holdout_round`** -- a second look at the *same* window, declared on
          purpose, and allowed only once the round before it has been spent. It
          exists because without it an honest second round has no sanctioned
          path, and people would reach for a seed change instead. A bump is
          visible in the registration, the look entry and the refusal; a changed
          seed is not.

        Everything else is excluded on purpose: `seed` and `n_boot` are
        computational; `thresholds`, `decision_rule` and `min_holdout_coverage`
        are the pre-registered rules, and changing one after a disappointing
        holdout is the move the funnel exists to prevent; `trial_scope`
        narrowing shrinks the trial count as well; `generator` and
        `factors_digest` are provenance the gate cannot verify.

        The candidate set is in neither the key nor the hash, so the same
        config with a fresh batch is refused: a screen that passed nothing
        cannot be retried until something passes. That is also why the holdout
        rows themselves cannot be the key -- they are read after the look is
        logged, and they depend on which candidates survived (@lyz1103 on #360).
        """
        return f"{self.dataset}#{self.holdout_round}"

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


def _dataset_of_version(version: object) -> str:
    """`GateConfig.dataset`'s rule, for a recorded `data_version` string."""
    text = str(version)
    if text == SYNTHETIC_DATA_VERSION:
        return f"{SYNTHETIC_DATA_VERSION}:default"
    if text.startswith(f"{SYNTHETIC_DATA_VERSION}:"):
        return text
    return holdout_authority.REAL_DATASET


def _look_key_of(entry: dict) -> str | None:
    """A look entry's key, deriving it for one written before `look_key` existed.

    An older entry carries `data_version` but no `look_key`, and predates
    `holdout_round`, so it is round 0 of whatever dataset its version maps to.
    Deriving it makes the budget retroactive: a look already spent still counts.

    `None` only when the entry records no dataset and no version at all, which
    `_looks_for` then treats as spending every key rather than nothing: an
    unattributable look is the one case where guessing wrong must fail closed.
    """
    params = entry.get("params") or {}
    key = params.get("look_key")
    if key is not None:
        return str(key)
    dataset = params.get("dataset")
    if dataset is None:
        version = params.get("data_version")
        if version is None:
            return None
        dataset = _dataset_of_version(version)
    return f"{dataset}#{params.get('holdout_round', 0)}"


def holdout_looks() -> list[dict]:
    """Every holdout look on the raw ledger, oldest first, with its key.

    The audit the budget rests on. Read from `runlog.read_entries` rather than
    the corrected view, so a ledger correction cannot retire a look.
    """
    looks = []
    for entry in runlog.read_entries():
        if entry.get("name") != LOOK_NAME:
            continue
        params = entry.get("params") or {}
        looks.append(
            {
                "look_key": _look_key_of(entry),
                "dataset": params.get("dataset"),
                "data_version": params.get("data_version"),
                "holdout_round": params.get("holdout_round", 0),
                "holdout_start": params.get("holdout_start"),
                "holdout_end": params.get("holdout_end"),
                "config_hash": params.get("config_hash"),
                "authorizer": params.get("authorizer"),
                "ts_utc": entry.get("ts_utc"),
            }
        )
    return looks


def _looks_for(look_key: str) -> list[dict]:
    """Looks already spent under `look_key`, plus any that name no dataset.

    An entry with neither a key nor a version cannot be attributed, so it
    counts against every key. Failing closed is the only safe reading: the
    alternative hands out a look because a record is incomplete.
    """
    return [
        look for look in holdout_looks() if look["look_key"] == look_key or look["look_key"] is None
    ]


def _overlapping(looks: list[dict], start: str, end: str) -> dict | None:
    """The first look whose holdout window overlaps [`start`, `end`], if any.

    Two windows overlap when each starts on or before the other ends. A look
    with no window recorded is from before windows were logged, so it is taken
    to cover whatever window is asked for: fail closed.
    """
    asked_start, asked_end = pd.Timestamp(start), pd.Timestamp(end)
    for look in looks:
        if look["holdout_start"] is None or look["holdout_end"] is None:
            return look
        spent_start = pd.Timestamp(look["holdout_start"])
        spent_end = pd.Timestamp(look["holdout_end"])
        if spent_start <= asked_end and asked_start <= spent_end:
            return look
    return None


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


def _checked_provenance(config: GateConfig) -> None:
    """A real config's `data_version` must be the live one.

    `data_version` is no longer the look key, but it is still the provenance
    every ledger entry records (`CLAUDE.md`), and a label nothing verifies is
    a label nothing can rely on. Checked only for the real dataset: a
    synthetic config names a simulated world that `shared_data` knows nothing
    about, and the check must not require a CRSP copy to run the tests.
    """
    if config.dataset != holdout_authority.REAL_DATASET:
        return
    from capstone import shared_data

    try:
        live = shared_data.data_version()
    except shared_data.SharedDataError as exc:
        raise ValueError(
            f"this config names real data ({config.data_version!r}) but the shared data "
            f"is not available to confirm it: {exc}"
        ) from exc
    if config.data_version != live:
        raise ValueError(
            f"data_version {config.data_version!r} is not the local shared data "
            f"({live!r}). Register the version actually being scored: the gate records "
            "it as provenance, so a label that is not true makes the record useless."
        )


def _checked_round_order(config: GateConfig) -> None:
    """Round *k* is available only once round *k-1* has been spent.

    Otherwise a first run could register round 7 and keep six untouched looks
    in reserve, which is the budget it was meant to bound (@CPH200 on #360).
    """
    if config.holdout_round == 0:
        return
    previous = f"{config.dataset}#{config.holdout_round - 1}"
    if not _looks_for(previous):
        raise ValueError(
            f"holdout_round {config.holdout_round} needs round "
            f"{config.holdout_round - 1} to have been spent first; the ledger holds no "
            f"look for {previous}. Rounds are consecutive so the count of looks taken "
            "is the count declared."
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


def load_registered(path: str | Path) -> GateConfig:
    """A registered file's config (`scripts/gate_crsp_v1.json`).

    The file's `gate_config` is what `register` hashes, including its
    `decision_rule` (which column decides `accepted`).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GateConfig.from_dict(data["gate_config"])


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


def _checked_authorizer(authorizer: object) -> str:
    """Who authorises this look at the holdout: a named person, never a default."""
    if not isinstance(authorizer, str) or not authorizer.strip():
        raise ValueError(
            "authorizer is required: name the person who authorises this look at the "
            "holdout (it is recorded with the look)"
        )
    return authorizer.strip()


def _checked_holdout(frame: object, ids: tuple, config: GateConfig) -> pd.DataFrame:
    """The provider's answer: exactly the requested columns, on holdout dates only.

    The look is already logged when this runs, so a bad answer leaves it
    spent; the messages say so.
    """
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("the holdout provider must return a DataFrame indexed by date")
    if not frame.columns.is_unique or set(frame.columns) != set(ids):
        raise ValueError(
            f"the holdout provider must return exactly the requested columns {list(ids)!r}, "
            f"got {list(frame.columns)!r}; the look is spent"
        )
    start, end = pd.Timestamp(config.holdout_start), pd.Timestamp(config.holdout_end)
    outside = (frame.index < start) | (frame.index > end)
    if outside.any():
        raise ValueError(
            f"the holdout provider returned {int(outside.sum())} dates outside the holdout "
            f"[{config.holdout_start}, {config.holdout_end}], e.g. "
            f"{frame.index[outside][0].date()}; the look is spent"
        )
    if not (frame.index.is_monotonic_increasing and frame.index.is_unique):
        raise ValueError("holdout dates must be strictly increasing; the look is spent")
    if np.isinf(frame.to_numpy(dtype=np.float64)).any():
        raise ValueError("holdout returns must be finite (NaN for missing); the look is spent")
    return frame[list(ids)]


def required_holdout_obs(config: GateConfig) -> int:
    """Returns a survivor needs inside the holdout: `min_holdout_coverage` of its
    business days, rounded up, and never fewer than `funnel.MIN_PERIOD_OBS`."""
    n_days = len(pd.bdate_range(config.holdout_start, config.holdout_end))
    return max(funnel.MIN_PERIOD_OBS, math.ceil(config.min_holdout_coverage * n_days))


def _checked_stage3(config: GateConfig) -> None:
    if config.factors_digest is not None:
        raise ValueError(
            "this config registered a factor frame (factors_digest) but the gate has no "
            "stage-3 test yet: register it with factors_digest=None"
        )


def run_gate(
    config: GateConfig,
    returns: pd.DataFrame,
    family: pd.Series,
    *,
    holdout: HoldoutProvider,
    authorizer: str,
) -> GateResult:
    """Run the registered gate once.

    Args:
        config: a config registered with `register`.
        returns: dates x variants (a `FunnelInput.returns`), the search period
            only: a date after `search_end` raises.
        family: hypothesis family per variant column.
        holdout: stage 4's source of holdout returns (`HoldoutProvider`):
            called at most once, after the look is logged, with the stage-3
            survivors' chosen variants and (holdout_start, holdout_end). Not
            called when nothing reaches stage 4.
        authorizer: who authorises the look at the holdout; required, recorded
            in the look entry.
        (The config's `decision_rule` decides `accepted`: the baseline is BH
        on the stage-1 bootstrap p-values at its q; the funnel is every
        stage passing. The other is reported (`funnel_pass`, the stage
        columns). It is in the config hash, and recorded in the look entry
        and the header.)

    Raises:
        ValueError: no authorizer, bad inputs, returns past `search_end`, too
            few bootstrap draws, a registered factor frame (no stage-3 test
            yet), or fewer ledger trials in scope than variants in `returns`
            (an unlogged search). All raised before the holdout is opened.
        GateNotRegistered: no registration entry has this config's hash.
        HoldoutAlreadyOpened: the holdout was looked at under this hash.
        LookupError: from `runlog.trial_count`, when the scope matches nothing.
    """
    authorizer = _checked_authorizer(authorizer)
    decision_rule = config.decision_rule
    if decision_rule.rule == "baseline" and config.thresholds.stage1 != "bootstrap":
        raise ValueError(
            "the baseline decides on the bootstrap family p-values: register stage1='bootstrap' "
            f"(this config has {config.thresholds.stage1!r})"
        )
    h = config.config_hash
    registration = _entries_for(REGISTRATION_NAME, h)
    if not registration:
        raise GateNotRegistered(f"gate config {h[:12]} is not registered; call register first")
    _checked_provenance(config)
    _checked_round_order(config)
    spent = _overlapping(_looks_for(config.look_key), config.holdout_start, config.holdout_end)
    if spent is not None:
        window = f"{spent['holdout_start']} to {spent['holdout_end']}"
        raise HoldoutAlreadyOpened(
            f"{config.dataset} round {config.holdout_round} was already opened over "
            f"{window} on {spent['ts_utc']} by {spent['authorizer']} under config "
            f"{str(spent['config_hash'])[:12]}, which overlaps the requested "
            f"{config.holdout_start} to {config.holdout_end}. A look is spent on the "
            "holdout, not on the configuration: the seed, the draws, a threshold, the "
            "decision rule, the trial scope, the generator and the data label do not "
            "open it again, and neither does re-slicing the same dates. To look at "
            f"this window a second time on purpose, register holdout_round "
            f"{config.holdout_round + 1} and say why."
        )
    labels = _checked_search(returns, family, config)
    _checked_stage3(config)
    th = config.thresholds
    n_trials = _trial_count(config.trial_scope)
    if n_trials < returns.shape[1]:
        raise ValueError(
            f"the ledger has {n_trials} trials in scope but {returns.shape[1]} variants were "
            "scored: log every trial before gating"
        )

    # Stages 1-3: the search period, which is all `returns` holds.
    search = returns
    stats = funnel.family_statistics(
        search,
        labels,
        search_end=config.search_end,
        n_boot=config.n_boot,
        seed=config.seed,
        q=th.q1,
    )["reuse"]
    stages = funnel.screen(stats, th)
    p1 = funnel.stage1_pvalues(stats, th.stage1)
    avg_corr = average_correlation(search) if search.shape[1] > 1 else float("nan")
    # Graded scores and batch diagnostics read only the search period, so they
    # are computed before the look (a failure here doesn't spend it).
    graded = grading.graded_scores(search, stats["best"], p1, n_trials, avg_corr, config.seed)
    batch_pbo = grading.pbo(search)

    # The look: logged before the provider is asked for anything.
    chosen = stats["best"].where(stages["stage3"])
    ids = tuple(chosen.dropna())
    runlog.log_run(
        LOOK_NAME,
        params={
            "config_hash": h,
            "look_key": config.look_key,
            "dataset": config.dataset,
            "holdout_round": config.holdout_round,
            "holdout_start": config.holdout_start,
            "holdout_end": config.holdout_end,
            "generator": config.generator,
            "data_version": config.data_version,
            "authorizer": authorizer,
            "decision_rule": dataclasses.asdict(decision_rule),
            "candidates": [str(b) for b in ids],
        },
        seed=config.seed,
        tags=[GATE_TAG, SYNTHETIC_TAG],
        notes="stage 4 opens the holdout under this config; it may not be opened again",
    )
    if ids:
        start, end = pd.Timestamp(config.holdout_start), pd.Timestamp(config.holdout_end)
        ho_returns = _checked_holdout(holdout(ids, start, end), ids, config)
    else:
        ho_returns = pd.DataFrame(index=pd.DatetimeIndex([]))
    # Every stage-3 survivor reaches stage 4. One whose chosen variant has too
    # few holdout returns (`required_holdout_obs`: none, or a sliver of the
    # window) is not tested: it fails, and stays in the family at p = 1.
    alive = stages["stage3"]
    ho_n_obs = pd.Series(
        {f: float(ho_returns[b].notna().sum()) for f, b in chosen.dropna().items()}, dtype=float
    )
    covered = (ho_n_obs >= required_holdout_obs(config)).reindex(stages.index, fill_value=False)
    tested = stats["best"].where(alive & covered)
    ho = funnel.holdout_statistics(ho_returns, tested)
    for column in ho.columns:
        stats[column] = ho[column]
    s4 = alive & funnel._final_gate(stats, alive, th, covered=covered)
    ho_mean = pd.Series({f: ho_returns[b].mean() for f, b in tested.dropna().items()}, dtype=float)

    card = pd.DataFrame(index=stats.index)
    card["best"] = stats["best"]
    card["k"] = labels.value_counts().reindex(card.index)
    card["generator"] = config.generator
    card["s1_stat"] = bh_adjusted(p1)
    card["s1_threshold"] = th.q1
    card["s1_pass"] = stages["stage1"]
    card["s2_stat"] = stats["p2"]
    card["s2_threshold"] = th.alpha2
    card["s2_pass"] = stages["stage2"]
    card["s3_stat"] = np.nan  # not measured: stage 3 passes stage 2's survivors
    card["s3_threshold"] = np.nan
    card["s3_pass"] = stages["stage3"]
    card["s3_measured"] = False
    card["s3_alpha"] = np.nan
    card["s3_t"] = np.nan
    card["s4_stat"] = grading.stage4_stat(stats, alive, th)
    card["s4_threshold"] = th.q4 if th.final == "bh" else th.d4
    card["s4_pass"] = s4
    funnel_pass = stages["stage1"] & stages["stage2"] & stages["stage3"] & s4
    if decision_rule.rule == "baseline":
        accepted_rows = card["s1_stat"] <= decision_rule.q  # BH-adjusted p; NaN is False
    else:
        accepted_rows = funnel_pass
    card["accepted"] = accepted_rows.astype(bool)
    card["funnel_pass"] = funnel_pass
    for column in graded.columns:
        card[column] = graded[column]
    card["ho_sharpe"] = stats["ho_sharpe"]
    card["ho_p"] = stats["ho_p"]
    card["ho_psr"] = 1.0 - stats["ho_p"]  # sharpe_test: pvalue_greater = 1 - psr
    card["ho_n_obs"] = ho_n_obs.reindex(card.index)  # every survivor, tested or not
    card["is_mean"] = [search[b].mean() for b in card["best"]]
    card["ho_mean"] = ho_mean.reindex(card.index)
    card["scale"] = [search[b].std() for b in card["best"]]
    card["expected_mean"] = grading.expected_mean(card).where(card["accepted"])
    card.index.name = "family"
    card = card[list(SCORECARD_COLUMNS)]

    header = GateHeader(
        config_hash=h,
        registered_utc=registration[0]["ts_utc"],
        generator=config.generator,
        data_version=config.data_version,
        factors_digest=config.factors_digest,
        thresholds=th,
        n_boot=config.n_boot,
        seed=config.seed,
        trial_scope=config.trial_scope,
        search_end=config.search_end,
        holdout_start=config.holdout_start,
        holdout_end=config.holdout_end,
        trial_count=n_trials,
        n_families=len(card),
        n_variants=returns.shape[1],
        n_search_obs=len(search),
        n_holdout_obs=int(ho_returns.notna().any(axis=1).sum()),
        search_fdr_estimate=grading.search_fdr_estimate(p1, stages["stage1"]),
        pbo=batch_pbo,
        avg_correlation=float(avg_corr),
        min_holdout_coverage=config.min_holdout_coverage,
        min_holdout_obs=required_holdout_obs(config),
        decision_rule=decision_rule,
    )
    accepted = tuple(card.index[card["accepted"]])
    return GateResult(card, header, accepted)
