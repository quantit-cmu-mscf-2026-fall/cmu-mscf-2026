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
   different hash, so it is refused too) and a hash whose holdout was already
   looked at (`HoldoutAlreadyOpened`). The look is a `gate-holdout-look`
   ledger entry written before the holdout provider is called, and it is
   written even when no family reaches stage 4: otherwise a screen that
   passed nothing could be repeated on another candidate set under the same
   registration until something did. Looks are read from the raw ledger, so a
   ledger correction cannot undo one.
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
5. **Stage 3 is not measured yet:** it passes stage 2's survivors through and
   the scorecard says so (`s3_measured` False). A config that registers a
   factor frame is refused until the spanning test is connected.

The gate's own ledger entries carry the tags "gate" and "synthetic" and are
never counted as trials (`_trial_count`).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import numbers
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capstone import funnel, grading, runlog
from capstone.evaluate import average_correlation, bh_adjusted
from capstone.funnel import Thresholds, thresholds_dict
from capstone.scorecard import SCORECARD_COLUMNS, GateHeader, GateResult, TrialScope

__all__ = [
    "GATE_TAG",
    "HOLDOUT_OPENS",
    "LOOK_NAME",
    "REGISTRATION_NAME",
    "SYNTHETIC_DATA_VERSION",
    "SYNTHETIC_TAG",
    "GateConfig",
    "GateNotRegistered",
    "HoldoutAlreadyOpened",
    "HoldoutProvider",
    "TrialScope",
    "register",
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


# Stage 4's source of holdout returns: (variant ids, holdout_start, holdout_end)
# -> dates x those variants, on holdout dates only. The gate ships no provider:
# the caller supplies one, and it is called only after the look is logged.
HoldoutProvider = Callable[[Sequence[str], pd.Timestamp, pd.Timestamp], pd.DataFrame]


class GateNotRegistered(RuntimeError):
    """`run_gate` was given a config whose hash is not on the ledger."""


class HoldoutAlreadyOpened(RuntimeError):
    """The holdout has been looked at under this config hash already."""


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
    h = config.config_hash
    registration = _entries_for(REGISTRATION_NAME, h)
    if not registration:
        raise GateNotRegistered(f"gate config {h[:12]} is not registered; call register first")
    if _entries_for(LOOK_NAME, h):
        raise HoldoutAlreadyOpened(f"the holdout was already opened under gate config {h[:12]}")
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
            "generator": config.generator,
            "data_version": config.data_version,
            "authorizer": authorizer,
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
    # A survivor whose chosen variant has no holdout returns can't be tested:
    # it fails stage 4.
    has_holdout = pd.Series(
        {f: bool(ho_returns[b].notna().any()) for f, b in chosen.dropna().items()}, dtype=bool
    )
    alive = stages["stage3"] & has_holdout.reindex(stages.index, fill_value=False)
    tested = stats["best"].where(alive)
    ho = funnel.holdout_statistics(ho_returns, tested)
    for column in ho.columns:
        stats[column] = ho[column]
    s4 = alive & funnel._final_gate(stats, alive, th)
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
    card["accepted"] = stages["stage1"] & stages["stage2"] & stages["stage3"] & s4
    for column in graded.columns:
        card[column] = graded[column]
    card["ho_sharpe"] = stats["ho_sharpe"]
    card["ho_p"] = stats["ho_p"]
    card["ho_psr"] = 1.0 - stats["ho_p"]  # sharpe_test: pvalue_greater = 1 - psr
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
        n_holdout_obs=len(ho_returns),
        search_fdr_estimate=grading.search_fdr_estimate(p1, stages["stage1"]),
        pbo=batch_pbo,
        avg_correlation=float(avg_corr),
    )
    accepted = tuple(card.index[card["accepted"]])
    return GateResult(card, header, accepted)
