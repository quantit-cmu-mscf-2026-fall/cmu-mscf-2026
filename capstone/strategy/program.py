"""General finite-history factor programs using the existing factor interpreter.

Schema six describes construction, not a paper allowlist or financial admission.
This version supports one fixed tail book with membership exits or a version-2
calendar window exiting to cash; cohorts, event portfolios, execution models,
and unsupported factor operations fail explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.factors import interpret, research_contracts, tree
from capstone.factors.research_contracts import canonical, digest
from capstone.strategy import contract, input_memo, program_schedule, program_types
from capstone.strategy.program_schedule import (
    CalendarWindowSchedule,
    ProgramSchedule,
    load_schedule,
    session_date,
)

CONSTRUCTIONS = ("fixed_tails_v1",)


def implementation_digest():
    """Bind execution dependencies, not unrelated paper catalogues or claim names."""
    files = {
        "program": Path(__file__),
        "schedule": Path(program_schedule.__file__),
        "schedule_dispatch": Path(__file__).with_name("schedule_dispatch.py"),
        "unit_types": Path(program_types.__file__),
        "tree": Path(tree.__file__),
        "interpreter": Path(interpret.__file__),
        "lookback": Path(contract.__file__),
        "identity": Path(research_contracts.__file__),
        "recorded_compatibility": Path(__file__).with_name("recorded.py"),
    }
    return digest(
        {
            "files": {
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
            },
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        }
    )


@dataclass(frozen=True)
class ProgramStrategySpec:
    expression: str
    source_ids: tuple[str, ...]
    method_spec_id: str
    claim_id: str
    mechanism_id: str
    data_version: str
    anchor: str
    side: str = "long_only"
    tail: float = 0.2
    weighting: str = "equal"
    schedule: ProgramSchedule = field(default_factory=ProgramSchedule)
    direction: str = "high"
    construction: str = "fixed_tails_v1"
    availability: str = "completed_session"
    execution: str = "following_session_open"
    exit_rule: str = "membership_or_rebalance"
    universe_rule: str = "point_in_time_members"
    gross: float = 1.0
    implementation: str = ""
    type_evidence_json: str = field(default="", init=False)
    gate_report: str = ""

    def __post_init__(self):
        node = tree.parse(self.expression)
        if "ema" in tree.funcs_used(node):
            raise ValueError("EMA requires an explicit infinite-memory construction version")
        type_evidence = canonical(asdict(program_types.infer_types(node)))
        object.__setattr__(self, "type_evidence_json", type_evidence)
        object.__setattr__(self, "expression", tree.unparse(node))
        if not isinstance(self.source_ids, (tuple, list)) or not self.source_ids:
            raise ValueError("at least one immutable source revision is required")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (*self.source_ids, self.method_spec_id, self.claim_id)
        ):
            raise ValueError("source, claim and method identities must be SHA-256 digests")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("duplicate source revisions do not create another program")
        object.__setattr__(self, "source_ids", tuple(sorted(self.source_ids)))
        for name in ("mechanism_id", "data_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError(f"{name} must be a nonempty bounded identity")
        if not isinstance(self.gate_report, str):
            raise ValueError("gate_report must be a string; it does not confer admission")
        object.__setattr__(self, "anchor", str(session_date(self.anchor, "anchor").date()))
        if (
            type(self.side) is not str
            or self.side not in {"long_only", "long_short"}
            or type(self.direction) is not str
            or self.direction not in {"high", "low"}
        ):
            raise ValueError("unsupported portfolio side or score direction")
        if (
            isinstance(self.tail, bool)
            or not isinstance(self.tail, (float, int))
            or not (0 < self.tail <= 0.5 or self.tail == 1)
            or not math.isfinite(self.tail)
        ):
            raise ValueError(
                "tail must be a finite proportion in (0, .5], or 1 for a whole universe"
            )
        if type(self.weighting) is not str or self.weighting not in {"equal", "capitalization"}:
            raise ValueError("unsupported weighting; no proxy weighting is substituted")
        if not isinstance(self.schedule, ProgramSchedule):
            raise ValueError("schedule must be a ProgramSchedule")
        if isinstance(self.schedule, CalendarWindowSchedule):
            raise ValueError("calendar_window schedules need the complete-history constructions")
        # The grammar has no constant score; tail one declares the whole eligible universe.
        if self.tail == 1:
            raise ValueError("whole-universe tail one needs a long-only high calendar window")
        if (
            type(self.construction) is not str
            or self.construction not in CONSTRUCTIONS
            or self.exit_rule != "membership_or_rebalance"
            or self.universe_rule != "point_in_time_members"
        ):
            raise ValueError("unsupported cohort, event, holding or exit construction")
        if self.availability != "completed_session" or self.execution != "following_session_open":
            raise ValueError("unsupported availability or execution timing")
        if isinstance(self.gross, bool) or self.gross != 1:
            raise ValueError("program profiles use fixed gross-one exposure")
        object.__setattr__(self, "tail", float(self.tail))
        object.__setattr__(self, "gross", 1.0)
        expected = implementation_digest()
        if self.implementation and self.implementation != expected:
            raise ValueError("program implementation changed; preserve and review a new version")
        object.__setattr__(self, "implementation", expected)

    @property
    def tree(self):
        return tree.parse(self.expression)

    @property
    def fields(self):
        required = set(tree.features_used(self.tree))
        if self.weighting == "capitalization":
            required.add("cap")
        return tuple(sorted(required))

    @property
    def type_evidence(self):
        return program_types.infer_types(self.tree)

    @property
    def factor_id(self):
        return tree.factor_id(self.tree)

    @property
    def construction_id(self):
        row = asdict(self)
        row["expression"] = tree.identity_key(self.tree)
        for name in (
            "source_ids",
            "method_spec_id",
            "claim_id",
            "mechanism_id",
            "gate_report",
        ):
            row.pop(name)
        return digest(row)[:16]

    @property
    def id(self):
        return digest(asdict(self))[:16]

    @property
    def variant_id(self):
        return self.id

    @property
    def hypothesis_id(self):
        # The controller assigns this stable allocation family, never a proposal alias.
        return self.mechanism_id

    def to_toml(self):
        return (
            f"schema_version = 6\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(canonical(asdict(self)))}\n"
        )

    @classmethod
    def from_toml(cls, text):
        row = tomllib.loads(text)
        if (
            set(row) != {"schema_version", "id", "payload"}
            or type(row["schema_version"]) is not int
            or row["schema_version"] != 6
        ):
            raise ValueError("invalid general program strategy")
        try:
            payload = json.loads(row["payload"])
            payload["schedule"] = load_schedule(payload["schedule"])
            supplied_types = payload.pop("type_evidence_json")
            spec = cls(**payload)
        except (TypeError, KeyError) as exc:
            raise ValueError("malformed general program specification") from exc
        if (
            supplied_types != spec.type_evidence_json
            or spec.id != row["id"]
            or text.replace("\r\n", "\n") != spec.to_toml()
        ):
            raise ValueError("general program specification changed")
        return spec


def _universe(panel, at):
    frame = panel.universe
    if frame.index.has_duplicates or frame.columns.has_duplicates or at not in frame.index:
        raise ValueError("distinct point-in-time membership is required at formation and decision")
    row = frame.loc[at]
    if row.isna().any() or not row.isin([True, False]).all():
        raise ValueError("membership must be explicitly known, never filled from future dates")
    return row.astype(bool).sort_index()


def _inputs(spec, panel, formation, dates, names):
    need = contract.lookback(spec.tree)
    at = dates.get_loc(formation)
    if at < need:
        raise contract.InsufficientHistoryError(
            f"program needs {need} earlier authoritative sessions"
        )
    window = dates[at - need : at + 1]
    historical = None
    columns = names
    if tree.funcs_used(spec.tree) & {"rank", "scale", "winsorize"}:
        if not window.isin(panel.universe.index).all():
            raise ValueError("historical cross-sectional membership has missing sessions")
        historical = panel.universe.reindex(window).sort_index(axis=1)
        if historical.isna().any().any() or not historical.isin([True, False]).all().all():
            raise ValueError("historical cross-sectional membership is unavailable")
        columns = historical.columns[historical.astype(bool).any(axis=0)].union(names).sort_values()
        historical = historical.reindex(columns=columns).astype(bool)
    inputs = {}
    for name in spec.fields:
        required = window if name in tree.features_used(spec.tree) else window[-1:]
        if name not in panel.fields or panel.availability is None or name not in panel.availability:
            raise ValueError(f"qualified feature and availability history required for {name}")
        frame = panel.fields[name]
        if frame.index.has_duplicates or not required.isin(frame.index).all():
            raise ValueError(f"missing or duplicate required observation sessions for {name}")
        if isinstance(frame, pd.DataFrame) and frame.columns.has_duplicates:
            raise ValueError(f"duplicate instrument observations for {name}")
        available = panel.availability[name]
        if available.index.has_duplicates or not required.isin(available.index).all():
            raise ValueError(f"required publication availability is missing for {name}")
        timestamps = pd.to_datetime(available.reindex(required))
        if timestamps.isna().any() or timestamps.dt.tz is not None:
            raise ValueError(f"{name} was unavailable at its formation decision")
        if not timestamps.equals(timestamps.dt.normalize()):
            raise ValueError(
                f"{name} availability precision requires certified completed-session date labels"
            )
        if (timestamps.to_numpy() > required.to_numpy()).any():
            raise ValueError(f"{name} was unavailable at its stated observation session")
        loaded = window if name in tree.features_used(spec.tree) else required
        values = frame.reindex(loaded)
        if isinstance(frame, pd.DataFrame):
            values = values.reindex(columns=columns)
        elif name not in tree.MARKET_FIELDS:
            raise ValueError(f"{name} requires per-instrument observations")
        values = values.astype(float)
        required_values = (
            values.reindex(index=required, columns=names)
            if isinstance(values, pd.DataFrame)
            else values.reindex(required)
        )
        if (
            name in tree.features_used(spec.tree)
            and not np.isfinite(required_values.to_numpy()).all()
        ):
            label = "capitalization" if name == "cap" else name
            raise ValueError(f"required {label} observations are missing or nonfinite")
        inputs[name] = values
    return inputs, historical, names, None


def require_runtime_capabilities(spec):
    """Check unsupported entity semantics before any target cache can act as authority."""
    if tree.features_used(spec.tree) & set(getattr(tree, "FUNDAMENTAL_FIELDS", ())):
        raise ValueError(
            "general accounting programs require qualified company/share-class alignment; "
            "use the reviewed company-aware construction route"
        )


class ExecutionIntegrationPending(ValueError):
    """The draft construction has no qualified shared execution route yet."""


def require_execution_integration(spec):
    """Fail closed until the subsequent shared fixed-share execution integration lands."""
    from capstone.strategy.recorded import require_current_implementation

    require_current_implementation(spec)
    if isinstance(spec, ProgramStrategySpec):
        raise ExecutionIntegrationPending("general program execution integration is pending")
    for member in getattr(spec, "specs", ()):
        require_execution_integration(member)


def program_weights(spec, panel, as_of):
    """One shared target construction for replay and planning, with no admission claim."""
    require_runtime_capabilities(spec)
    at = session_date(as_of, "decision")
    if panel.data_version != spec.data_version:
        raise ValueError("program and panel data versions differ")
    dates = program_schedule.session_calendar(panel.calendar, spec.anchor, at)
    formations = spec.schedule.formations(dates, spec.anchor, at)
    if formations.empty:
        raise contract.InsufficientHistoryError("no completed construction formation")
    formation = formations[-1]
    # The formation book reads only inputs as of `formation`, never `at` or costs.
    book = input_memo.panel_cached(
        "formation",
        spec,
        panel,
        (str(formation),),
        lambda: _formation_book(spec, panel, formation, dates),
    )
    held_sessions = dates[(dates >= formation) & (dates <= at)]
    membership = panel.universe
    if not held_sessions.isin(membership.index).all():
        raise ValueError("held membership history has missing sessions")
    held = membership.reindex(index=held_sessions, columns=book.index)
    if held.isna().any().any() or not held.isin([True, False]).all().all():
        raise ValueError("held membership history is unavailable")
    active = held.astype(bool).all(axis=0)
    book = book.where(active, 0)
    book = book[book != 0].astype(float).sort_index().rename("weight")
    book.index.name = "permno"
    if not np.isfinite(book.to_numpy()).all() or book.abs().sum() > 1 + 1e-10:
        raise ValueError("constructed book violates fixed exposure")
    return book


def _formation_book(spec, panel, formation, dates):
    """The formation book before held-membership exits."""
    membership = _universe(panel, formation)
    names = membership.index[membership]
    if not len(names):
        raise ValueError("formation universe is empty")
    inputs, historical, names, _ = _inputs(spec, panel, formation, dates, names)
    signal_inputs = {name: inputs[name] for name in tree.features_used(spec.tree)}
    score = (
        interpret.evaluate(
            spec.tree,
            signal_inputs,
            strict_domains=True,
            cross_sectional_universe=historical,
        )
        .loc[formation]
        .reindex(names)
    )
    if score.isna().any() or not np.isfinite(score.to_numpy()).all():
        raise ValueError("required formation scores are missing or nonfinite")
    if score.nunique() < 2 and spec.tail < 1:
        raise ValueError("formation has no cross-sectional score distinction")
    score = score.sort_index().sort_values(ascending=spec.direction == "high", kind="stable")
    count = max(1, math.floor(len(score) * spec.tail))
    if spec.tail < 1 and len(score) < 2 * count:
        raise ValueError("formation requires distinct nonoverlapping tails")
    low, high = score.iloc[:count], score.iloc[-count:]

    def leg(selected):
        values = pd.Series(1.0, index=selected.index)
        if spec.weighting == "capitalization":
            values = inputs["cap"].loc[formation].reindex(selected.index)
            if (
                values.isna().any()
                or not np.isfinite(values.to_numpy()).all()
                or (values <= 0).any()
            ):
                raise ValueError("positive point-in-time formation capitalization is required")
        scaled = values / values.max()
        return scaled / scaled.sum()

    book = leg(high)
    if spec.side == "long_short":
        book = book.mul(0.5).add(leg(low).mul(-0.5), fill_value=0)
    return book


def decision_due(spec, panel, as_of):
    return spec.schedule.decision_due(panel.calendar, spec.anchor, as_of)
