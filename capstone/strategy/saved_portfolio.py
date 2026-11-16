"""Saved, prior-only funded-capital allocations over existing Program books."""

from __future__ import annotations

import hashlib
import json
import platform
import tomllib
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd
import scipy

from capstone.strategy.portfolio import portfolio_target
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule, session_calendar, session_date
from capstone.strategy.research import canonical, digest
from capstone.strategy.robust_allocation import AllocationPolicyV2
from capstone.strategy.selected_portfolio import SelectedPortfolioSpec

END = "2020-12-31"  # DISCOVERY_END: later dates need the authorized current route

MODE = "saved_funded_coefficients_v1"
# v4: retrospective fixed pool (allocation-state v1). v5: chronological
# prior-qualified-prefix membership (allocation-state v2). Both keep replaying/planning.
FIXED_POOL_EXECUTION_VERSION = "saved-program-portfolio-v4"
EXECUTION_VERSION = "saved-program-portfolio-v5"
# v6 (fixed pool) / v7 (chronological) replay the same pools under the held-action rules
# (contracts.HELD_ACTION_VERSIONS). A registered study chooses the replay version; the
# spec identity and its frozen order plans keep v4/v5.
HELD_EXECUTION_VERSIONS = {
    FIXED_POOL_EXECUTION_VERSION: "saved-program-portfolio-v6",
    EXECUTION_VERSION: "saved-program-portfolio-v7",
}
# v8 (fixed pool) / v9 (chronological) add capped fallback exits
# (contracts.FALLBACK_VERSIONS); new registrations replay under these.
FALLBACK_EXECUTION_VERSIONS = {
    FIXED_POOL_EXECUTION_VERSION: "saved-program-portfolio-v8",
    EXECUTION_VERSION: "saved-program-portfolio-v9",
}
EXECUTION_VERSIONS = (
    FIXED_POOL_EXECUTION_VERSION,
    EXECUTION_VERSION,
    *HELD_EXECUTION_VERSIONS.values(),
    *FALLBACK_EXECUTION_VERSIONS.values(),
)
CHRONOLOGICAL_VERSIONS = (
    EXECUTION_VERSION,
    HELD_EXECUTION_VERSIONS[EXECUTION_VERSION],
    FALLBACK_EXECUTION_VERSIONS[EXECUTION_VERSION],
)
FIXED_POOL = "retrospective_fixed_pool"
CHRONOLOGICAL = "prior_qualified_prefix"
# A decision refuses a pool member whose newest receipt is older than this many decision
# periods. Absent (None) on every earlier spec: identity and behaviour pinned.
FRESHNESS_VERSION = "max-receipt-age-v1"


def implementation_digest():
    from capstone.strategy import allocation_state, portfolio, robust_allocation

    paths = {
        "saved_portfolio": Path(__file__),
        "allocation_state": Path(allocation_state.__file__),
        "optimizer": Path(robust_allocation.__file__),
        "portfolio": Path(portfolio.__file__),  # also holds the book combination
    }
    return digest(
        {
            "files": {
                key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()
            },
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
        }
    )


@dataclass(frozen=True, kw_only=True)
class SavedProgramPortfolioSpec(SelectedPortfolioSpec):
    """Definition identity is stable; every saved state has its own prefix identity.

    The controller supplies the canonical pool and qualified funded returns.
    Neither a saved allocation nor a positive model score grants admission.
    """

    profile: str
    data_version: str
    anchor: str
    policy: AllocationPolicyV2 = field(default_factory=AllocationPolicyV2)
    allocation_mode: str = "median_window_sharpe"
    mode: str = MODE
    implementation: str = ""
    membership_clock: str = FIXED_POOL
    membership: tuple = ()
    max_receipt_age: int | None = None  # decision periods ; None = not registered

    def __post_init__(self):
        if (
            type(self.profile) is not str
            or self.profile not in {"long_only", "long_short"}
            or self.mode != MODE
        ):
            raise ValueError("explicit saved-allocation profile and mode required")
        if not isinstance(self.policy, AllocationPolicyV2):
            raise ValueError("saved funded allocations require policy V2")
        if type(self.allocation_mode) is not str or self.allocation_mode not in {
            "equal_weight",
            "shrunk_max_sharpe",
            "median_window_sharpe",
        }:
            raise ValueError("unsupported saved allocation objective")
        if not isinstance(self.data_version, str) or not self.data_version:
            raise ValueError("saved portfolio data version required")
        anchor = str(session_date(self.anchor, "portfolio anchor").date())
        object.__setattr__(self, "anchor", anchor)
        if not isinstance(self.specs, (tuple, list)) or any(
            type(member) is not ProgramStrategySpec for member in self.specs
        ):
            raise ValueError("saved allocation pool requires schema-six programs")
        members = tuple(sorted(self.specs, key=lambda member: member.id))
        if len({member.id for member in members}) != len(members) or any(
            type(member) is not ProgramStrategySpec
            or member.side != self.profile
            or member.data_version != self.data_version
            or member.anchor != anchor
            or member.schedule != ProgramSchedule(kind="month_end", interval=1)
            for member in members
        ):
            raise ValueError(
                "distinct schema-six programs need one profile and synchronized monthly clocks"
            )
        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("actual funded-return frame required")
        object.__setattr__(self, "specs", members)
        if self.membership_clock not in {FIXED_POOL, CHRONOLOGICAL} or (
            self.membership_clock == FIXED_POOL and self.membership
        ):
            raise ValueError("unknown saved allocation membership clock")
        age = self.max_receipt_age
        if age is not None and (
            type(age) is not int or age < 1 or self.membership_clock != CHRONOLOGICAL
        ):
            raise ValueError("max_receipt_age is a positive period count of a chronological pool")
        from capstone.strategy.allocation_state import membership_events

        events = membership_events(self.membership)
        if {row["strategy_id"] for row in events} - {spec.id for spec in members}:
            raise ValueError("membership events must identify registered pool candidates")
        object.__setattr__(self, "membership", events)
        object.__setattr__(self, "returns", self.returns.copy(deep=True))
        try:
            object.__setattr__(self, "schedule", tuple(json.loads(canonical(self.schedule))))
            self._validate_records()
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError("malformed saved allocation records") from exc
        current = implementation_digest()
        if self.implementation and self.implementation != current:
            raise ValueError("saved allocation implementation is incompatible")
        object.__setattr__(self, "implementation", current)

    def _validate_records(self):
        from capstone.strategy.allocation_state import FIXED_POOL_VERSION, VERSION, verify_decision

        chronological, candidates = self.chronological, [spec.id for spec in self.specs]
        previous = None
        for row in self.schedule:
            verify_decision(row)
            if (
                row["profile"] != self.profile
                or row["version"] != (VERSION if chronological else FIXED_POOL_VERSION)
                or (
                    not set(row["pool_ids"]) <= set(candidates)
                    if chronological
                    else row["pool_ids"] != candidates
                )
                or row["policy"] != asdict(self.policy)
                or row.get("mode", self.allocation_mode) != self.allocation_mode
                or row.get("formal_admission") is not False
                or row.get("source_return_basis") != "funded_total_return"
                or row.get("coefficient_units") != "fraction_of_funded_capital"
                or row["previous_state_id"] != (None if previous is None else previous["id"])
                or previous is not None
                and row["decision_session"] <= previous["decision_session"]
            ):
                raise ValueError(
                    "saved allocation policy, pool or previous-state chain is incompatible"
                )
            previous = row

    @property
    def side(self):
        return self.profile

    @property
    def chronological(self):
        return self.membership_clock == CHRONOLOGICAL

    @property
    def execution_version(self):
        return EXECUTION_VERSION if self.chronological else FIXED_POOL_EXECUTION_VERSION

    @property
    def replay_versions(self):
        base = self.execution_version
        return (base, HELD_EXECUTION_VERSIONS[base], FALLBACK_EXECUTION_VERSIONS[base])

    @property
    def fields(self):
        return tuple(sorted({name for spec in self.specs for name in spec.fields}))

    @property
    def gate_report(self):
        return ""

    @property
    def expression(self):
        return MODE

    @property
    def hypothesis_id(self):
        return "saved_program_portfolio"

    @property
    def definition(self):
        from capstone.strategy.allocation_state import BASIS

        # Fixed-pool definitions stay byte-identical to v4; chronological ones say so.
        return {
            "version": 1,
            "profile": self.profile,
            "data_version": self.data_version,
            "anchor": self.anchor,
            "clock": asdict(ProgramSchedule(kind="month_end", interval=1)),
            "mode": self.mode,
            "allocation_mode": self.allocation_mode,
            "return_basis": BASIS,
            "source_return_basis": "funded_total_return",
            "cash_reference": "zero_return_cash",
            "policy": asdict(self.policy),
            "specifications": {spec.id: spec.to_toml() for spec in self.specs},
            "implementation": self.implementation,
            **({"version": 2, "membership_clock": CHRONOLOGICAL} if self.chronological else {}),
            **(
                {}
                if self.max_receipt_age is None
                else {
                    "receipt_freshness": {
                        "version": FRESHNESS_VERSION,
                        "max_receipt_age": self.max_receipt_age,
                        "unit": "decision_periods",
                    }
                }
            ),
        }

    @property
    def id(self):
        return digest(self.definition)

    def decision_prefix(self, as_of):
        at = session_date(as_of)
        return deepcopy(
            [row for row in self.schedule if session_date(row["decision_session"]) <= at]
        )

    def prefix_id(self, as_of):
        return digest({"definition_id": self.id, "decisions": self.decision_prefix(as_of)})

    def frozen(self, as_of):
        at = session_date(as_of)
        return replace(
            self,
            schedule=tuple(self.decision_prefix(at)),
            returns=self.returns.loc[self.returns.index < at].copy(deep=True),
            membership=tuple(
                row for row in self.membership if session_date(row["available_at"]) < at
            ),
        )

    def decision(self, as_of):
        known = self.decision_prefix(as_of)
        if not known:
            raise ValueError("saved allocation decision is missing")
        return known[-1]

    def members(self, as_of):
        known = self.decision_prefix(as_of)
        if not known:
            return ()
        weights = known[-1]["weights"]
        return tuple(spec for spec in self.specs if weights.get(spec.id, 0) > 0)

    def current_bound(self, through):
        """Post-2020 only as a live history whose latest decisions carry current provenance."""
        later = [row for row in self.decision_prefix(through) if row["decision_session"] > END]
        return bool(later) and all("current_provenance" in row for row in later)

    def validate_clock(self, panel, through):
        at = session_date(through)
        late = at > pd.Timestamp(END) or panel.universe.index.max() > pd.Timestamp(END)
        if late and not self.current_bound(at):
            raise ValueError("saved allocation research cannot open the sealed holdout")
        calendar = session_calendar(panel.calendar, self.anchor, at)
        expected = ProgramSchedule(kind="month_end", interval=1).formations(
            calendar, self.anchor, at
        )
        known = self.decision_prefix(at)
        if [row["decision_session"] for row in known] != [str(day.date()) for day in expected]:
            raise ValueError("every completed monthly allocation needs an explicit saved decision")
        for row in known:
            index = calendar.get_loc(session_date(row["decision_session"]))
            if row["execution_session"] != str(calendar[index + 1].date()):
                raise ValueError(
                    "saved allocation execution must be the next authoritative session"
                )
        if panel.data_version != self.data_version:
            raise ValueError("saved allocation data version differs")

    def verify_history(self, panel, through):
        """Boundary verification only; planning never invokes the optimizer."""
        from capstone.strategy.allocation_state import make_decision

        self.validate_clock(panel, through)
        index = self.returns.index
        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.has_duplicates
            or index.hasnans
            or index.tz is not None
            or not index.equals(index.normalize())
        ):
            raise ValueError("funded returns require unique completed-session dates")
        if len(index) and index.max() > pd.Timestamp(END) and not self.current_bound(through):
            raise ValueError("funded allocation history cannot open the sealed holdout")
        previous = None
        for row in self.decision_prefix(through):
            if "current_provenance" in row:
                # The authorized current-feature route is not part of this module yet.
                raise ValueError("current-provenance decisions need the current-feature route")
            past = index[index < session_date(row["decision_session"])].sort_values()
            expected_sessions = pd.DatetimeIndex(panel.calendar)
            if len(past) and not past.equals(
                expected_sessions[
                    (expected_sessions >= past[0])
                    & (expected_sessions < session_date(row["decision_session"]))
                ]
            ):
                raise ValueError("funded return history has missing or non-session observations")
            expected = make_decision(
                self.returns,
                as_of=row["decision_session"],
                execution_session=row["execution_session"],
                **(
                    {"membership": self.membership}
                    if self.chronological
                    else {"pool_ids": [spec.id for spec in self.specs]}
                ),
                profile=self.profile,
                input_evidence_id=row["input_evidence_id"],
                previous=previous,
                mode=self.allocation_mode,
                policy=self.policy,
            )
            if canonical(expected) != canonical(row):
                raise ValueError(
                    "saved allocation does not derive from actual prior funded returns"
                )
            previous = row
        return self.prefix_id(through)

    def weights(self, panel, as_of):
        self.validate_clock(panel, as_of)
        row, members = self.decision(as_of), self.members(as_of)
        evidence = {
            "definition_id": self.id,
            "decision_prefix_id": self.prefix_id(as_of),
            "decision_id": row["id"],
            "cash_weight": row["cash_weight"],
        }
        if members:
            allocated = {
                "schema_version": 4,
                "side": self.side,
                "weights": {spec.id: row["weights"][spec.id] for spec in members},
            }
            book, details = portfolio_target(members, panel, as_of, allocated)
            book.attrs["allocation_attribution"] = details["attribution"]
        else:
            book = pd.Series(
                index=pd.Index([], dtype="int64", name="permno"), dtype=float, name="weight"
            )
            book.attrs["allocation_attribution"] = {
                "kind": "target-weight attribution, before integer rounding",
                "strategies": {},
            }
        book.attrs["saved_allocation"] = evidence
        return book

    def to_toml(self):
        state = {
            "definition": self.definition,
            "decisions": self.schedule,
            "state_id": digest({"definition_id": self.id, "decisions": self.schedule}),
        }
        if self.chronological:
            state["membership"] = list(self.membership)
        return (
            f"schema_version = 7\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(canonical(state))}\n"
        )

    def plan_inputs(self, panel, as_of):
        book, row = self.weights(panel, as_of), self.decision(as_of)
        allocated = {
            "version": MODE,
            "mode": self.allocation_mode,
            "id": row["id"],
            "side": self.side,
            "weights": row["weights"],
            "cash_weight": row["cash_weight"],
            "decision_id": row["id"],
            "decision_prefix_id": self.prefix_id(as_of),
        }
        return book, allocated, {"attribution": book.attrs["allocation_attribution"]}

    @classmethod
    def from_toml(cls, text, *, returns=None):
        try:
            return cls._from_toml(text, returns=returns)
        except (TypeError, KeyError, AttributeError, OverflowError) as exc:
            raise ValueError("malformed saved portfolio content") from exc

    @classmethod
    def _from_toml(cls, text, *, returns=None):
        document = tomllib.loads(text)
        if (
            set(document) != {"schema_version", "id", "payload"}
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 7
        ):
            raise ValueError("unsupported saved portfolio schema")
        state = json.loads(document["payload"])
        definition = state["definition"]
        spec = cls(
            specs=tuple(
                ProgramStrategySpec.from_toml(value)
                for _, value in sorted(definition["specifications"].items())
            ),
            returns=pd.DataFrame(index=pd.DatetimeIndex([])) if returns is None else returns,
            profile=definition["profile"],
            data_version=definition["data_version"],
            anchor=definition["anchor"],
            policy=AllocationPolicyV2(**definition["policy"]),
            allocation_mode=definition["allocation_mode"],
            schedule=tuple(state["decisions"]),
            implementation=definition["implementation"],
            membership_clock=definition.get("membership_clock", FIXED_POOL),
            membership=tuple(state.get("membership", ())),
            max_receipt_age=_freshness(definition.get("receipt_freshness")),
        )
        if spec.id != document["id"] or spec.to_toml() != text.replace("\r\n", "\n"):
            raise ValueError("noncanonical saved portfolio content or identity")
        return spec


def _freshness(row):
    if row is None:
        return None
    if row != {"version": FRESHNESS_VERSION, "max_receipt_age": row.get("max_receipt_age"),
               "unit": "decision_periods"}:  # fmt: skip
        raise ValueError("unsupported receipt freshness policy")
    return row["max_receipt_age"]
