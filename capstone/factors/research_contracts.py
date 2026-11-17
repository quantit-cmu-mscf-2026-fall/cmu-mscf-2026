"""Opt-in research contracts; these confer no financial or broker admission."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal

from capstone.artifacts import canonical, digest

FAMILIES = (
    "liquidity_supply",
    "constraints_and_defensive",
    "information_networks",
    "forced_flows",
    "accounting_and_events",
    "cross_asset_watchlist",
)
TRACKS = ("discovery", "replication_control", "refinement")
Track = Literal["discovery", "replication_control", "refinement"]

# canonical/digest live in capstone.artifacts; re-exported for the modules that import them here.
__all__ = ["canonical", "digest"]


@dataclass(frozen=True)
class ResearchPolicy:
    """Controller-owned reservations, frozen before proposals or results exist."""

    reservations: tuple[tuple[str, str, int], ...]
    version: int = 1
    universe: str = "sp500"
    discovery_end: str = "2020-12-31"
    max_repairs: int = 1
    model_rejection: bool = False

    def __post_init__(self):
        date.fromisoformat(self.discovery_end)
        if self.version != 1 or self.universe != "sp500":
            raise ValueError("unsupported research policy")
        if self.discovery_end > "2020-12-31" or self.max_repairs != 1 or self.model_rejection:
            raise ValueError("scope or uncalibrated rejection is not authorized")
        keys = [(family, track) for family, track, _ in self.reservations]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate reservation")
        for family, track, limit in self.reservations:
            if family not in FAMILIES or track not in TRACKS or type(limit) is not int or limit < 0:
                raise ValueError("invalid canonical family, track or reservation")
        object.__setattr__(
            self, "reservations", tuple(sorted((f, t, n) for f, t, n in self.reservations))
        )

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class AssignedBrief:
    """A registry assignment, never a family label chosen by a generating agent."""

    packet_id: str
    family: str
    track: Track
    method_key: str

    def __post_init__(self):
        if self.family not in FAMILIES or self.track not in TRACKS:
            raise ValueError("unknown family or track")
        if not self.packet_id or not self.method_key:
            raise ValueError("packet and canonical method identities are required")


@dataclass(frozen=True)
class SourceClaim:
    source_url: str
    method_location: str
    claim: str
    falsifier: str
    required_inputs: tuple[str, ...]
    full_methods: bool
    universe: str = "sp500"

    def __post_init__(self):
        if not self.source_url.startswith("https://"):
            raise ValueError("primary source link required")
        if type(self.full_methods) is not bool:
            raise ValueError("full_methods must be a boolean from the source inventory")
        if not self.required_inputs or any(
            not isinstance(i, str) or not i for i in self.required_inputs
        ):
            raise ValueError("explicit required inputs are required")
        if not all((self.method_location, self.claim, self.falsifier)):
            raise ValueError("claim evidence and falsifier required")
        object.__setattr__(self, "required_inputs", tuple(sorted(set(self.required_inputs))))


@dataclass(frozen=True)
class MethodSpec:
    """New semantics receive a new identity; legacy strategy identities are untouched."""

    claim_id: str
    variant: Literal["reference", "adaptation"]
    direction: str
    selection: str
    weighting: str
    formation: str
    holding: str
    exits: str
    availability: str
    profile: Literal["long_only", "long_short"]
    deviations: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self):
        if self.version != 1 or self.variant not in ("reference", "adaptation"):
            raise ValueError("unsupported method version or variant")
        if self.profile not in ("long_only", "long_short"):
            raise ValueError("unknown profile")
        if not all(
            (
                self.claim_id,
                self.direction,
                self.selection,
                self.weighting,
                self.formation,
                self.holding,
                self.exits,
                self.availability,
            )
        ):
            raise ValueError("incomplete construction specification")
        if self.variant == "adaptation" and not self.deviations:
            raise ValueError("adaptations must disclose deviations")
        object.__setattr__(self, "deviations", tuple(self.deviations))

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ConstructionEvidence:
    spec_id: str
    verifier_version: str
    independent_example: bool
    timing_passed: bool
    construction_passed: bool

    @property
    def passed(self):
        return bool(self.spec_id and self.verifier_version) and all(
            (self.independent_example, self.timing_passed, self.construction_passed)
        )


@dataclass(frozen=True)
class TransitionEvent:
    brief_id: str
    before: str
    after: str
    reason: str
    evidence_id: str


def qualify(claim: SourceClaim, qualified_inputs: frozenset[str]) -> tuple[str, tuple[str, ...]]:
    """Availability is supplied by the controller, never accepted from agent output."""
    if not claim.full_methods:
        return "methods_review_needed", ()
    if claim.universe != "sp500":
        return "outside_scope", ()
    missing = tuple(sorted(set(claim.required_inputs) - qualified_inputs))
    return ("needs_data", missing) if missing else ("specification_ready", ())
