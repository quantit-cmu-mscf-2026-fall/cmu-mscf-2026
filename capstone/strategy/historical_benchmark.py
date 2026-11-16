"""Versioned historical controls over the existing benchmark algorithms."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.strategy.program_schedule import session_date
from capstone.strategy.reference import BenchmarkSpec
from capstone.strategy.research import digest

# v5 keeps its behaviour; v6 adds held carry, the capped fallback exit and the declared
# cash-in-lieu default; v7 adds valued held actions and deferred-entry requalification;
# v8 adds capped fallback exits. The version-specific behaviour lives in the replay and
# registration code that selects among these identifiers.
EXECUTION_VERSIONS = (
    "observed-benchmark-next-open-v5",
    "observed-benchmark-next-open-v6",
    "observed-benchmark-next-open-v7",
    "observed-benchmark-next-open-v8",
)
# Direct-replay default stays v7; registrations pick the version they declare.
EXECUTION_VERSION = EXECUTION_VERSIONS[2]
ROUTE = "benchmark-history-v1"
# version -> eligibility rule; v1 identities keep verifying under their own rule.
ELIGIBILITY_RULES = {
    1: "formation_required_history_v1",
    # v1 plus eligible_members: qualified same-day quote and positive sizing price.
    2: "formation_required_history_and_qualified_quote_v2",
}
KINDS = ("cash", "cap_weight_universe_proxy", "equal_weight_universe_proxy", "known_momentum_12_1")


def implementation_digest():
    from capstone import backtest
    from capstone.factors import evaluation, interpret, tree
    from capstone.strategy import reference

    files = {
        "historical_benchmark": Path(__file__),
        "reference": Path(reference.__file__),
        "factor_evaluation": Path(evaluation.__file__),
        "interpreter": Path(interpret.__file__),
        "tree": Path(tree.__file__),
        "weight_conversion": Path(backtest.__file__),
    }
    return digest(
        {
            "files": {
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
            },
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }
    )


@dataclass(frozen=True)
class HistoricalBenchmarkSpec:
    kind: str
    data_version: str
    anchor: str
    side: str = "long_only"
    version: int = 1
    eligibility_rule: str = "formation_required_history_v1"
    phase_rule: str = "new_account_at_evaluation_prior_session_v1"
    implementation: str = ""

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError("known historical benchmark kind required")
        if type(self.side) is not str or self.side not in {"long_only", "long_short"}:
            raise ValueError("separate historical benchmark profile required")
        if self.kind.endswith("universe_proxy") and self.side != "long_only":
            raise ValueError("universe proxies are long-only funded controls")
        if (
            type(self.version) is not int
            or ELIGIBILITY_RULES.get(self.version) != self.eligibility_rule
            or self.phase_rule != "new_account_at_evaluation_prior_session_v1"
        ):
            raise ValueError("unsupported historical benchmark adaptation")
        if not isinstance(self.data_version, str) or not self.data_version:
            raise ValueError("historical benchmark data version required")
        object.__setattr__(
            self, "anchor", str(session_date(self.anchor, "benchmark anchor").date())
        )
        current = implementation_digest()
        if self.implementation and self.implementation != current:
            raise ValueError("historical benchmark implementation incompatible")
        object.__setattr__(self, "implementation", current)

    @property
    def legacy_benchmark_id(self):
        return self.legacy.id

    @property
    def legacy(self):
        return BenchmarkSpec(self.kind, self.data_version, self.anchor, self.side)

    @property
    def specs(self):
        """No generic engine leaves: existing dispatch skips its cache and refuses
        non-PortfolioSpec wrappers. Only the explicit v5/v6 constructor may route it.
        """
        return ()

    @property
    def id(self):
        return digest(asdict(self))[:16]

    def to_record(self):
        return {
            "schema_version": 1,
            "id": self.id,
            "payload": asdict(self),
            "legacy_benchmark_id": self.legacy_benchmark_id,
        }

    @classmethod
    def from_record(cls, record):
        if (
            not isinstance(record, dict)
            or set(record) != {"schema_version", "id", "payload", "legacy_benchmark_id"}
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
        ):
            raise ValueError("invalid historical benchmark record")
        try:
            spec = cls(**record["payload"])
        except (TypeError, KeyError) as error:
            raise ValueError("invalid historical benchmark payload") from error
        if spec.to_record() != record:
            raise ValueError("historical benchmark content or identity changed")
        return spec
