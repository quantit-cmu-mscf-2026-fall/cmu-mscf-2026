"""Source-bound agent programs lowered to the shared strategy interface.

The model selects explicit construction rules, never executable Python or data
qualification. Unsupported mechanisms remain blocked rather than becoming a
different price formula. Construction verification is separate from admission.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from capstone.factors.reference import sessions
from capstone.factors.research_contracts import canonical, digest
from capstone.strategy.diversified import DiversifiedStrategySpec, diversified_weights

RULES = {
    "weekly_reversal": {
        "signal": "calendar_week_compounded_return_deviation",
        "selection": "all_signed_deviations",
        "weighting": "absolute_deviation",
        "formation": "exchange_week_end",
        "holding": "until_next_formation",
    },
    "monthly_low_volatility": {
        "signal": "monthly_total_return_volatility_60m_minimum_24m",
        "selection": "quintile_tails",
        "weighting": "formation_capitalization",
        "formation": "exchange_month_end",
        "holding": "until_next_formation",
    },
    "turn_of_month": {
        "signal": "last_session_and_first_three_sessions",
        "selection": "entire_point_in_time_universe",
        "weighting": "equal_at_entry",
        "formation": "calendar_window_entry",
        "holding": "calendar_window_only",
    },
}
COMMON = {
    "exits": "membership_exit_or_scheduled_exit",
    "availability": "completed_session",
    "execution": "following_session_close",
    "gross": 1.0,
}


def validate_program(method, program, side):
    if method not in RULES or side not in {"long_only", "long_short"}:
        raise ValueError("unsupported mechanism or profile; do not substitute a proxy")
    if method == "turn_of_month" and side != "long_only":
        raise ValueError("calendar source does not authorize a short-rest-of-month strategy")
    expected = RULES[method] | COMMON | {"profile": side}
    if program != expected:
        wrong = sorted(k for k in set(program) | set(expected) if program.get(k) != expected.get(k))
        raise ValueError("source construction mismatch: " + ", ".join(wrong))
    if isinstance(program["gross"], bool):
        raise ValueError("gross must be a numeric exposure")


@dataclass(frozen=True)
class CompiledStrategySpec:
    method: str
    program_json: str
    source_digest: str
    method_spec_id: str
    data_version: str
    anchor: str
    side: str
    implementation: str = ""
    gate_report: str = ""
    gross: float = 1.0

    def __post_init__(self):
        program = json.loads(self.program_json)
        validate_program(self.method, program, self.side)
        if canonical(program) != self.program_json or self.gross != 1:
            raise ValueError("noncanonical construction or exposure")
        if any(len(v) != 64 for v in (self.source_digest, self.method_spec_id)):
            raise ValueError("source and grounded method identities required")
        if not self.data_version or pd.Timestamp(self.anchor).isoformat()[:10] != self.anchor:
            raise ValueError("data version and session anchor required")
        expected = hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("diversified.py").read_bytes()
            + (Path(__file__).parents[1] / "factors/reference.py").read_bytes()
        ).hexdigest()
        if self.implementation and self.implementation != expected:
            raise ValueError("compiler implementation changed")
        object.__setattr__(self, "implementation", expected)

    @property
    def id(self):
        return digest(asdict(self))[:16]

    @property
    def hypothesis_id(self):
        # A source rename or different prose cannot create another allocation family.
        return "compiled-mechanism/" + self.method

    @property
    def claim_id(self):
        return self.hypothesis_id

    @property
    def factor_id(self):
        return digest({"method": self.method, "program": self.program_json})[:16]

    @property
    def variant_id(self):
        return self.id

    @property
    def fields(self):
        return ("returns", "cap") if self.method == "monthly_low_volatility" else ("returns",)

    def to_toml(self):
        return (
            f"schema_version = 5\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(canonical(asdict(self)))}\n"
        )

    @classmethod
    def from_toml(cls, text):
        row = tomllib.loads(text)
        if set(row) != {"schema_version", "id", "payload"} or row["schema_version"] != 5:
            raise ValueError("invalid compiled strategy")
        spec = cls(**json.loads(row["payload"]))
        if row["id"] != spec.id or text.replace("\r\n", "\n") != spec.to_toml():
            raise ValueError("compiled strategy changed")
        return spec


def _window(calendar, decision):
    """Return window for the first close-to-close return after next-close fill.

    Only the known exchange calendar is inspected ahead. No future price, return,
    membership or capitalization is read. This explicitly matches return_units.
    """
    position = calendar.get_loc(pd.Timestamp(decision)) + 2
    if position >= len(calendar) - 1:
        raise ValueError("calendar needs three future sessions for the next-close plan")
    month = calendar.to_period("M")
    first_three = (month[:position] == month[position]).sum() < 3
    last = month[position] != month[position + 1]
    return bool(first_three or last)


def calendar_weights(spec, panel, as_of):
    at = pd.Timestamp(as_of)
    calendar = sessions(panel.calendar, at)
    if not _window(calendar, at):
        return pd.Series(dtype=float, name="weight", index=pd.Index([], name="permno"))
    # Freeze names and equal weights at the window's decision entry, not each day.
    position = calendar.get_loc(at)
    entry = position
    while entry > 0 and _window(calendar, calendar[entry - 1]):
        entry -= 1
    formed = calendar[entry]
    if formed < pd.Timestamp(spec.anchor) or formed not in panel.universe.index:
        raise ValueError("calendar window formation history unavailable")
    names = panel.universe.loc[formed].fillna(False).astype(bool).sort_index()
    if not names.any():
        raise ValueError("empty calendar formation universe")
    book = pd.Series(1 / names.sum(), index=names.index[names], name="weight")
    active = panel.universe.loc[at].reindex(book.index).fillna(False)
    return book[active].rename_axis("permno")


def compiled_weights(spec, panel, as_of):
    if spec.data_version != panel.data_version or pd.Timestamp(as_of) < pd.Timestamp(spec.anchor):
        raise ValueError("construction data version or anchor differs")
    if spec.method == "turn_of_month":
        return calendar_weights(spec, panel, as_of)
    native = DiversifiedStrategySpec(spec.method, spec.data_version, spec.anchor, spec.side)
    return diversified_weights(native, panel, as_of)


def decision_due(spec, panel, as_of):
    if spec.method != "turn_of_month":
        from capstone.strategy.diversified import decision_due as native_due

        return native_due(
            DiversifiedStrategySpec(spec.method, spec.data_version, spec.anchor, spec.side),
            panel,
            as_of,
        )
    calendar = sessions(panel.calendar, as_of)
    position = calendar.get_loc(pd.Timestamp(as_of))
    return position == 0 or _window(calendar, calendar[position - 1]) != _window(calendar, as_of)
