"""Disclosed S&P 500 adaptations for a bounded mechanism-diversity study."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.factors.reference import field, month_ends, sessions
from capstone.strategy.research import canonical, digest

METHODS = {
    "weekly_reversal": "liquidity_supply",
    "monthly_low_volatility": "constraints_and_defensive",
}


@dataclass(frozen=True)
class DiversifiedStrategySpec:
    method: str
    data_version: str
    anchor: str
    side: str = "long_only"
    method_hash: str = ""
    gross: float = 1.0
    gate_report: str = ""
    version: int = 1

    def __post_init__(self):
        if self.method not in METHODS or self.side not in {"long_only", "long_short"}:
            raise ValueError("unknown diversity-study method or profile")
        if self.version != 1 or self.gross != 1 or not self.data_version:
            raise ValueError("unsupported diversity-study version or exposure")
        anchor = pd.Timestamp(self.anchor)
        if pd.isna(anchor) or anchor.tz is not None or anchor != anchor.normalize():
            raise ValueError("anchor must be a timezone-free session date")
        expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if self.method_hash and self.method_hash != expected:
            raise ValueError("construction implementation changed; use a new specification")
        object.__setattr__(self, "method_hash", expected)
        object.__setattr__(self, "anchor", str(anchor.date()))

    @property
    def id(self):
        return digest(asdict(self))[:16]

    @property
    def claim_id(self):
        return "diversity-adaptation/" + self.method

    @property
    def hypothesis_id(self):
        return self.claim_id

    @property
    def variant_id(self):
        return self.id

    @property
    def factor_id(self):
        return digest({"method": self.method, "implementation": self.method_hash})[:16]

    @property
    def fields(self):
        return ("returns", "cap") if self.method == "monthly_low_volatility" else ("returns",)

    def to_toml(self):
        return (
            f"schema_version = 4\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(canonical(asdict(self)))}\n"
        )

    @classmethod
    def from_toml(cls, text):
        data = tomllib.loads(text)
        if set(data) != {"schema_version", "id", "payload"} or data["schema_version"] != 4:
            raise ValueError("invalid diversity specification")
        spec = cls(**json.loads(data["payload"]))
        if data["id"] != spec.id or text.replace("\r\n", "\n") != spec.to_toml():
            raise ValueError("diversity specification changed")
        return spec


def formation_dates(spec, calendar, as_of):
    dates = sessions(calendar, as_of)
    if spec.method == "weekly_reversal":
        weeks = dates.to_period("W-FRI")
        complete = dates[:-1][weeks[:-1] != weeks[1:]]
        return complete[(complete >= pd.Timestamp(spec.anchor)) & (complete <= pd.Timestamp(as_of))]
    ends = month_ends(calendar, as_of)
    return ends[ends >= pd.Timestamp(spec.anchor)]


def decision_due(spec, panel, as_of):
    return pd.Timestamp(as_of) in formation_dates(spec, panel.calendar, as_of)


def _weekly(spec, panel, formation):
    returns = field(panel, "returns", formation)
    dates = sessions(panel.calendar)
    expected = dates[dates.to_period("W-FRI") == formation.to_period("W-FRI")]
    if not expected.isin(returns.index).all():
        raise ValueError("weekly formation requires the complete exchange week")
    window = returns.loc[expected]
    if (window < -1).any().any():
        raise ValueError("invalid total return")
    eligible = panel.universe.loc[formation].fillna(False).astype(bool)
    score = ((1 + window).prod(skipna=False) - 1).where(eligible).dropna().sort_index()
    if len(score) < 2:
        raise ValueError("weekly reversal requires two complete formation histories")
    opposite = score.mean() - score
    if spec.side == "long_only":
        opposite = opposite.clip(lower=0)
    scale = opposite.abs().sum()
    return opposite / scale if scale > 0 else opposite * 0


def _low_volatility(spec, panel, formation):
    first = (formation.to_period("M") - 59).start_time
    returns = field(panel, "returns", formation).loc[first:]
    calendar = sessions(panel.calendar)
    required = calendar[(calendar >= first) & (calendar <= formation)]
    if not required.isin(returns.index).all() or (returns < -1).any().any():
        raise ValueError("low-volatility input history is incomplete or invalid")
    # Missing observations must not become a zero-return month through prod's identity.
    groups = returns.index.to_period("M")
    monthly = (1 + returns).groupby(groups).prod(min_count=1) - 1
    monthly = monthly.where(returns.notna().groupby(groups).all())
    trailing = monthly.iloc[-60:]
    complete_recent = trailing.iloc[-24:].notna().all() & (len(trailing) >= 24)
    risk = trailing.std(ddof=1).where(complete_recent)
    eligible = panel.universe.loc[formation].fillna(False).astype(bool)
    caps = field(panel, "cap", formation).loc[formation].where(eligible)
    valid = risk.notna() & caps.notna() & caps.gt(0) & eligible
    table = pd.DataFrame({"risk": risk[valid], "cap": caps[valid]}).sort_index()
    table = table.sort_values("risk", kind="stable")
    if len(table) < 5:
        raise ValueError("monthly quintiles require five qualified risk histories")
    n = len(table) // 5
    low, high = table.iloc[:n], table.iloc[-n:]
    long = low.cap / low.cap.sum()
    if spec.side == "long_only":
        return long
    short = high.cap / high.cap.sum()
    return long.mul(0.5).add(short.mul(-0.5), fill_value=0)


def diversified_weights(spec, panel, as_of):
    at = pd.Timestamp(as_of)
    if at not in panel.universe.index or at < pd.Timestamp(spec.anchor):
        raise ValueError("decision is not a supported session")
    dates = formation_dates(spec, panel.calendar, at)
    if dates.empty:
        raise ValueError("no completed construction interval")
    formation = dates[-1]
    book = (
        _weekly(spec, panel, formation)
        if spec.method == "weekly_reversal"
        else _low_volatility(spec, panel, formation)
    )
    active = panel.universe.loc[at].reindex(book.index).fillna(False)
    book = book.where(active, 0).sort_index().rename("weight")
    book = book[book != 0]
    book.index.name = "permno"
    if not np.isfinite(book).all() or book.abs().sum() > 1 + 1e-10:
        raise ValueError("invalid constructed book")
    return book
