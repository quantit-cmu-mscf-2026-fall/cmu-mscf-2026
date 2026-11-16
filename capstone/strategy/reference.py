"""Version-two reference strategies; the legacy formula contract remains unchanged."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from capstone.factors import reference as features
from capstone.factors.evaluation import Panel, TradingSpec
from capstone.strategy.recorded import ImplementationMismatch, require_current_implementation
from capstone.strategy.research import canonical, claim_for, digest, method_digest

METHOD_REVISION = 3
REQUIRED = {
    "reversal_36m": ("returns", "mkt_return"),
    "volume_momentum_6m": ("returns", "volume", "shares"),
    "momentum_6m": ("returns",),
    "momentum_6m_skip_week": ("returns",),
    "high_52week": ("close",),
    "ff3_idiosyncratic_volatility": ("returns", "cap", "rf", "mkt_excess", "smb", "hml"),
}


@dataclass(frozen=True)
class BenchmarkSpec:
    kind: str
    data_version: str
    anchor: str
    side: str = "long_only"

    def __post_init__(self):
        if self.kind not in {
            "cash",
            "cap_weight_universe_proxy",
            "equal_weight_universe_proxy",
            "known_momentum_12_1",
        }:
            raise ValueError("unknown benchmark")
        if self.side not in {"long_only", "long_short"}:
            raise ValueError("unknown benchmark profile")

    @property
    def id(self):
        return digest(asdict(self))[:16]


def benchmark_target(spec, panel, as_of):
    from capstone.backtest import to_weights
    from capstone.factors.evaluation import factor_signal
    from capstone.factors.tree import parse

    names = panel.universe.loc[as_of].fillna(False).astype(bool)
    weights = pd.Series(0.0, index=names.index)
    if spec.kind == "cash":
        return weights
    if spec.kind == "equal_weight_universe_proxy":
        weights.loc[names] = 1.0
    elif spec.kind == "cap_weight_universe_proxy":
        weights.loc[names] = features.field(panel, "cap", as_of).loc[as_of, names]
        if weights.loc[names].isna().any() or (weights.loc[names] <= 0).any():
            raise ValueError("benchmark needs positive point-in-time capitalization")
    else:
        signal = factor_signal(parse("shift(ts_sum(returns, 231), 21)"), panel).loc[:as_of]
        weights = to_weights(signal, demean=True).iloc[-1].where(names, 0)
        if spec.side == "long_short":
            return weights
        weights = weights.clip(lower=0)
    return weights / weights.sum() if weights.sum() else weights


def decision_due(spec, panel, as_of):
    from capstone.strategy.diversified import DiversifiedStrategySpec
    from capstone.strategy.diversified import decision_due as diversity_due

    if isinstance(spec, DiversifiedStrategySpec):
        return diversity_due(spec, panel, as_of)
    if hasattr(spec, "members"):
        boundary = any(
            pd.Timestamp(f["decision_session"]) == pd.Timestamp(as_of) for f in spec.schedule
        )
        return boundary or any(decision_due(s, panel, as_of) for s in spec.members(as_of))
    if hasattr(spec, "specs"):
        return any(decision_due(s, panel, as_of) for s in spec.specs)
    if getattr(spec, "method", "").endswith("skip_week"):
        dates = features.sessions(panel.calendar, as_of)
        ends = features.month_ends(dates, as_of)
        return any(dates.get_loc(d) + 5 == dates.get_loc(pd.Timestamp(as_of)) for d in ends)
    if isinstance(spec, BenchmarkSpec) and spec.kind == "known_momentum_12_1":
        dates = pd.DatetimeIndex(panel.universe.index)
        return (
            dates.get_loc(pd.Timestamp(as_of)) - dates.get_loc(pd.Timestamp(spec.anchor))
        ) % 21 == 0
    return pd.Timestamp(as_of) in set(features.month_ends(panel.calendar, as_of))


@dataclass(frozen=True)
class ReferenceStrategySpec:
    claim_id: str
    data_version: str
    anchor: str
    side: str = "long_short"
    method_hash: str = ""
    gate_report: str = ""
    gross: float = 1.0
    method_revision: int = METHOD_REVISION

    def __post_init__(self):
        claim_for(self.claim_id, executable=True)
        expected = method_digest(self.claim_id)
        if self.method_hash and self.method_hash != expected:
            raise ImplementationMismatch(
                "research method changed; a new construction review is required"
            )
        object.__setattr__(self, "method_hash", expected)
        stamp = pd.Timestamp(self.anchor)
        if stamp.tz is not None or stamp != stamp.normalize():
            raise ValueError("anchor must be a session date")
        object.__setattr__(self, "anchor", stamp.date().isoformat())
        if self.side not in {"long_only", "long_short"} or self.gross != 1.0:
            raise ValueError("reference profiles use long_only or long_short at gross one")
        if not self.data_version or self.method_revision != METHOD_REVISION:
            raise ValueError("data version and current method revision are required")

    @property
    def method(self):
        return claim_for(self.claim_id, executable=True)["method"]

    @property
    def fields(self):
        return tuple(sorted(REQUIRED[self.method]))

    @property
    def hypothesis_id(self):
        return self.claim_id

    @property
    def expression(self):
        return f"reference-v{self.method_revision}/{self.method}/{self.side}"

    @property
    def trading(self):
        # Compatibility for preflight field/sector checks; reference timing is calendar-based.
        return TradingSpec(hold=21)

    @property
    def factor_id(self):
        return self.method_hash[:16]

    @property
    def variant_id(self):
        return self.id

    @property
    def id(self):
        return digest(asdict(self))[:16]

    def to_toml(self):
        return (
            f"schema_version = 2\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(canonical(asdict(self)))}\n"
        )

    @classmethod
    def from_toml(cls, text):
        document = tomllib.loads(text)
        if set(document) != {"schema_version", "id", "payload"} or document["schema_version"] != 2:
            raise ValueError("invalid reference specification schema")
        spec = cls(**json.loads(document["payload"]))
        if spec.id != document["id"] or text.replace("\r\n", "\n") != spec.to_toml():
            raise ValueError("reference specification content or identifier was altered")
        return spec


def _ranked(values):
    clean = values.dropna().sort_index()
    return clean.sort_values(kind="stable")


def _legs(spec, panel, formation, calendar):
    method = spec.method
    if method == "reversal_36m":
        score = -features.formation_return(panel, formation, 36, market_adjusted=True)
        count = 35
    elif method.startswith("momentum") or method == "volume_momentum_6m":
        score = features.formation_return(panel, formation, 6)
        if method == "volume_momentum_6m":
            turnover_values = features.mean_turnover(panel, formation)
            score = score.where(turnover_values.notna())
        count = max(1, int(score.notna().sum() * 0.1))
    elif method == "high_52week":
        score = features.high_nearness(panel, formation)
        count = max(1, int(score.notna().sum() * 0.3))
    else:
        score = -features.ff3_residual_volatility(panel, formation)
        count = max(1, int(score.notna().sum() * 0.2))
    eligible = panel.universe.loc[formation].fillna(False).astype(bool)
    ordered = _ranked(score.where(eligible))
    if method != "reversal_36m":
        fraction = 0.3 if method == "high_52week" else (0.2 if method.startswith("ff3") else 0.1)
        count = max(1, int(len(ordered) * fraction))
    if len(ordered) < 2 * count or ordered.nunique() < 2:
        raise ValueError("insufficient eligible instruments or cross-sectional variation")
    long_names, short_names = ordered.tail(count).index, ordered.head(count).index
    if method == "volume_momentum_6m":
        turnover = _ranked(turnover_values.where(eligible & score.notna()))
        third = max(1, len(turnover) // 3)
        long_names = long_names.intersection(turnover.head(third).index)
        short_names = short_names.intersection(turnover.tail(third).index)
        if not len(long_names) or not len(short_names):
            raise ValueError("turnover-conditioned portfolio has an empty leg")

    def weights(names):
        w = pd.Series(1.0, index=names)
        if method == "ff3_idiosyncratic_volatility":
            w = features.field(panel, "cap", formation).loc[formation, names]
            if w.isna().any() or (w <= 0).any():
                raise ValueError("positive formation-date capitalization is required")
        return w / w.sum()

    return weights(long_names), weights(short_names)


def cohort_books(spec: ReferenceStrategySpec, panel: Panel, as_of, calendar) -> list[dict]:
    require_current_implementation(spec)
    dates = features.sessions(calendar, as_of)
    as_of = pd.Timestamp(as_of)
    if as_of not in panel.universe.index or pd.Timestamp(spec.anchor) not in dates:
        raise ValueError("snapshot and calendar must include the decision and anchor sessions")
    if panel.data_version != spec.data_version:
        raise ValueError("snapshot data version differs from the strategy specification")
    ends = features.month_ends(dates, as_of)
    if spec.method.endswith("skip_week"):
        ends = pd.DatetimeIndex([d for d in ends if dates.get_loc(d) + 5 <= dates.get_loc(as_of)])
    if ends.empty:
        return []
    latest_period = ends[-1].to_period("M")
    holding = 36 if spec.method == "reversal_36m" else (1 if spec.method.startswith("ff3") else 6)
    formations = [
        d
        for d in ends
        if d >= pd.Timestamp(spec.anchor)
        and 0 <= (latest_period - d.to_period("M")).n < holding
        and (spec.method != "reversal_36m" or d.month == 12)
    ]
    slots = math.ceil(holding / 12) if spec.method == "reversal_36m" else holding
    out = []
    for formation in formations:
        long, short = _legs(spec, panel, formation, dates)
        book = (
            long if spec.side == "long_only" else long.mul(0.5).add(short.mul(-0.5), fill_value=0)
        )
        # A departed name cannot silently be replaced by today's constituents.
        active = panel.universe.loc[as_of].reindex(book.index).fillna(False)
        book = book.where(active, 0) / slots
        out.append({"formation": formation, "long": long, "short": short, "book": book})
    return out


def reference_weights(spec, panel, as_of, calendar):
    require_current_implementation(spec)
    books = cohort_books(spec, panel, as_of, calendar)
    if not books:
        return pd.Series(dtype=float, name="weight")
    result = pd.concat([b["book"] for b in books]).groupby(level=0, sort=True).sum()
    result = result[result != 0].rename("weight")
    result.index.name = "permno"
    if not np.isfinite(result).all() or result.abs().sum() > 1 + 1e-10:
        raise ValueError("invalid reference portfolio")
    return result


def construction_check(spec, candidate_weights, panel, as_of, calendar):
    expected = reference_weights(spec, panel, as_of, calendar)
    actual = candidate_weights.sort_index().astype(float)
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-12, rtol=0)
    return {
        "construction_verified": True,
        "strategy_id": spec.id,
        "method_hash": spec.method_hash,
        "reference_weights_hash": digest(expected.to_dict()),
    }
