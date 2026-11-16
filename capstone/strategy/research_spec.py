"""Bounded research adaptations. These specifications never confer admission."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.factors import reference as features
from capstone.factors.evaluation import TradingSpec
from capstone.strategy.recorded import ImplementationMismatch, require_current_implementation
from capstone.strategy.reference import ReferenceStrategySpec, cohort_books
from capstone.strategy.research import canonical, digest


@dataclass(frozen=True)
class ResearchStrategySpec:
    claim_id: str
    method: str
    construction: str
    data_version: str
    anchor: str
    side: str = "long_short"
    tail: float = 0.1
    weighting: str = "equal"
    method_hash: str = ""
    gross: float = 1.0
    gate_report: str = ""

    def __post_init__(self):
        anchor = pd.Timestamp(self.anchor)
        if anchor.tz is not None or anchor != anchor.normalize():
            raise ValueError("research anchor must be a session date")
        if self.side not in {"long_only", "long_short"} or self.gross != 1:
            raise ValueError("research profiles must use the fixed gross-one convention")
        if self.construction not in {"alternate_weighting", "fundamental"}:
            raise ValueError("unsupported research construction")
        if self.weighting not in {"equal", "capitalization"} or self.tail not in {0.1, 0.2, 0.3}:
            raise ValueError("construction lies outside the frozen study")
        if self.construction == "fundamental" and (
            self.method not in {"value", "profitability"} or self.claim_id != f"paper-{self.method}"
        ):
            raise ValueError("unknown fundamental paper family")
        base_hash = self.base.method_hash if self.construction == "alternate_weighting" else None
        if self.construction == "alternate_weighting" and self.method != self.base.method:
            raise ValueError("reference method and family disagree")
        if self.construction == "alternate_weighting":
            expected_weighting = "equal" if self.method.startswith("ff3") else "capitalization"
            if self.weighting != expected_weighting:
                raise ValueError("alternate weighting must match its declared method")
        expected = digest(
            {
                "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "compatibility_policy": hashlib.sha256(
                    Path(__file__).with_name("recorded.py").read_bytes()
                ).hexdigest(),
                "base": base_hash,
                "method": self.method,
                "tail": self.tail,
                "construction": self.construction,
                "weighting": self.weighting,
            }
        )
        if self.method_hash and self.method_hash != expected:
            raise ImplementationMismatch(
                "research method changed; preserve the old identity and review a new version"
            )
        object.__setattr__(self, "method_hash", expected)
        object.__setattr__(self, "anchor", str(pd.Timestamp(self.anchor).date()))
        if not self.data_version:
            raise ValueError("research data version is required")

    @property
    def base(self):
        return ReferenceStrategySpec(self.claim_id, self.data_version, self.anchor, self.side)

    @property
    def fields(self):
        return (
            tuple(sorted(set(self.base.fields) | {"cap"}))
            if self.construction == "alternate_weighting"
            else ("cap",)
        )

    @property
    def id(self):
        return digest(asdict(self))[:16]

    @property
    def variant_id(self):
        return self.id

    @property
    def factor_id(self):
        return self.method_hash[:16]

    @property
    def hypothesis_id(self):
        return self.claim_id

    @property
    def expression(self):
        return f"research-v1/{self.method}/{self.construction}/{self.side}"

    @property
    def trading(self):
        return TradingSpec(hold=21)

    def to_toml(self):
        return (
            f"schema_version = 3\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(canonical(asdict(self)))}\n"
        )

    @classmethod
    def from_toml(cls, text):
        document = tomllib.loads(text)
        if set(document) != {"schema_version", "id", "payload"} or document["schema_version"] != 3:
            raise ValueError("invalid research specification")
        spec = cls(**json.loads(document["payload"]))
        if spec.id != document["id"] or text.replace("\r\n", "\n") != spec.to_toml():
            raise ValueError("research specification changed")
        return spec


def candidate_spec(candidate, data_version, anchor):
    if candidate["construction"] == "reference":
        return ReferenceStrategySpec(candidate["family"], data_version, anchor, candidate["side"])
    weighting = candidate.get("weighting", "equal")
    if candidate["construction"] == "alternate_weighting":
        weighting = "equal" if candidate["method"].startswith("ff3") else "capitalization"
    return ResearchStrategySpec(
        candidate["family"],
        candidate["method"],
        candidate["construction"],
        data_version,
        anchor,
        candidate["side"],
        candidate.get("tail", 0.1),
        weighting,
    )


def _reweight(names, panel, formation, mode):
    weights = pd.Series(1.0, index=names)
    if mode == "capitalization":
        weights = features.field(panel, "cap", formation).loc[formation, names]
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError("formation capitalization is unavailable")
    return weights / weights.sum()


def _alternate(spec, panel, as_of):
    books = cohort_books(spec.base, panel, as_of, panel.calendar)
    slots = 3 if spec.method == "reversal_36m" else (1 if spec.method.startswith("ff3") else 6)
    mode = "equal" if spec.method.startswith("ff3") else "capitalization"
    combined = []
    for cohort in books:
        long = _reweight(cohort["long"].index, panel, cohort["formation"], mode)
        short = _reweight(cohort["short"].index, panel, cohort["formation"], mode)
        book = (
            long if spec.side == "long_only" else long.mul(0.5).add(short.mul(-0.5), fill_value=0)
        )
        active = panel.universe.loc[as_of].reindex(book.index).fillna(False)
        combined.append(book.where(active, 0) / slots)
    return pd.concat(combined).groupby(level=0).sum() if combined else pd.Series(dtype=float)


def fundamental_legs(spec, panel, formation):
    if panel.fundamentals is None or panel.company is None or panel.sic is None:
        raise ValueError("qualified accounting vintages, company identities and SIC are required")
    fund = panel.fundamentals.copy()
    needed = {
        "permco",
        "datadate",
        "published_at",
        "vintage_at",
        "book",
        "revenue",
        "cogs",
        "assets",
    }
    if not needed <= set(fund):
        raise ValueError("fundamental records are incomplete")
    for key in ("datadate", "published_at", "vintage_at"):
        fund[key] = pd.to_datetime(fund[key])
    fund = fund[
        (fund.datadate.dt.year == formation.year - 1)
        & (fund.published_at.dt.normalize() < formation)
        & (fund.vintage_at <= fund.published_at)
    ]
    # Both papers form in June using previous-calendar-year fiscal accounts.
    fund = fund[fund.datadate.map(lambda d: d + pd.DateOffset(months=6)) <= formation]
    if fund.duplicated(["permco", "datadate", "published_at"]).any():
        raise ValueError("ambiguous company accounting vintages")
    fund = (
        fund.sort_values(["permco", "datadate", "published_at"])
        .drop_duplicates("permco", keep="last")
        .set_index("permco")
    )
    cap = features.field(panel, "cap", formation)
    company = panel.company.loc[formation]
    eligible = panel.universe.loc[formation].fillna(False) & panel.sic.loc[formation].notna()
    eligible &= ~panel.sic.loc[formation].between(6000, 6999)
    valid = pd.DataFrame({"permco": company, "cap": cap.loc[formation]})
    valid = valid[valid.permco.notna() & valid.cap.gt(0)]
    current_caps = valid.groupby("permco").cap.sum()
    if spec.method == "value":
        ends = features.month_ends(panel.calendar, formation)
        december = ends[(ends.year == formation.year - 1) & (ends.month == 12)]
        if len(december) != 1:
            raise ValueError("December market equity is required")
        at = december[0]
        denominator = cap.loc[at].groupby(panel.company.loc[at]).sum(min_count=1)
        score = fund.book.where(fund.book > 0) / denominator.where(denominator > 0)
    else:
        score = (fund.revenue - fund.cogs) / fund.assets.where(fund.assets > 0)
    # One eligible representative security per company, largest class then permno.
    reps = valid.loc[eligible.reindex(valid.index).fillna(False)].reset_index()
    reps = reps.rename(columns={reps.columns[0]: "permno"})
    reps = reps.sort_values(
        ["permco", "cap", "permno"], ascending=[True, False, True]
    ).drop_duplicates("permco")
    score = score.reindex(reps.permco).dropna().sort_index().sort_values(kind="stable")
    if not np.isfinite(score).all():
        raise ValueError("nonfinite accounting ratios")
    count = max(1, int(len(score) * spec.tail))
    if len(score) < 2 * count or score.nunique() < 2:
        raise ValueError("insufficient eligible fundamental companies")
    representatives = reps.set_index("permco").permno

    def leg(companies):
        values = pd.Series(1.0, index=companies)
        if spec.weighting == "capitalization":
            values = current_caps.reindex(companies)
        values = values / values.sum()
        return pd.Series(
            values.to_numpy(), index=representatives.reindex(companies).astype(int).to_numpy()
        )

    return leg(score.tail(count).index), leg(score.head(count).index)


def research_weights(spec, panel, as_of):
    require_current_implementation(spec)
    as_of = pd.Timestamp(as_of)
    if panel.data_version != spec.data_version or as_of not in panel.universe.index:
        raise ValueError("research snapshot or data version differs")
    if as_of < pd.Timestamp(spec.anchor):
        raise ValueError("research decision precedes its anchor")
    if spec.construction == "alternate_weighting":
        result = _alternate(spec, panel, as_of)
    else:
        ends = features.month_ends(panel.calendar, as_of)
        june = ends[(ends.month == 6) & (ends >= pd.Timestamp(spec.anchor))]
        if june.empty:
            return pd.Series(dtype=float, name="weight")
        long, short = fundamental_legs(spec, panel, june[-1])
        result = (
            long if spec.side == "long_only" else long.mul(0.5).add(short.mul(-0.5), fill_value=0)
        )
        result = result.where(panel.universe.loc[as_of].reindex(result.index).fillna(False), 0)
    result = result[result != 0].sort_index().rename("weight")
    result.index.name = "permno"
    if not np.isfinite(result).all() or result.abs().sum() > 1 + 1e-10:
        raise ValueError("invalid research portfolio")
    return result
