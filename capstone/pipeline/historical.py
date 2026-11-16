"""Qualified discovery inputs. Missing source attestations block, never manufacture data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.factors.evaluation import Panel, _market_return, build_panel
from capstone.strategy import input_memo

CUTOFF = pd.Timestamp("2020-12-31")
EVENT_COLUMNS = ("event_id", "event_date", "permno", "kind")


@dataclass
class HistoricalInputs:
    panel: Panel
    opens: pd.DataFrame
    closes: pd.DataFrame
    events: pd.DataFrame
    total_returns: pd.DataFrame
    half_spread_bps: pd.DataFrame
    qualification: dict
    schedules: dict = field(default_factory=dict, repr=False)
    symbols: pd.DataFrame | None = None

    def __post_init__(self):
        # Every book built from these inputs (replay, saved-portfolio verify_history,
        # frozen plans) checks eligibility against the raw closes the planner sizes with.
        if self.panel is not None and self.panel.sizing_prices is None:
            self.panel.sizing_prices = self.closes


# Kinds written only by valued corporate-action handling (valued distributions, terminal values).
VALUED = ("distribution_value", "terminal_value")


def validate_events(events, through=None):
    """The authoritative event table; dates after `through` (default CUTOFF) are refused.

    Only an authorized current-events route passes `through`.
    Validated once per table object inside an evaluation scope; callers own a copy.
    """
    check = input_memo.cached(
        "validated_events", (events,), (through,), lambda: _validated(events, through)
    )
    return check.copy()


def _validated(events, through):
    if not set(EVENT_COLUMNS) <= set(events):
        raise ValueError("authoritative event table is incomplete")
    if events.event_id.isna().any() or events.event_id.duplicated().any():
        raise ValueError("corporate action identifiers must be unique")
    if any(not isinstance(value, str) or not value for value in events.event_id):
        raise ValueError("corporate action identifiers must be nonempty strings")
    out = events.copy()
    out["event_date"] = pd.to_datetime(out.event_date)
    if out.event_date.isna().any() or (out.event_date > pd.Timestamp(through or CUTOFF)).any():
        raise ValueError("corporate action lies outside discovery")
    allowed = {"dividend", "split", "settlement", "conversion", "unsupported", *VALUED}
    if not set(out.kind) <= allowed:
        raise ValueError("unknown corporate action kind")
    if out.duplicated(["event_date", "permno"], keep=False).any():
        if "sequence" not in out or out.sequence.isna().any():
            raise ValueError("concurrent corporate actions require source-defined sequencing")
        if out.duplicated(["event_date", "permno", "sequence"]).any():
            raise ValueError("corporate action sequencing is ambiguous")
    for _, event in out.iterrows():
        if not np.isfinite(event.permno) or event.permno <= 0 or int(event.permno) != event.permno:
            raise ValueError("corporate actions require positive whole instrument identifiers")
        if event.kind in {"dividend", "distribution_value"}:
            if pd.isna(event.get("pay_date")) or pd.Timestamp(event.pay_date) < event.event_date:
                raise ValueError("dividends need an observed payment date after entitlement")
        if event.kind in {"dividend", "settlement", "conversion", *VALUED}:
            if not np.isfinite(event.get("amount", np.nan)) or event.amount < 0:
                raise ValueError("cash consideration must be explicit and nonnegative")
        if event.kind in {"split", "conversion"}:
            if not np.isfinite(event.get("ratio", np.nan)) or event.ratio <= 0:
                raise ValueError("share conversion ratio must be positive")
        if event.kind == "conversion" and pd.isna(event.get("new_permno")):
            raise ValueError("a stock conversion needs its destination instrument")
        if event.kind == "conversion" and (
            event.new_permno == event.permno
            or event.new_permno <= 0
            or int(event.new_permno) != event.new_permno
        ):
            raise ValueError("stock conversion destination must be a different valid instrument")
    order = ["event_date", "permno", *(["sequence"] if "sequence" in out else []), "event_id"]
    return out.sort_values(order).reset_index(drop=True)


def build_inputs(
    daily, calendar, events, *, data_version, qualification, fundamentals=None, factors=None
):
    dates = pd.DatetimeIndex(pd.to_datetime(daily.date))
    if dates.isna().any() or dates.max() > CUTOFF:
        raise ValueError("daily observations cross the discovery boundary")
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing or calendar.tz is not None:
        raise ValueError("an ordered, unique exchange calendar is required")
    expected = calendar[(calendar >= dates.min()) & (calendar <= dates.max())]
    if not expected.equals(pd.DatetimeIndex(sorted(dates.unique()))):
        raise ValueError("daily inputs omit or invent declared exchange sessions")
    panel = build_panel(daily, data_version)
    # This phase does not inherit legacy synthetic loss or cross-sectional quote fills.
    panel.fields["returns"] = daily.pivot(index="date", columns="permno", values="dlyret")
    panel.fields["mkt_return"] = _market_return(
        panel.fields["returns"], panel.fields["cap"], panel.universe
    )
    mid = (daily.dlyask + daily.dlybid) / 2
    spread = (daily.dlyask - daily.dlybid) / mid
    valid = daily.dlybid.gt(0) & spread.gt(0) & spread.le(sd.MAX_SPREAD)
    quotes = daily.assign(qualified_spread=spread.where(valid) / 2)
    panel.half_spread = quotes.pivot(
        index="date", columns="permno", values="qualified_spread"
    ).sort_index()
    # CRSP daily capitalization and shares are thousands; accounting inputs are USD millions.
    panel.fields["shares"] = panel.fields["shares"] * 1000
    panel.fields["cap"] = panel.fields["cap"] * 1000
    for name, values in list(panel.fields.items()):
        if isinstance(values, pd.Series):
            panel.fields[name] = pd.DataFrame({n: values for n in panel.universe.columns})
    panel.calendar = calendar
    panel.availability = {
        n: pd.Series(panel.universe.index, index=panel.universe.index) for n in panel.fields
    }
    panel.company = daily.pivot(index="date", columns="permno", values="permco").sort_index()
    if fundamentals is not None:
        if qualification["method_blockers"].get("value"):
            raise ValueError("accounting vintages are not qualified")
        fund = fundamentals.copy()
        required = {
            "permco",
            "datadate",
            "published_at",
            "vintage_at",
            "book",
            "revenue",
            "cogs",
            "assets",
        }
        if not required <= set(fund):
            raise ValueError("fundamental source lacks required values or vintage times")
        fund = fund[sorted(required)].copy()
        for col in ("datadate", "published_at", "vintage_at"):
            fund[col] = pd.to_datetime(fund[col])
        if fund[["datadate", "published_at", "vintage_at"]].isna().any().any():
            raise ValueError("accounting publication and vintage times cannot be missing")
        if (fund.datadate > CUTOFF).any() or (fund.vintage_at > fund.published_at).any():
            raise ValueError("unqualified restatement or future fiscal observation")
        if fund.duplicated(["permco", "datadate", "published_at"]).any():
            raise ValueError("ambiguous accounting source records")
        fund[["book", "revenue", "cogs", "assets"]] *= 1_000_000
        panel.fundamentals = fund.sort_values(["permco", "datadate", "published_at"])
    if factors is not None:
        if qualification["method_blockers"].get("ff3_idiosyncratic_volatility"):
            raise ValueError("regression factor vintage is not qualified")
        required = {"rf", "mkt_excess", "smb", "hml", "published_at", "vintage_at"}
        if not required <= set(factors) or factors.index.max() > CUTOFF:
            raise ValueError("factor history is incomplete or outside discovery")
        if factors[["published_at", "vintage_at"]].isna().any().any():
            raise ValueError("factor publication and vintage times cannot be missing")
        if (pd.to_datetime(factors.vintage_at) > pd.to_datetime(factors.published_at)).any():
            raise ValueError("factor revisions cannot masquerade as original publication")
        extra = {"rmw", "cma", "momentum"} & set(factors)
        for name in sorted((required - {"published_at", "vintage_at"}) | extra):
            panel.fields[name] = pd.DataFrame({n: factors[name] for n in panel.universe.columns})
            panel.availability[name] = pd.to_datetime(factors.published_at)
    opens = daily.pivot(index="date", columns="permno", values="dlyopen").sort_index()
    closes = daily.pivot(index="date", columns="permno", values="dlyprc").sort_index()
    result = HistoricalInputs(
        panel,
        opens,
        closes,
        validate_events(events),
        panel.fields["returns"],
        panel.half_spread * 10000,
        {**qualification, "historical_observations_loaded": True},
    )
    if "ticker" in daily:
        result.symbols = daily.pivot(index="date", columns="permno", values="ticker").sort_index()
    return result
