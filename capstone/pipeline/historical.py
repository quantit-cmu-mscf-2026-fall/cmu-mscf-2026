"""Qualified discovery inputs. Missing source attestations block, never manufacture data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.factors.evaluation import COLUMNS, DAILY_FILE, Panel, _market_return, build_panel
from capstone.pipeline.study import file_digest
from capstone.strategy import input_memo
from capstone.strategy.research import digest

CUTOFF = pd.Timestamp("2020-12-31")
EVENT_COLUMNS = ("event_id", "event_date", "permno", "kind")
REQUIREMENTS = {
    "calendar": {"actual_exchange_sessions"},
    "events": {"complete_event_coverage", "authoritative_cash_and_conversions"},
    "fundamentals": {
        "as_reported_vintages",
        "publication_times",
        "company_links",
        "complete_share_class_coverage",
        "usd_millions",
    },
    "factors": {"publication_times", "qualified_historical_vintage"},
}


class DataBlocked(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("historical inputs blocked: " + ", ".join(report["global_blockers"]))


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


def source_status(descriptor=None, *, root=None, verify=True):
    """Which declared sources are present, manifest-bound and qualified.

    A source file is only hashed here (its bytes streamed through SHA-256 and compared with the
    manifest); no row is parsed. Observations reach the process only through `sd.load`.
    """
    root = Path(root or sd.cache_dir())
    descriptor = descriptor or {}
    manifest = root / "SHA256SUMS"
    hashes = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                sha, name = line.split(maxsplit=1)
                hashes[name.lstrip("*")] = sha
    result = {
        "schema_version": 1,
        "source_hashes": {},
        "sources": {},
        "global_blockers": [],
        "method_blockers": {},
    }
    if descriptor.get("schema_version") != 1:
        result["global_blockers"].append("source_descriptor_schema_missing_or_unsupported")
    daily_name = DAILY_FILE + ".parquet"
    if daily_name not in hashes or not (root / daily_name).exists():
        result["global_blockers"].append("daily_source_missing")
    elif verify and file_digest(root / daily_name) != hashes[daily_name]:
        result["global_blockers"].append("daily_checksum_mismatch")
    else:
        result["source_hashes"]["daily"] = hashes[daily_name]
    for role, required in REQUIREMENTS.items():
        item = descriptor.get("sources", {}).get(role, {})
        name = item.get("dataset", "")
        reasons = []
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            reasons.append(f"{role}_source_missing")
        else:
            filename = name + ".parquet"
            expected = hashes.get(filename)
            if not expected or expected != item.get("sha256"):
                reasons.append(f"{role}_manifest_unbound")
            elif not (root / filename).exists():
                reasons.append(f"{role}_file_missing")
            elif verify and file_digest(root / filename) != expected:
                reasons.append(f"{role}_checksum_mismatch")
            else:
                result["source_hashes"][role] = expected
        if not required <= set(item.get("qualifications", [])):
            reasons.append(f"{role}_qualification_missing")
        try:
            first, last = pd.Timestamp(item["start"]), pd.Timestamp(item["end"])
            needed = pd.Timestamp(
                {
                    "calendar": "1990-01-01",
                    "events": "1996-01-01",
                    "fundamentals": "1994-01-01",
                    "factors": "1995-01-01",
                }[role]
            )
            if pd.isna(first) or pd.isna(last) or first > needed or last < CUTOFF:
                reasons.append(f"{role}_coverage_incomplete")
            if role == "calendar" and last <= CUTOFF:
                reasons.append("calendar_next_session_missing")
        except (KeyError, ValueError, TypeError):
            reasons.append(f"{role}_coverage_missing")
        result["sources"][role] = {"dataset": name, "qualified": not reasons, "blockers": reasons}
        if role in {"calendar", "events"}:
            result["global_blockers"] += reasons
        else:
            methods = (
                ("value", "profitability")
                if role == "fundamentals"
                else ("ff3_idiosyncratic_volatility",)
            )
            for method in methods:
                result["method_blockers"][method] = reasons
    result["descriptor_hash"] = digest(descriptor)
    result["historical_observations_loaded"] = False
    result["holdout_opened"] = False
    return result


def preflight(descriptor_path=None):
    path = Path(descriptor_path) if descriptor_path else sd.cache_dir() / "historical-sources.json"
    descriptor = json.loads(path.read_text()) if path.exists() else {}
    report = source_status(descriptor)
    report["descriptor_present"] = path.exists()
    return report, descriptor


def discovery_read(
    name,
    columns=None,
    *,
    start="1990-01-01",
    end="2020-12-31",
    date_column="date",
    knowledge_columns=(),
):
    if pd.Timestamp(end) > CUTOFF or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("historical loader cannot read the sealed holdout")
    kwargs = {"knowledge_columns": tuple(knowledge_columns)} if knowledge_columns else {}
    return sd.load(
        name,
        columns=columns,
        start=start,
        end=end,
        auto_sync=False,
        date_column=date_column,
        **kwargs,
    )


def read_calendar(name):
    """Only session dates may cross the boundary, never a market observation."""
    frame = sd.load(name, columns=["date"], start="1990-01-01", end="2021-01-31", auto_sync=False)
    # The verified daily CRSP file can supply dates for many securities per session.
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date)).drop_duplicates().sort_values()
    if dates.hasnans or dates.tz is not None:
        raise ValueError("calendar session dates must be nonmissing and timezone naive")
    future = dates[dates > CUTOFF]
    if future.empty:
        raise ValueError("calendar needs the first following session date")
    return dates[dates <= future.min()]


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


def load_inputs(descriptor_path=None):
    report, descriptor = preflight(descriptor_path)
    if report["global_blockers"]:
        raise DataBlocked(report)
    sources = descriptor["sources"]
    daily = discovery_read(DAILY_FILE, sorted(set(COLUMNS + ["permco", "ticker"])))
    calendar = read_calendar(sources["calendar"]["dataset"])
    if (
        pd.Timestamp(daily.date.min()) != calendar[0]
        or pd.Timestamp(daily.date.max()) != calendar[calendar <= CUTOFF][-1]
    ):
        raise ValueError("historical daily coverage must include full 1990–2020 sessions")
    events = discovery_read(sources["events"]["dataset"], date_column="event_date")
    fundamentals = factors = None
    if report["sources"]["fundamentals"]["qualified"]:
        fundamentals = discovery_read(
            sources["fundamentals"]["dataset"],
            date_column="datadate",
            knowledge_columns=("published_at", "vintage_at"),
        )
    if report["sources"]["factors"]["qualified"]:
        factors = discovery_read(
            sources["factors"]["dataset"], knowledge_columns=("published_at", "vintage_at")
        ).set_index("date")
    result = build_inputs(
        daily,
        calendar,
        events,
        data_version=sd.data_version(),
        qualification=report,
        fundamentals=fundamentals,
        factors=factors,
    )
    from capstone.pipeline.event_checks import verify_event_returns

    result.qualification["event_return_identities"] = verify_event_returns(result)
    after, _ = preflight(descriptor_path)
    if (
        after["source_hashes"] != report["source_hashes"]
        or after["descriptor_hash"] != report["descriptor_hash"]
    ):
        raise ValueError("historical sources changed while they were being loaded")
    return result
