"""Observation-only historical views; slicing grants no input qualification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd

from capstone.pipeline.historical import CUTOFF, validate_events
from capstone.strategy.program_schedule import session_date


def _dated(frame, through):
    if frame is None:
        return None
    if (
        not isinstance(frame, (pd.DataFrame, pd.Series))
        or not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.has_duplicates
        or frame.index.hasnans
        or frame.index.tz is not None
        or not frame.index.equals(frame.index.normalize())
    ):
        raise ValueError("prefix observations need unique normalized session dates")
    return frame.loc[frame.index <= through].sort_index().copy(deep=True)


def _stock_frame(frame, through, names):
    selected = _dated(frame, through)
    if isinstance(selected, pd.DataFrame):
        if selected.columns.has_duplicates:
            raise ValueError("prefix observations need unique instrument columns")
        selected = selected.reindex(columns=names)
    return selected


def _fundamentals(frame, through, company):
    if frame is None:
        return None
    required = {"permco", "datadate", "published_at", "vintage_at"}
    if not isinstance(frame, pd.DataFrame) or not required <= set(frame):
        raise ValueError("prefix accounting needs explicit publication and vintage dates")
    if not isinstance(company, pd.DataFrame):
        raise ValueError("company-level accounting needs dated security-company links")
    companies = set(company.stack(future_stack=True).dropna())
    keep = frame.permco.isin(companies)
    for key in ("datadate", "published_at", "vintage_at"):
        values = pd.to_datetime(frame[key])
        if values.dt.tz is not None or values.isna().any():
            raise ValueError("prefix accounting knowledge dates must be explicit")
        keep &= values <= through
    return frame.loc[keep].sort_values(sorted(required)).reset_index(drop=True).copy()


def input_prefix(inputs, through):
    """Keep observed sessions and known instruments through an explicit cutoff.

    Retain one following date-only calendar label for timing. Effective corporate
    events through the cutoff retain their supplied later payment metadata; this
    is not an announcement-time feed. The caller must separately establish as-of
    source/event qualification. Original full-source attestations are provenance,
    never a new completed-prefix receipt or admission decision.
    """
    cutoff = session_date(through, "completed prefix cutoff")
    calendar = pd.DatetimeIndex(inputs.panel.calendar)
    if (
        cutoff > CUTOFF
        or cutoff not in calendar
        or calendar.has_duplicates
        or calendar.hasnans
        or calendar.tz is not None
        or not calendar.equals(calendar.normalize())
    ):
        raise ValueError("prefix cutoff needs an authoritative discovery session")
    calendar = calendar.sort_values()
    if calendar.get_loc(cutoff) + 1 == len(calendar):
        raise ValueError("prefix timing needs the next date-only session label")
    calendar = calendar[: calendar.get_loc(cutoff) + 2]
    universe = _dated(inputs.panel.universe, cutoff)
    if (
        universe.empty
        or universe.columns.has_duplicates
        or universe.isna().any().any()
        or not universe.isin([True, False]).all().all()
    ):
        raise ValueError("prefix universe requires explicit membership observations")
    expected = calendar[(calendar >= universe.index[0]) & (calendar <= cutoff)]
    if universe.index[-1] != cutoff or not universe.index.equals(expected):
        raise ValueError("prefix universe must cover its authoritative session history")
    names = set(universe.columns[universe.astype(bool).any(axis=0)])
    if "event_date" not in inputs.events:
        raise ValueError("prefix events need explicit effective dates")
    dates = pd.to_datetime(inputs.events.event_date)
    if dates.dt.tz is not None or dates.isna().any():
        raise ValueError("prefix events need explicit normalized effective dates")
    events = validate_events(inputs.events.loc[dates <= cutoff].copy())
    if not events.event_date.equals(events.event_date.dt.normalize()):
        raise ValueError("prefix effective events require normalized session dates")
    # A known conversion can create a held instrument outside the equity universe.
    while True:
        reachable = {
            int(row.new_permno)
            for row in events.itertuples()
            if row.kind == "conversion" and row.permno in names
        }
        if reachable <= names:
            break
        names.update(reachable)
    names = sorted(names)
    events = events.loc[events.permno.isin(names)].reset_index(drop=True)
    fields, availability = {}, {}
    supplied = inputs.panel.availability or {}
    for name, frame in inputs.panel.fields.items():
        values = _stock_frame(frame, cutoff, names)
        if name in supplied:
            observed = _dated(supplied[name], cutoff)
            known = pd.to_datetime(observed)
            if not isinstance(known, pd.Series) or known.dt.tz is not None:
                raise ValueError("prefix feature availability needs explicit dates")
            known = known.where(known <= cutoff)
            # Future revisions may not expose their numeric values or dates.
            allowed = known.reindex(values.index).notna()
            values.loc[~allowed] = float("nan")
            availability[name] = known
        else:
            # Unknown availability remains a blocker, not silently usable values.
            values.loc[:] = float("nan")
        fields[name] = values
    qualification = deepcopy(inputs.qualification)
    qualification["prefix_view"] = {
        "cutoff": str(cutoff.date()),
        "source_qualification_recomputed": False,
        "event_scope": "observed_effective_events; supplied_payment_metadata",
        "formal_admission": False,
    }
    company = _stock_frame(inputs.panel.company, cutoff, names)
    panel = replace(
        inputs.panel,
        fields=fields,
        universe=universe.reindex(columns=names, fill_value=False),
        half_spread=_stock_frame(inputs.panel.half_spread, cutoff, names),
        sic=_stock_frame(inputs.panel.sic, cutoff, names),
        company=company,
        fundamentals=_fundamentals(inputs.panel.fundamentals, cutoff, company),
        calendar=calendar,
        availability=availability,
        target_cache=None,
    )
    return replace(
        inputs,
        panel=panel,
        opens=_stock_frame(inputs.opens, cutoff, names),
        closes=_stock_frame(inputs.closes, cutoff, names),
        events=events,
        total_returns=_stock_frame(inputs.total_returns, cutoff, names),
        half_spread_bps=_stock_frame(inputs.half_spread_bps, cutoff, names),
        qualification=qualification,
        schedules={},
        symbols=_stock_frame(inputs.symbols, cutoff, names),
    )
