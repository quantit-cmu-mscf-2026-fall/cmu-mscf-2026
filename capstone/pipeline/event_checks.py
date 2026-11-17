"""Independent observed-event versus CRSP total-return identities, before performance."""

import math

import pandas as pd

from capstone.pipeline.historical import validate_events


def return_identity(previous_value, current_value, expected_return, *, tolerance=1e-8):
    """The return implied by two observed values, if it matches the observed total return."""
    if (
        not all(math.isfinite(v) for v in (previous_value, current_value, expected_return))
        or previous_value <= 0
    ):
        raise ValueError("return identity needs finite observed values")
    actual = current_value / previous_value - 1
    if abs(actual - expected_return) > tolerance:
        raise ValueError("corporate action and observed total return do not reconcile")
    return actual


def _event_return(inputs, sequence, *, tolerance=1e-6):
    """Check one unit's economic entitlement; whole-share execution is a separate check."""
    day, name = pd.Timestamp(sequence.event_date.iloc[0]), int(sequence.permno.iloc[0])
    if (sequence.kind == "unsupported").any():
        raise ValueError("unsupported observed event")
    dates = inputs.closes.index
    earlier = dates[dates < day]
    if earlier.empty:
        raise ValueError("corporate action lacks its prior raw closing valuation")
    prior = float(inputs.closes.loc[earlier[-1], name])
    expected = float(inputs.total_returns.loc[day, name])
    # A unit return must not require cash-in-lieu for a hypothetical fractional share.
    positions, claims = {name: 1.0}, 0.0
    for event in sequence.itertuples(index=False):
        n = int(event.permno)
        quantity = positions.get(n, 0.0)
        if event.kind == "dividend":
            claims += quantity * float(event.amount)
        elif event.kind == "split":
            positions[n] = quantity * float(event.ratio)
        elif event.kind == "conversion":
            positions.pop(n, None)
            destination = int(event.new_permno)
            positions[destination] = positions.get(destination, 0) + quantity * float(event.ratio)
            claims += quantity * float(event.amount)
        elif event.kind == "settlement":
            positions.pop(n, None)
            claims += quantity * float(event.amount)
    value = claims
    for target, quantity in positions.items():
        if quantity:
            close = float(inputs.closes.loc[day, target])
            if not math.isfinite(close) or close <= 0:
                raise ValueError("corporate action lacks a current raw closing valuation")
            value += quantity * close
    return return_identity(prior, value, expected, tolerance=tolerance)


def verify_event_returns(inputs, *, tolerance=1e-6):
    checks = []
    events = validate_events(inputs.events)
    for (event_day, event_name), sequence in events.groupby(["event_date", "permno"], sort=True):
        context = {"date": str(pd.Timestamp(event_day).date()), "permno": int(event_name)}
        if (sequence.kind == "unsupported").any():
            checks.append({**context, "status": "unsupported"})
            continue  # Completion is blocked if this instrument is held, in ActionBook.
        try:
            actual = _event_return(inputs, sequence, tolerance=tolerance)
        except (ValueError, KeyError) as error:
            raise ValueError(f"event identity failed:{context}:{error}") from error
        checks.append({**context, "status": "verified", "return": actual})
    return {"tolerance": tolerance, "checks": checks, "verified": True}


def classify_event_returns(inputs, *, tolerance=1e-6):
    """Retain an unverified event as a held-position blocker, never remove its row."""
    events, audit = validate_events(inputs.events), []
    for (day, name), sequence in events.groupby(["event_date", "permno"], sort=True):
        if (sequence.kind == "unsupported").any():
            continue
        try:
            _event_return(inputs, sequence, tolerance=tolerance)
        except (ValueError, KeyError) as error:
            events.loc[sequence.index, "kind"] = "unsupported"
            events.loc[sequence.index, "reason"] = "unverified_observed_return_identity"
            audit.append(
                {
                    "date": str(day.date()),
                    "permno": int(name),
                    "event_ids": sequence.event_id.tolist(),
                    "reason": str(error),
                }
            )
    return validate_events(events), audit
