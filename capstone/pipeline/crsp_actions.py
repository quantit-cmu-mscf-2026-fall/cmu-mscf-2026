"""Translate observed CRSP event terms; incomplete events remain explicit blockers."""

import math
from dataclasses import replace

import pandas as pd

from capstone.pipeline.historical import CUTOFF, validate_events


def normalize(distributions, delistings, calendar):
    """Return position events and a disposition for every source row in discovery."""
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("CRSP normalization needs the verified session calendar")
    for frame, date in ((distributions, "disexdt"), (delistings, "delistingdt")):
        if frame[date].isna().any() or (pd.to_datetime(frame[date]) > CUTOFF).any():
            raise ValueError("corporate-action source crosses the discovery boundary")
    if distributions.duplicated(["permno", "disexdt", "disseqnbr"]).any():
        raise ValueError("duplicate CRSP distribution identity")
    if any(n < 1 or int(n) != n for n in distributions.disseqnbr):
        raise ValueError("source-defined distribution sequencing must be positive whole numbers")
    if delistings.deldlydt.isna().any():
        raise ValueError("terminal events need an observed effective date")
    terminal = {}
    for row in delistings.itertuples(index=False):
        key = (int(row.permno), pd.Timestamp(row.deldlydt))
        if key in terminal:
            raise ValueError("ambiguous delisting source")
        terminal[key] = row
    events, audit, matched = [], [], set()

    def effective(date):
        offset = calendar.searchsorted(pd.Timestamp(date))
        return calendar[offset] if offset < len(calendar) else None

    for row in distributions.sort_values(["permno", "disexdt", "disseqnbr"]).itertuples(
        index=False
    ):
        day = effective(row.disexdt)
        identity = f"crsp-dis:{row.permno}:{pd.Timestamp(row.disexdt).date()}:{row.disseqnbr}"
        if day is None or day > CUTOFF or day < pd.Timestamp("1996-01-01"):
            audit.append({"source_id": identity, "status": "outside_replay_window"})
            continue
        event = {
            "event_id": identity,
            "event_date": day,
            "permno": int(row.permno),
            "sequence": int(row.disseqnbr),
            "kind": "unsupported",
            "source_ex_date": str(pd.Timestamp(row.disexdt).date()),
            "reason": "unsupported_distribution_terms",
        }
        if row.distype == "TSOO":
            audit.append({"source_id": identity, "status": "shares_outstanding_observation_only"})
            continue
        valid_cash = pd.notna(row.disdivamt) and math.isfinite(row.disdivamt) and row.disdivamt >= 0
        payment = pd.Timestamp(row.dispaydt)
        if row.distype in {"CD", "SD", "ROC", "CG"} and row.dispaymenttype == "USD":
            if valid_cash and pd.notna(payment) and payment >= day:
                event.update(
                    kind="dividend",
                    amount=float(row.disdivamt),
                    pay_date=payment,
                    reason="observed_usd_distribution",
                )
            else:
                event["reason"] = "missing_cash_amount_or_payment_date"
        elif row.distype == "FRS" and row.dispaymenttype == "SS":
            ratio = 1 + float(row.disfacshr)
            if math.isfinite(ratio) and ratio > 0:
                event.update(kind="split", ratio=ratio, reason="observed_same_security_split")
        elif row.distype == "CP" and row.dispaymenttype == "USD":
            key = (int(row.permno), day)
            closing = terminal.get(key)
            if closing is not None and closing.delpaymenttype == "CASH":
                if valid_cash and pd.notna(payment) and payment <= day:
                    event.update(
                        kind="settlement",
                        amount=float(row.disdivamt),
                        observed_pay_date=str(payment.date()),
                        reason="observed_cash_terminal_distribution",
                    )
                    matched.add(key)
                else:
                    event["reason"] = "terminal_cash_payment_not_available_on_effective_session"
        events.append(event)
        audit.append({"source_id": identity, "status": event["kind"], "reason": event["reason"]})
    for key, row in terminal.items():
        day = effective(row.deldlydt)
        identity = f"crsp-del:{row.permno}:{pd.Timestamp(row.delistingdt).date()}"
        if day is None or day > CUTOFF or day < pd.Timestamp("1996-01-01"):
            audit.append({"source_id": identity, "status": "outside_replay_window"})
        elif key in matched:
            audit.append({"source_id": identity, "status": "represented_by_terminal_distribution"})
        else:
            events.append(
                {
                    "event_id": identity,
                    "event_date": day,
                    "permno": int(row.permno),
                    "sequence": 1000000,
                    "kind": "unsupported",
                    "reason": "unqualified_terminal_conversion_or_cash_terms",
                }
            )
            audit.append(
                {
                    "source_id": identity,
                    "status": "unsupported",
                    "reason": "unqualified_terminal_conversion_or_cash_terms",
                }
            )
    result = pd.DataFrame(events)
    if result.empty:
        result = pd.DataFrame(columns=["event_id", "event_date", "permno", "kind", "sequence"])
    if len(audit) != len(distributions) + len(delistings):
        raise AssertionError("every source event requires a disposition")
    return validate_events(result), sorted(audit, key=lambda row: row["source_id"])


# program-events-v5 / observed-benchmark-next-open-v7 (a declared rule): an
# `unsupported` CRSP distribution or delisting whose terms the archive lacks is valued
# from CRSP's own total return on its effective session. Non-terminal: the distributed
# property is worth (1 + TR) * prior close - close per share, booked as cash on that
# session; terminal (a delisting reason in the group and no later valid close): the
# holder receives (1 + TR) * prior close per share and the position retires.
# Equivalent to liquidating what was distributed at the effective-session close
# without trading cost.
HELD_VALUE_RULE = "crsp_total_return_value_v1"
VALUABLE = {
    "unsupported_distribution_terms",
    "missing_cash_amount_or_payment_date",
    "unqualified_terminal_conversion_or_cash_terms",
    "terminal_cash_payment_not_available_on_effective_session",
}
TERMINAL = {
    "unqualified_terminal_conversion_or_cash_terms",
    "terminal_cash_payment_not_available_on_effective_session",
}


def _price(frame, day, name):
    try:
        value = float(frame.loc[day, name])
    except KeyError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def valued_events(events, closes, total_returns, *, as_of=False):
    """(events with valuable `unsupported` groups re-typed, audit). Each event group is
    one (session, permno), walked in source sequence: observed same-session dividends
    are paid as usual and splits applied as usual, and both are removed from the value,
    which is stated per share held at the first unsupported event. A group stays
    `unsupported` with a specific `reason` when it holds another kind (or, if terminal,
    a split or a later dividend), lacks an observation, or implies a negative value.

    `as_of` (`as_of_event_session_v1`): a group is terminal from its CRSP delisting
    record alone. Without it (v5/v6 routes) a terminal group followed by any valid close
    is `relisted_after_terminal_event`, which reads closes after the event session."""
    events, audit = validate_events(events), []
    if "reason" not in events:
        events["reason"] = None
    events["pay_date"] = pd.to_datetime(events.get("pay_date", pd.NaT))
    sessions = closes.index
    for (day, name), group in events.groupby(["event_date", "permno"], sort=True):
        if not (group.kind == "unsupported").any():
            continue
        mine = group[group.kind == "unsupported"]
        reasons, ids = set(mine.reason), mine.event_id.tolist()
        position = sessions.searchsorted(day)
        prior = _price(closes, sessions[position - 1], name) if position else math.nan
        close, total = _price(closes, day, name), _price(total_returns, day, name)
        terminal = bool(reasons & TERMINAL)
        later = (
            closes[name].loc[closes.index > day]
            if name in closes and not as_of
            else pd.Series(dtype=float)
        )
        units, paid, first, mixed, settled = 1.0, 0.0, None, False, False
        for row in group.itertuples():
            if row.kind == "unsupported":
                first = units if first is None else first
            elif row.kind == "dividend":
                paid += units * float(row.amount)
                mixed |= terminal and first is not None
            elif row.kind == "split":
                units *= float(row.ratio)
                mixed |= terminal
            elif row.kind == "settlement":  # observed terminal cash, after the unsupported
                paid += units * float(row.amount)
                settled, mixed = True, mixed or terminal or first is None
            else:
                mixed = True
        held = 0 if terminal or settled else units * close
        value = ((1 + total) * prior - paid - held) / first
        if mixed:
            reason = "mixed_event_group_unvalued"
        elif not reasons <= VALUABLE:
            reason = sorted(reasons - VALUABLE, key=str)[0]
        elif (terminal or settled) and (later > 0).any():
            reason = "relisted_after_terminal_event"
        elif not (prior > 0 and math.isfinite(value)):
            reason = "total_return_value_unavailable"
        elif value < -1e-9 * prior:
            reason = "negative_total_return_residual"
        else:
            kind, value = ("terminal_value" if terminal else "distribution_value"), max(value, 0)
            events.loc[mine.index, "kind"] = kind
            events.loc[mine.index, "amount"] = [value] + [0.0] * (len(mine) - 1)
            events.loc[mine.index, "pay_date"] = day
            events.loc[mine.index, "reason"] = HELD_VALUE_RULE
            audit.append({"event_ids": ids, "status": kind, "amount": value})
            continue
        events.loc[mine.index, "reason"] = reason
        audit.append({"event_ids": ids, "status": "unsupported", "reason": reason})
    # actions.CASH_IN_LIEU_CLOSE: the surviving security's effective-session close.
    shares = events.index[events.kind.isin(["split", "conversion"])]
    events["cash_in_lieu_close"] = math.nan
    for i in shares:
        row = events.loc[i]
        survivor = int(row.new_permno) if row.kind == "conversion" else int(row.permno)
        events.loc[i, "cash_in_lieu_close"] = _price(closes, row.event_date, survivor)
    return validate_events(events), audit


def held_inputs(inputs, execution_version):
    """Historical inputs as a replay under `execution_version` consumes them: valued
    events for program-events-v5 / benchmark v7, the archive's own table otherwise."""
    from capstone.execution.contracts import FALLBACK_VERSIONS, HELD_ACTION_VERSIONS

    if execution_version not in HELD_ACTION_VERSIONS:
        return inputs
    as_of = execution_version in FALLBACK_VERSIONS
    events = valued_events(inputs.events, inputs.closes, inputs.total_returns, as_of=as_of)[0]
    return replace(inputs, events=events)
