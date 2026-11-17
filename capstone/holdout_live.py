"""The post-holdout read route for live-era CRSP.

A separate module: `holdout_authority` is unchanged by it, and its
`guard_shared_read` still refuses every live-era read. The holdout's end is read from
`holdout_authority.REAL_PERIOD` at each call and is never repeated here.
"""

from __future__ import annotations

from capstone import holdout_authority as authority


class LiveReadRefused(authority.HoldoutSealed):
    """A post-holdout (live-era) CRSP read the live route refuses; `reason` is its type."""

    def __init__(self, reason, message):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


LIVE_READ_REASONS = (
    "live_read_not_crsp",  # only licensed `crsp_*` files have a live-era route
    "live_read_date_column",  # the file's authoritative date filter is required
    "live_read_unbounded",  # no start, or a bound that is not a plain date
    "live_read_touches_holdout",  # the window starts on or before the holdout's end
    "live_read_inverted",  # the window ends before it starts
)


def guard_live_read(name, start, end, date_column):
    """The post-holdout read route for live-era CRSP .

    Additional to `holdout_authority.guard_shared_read`: that module is unchanged, and its
    guard still refuses any protected read ending after `DISCOVERY_END` without a holdout
    access. This route admits only a window that starts after the sealed holdout's end
    (`REAL_PERIOD[1]`, read here, never repeated) and ends on or after its start (or is
    open-ended), so no row inside the holdout can match it. Anything else is
    `LiveReadRefused` with a typed
    `reason` (`LIVE_READ_REASONS`). It opens nothing and needs no access: the holdout
    authority, its records and its terminal look are not touched."""
    if not isinstance(name, str) or not name.casefold().startswith("crsp_"):
        raise LiveReadRefused("live_read_not_crsp", f"{name!r} is not a crsp_* shared file")
    try:
        licensed = authority.protected_dataset(name) == authority.REAL_DATASET
    except authority.HoldoutSealed as error:  # a path alias, not a canonical file name
        raise LiveReadRefused("live_read_not_crsp", f"{name!r}: {error}") from error
    if not licensed:
        raise LiveReadRefused("live_read_not_crsp", f"{name!r} is not a licensed CRSP file")
    folded = name.casefold()
    expected = "event_date" if ("event" in folded or "action" in folded) else "date"
    if date_column != expected:
        raise LiveReadRefused(
            "live_read_date_column", f"{name} filters on {expected!r}, not {date_column!r}"
        )
    holdout_end = authority.REAL_PERIOD[1]
    try:
        first = None if start is None else authority._day(start)
        last = None if end is None else authority._day(end)
    except authority.HoldoutSealed as error:
        raise LiveReadRefused("live_read_unbounded", f"{start!r}..{end!r}: {error}") from error
    if first is None:
        raise LiveReadRefused("live_read_unbounded", "a live read needs an explicit start")
    if first <= holdout_end or (last is not None and last <= holdout_end):
        raise LiveReadRefused(
            "live_read_touches_holdout",
            f"window {first}..{last or 'open'} is not after the sealed holdout's end {holdout_end}",
        )
    if last is not None and last < first:
        raise LiveReadRefused("live_read_inverted", f"window {first}..{last} ends before it starts")
    return first, last
