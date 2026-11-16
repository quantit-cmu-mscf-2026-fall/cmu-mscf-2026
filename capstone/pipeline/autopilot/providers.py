"""Offline market-data providers for the autopilot `daily` stage.

A provider answers these questions about decision sessions `D` (`YYYY-MM-DD`):

- `sessions(after)`: the completed decision sessions after `after` (None: all), ascending.
  A session is complete when its next session (the execution session) is known too, so
  its `market` document exists.
- `market(D)`: the `lifecycle simulate --market` document, `{"session": <execution
  session>, "prices": {permno: price}}`.
- `marks(D)`: the `lifecycle carry --marks` document, `{"session": D, "prices": ...}`.
- `events(D)`: the `lifecycle carry --events` rows dated D, or None.
- `panel(D)` (optional method): the current-data panel through D in the JSON panel
  document format, or None.

A missing document is `MarketDataMissing`; a provider that is not configured explicitly
is `ProviderNotConfigured`. Neither ever falls back to a default location.

Everything here is offline: no provider opens a network connection or reads a
credential, and no call is paid. A live provider would have to be injected by the
caller (`services["market_data_provider"]`) and count its calls against the caller's
call budget; this module ships none.

- `FixtureProvider`: in-memory documents (tests construct it from synthetic panels), or
  the folder `provider_config.root` holding `<D>/market.json`, `<D>/marks.json` and an
  optional `<D>/events.json` (`provider = "fixture"`).
- `ArchiveReplayProvider`: replays a qualified historical input archive
  (`historical_bundle.load_inputs`): execution prices are the next session's opens,
  marks are D's closes, events are the archive's event rows dated D. It refuses unless
  `provider = "archive_replay"` and `provider_config.root` names the archive explicitly;
  it never reads `[autopilot] input_archive` or a shared-data cache on its own. The
  archive's recorded cutoff is checked before any frame is loaded.

The sealed holdout: no market, marks or events document, and no panel row or panel
calendar day, may carry a date inside 2021-01-01..2025-12-31 (`HOLDOUT`).
`require_outside_holdout` raises `HoldoutPrices` (not a `ProviderDataError`: the `daily`
stage turns it into a STOP, never a blocker that waits). A 2020-12-31 decision session
executes on a 2021 open, so it can never be simulated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import pandas as pd

DAY = "%Y-%m-%d"
ARCHIVE_CUTOFF = "2020-12-31"  # capstone.pipeline.historical.CUTOFF
HOLDOUT = ("2021-01-01", "2025-12-31")  # sealed research holdout, both ends inclusive


class MarketDataProvider(Protocol):
    """Supplies each decision session's files. Tests pass a fixture provider through
    `services["market_data_provider"]`; the archive-replay provider refuses unless
    `[daily] provider = "archive_replay"` and `provider_config` name it explicitly."""

    def sessions(self, after: str | None) -> list[str]:
        """Completed decision sessions (YYYY-MM-DD) after `after`, ascending."""

    def market(self, session: str) -> dict:
        """The `simulate --market` document for the session."""

    def marks(self, session: str) -> dict:
        """The `carry --marks` document for the session."""

    def events(self, session: str) -> list | None:
        """The `carry --events` rows, or None when there are none."""


class ProviderNotConfigured(ValueError):
    code = "provider_not_configured"


class MarketDataMissing(LookupError):
    code = "market_data_missing"


class HoldoutPrices(Exception):
    """A provider document dated inside the sealed holdout: a STOP, never a blocker."""

    code = "holdout_prices"


def in_holdout(day):
    return HOLDOUT[0] <= day <= HOLDOUT[1]


def require_outside_holdout(session, market=None, marks=None, events=None, panel=None):
    """Refuse any market / marks / events date, panel frame row or panel calendar day
    inside `HOLDOUT` (`HoldoutPrices`).

    Dates that are not YYYY-MM-DD are left to the schema checks; a date-like value is
    normalized first, so `2021-01-04T00:00:00` is refused too."""
    found = []
    for kind, document in (("market", market), ("marks", marks)):
        if isinstance(document, dict):
            found.append((f"{kind}.json session", document.get("session")))
    for row in events if isinstance(events, list) else []:
        if isinstance(row, dict):
            found.append((f"event {row.get('event_id')!r} event_date", row.get("event_date")))
    found += [(f"panel.json {where}", day) for where, day in panel_days(panel)]
    for where, value in found:
        try:
            day = _day(value) if isinstance(value, str) else None
        except (TypeError, ValueError):
            day = None
        if day is not None and in_holdout(day):
            raise HoldoutPrices(
                f"session {session}: {where} {value!r} is inside the sealed holdout "
                f"{HOLDOUT[0]}..{HOLDOUT[1]}; holdout-era prices never enter a simulation"
            )


class ProviderDataError(Exception):
    """A drop-folder problem that blocks (the tick waits); `code` is the blocker code."""

    code = "data_invalid"


class DataMissing(ProviderDataError, LookupError):
    code = "data_missing"


class DataStale(ProviderDataError):
    code = "data_stale"


class DataInvalid(ProviderDataError, ValueError):
    code = "data_invalid"


def _day(value):
    return pd.Timestamp(value).strftime(DAY)


def _is_day(text):
    try:
        return isinstance(text, str) and _day(text) == text
    except (TypeError, ValueError):
        return False


def _panel_frames(panel):
    """(name, frame) of the universe, spreads, fields and availability of a panel."""
    frames = [("universe", panel.get("universe")), ("half_spread", panel.get("half_spread"))]
    for group, label in (("fields", "field"), ("availability", "availability")):
        table = panel.get(group)
        if isinstance(table, dict):
            frames += [(f"{label}:{name}", frame) for name, frame in table.items()]
    return frames


def panel_days(panel):
    """(where, day) of every frame row and calendar day of a JSON panel document;
    parts that are not lists are left to the schema checks."""
    if not isinstance(panel, dict):
        return []
    calendar = panel.get("calendar")
    found = [("calendar day", day) for day in calendar] if isinstance(calendar, list) else []
    for name, frame in _panel_frames(panel):
        index = frame.get("index") if isinstance(frame, dict) else None
        if isinstance(index, list):
            found += [(f"`{name}` row", day) for day in index]
    return found


def _prices(row):
    """{permno: price} for the positive finite prices of one row."""
    out = {}
    for name, value in row.items():
        if value is not None and math.isfinite(float(value)) and float(value) > 0:
            out[str(int(name))] = float(value)
    return out


class FixtureProvider:
    """Documents per session: `{D: {"market": {...}, "marks": {...}, "events": [...]}}`."""

    def __init__(self, documents):
        self.documents = {_day(day): dict(item) for day, item in documents.items()}
        self.reads = []  # (kind, session), so a test can prove what was read

    @classmethod
    def from_closes(cls, closes, *, opens=None, events=None):
        """Every session but the last of a synthetic close frame (index: sessions)."""
        closes = closes.sort_index()
        opens = closes if opens is None else opens.sort_index()
        days = [_day(d) for d in closes.index]
        documents = {}
        for index, day in enumerate(days[:-1]):
            following = closes.index[index + 1]
            documents[day] = {
                "market": {"session": days[index + 1], "prices": _prices(opens.loc[following])},
                "marks": {"session": day, "prices": _prices(closes.iloc[index])},
                "events": (events or {}).get(day),
            }
        return cls(documents)

    @classmethod
    def from_folder(cls, root):
        root = Path(root)
        if not root.is_dir():
            raise ProviderNotConfigured(f"fixture provider root {root} is not a folder")
        documents = {}
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            item = {}
            for kind in ("market", "marks", "events"):
                path = folder / f"{kind}.json"
                if path.is_file():
                    item[kind] = json.loads(path.read_text(encoding="utf-8"))
            documents[folder.name] = item
        return cls(documents)

    def sessions(self, after):
        return [d for d in sorted(self.documents) if after is None or d > after]

    def _get(self, kind, session):
        self.reads.append((kind, session))
        document = self.documents.get(_day(session), {}).get(kind)
        if document is None and kind not in ("events", "panel"):
            raise MarketDataMissing(f"no {kind} document for session {session}")
        return document

    def market(self, session):
        return self._get("market", session)

    def marks(self, session):
        return self._get("marks", session)

    def events(self, session):
        return self._get("events", session) or None

    def panel(self, session):
        return self._get("panel", session)


class ArchiveReplayProvider:
    """Sessions, opens, closes and events of one qualified historical input archive.

    `historical-inputs.json` is read first and checked against the research cutoff
    (2020-12-31) before any frame is loaded: an archive recording another cutoff is
    refused, the session list comes from its recorded calendar (decision sessions whose
    execution session is on or before the cutoff too), and a session after the cutoff is
    `data_invalid` without loading anything."""

    def __init__(self, root):
        self.root = Path(root)
        self._inputs = None
        self._metadata = None
        self.reads = []

    def metadata(self):
        if self._metadata is None:
            path = self.root / "historical-inputs.json"
            if not path.is_file():
                raise ProviderNotConfigured(f"{self.root} is not a historical input archive")
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                calendar = sorted(_day(d) for d in metadata["calendar"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ProviderNotConfigured(f"{path} is unreadable: {error}") from error
            if metadata.get("cutoff") != ARCHIVE_CUTOFF:
                raise ProviderNotConfigured(
                    f"{self.root} records cutoff {metadata.get('cutoff')!r}, not {ARCHIVE_CUTOFF}"
                )
            self._metadata = {"calendar": calendar}
        return self._metadata

    def _within_cutoff(self, session):
        """A session after the archive cutoff is refused before any frame is loaded."""
        self.metadata()
        if _day(session) > ARCHIVE_CUTOFF:
            raise DataInvalid(
                f"archive_replay serves sessions on or before {ARCHIVE_CUTOFF}, not {session}"
            )

    def inputs(self):
        if self._inputs is None:
            from capstone.pipeline.historical_bundle import load_inputs

            self.metadata()  # the recorded cutoff, before any frame is materialized
            self._inputs = load_inputs(self.root)
        return self._inputs

    def _calendar(self):
        return [d for d in self.metadata()["calendar"] if d <= ARCHIVE_CUTOFF]

    def sessions(self, after):
        days = self._calendar()[:-1]  # the last session has no execution session yet
        return [d for d in days if after is None or d > after]

    def _next(self, session):
        self._within_cutoff(session)
        days = self._calendar()
        if session not in days or days.index(session) + 1 >= len(days):
            raise MarketDataMissing(f"session {session} has no next session in {self.root}")
        return days[days.index(session) + 1]

    def market(self, session):
        self.reads.append(("market", session))
        following = self._next(session)
        opens = self.inputs().opens
        return {"session": following, "prices": _prices(opens.loc[pd.Timestamp(following)])}

    def marks(self, session):
        self.reads.append(("marks", session))
        self._next(session)
        closes = self.inputs().closes
        return {"session": session, "prices": _prices(closes.loc[pd.Timestamp(session)])}

    def events(self, session):
        self.reads.append(("events", session))
        self._within_cutoff(session)
        table = self.inputs().events
        if table is None or table.empty:
            return None
        dates = pd.to_datetime(table["event_date"]).dt.strftime(DAY)
        rows = table[dates == session]
        if rows.empty:
            return None
        text = rows.assign(event_date=session).to_json(orient="records", date_format="iso")
        return json.loads(text)


def build(section, services):
    """The injected `services["market_data_provider"]`, else the configured provider."""
    injected = services.get("market_data_provider")
    if injected is not None:
        return injected
    kind, config = section["provider"], section["provider_config"]
    root = config.get("root")
    if kind not in ("fixture", "archive_replay"):
        raise ProviderNotConfigured(f"[daily] provider {kind!r} is not fixture | archive_replay")
    if not isinstance(root, str) or not root.strip():
        raise ProviderNotConfigured(
            f"[daily] provider = {kind!r} needs provider_config.root named explicitly"
        )
    if kind == "fixture":
        return FixtureProvider.from_folder(root)
    return ArchiveReplayProvider(root)
