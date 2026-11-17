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
  document format, or None. Sessions after
  2020-12-31 are bundled from it (`lifecycle bundle --panel ... --current-authorization`).

A missing document is `MarketDataMissing`; a provider that is not configured explicitly
is `ProviderNotConfigured`. Neither ever falls back to a default location. The
drop-folder provider raises the typed `DataMissing` / `DataStale` / `DataInvalid`
(`ProviderDataError.code`: `data_missing`, `data_stale`, `data_invalid`), which the
stage turns into blockers of the same code, so a tick waits instead of failing.

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
- `DropFolderProvider` (`provider = "drop_folder"`): a vendor-neutral folder that an
  external scheduled export writes, one `<D>/` per session in the schema below, with
  `manifest.json` written last. It validates schema, dates, completeness and staleness.
  `VendorAdapter` is what such an export implements; `export_session` writes one
  session's folder in the schema. No concrete adapter ships here.

Drop-folder schema (`DROP_SCHEMA`), per decision session `<root>/<D>/`:

- `market.json`: `{"session": E, "prices": {permno: price}}`, E the execution session
  (the next trading session after D, which must be the next session folder once one
  exists), prices the positive finite opens of E;
- `marks.json`: `{"session": D, "prices": {permno: price}}`, D's closes;
- `events.json` (optional): a list of carry event rows, each with `event_id`,
  `event_date == D`, integer `permno` and `kind`;
- `panel.json` (required for D after 2020-12-31): the JSON panel whose universe ends
  on D and whose frames hold no row after D; its `calendar` (the exchange calendar,
  known ahead) holds D and E. For a live D (after 2025-12-31) no frame row and no
  calendar day is before the drop's first live session (`live_start`: the earliest
  complete session folder after 2025-12-31, or D itself when earlier):
  `export_session` trims the panel there before writing (`trim_panel`), and the
  provider refuses an untrimmed panel (`data_invalid`). For an archive-era D no frame
  row and no calendar day may lie inside the holdout (`HoldoutPrices`);
- `manifest.json` (written last, atomically): `{"schema": DROP_SCHEMA, "session": D,
  "vendor": <name>, "exported_at": <ISO-8601 time, not before E>, "files": {<file
  name>: <sha256 of its bytes>}}`, naming `market.json`, `marks.json` and every other
  file present.

A folder without `manifest.json` is an export in progress: it is not listed, and it is
`data_missing` once a later session folder is complete. Folders whose name starts with
`.` or `_` are staging space and ignored; any other name that is not a date is
`data_invalid`. The newest complete session older than `max_lag_days` (calendar days,
default 6) before the clock's date is `data_stale`.

`DROP_SCHEMA` is `autopilot-drop-v2` (the live panel trim above). An
`autopilot-drop-v1` folder is still read under the same checks, so a v1 live panel is
accepted only where it is already trimmed at `live_start`.

The sealed holdout: no market, marks or events document,
and no panel row or panel calendar day, may carry a date inside 2021-01-01..2025-12-31
(`HOLDOUT`), whatever the decision session.
`require_outside_holdout` raises `HoldoutPrices` (not a `ProviderDataError`: the `daily`
stage turns it into the STOP `holdout_prices`, never a blocker that waits). The
drop-folder schema (`check_documents`, so `export_session` too) applies it, and the
`daily` stage applies it to every provider's documents before anything is written. A
2020-12-31 decision session executes on a 2021 open, so it can never be simulated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from capstone.pipeline.autopilot import intents

DAY = "%Y-%m-%d"
ARCHIVE_CUTOFF = "2020-12-31"  # capstone.pipeline.historical.CUTOFF
DROP_SCHEMA = "autopilot-drop-v2"  # live panels trimmed at the first live session
READ_SCHEMAS = (DROP_SCHEMA, "autopilot-drop-v1")  # v1: read only where already trimmed
MANIFEST = "manifest.json"
DROP_FILES = ("market.json", "marks.json", "events.json", "panel.json")
MAX_LAG_DAYS = 6
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


class VendorAdapter(Protocol):
    """What a vendor export must provide to fill a drop folder (`export_session`).

    An adapter is the only vendor-specific code: it reads its vendor (network, files,
    a terminal export) and answers in the drop-folder documents. It must be
    deterministic for a session once that session has closed (the folder is written
    once and a different re-export is refused), must never return a row dated after
    the session it is asked about (marks, events, panel), and returns None while the
    session is not complete yet (its execution session has not opened). No concrete
    adapter ships in this repository.

    Its `panel` may carry the vendor's whole history: `export_session` trims a live
    session's panel at the drop's first live session (`trim_panel`) before anything is
    written, so no pre-live row (and no 2021-2025 row) reaches the drop folder.
    """

    vendor: str  # the `current_authority` vendor name the run's policy authorizes

    def session_documents(self, session: str) -> dict | None:
        """`{"market", "marks", "events"?, "panel"?}` for decision session D, or None."""


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


def _before(value, start):
    try:
        return isinstance(value, str) and _day(value) < start
    except (TypeError, ValueError):
        return False  # not a date: the schema checks refuse it


def trim_panel(panel, start):
    """The JSON panel document without any frame row or calendar day before `start`
    (a live panel holds live sessions only); the input is not changed."""
    start = _day(start)

    def keep(frame):
        if not isinstance(frame, dict) or not isinstance(frame.get("index"), list):
            return frame
        index, values = frame["index"], frame.get("values")
        rows = [i for i, day in enumerate(index) if not _before(day, start)]
        out = dict(frame, index=[index[i] for i in rows])
        if isinstance(values, list) and len(values) == len(index):
            out["values"] = [values[i] for i in rows]
        return out

    out = dict(panel)
    if isinstance(panel.get("calendar"), list):
        out["calendar"] = [day for day in panel["calendar"] if not _before(day, start)]
    for key in ("universe", "half_spread"):
        if key in panel:
            out[key] = keep(panel[key])
    for group in ("fields", "availability"):
        if isinstance(panel.get(group), dict):
            out[group] = {name: keep(frame) for name, frame in panel[group].items()}
    return out


def require_trimmed(panel, start, where):
    """`DataInvalid` when a live panel holds a frame row or calendar day before `start`."""
    early = [(part, day) for part, day in panel_days(panel) if _before(day, start)]
    if early:
        part, day = early[0]
        raise DataInvalid(
            f"{where}: panel.json {part} {day} is before the first live session {start} "
            f"({len(early)} rows and days in all); the export trims the panel there"
        )


def live_start(session, complete):
    """The drop's first live session for a live `session`: the earliest complete session
    folder after the holdout, or the session itself when earlier; None for a session on
    or before 2025-12-31."""
    session = _day(session)
    if session <= HOLDOUT[1]:
        return None
    return min([day for day in complete if day > HOLDOUT[1]] + [session])


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


# ---------------------------------------------------------------------------
# the drop folder


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _check_prices(prices, where):
    if not isinstance(prices, dict) or not prices:
        raise DataInvalid(f"{where}: `prices` is a non-empty {{permno: price}} table")
    for key, value in prices.items():
        if not (isinstance(key, str) and key.isdigit()):
            raise DataInvalid(f"{where}: price key {key!r} is not a permno")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DataInvalid(f"{where}: price of {key} is not a number")
        if not math.isfinite(value) or value <= 0:
            raise DataInvalid(f"{where}: price of {key} is not positive and finite")


def _check_frame(frame, name, session, where):
    if not isinstance(frame, dict) or not isinstance(frame.get("index"), list):
        raise DataInvalid(f"{where}: panel `{name}` has no `index` list")
    index = frame["index"]
    if not all(_is_day(d) for d in index) or index != sorted(set(index)):
        raise DataInvalid(f"{where}: panel `{name}` index is not ascending YYYY-MM-DD dates")
    if index and index[-1] > session:
        raise DataInvalid(f"{where}: panel `{name}` holds {index[-1]}, after the session")
    values = frame.get("values")
    if not isinstance(values, list) or len(values) != len(index):
        raise DataInvalid(f"{where}: panel `{name}` values do not match its index")
    if "columns" in frame:
        width = len(frame["columns"])
        if any(not isinstance(row, list) or len(row) != width for row in values):
            raise DataInvalid(f"{where}: panel `{name}` rows do not match its columns")
    return index


def check_documents(session, documents, *, vendor=None, exported_at=None, first_live=None):
    """Validate one session's drop-folder documents (`DataInvalid` / `DataMissing`).

    `first_live`: the drop's first live session (`live_start`) for a live session's
    panel trim; None means the session itself."""
    where = f"drop session {session}"
    for kind in ("market", "marks"):
        if documents.get(kind) is None:
            raise DataMissing(f"{where}: no {kind}.json")
    market, marks = documents["market"], documents["marks"]
    if not isinstance(market, dict) or set(market) != {"session", "prices"}:
        raise DataInvalid(f"{where}: market.json is {{session, prices}}")
    if not _is_day(market["session"]) or market["session"] <= session:
        raise DataInvalid(f"{where}: market.json session must be a date after {session}")
    require_outside_holdout(session, market, documents["marks"], documents.get("events"))
    _check_prices(market["prices"], f"{where} market.json")
    if not isinstance(marks, dict) or set(marks) != {"session", "prices"}:
        raise DataInvalid(f"{where}: marks.json is {{session, prices}}")
    if marks["session"] != session:
        raise DataInvalid(f"{where}: marks.json names session {marks['session']!r}")
    _check_prices(marks["prices"], f"{where} marks.json")
    events = documents.get("events")
    if events is not None:
        if not isinstance(events, list):
            raise DataInvalid(f"{where}: events.json is a list of rows")
        for row in events:
            if not isinstance(row, dict) or not {"event_id", "permno", "kind"} <= set(row):
                raise DataInvalid(f"{where}: an event row lacks event_id / permno / kind")
            if row.get("event_date") != session:
                raise DataInvalid(f"{where}: event {row['event_id']!r} is not dated {session}")
            if isinstance(row["permno"], bool) or not isinstance(row["permno"], int):
                raise DataInvalid(f"{where}: event {row['event_id']!r} permno is not an int")
    panel = documents.get("panel")
    if panel is None and session > ARCHIVE_CUTOFF:
        raise DataMissing(f"{where}: a session after {ARCHIVE_CUTOFF} needs panel.json")
    if panel is not None:
        keys = {"data_version", "calendar", "universe", "half_spread", "fields", "availability"}
        if not isinstance(panel, dict) or not keys <= set(panel):
            raise DataInvalid(f"{where}: panel.json needs {sorted(keys)}")
        if session > HOLDOUT[1]:  # live sessions only, before any other panel check
            require_trimmed(panel, _day(first_live or session), where)
        require_outside_holdout(session, panel=panel)
        if not isinstance(panel["fields"], dict) or not panel["fields"]:
            raise DataInvalid(f"{where}: panel.json has no fields")
        calendar = panel["calendar"]
        if not isinstance(calendar, list) or not {session, market["session"]} <= set(calendar):
            raise DataInvalid(
                f"{where}: panel.json calendar must hold the session and its execution session"
            )
        universe = _check_frame(panel["universe"], "universe", session, where)
        if not universe or universe[-1] != session:
            raise DataInvalid(f"{where}: panel.json universe does not end on the session")
        _check_frame(panel["half_spread"], "half_spread", session, where)
        for name, frame in panel["fields"].items():
            _check_frame(frame, f"field:{name}", session, where)
        for name, series in dict(panel["availability"]).items():
            _check_frame(series, f"availability:{name}", session, where)
    if vendor is not None and (not isinstance(vendor, str) or not vendor):
        raise DataInvalid(f"{where}: manifest vendor is a non-empty name")
    if exported_at is not None:
        try:
            exported = pd.Timestamp(exported_at)
        except (TypeError, ValueError) as error:
            raise DataInvalid(f"{where}: exported_at {exported_at!r} is not a time") from error
        if _day(exported) < market["session"]:
            raise DataInvalid(
                f"{where}: exported {_day(exported)}, before its execution session "
                f"{market['session']} opened"
            )


class DropFolderProvider:
    """Per-session documents an external scheduled export drops into `root`."""

    def __init__(self, root, *, max_lag_days=MAX_LAG_DAYS, today: Callable | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ProviderNotConfigured(f"drop_folder root {self.root} is not a folder")
        if isinstance(max_lag_days, bool) or not isinstance(max_lag_days, int) or max_lag_days < 1:
            raise ProviderNotConfigured("drop_folder max_lag_days is a whole number >= 1")
        self.max_lag_days = max_lag_days
        self.today = today or (lambda: datetime.now(UTC).isoformat(timespec="seconds"))
        self._verified = {}
        self.reads = []

    def _today(self):
        return _day(pd.Timestamp(self.today()))

    def _folders(self):
        """(every session folder, the complete ones), ascending; names only."""
        names = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name[:1] in (".", "_"):
                continue
            if not _is_day(path.name):
                raise DataInvalid(f"drop folder {path} is not a YYYY-MM-DD session folder")
            names.append(path.name)
        return names, [n for n in names if (self.root / n / MANIFEST).is_file()]

    def sessions(self, after):
        names, complete = self._folders()
        newest = complete[-1] if complete else None
        for name in names:
            if name not in complete and newest is not None and name < newest:
                if after is None or name > after:
                    raise DataMissing(
                        f"drop session {name} has no {MANIFEST} but {newest} is complete"
                    )
        return [n for n in complete if after is None or n > after]

    def check_fresh(self):
        """`data_stale` when the newest complete session lags the clock too far."""
        _, complete = self._folders()
        today = self._today()
        if not complete:
            raise DataMissing(f"drop folder {self.root} has no complete session yet")
        lag = (pd.Timestamp(today) - pd.Timestamp(complete[-1])).days
        if lag > self.max_lag_days:
            raise DataStale(
                f"newest complete drop session {complete[-1]} is {lag} days before {today} "
                f"(max_lag_days {self.max_lag_days}): the export has stopped"
            )

    def _load(self, session):
        """The verified documents of one complete session folder."""
        session = _day(session)
        if session in self._verified:
            return self._verified[session]
        folder = self.root / session
        where = f"drop session {session}"
        if not (folder / MANIFEST).is_file():
            raise DataMissing(f"{where}: no {MANIFEST} (export incomplete)")
        try:
            manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DataInvalid(f"{where}: {MANIFEST} is unreadable: {error}") from error
        keys = {"schema", "session", "vendor", "exported_at", "files"}
        if not isinstance(manifest, dict) or set(manifest) != keys:
            raise DataInvalid(f"{where}: {MANIFEST} keys are {sorted(keys)}")
        if manifest["schema"] not in READ_SCHEMAS or manifest["session"] != session:
            raise DataInvalid(f"{where}: {MANIFEST} is not {READ_SCHEMAS} for {session}")
        files = manifest["files"]
        if not isinstance(files, dict) or not set(files) <= set(DROP_FILES):
            raise DataInvalid(f"{where}: {MANIFEST} files must be among {list(DROP_FILES)}")
        present = {p.name for p in folder.iterdir() if p.name in DROP_FILES}
        if present - set(files):
            raise DataInvalid(f"{where}: {sorted(present - set(files))} not in {MANIFEST}")
        documents = {}
        for name, digest in sorted(files.items()):
            path = folder / name
            if not path.is_file():
                raise DataMissing(f"{where}: {name} is listed in {MANIFEST} but absent")
            data = path.read_bytes()
            if _sha256(data) != digest:
                raise DataInvalid(f"{where}: {name} does not match its {MANIFEST} digest")
            try:
                documents[name.removesuffix(".json")] = json.loads(data.decode("utf-8"))
            except ValueError as error:
                raise DataInvalid(f"{where}: {name} is not JSON: {error}") from error
        names, complete = self._folders()
        check_documents(
            session,
            documents,
            vendor=manifest["vendor"],
            exported_at=manifest["exported_at"],
            first_live=live_start(session, complete),
        )
        today = self._today()
        if documents["market"]["session"] > today:
            raise DataInvalid(f"{where}: execution session is after today {today}")
        later = [n for n in names if n > session]
        execution = documents["market"]["session"]
        if later and later[0] != execution:
            if later[0] > execution:
                raise DataMissing(f"{where}: its execution session {execution} has no folder")
            raise DataInvalid(
                f"{where}: execution session {execution} skips session folder {later[0]}"
            )
        self._verified[session] = documents
        return documents

    def _get(self, kind, session):
        self.reads.append((kind, session))
        return self._load(session).get(kind)

    def market(self, session):
        return self._get("market", session)

    def marks(self, session):
        return self._get("marks", session)

    def events(self, session):
        return self._get("events", session) or None

    def panel(self, session):
        return self._get("panel", session)


def _write_bytes(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _encoded(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def export_session(root, vendor, session, documents, *, exported_at):
    """Write one session's drop folder in `DROP_SCHEMA`, `manifest.json` last.

    The scheduled export (any vendor, through a `VendorAdapter`) calls this. A live
    session's panel is trimmed at the drop's first live session (`live_start`) before
    it is checked or written, so the vendor's earlier history never reaches the folder
    A folder whose manifest already records different documents is refused
    (`DataInvalid`); the same documents are a no-op. Returns the folder."""
    session = _day(session)
    documents = {k: v for k, v in documents.items() if v is not None}
    if unknown := sorted(set(documents) - {n.removesuffix(".json") for n in DROP_FILES}):
        raise DataInvalid(f"drop session {session}: unknown documents {unknown}")
    complete = []
    if Path(root).is_dir():
        complete = sorted(
            p.name for p in Path(root).iterdir() if _is_day(p.name) and (p / MANIFEST).is_file()
        )
    first_live = live_start(session, complete)
    if first_live is not None and isinstance(documents.get("panel"), dict):
        documents["panel"] = trim_panel(documents["panel"], first_live)
    check_documents(
        session, documents, vendor=vendor, exported_at=exported_at, first_live=first_live
    )
    folder = Path(root) / session
    payload = {f"{kind}.json": _encoded(doc) for kind, doc in sorted(documents.items())}
    files = {name: _sha256(data) for name, data in payload.items()}
    if (folder / MANIFEST).is_file():
        known = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        if known.get("files") != files:
            raise DataInvalid(f"drop session {session} was exported with other documents")
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        _write_bytes(folder / name, data)
    manifest = {
        "schema": DROP_SCHEMA,
        "session": session,
        "vendor": vendor,
        "exported_at": str(exported_at),
        "files": files,
    }
    # a kill here leaves documents without a manifest: the folder is not a session yet,
    # and the next export writes the same bytes again
    intents.fault("export:before_manifest")
    _write_bytes(folder / MANIFEST, _encoded(manifest))
    return folder


def export_from(adapter: VendorAdapter, root, sessions, *, exported_at):
    """Export every session the adapter can answer; returns the sessions written."""
    written = []
    for session in sessions:
        documents = adapter.session_documents(_day(session))
        if documents is None:
            break  # not complete yet: later sessions wait for it
        export_session(root, adapter.vendor, session, documents, exported_at=exported_at)
        written.append(_day(session))
    return written


DROP_KEYS = {"root", "max_lag_days"}


def build(section, services):
    """The injected `services["market_data_provider"]`, else the configured provider."""
    injected = services.get("market_data_provider")
    if injected is not None:
        return injected
    kind, config = section["provider"], section["provider_config"]
    root = config.get("root")
    if kind not in ("fixture", "archive_replay", "drop_folder"):
        raise ProviderNotConfigured(
            f"[daily] provider {kind!r} is not fixture | archive_replay | drop_folder"
        )
    if not isinstance(root, str) or not root.strip():
        raise ProviderNotConfigured(
            f"[daily] provider = {kind!r} needs provider_config.root named explicitly"
        )
    if kind == "fixture":
        return FixtureProvider.from_folder(root)
    if kind == "drop_folder":
        if unknown := sorted(set(config) - DROP_KEYS):
            raise ProviderNotConfigured(f"drop_folder provider_config has unknown keys {unknown}")
        return DropFolderProvider(
            root,
            max_lag_days=config.get("max_lag_days", MAX_LAG_DAYS),
            today=services.get("clock"),
        )
    return ArchiveReplayProvider(root)
