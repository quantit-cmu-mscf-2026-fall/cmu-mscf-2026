"""The deploy registry: `deploy/strategies.yml` entries.

The registry file's schema is in its header: one entry per strategy with
`id`, `module`, `gate_report` and `enabled`, and three optional keys, `role`,
`state` and `version`, that are written only when they differ from their
defaults. Rules:

- No gate report, no entry: an entry without one is refused when built or
  read.
- `enabled` defaults to false. Turning a strategy on is a reviewed edit of
  the registry file, never a side effect of reading or rewriting it, and
  registering again never turns it off (or on).
- `register` writes a spec's file (`deploy/strategies/<id>.toml`) and its entry, only
  under the given root. Registering the same spec twice changes no byte of either file.
  A program or saved-portfolio entry needs a JSON gate report of its module's schema
  (`check_program_gate_report`), except a "plumbing-test" one.
- A file that is read and rendered again is byte for byte what it was. The
  optional keys are never written at their defaults, so a file without them
  reads and writes exactly as before.
- `module` is one of `MODULES` (a legacy `StrategySpec`, a typed program, a saved program
  portfolio); `registry_spec` reads an entry's spec file, `deploy/strategies/<id>.toml`
  (`SPEC_DIR`), with that module's `from_toml`.
- `role`: "deploy" (the default), "incubation-only" for a family the gate
  labelled incubate, or "plumbing-test" for an entry that tests plumbing and
  claims no edge. Reading never changes a role.
- `state`: "active" (the default), "suspended" or "retired". The state is
  recorded here and changes only through `set_state`, which appends a hash-chained
  event to `deploy/lifecycle.jsonl` before rewriting the entry. `version` is an
  optional label (for example an allocation version).
- `register` writes a `registered` event for every new entry. A registry whose
  entries disagree with their events is refused when read (a contract entry
  with no events at all is a legacy entry that predates the file). A crash
  between the event and the entry is the recoverable
  `lifecycle_entry_rewrite_pending:<id>`, completed by the same call again.
  Retired is terminal, and registering never changes a state: it refuses a
  suspended or retired id. Rollback is a revert of the registry commit, which takes
  an entry and its events back together.
- A re-anchored version (`register_version`): a program re-anchored on a later panel
  (only `anchor` and `data_version` moved) has a new id; it is registered as a new entry
  with `version = "reanchor-of:<parent id>"`, the parent's gate report and role
  `incubation-only`. Its gate report names the parent, which `check_program_gate_report`
  accepts through that label.
- `promote` turns an active "incubation-only" entry into a "deploy" one only on a
  confirm record whose confirming e-value reached 1/alpha (`check_confirm_record`);
  the record is kept in the `role` event.
- These events only RECORD a state or role a person confirmed. Nothing in this
  module decides to suspend, retire or promote a strategy.

The project has no YAML dependency, so this module reads and writes the
registry's own narrow schema: a header of comments, then `strategies: []` or
a block list of entries with the keys in `KEYS` (the first four always) whose values are plain,
single-quoted or double-quoted scalars. Anything else is refused rather than
guessed at.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args

from capstone.artifacts import canonical, digest
from capstone.faults import maybe_fault

MODULE = "capstone.strategy.contract"
PROGRAM_MODULE = "capstone.strategy.program"
SAVED_PORTFOLIO_MODULE = "capstone.strategy.saved_portfolio"
MODULES = (MODULE, PROGRAM_MODULE, SAVED_PORTFOLIO_MODULE)
REGISTRY_FILE = Path("deploy") / "strategies.yml"
SPEC_DIR = Path("deploy") / "strategies"  # one spec file per registry entry: <id>.toml
LIFECYCLE_FILE = Path("deploy") / "lifecycle.jsonl"
KEYS = ("id", "module", "gate_report", "enabled", "role", "state", "version")
Role = Literal["deploy", "incubation-only", "plumbing-test"]
ROLES: tuple[str, ...] = get_args(Role)
INCUBATION_ONLY = "incubation-only"
PLUMBING_TEST = "plumbing-test"
State = Literal["active", "suspended", "retired"]
STATES: tuple[str, ...] = get_args(State)
TRANSITIONS = frozenset(
    {
        ("active", "suspended"),
        ("suspended", "active"),
        ("active", "retired"),
        ("suspended", "retired"),
    }
)
# The gate report a program / saved-portfolio entry needs: these identity keys here, the
# full versioned schema in `capstone.program_gate.check_schema`.
REPORT_SCHEMAS = {
    PROGRAM_MODULE: "program-gate-report-v1",
    SAVED_PORTFOLIO_MODULE: "saved-portfolio-gate-report-v1",
}
REPORT_KEYS = ("schema", "procedure_id", "strategy_ids", "holdout_opened")
# The `version` label of a re-anchored program version names its parent entry.
LINEAGE_PREFIX = "reanchor-of:"


@dataclass(frozen=True)
class RegistryEntry:
    id: str
    module: str
    gate_report: str
    enabled: bool = False
    role: Role = "deploy"
    state: State = "active"
    version: str = ""

    def __post_init__(self) -> None:
        if not self.gate_report:
            raise ValueError(f"strategy {self.id!r} has no gate report: no gate report, no deploy")
        if not self.id or not self.module:
            raise ValueError("an entry needs an id and a module")
        if self.module not in MODULES:
            raise ValueError(
                f"strategy {self.id!r}: module must be one of {MODULES}: {self.module!r}"
            )
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        if self.role not in ROLES:
            raise ValueError(f"strategy {self.id!r}: role must be one of {ROLES}: {self.role!r}")
        if self.state not in STATES:
            raise ValueError(f"strategy {self.id!r}: state must be one of {STATES}: {self.state!r}")
        if not isinstance(self.version, str):
            raise ValueError(f"strategy {self.id!r}: version must be a string")


def basic_string(text: str) -> str:
    """A TOML basic string (also a YAML double-quoted scalar)."""
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _scalar(text: str) -> str | bool:
    text = text.strip()
    if text.startswith('"'):
        return json.loads(text)
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1].replace("''", "'")
    if text in ("true", "false"):
        return text == "true"
    if not text or text[0] in "[{&*!|>%@`#":
        raise ValueError(f"registry value {text!r} is not a plain scalar")
    return text


def _parse(text: str) -> tuple[str, list[RegistryEntry]]:
    """(header, entries) of a registry file."""
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    header = "".join(line + "\n" for line in lines[:i])
    if i == len(lines):
        raise ValueError("registry file has no `strategies:` key")
    top = lines[i].split("#", 1)[0].rstrip()
    if top == "strategies: []":
        rest = lines[i + 1 :]
        if any(line.strip() and not line.lstrip().startswith("#") for line in rest):
            raise ValueError("registry file has content after `strategies: []`")
        return header, []
    if top != "strategies:":
        raise ValueError(f"registry file must start with `strategies:`, got {lines[i]!r}")
    raw: list[dict] = []
    for line in lines[i + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        body = line.strip()
        if body.startswith("- "):
            raw.append({})
            body = body[2:].strip()
        elif not raw or not line.startswith("    "):
            raise ValueError(f"registry line {line!r} is not part of an entry")
        key, sep, value = body.partition(":")
        if not sep or key not in KEYS or key in raw[-1]:
            raise ValueError(f"registry line {line!r}: expected one of {KEYS}, once each")
        raw[-1][key] = _scalar(value)
    entries = []
    for item in raw:
        if not item.get("gate_report"):
            raise ValueError(f"registry entry {item.get('id')!r} has no gate_report")
        entries.append(RegistryEntry(**item))
    return header, entries


def _render(header: str, entries: list[RegistryEntry]) -> str:
    if not entries:
        return header + "strategies: []\n"
    body = "".join(
        f"  - id: {basic_string(e.id)}\n"
        f"    module: {basic_string(e.module)}\n"
        f"    gate_report: {basic_string(e.gate_report)}\n"
        f"    enabled: {'true' if e.enabled else 'false'}\n"
        + (f"    role: {basic_string(e.role)}\n" if e.role != "deploy" else "")
        + (f"    state: {basic_string(e.state)}\n" if e.state != "active" else "")
        + (f"    version: {basic_string(e.version)}\n" if e.version else "")
        for e in entries
    )
    return header + "strategies:\n" + body


def parse_registry(text: str) -> tuple[RegistryEntry, ...]:
    """The entries of a registry file's text, in file order (e.g. read at a git tag).
    Text alone cannot show its states are backed by events: `verify_lifecycle` does
    (and `read_registry` runs it)."""
    return tuple(_parse(text)[1])


def read_registry(root: Path) -> tuple[RegistryEntry, ...]:
    """The entries of `<root>/deploy/strategies.yml`, in file order, checked
    against `<root>/deploy/lifecycle.jsonl` (absent: no events)."""
    entries = parse_registry(_read(root, REGISTRY_FILE))
    verify_lifecycle(_lifecycle_text(root), entries)
    return entries


def _read(root: Path, path: Path) -> str:
    return (Path(root) / path).read_bytes().decode("utf-8")


def _lifecycle_text(root: Path) -> str:
    path = Path(root) / LIFECYCLE_FILE
    return path.read_bytes().decode("utf-8") if path.exists() else ""


def spec_from_toml(module: str, text: str):
    """A spec file's content, read with its registry module's `from_toml`."""
    if module == MODULE:
        from capstone.strategy.contract import StrategySpec

        return StrategySpec.from_toml(text)
    if module == PROGRAM_MODULE:
        from capstone.strategy.program import ProgramStrategySpec

        return ProgramStrategySpec.from_toml(text)
    if module == SAVED_PORTFOLIO_MODULE:
        from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec

        return SavedProgramPortfolioSpec.from_toml(text)
    raise ValueError(f"module must be one of {MODULES}: {module!r}")


def registry_spec(root: Path, strategy_id: str, module: str | None = None):
    """The spec file of a registered strategy, read with its entry's module
    (`module` if given; the contract for an id not in the registry); refused
    if its content is not its id."""
    if module is None:
        entries = _parse(_read(root, REGISTRY_FILE))[1]
        module = next((e.module for e in entries if e.id == strategy_id), MODULE)
    path = Path(root) / SPEC_DIR / f"{strategy_id}.toml"
    if module == MODULE:
        from capstone.strategy.contract import load_spec

        spec = load_spec(path)
    else:
        spec = spec_from_toml(module, _read(root, path))
    if spec.id != strategy_id:
        raise ValueError(f"spec file {strategy_id}.toml holds spec id {spec.id}")
    return spec


def check_program_gate_report(report: Any, entry: RegistryEntry) -> None:
    """A program / saved-portfolio entry's gate report must be a JSON object of
    its module's schema (`capstone.program_gate`) that names the entry: every key
    of the schema, an unchanged `id`, and either the search stage
    (`holdout_opened: false`, nothing accepted) or a final report whose holdout
    look the authority store recorded. The content is re-verified (re-run on its
    bound inputs) by execution evidence, not here."""
    from capstone.program_gate import check_schema  # heavy; only for programs

    where = f"program gate report {entry.gate_report} of strategy {entry.id}"
    if not isinstance(report, dict):
        raise ValueError(f"{where} is not a JSON object in the repository")
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"{where} lacks {missing}")
    schema = REPORT_SCHEMAS[entry.module]
    if report["schema"] != schema:
        raise ValueError(f"{where} has schema {report['schema']!r}, not {schema!r}")
    if not isinstance(report["procedure_id"], str) or not report["procedure_id"]:
        raise ValueError(f"{where} has no procedure_id")
    ids = report["strategy_ids"]
    named = (entry.id, lineage_parent(entry.version))  # a re-anchored version: its parent
    if not isinstance(ids, list) or not any(sid in ids for sid in named if sid):
        raise ValueError(f"{where}: strategy_ids does not name {entry.id}")
    try:
        check_schema(report)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


def lineage_parent(version: str) -> str | None:
    """The parent id a re-anchored version's `version` label names, else None."""
    if isinstance(version, str) and version.startswith(LINEAGE_PREFIX):
        return version[len(LINEAGE_PREFIX) :] or None
    return None


def _report(root: Path, gate_report: str) -> Any:
    if "://" in gate_report:
        return None
    try:
        return json.loads(_read(root, Path(gate_report)))
    except (OSError, ValueError):
        return None


def register(
    spec,
    root: Path,
    *,
    enabled: bool = False,
    role: Role = "deploy",
    gate_report: str | None = None,
    version: str = "",
    at: str = "",
    reviewer: str = "",
) -> RegistryEntry:
    """Write the spec's file and its registry entry; return the entry in the file.

    The module is the spec class's for a program or saved portfolio, else the
    contract (whose `load_spec` reads every older spec schema); the gate report
    is `gate_report`, else the spec's own (a spec naming one must agree).
    A new entry gets a `registered` lifecycle event (role, state, module,
    version, `reviewer`, `at`) before it is written; `at` is the caller's
    timestamp and is required for a program or saved portfolio (a legacy
    contract caller that passes none records "").
    An id already registered is left exactly as it is (its spec file must
    still match), so registering twice is a no-op and never flips `enabled`
    or changes `role`; a suspended or retired id is refused, since
    registering never changes a state.
    """
    module = type(spec).__module__
    module = module if module in REPORT_SCHEMAS else MODULE  # the contract, as before
    report = spec.gate_report if gate_report is None else gate_report
    if spec.gate_report and report != spec.gate_report:
        raise ValueError(f"spec {spec.id} names gate report {spec.gate_report}, not {report}")
    entry = RegistryEntry(spec.id, module, report, enabled, role, version=version)
    if not (isinstance(at, str) and isinstance(reviewer, str)):
        raise ValueError("at and reviewer must be strings")
    if module != MODULE and not at:
        raise ValueError("registering a program or saved portfolio needs a timestamp (`at`)")
    if module != MODULE and role != PLUMBING_TEST:
        check_program_gate_report(_report(root, report), entry)
    entries = parse_registry(_read(root, REGISTRY_FILE))
    existing = next((e for e in entries if e.id == spec.id), None)
    if existing is not None and existing.state != "active":
        raise ValueError(
            f"strategy {spec.id} is {existing.state}; registering never changes a state "
            "(set_state resumes a suspended entry; a retired one is final)"
        )
    spec_path = Path(root) / SPEC_DIR / f"{spec.id}.toml"
    if spec_path.exists():
        registry_spec(root, spec.id, module)
    else:
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        if module == MODULE:
            from capstone.strategy.contract import save_spec

            save_spec(spec, spec_path)
        else:
            spec_path.write_bytes(spec.to_toml().encode("utf-8"))
    if existing is not None:
        return existing
    event = {
        "field": "registered",
        "id": spec.id,
        "role": role,
        "state": "active",
        "module": module,
        "version": version,
        "reviewer": reviewer,
        "at": at,
    }
    return _commit(root, event, entry)


def register_version(spec, root: Path, *, parent_id: str, at: str, reviewer: str) -> RegistryEntry:
    """Register `spec`, a re-anchoring of the registered program `parent_id`, as a
    new version of it; return the entry in the file.

    A re-anchoring is the parent's spec with only `anchor` and `data_version`
    moved (the live handoff). The new entry has the parent's gate report, role
    `incubation-only` (promotion is earned again, on the new data) and
    `version = "reanchor-of:<parent_id>"`. Refused unless the parent is an
    active program entry. Idempotent like `register`: a version already
    registered with this label is returned as it is (whatever its state), and a
    `registered` event a crash left pending is completed by the same call."""
    from capstone.strategy.program import ProgramStrategySpec

    version = LINEAGE_PREFIX + parent_id
    entries = parse_registry(_read(root, REGISTRY_FILE))
    existing = next((e for e in entries if e.id == spec.id), None)
    if existing is not None:
        if existing.version != version:
            raise ValueError(f"strategy {spec.id} is registered, but not as {version}")
        registry_spec(root, spec.id, existing.module)  # its spec file still matches
        return existing
    parent = next((e for e in entries if e.id == parent_id), None)
    if parent is None or parent.module != PROGRAM_MODULE:
        raise ValueError(f"{parent_id} is not a registered program; {spec.id} has no parent")
    if parent.state != "active":
        raise ValueError(f"{parent_id} is {parent.state}; a new version needs an active parent")
    if not isinstance(spec, ProgramStrategySpec) or spec.id == parent_id:
        raise ValueError(f"{spec.id} is not a new program version of {parent_id}")
    held = registry_spec(root, parent_id, PROGRAM_MODULE)
    if replace(spec, anchor=held.anchor, data_version=held.data_version).id != parent_id:
        raise ValueError(f"{spec.id} is not a re-anchoring of {parent_id}")
    return register(
        spec,
        root,
        role=INCUBATION_ONLY,
        gate_report=parent.gate_report,
        version=version,
        at=at,
        reviewer=reviewer,
    )


def read_lifecycle(root: Path) -> tuple[dict[str, Any], ...]:
    """The verified lifecycle events of `<root>`, oldest first."""
    return verify_lifecycle(_lifecycle_text(root), parse_registry(_read(root, REGISTRY_FILE)))


# ---------------------------------------------------------------------------
# Lifecycle events: deploy/lifecycle.jsonl, one canonical JSON event per line,
# each carrying the digest of the event before it ("" for the first).

REGISTERED_EVENT = frozenset(
    {"prev_hash", "field", "id", "role", "state", "module", "version", "reviewer", "at"}
)
STATE_EVENT = frozenset({"prev_hash", "field", "id", "from", "to", "reason", "reviewer", "at"})
ROLE_EVENT = STATE_EVENT | {"confirm_record"}
EVENT_KEYS = {"registered": REGISTERED_EVENT, "state": STATE_EVENT, "role": ROLE_EVENT}
PENDING = "lifecycle_entry_rewrite_pending"


def _refused(message: str) -> ValueError:
    return ValueError(f"lifecycle {LIFECYCLE_FILE.as_posix()}: {message}")


class LifecycleRewritePending(ValueError):
    """The last event is written and its entry not yet (a crash between the two
    writes). Recoverable: the same `register`/`set_state`/`promote` call again rewrites the
    entry without a second event. Until then reads refuse."""

    def __init__(self, event: dict[str, Any]):
        self.strategy_id = event["id"]
        self.event = event
        super().__init__(
            f"lifecycle {LIFECYCLE_FILE.as_posix()}: {PENDING}:{self.strategy_id} (its last "
            f"{event['field']} event is written, the entry is not; repeat the same call)"
        )


def check_confirm_record(record: Any, strategy_id: str):
    """The `ConfirmState` of a promotion record `{"strategy_id": ...,
    "confirm_state": ConfirmState.to_dict()}` that promotes `strategy_id`
    (the confirming e-value reached 1/alpha); else refused."""
    from capstone.incubation.confirm import ConfirmState

    if not isinstance(record, Mapping) or set(record) != {"strategy_id", "confirm_state"}:
        raise ValueError(
            'a promotion needs a confirm record {"strategy_id", "confirm_state": '
            "ConfirmState.to_dict()}"
        )
    if record["strategy_id"] != strategy_id:
        raise ValueError(f"the confirm record is for {record['strategy_id']!r}, not {strategy_id}")
    try:
        state = ConfirmState.from_dict(record["confirm_state"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"the confirm record is not a serialized ConfirmState: {exc}") from exc
    threshold = 1 / state.config.alpha
    if not (
        state.decision == "promote" and state.reason == "confirmed" and state.e_value >= threshold
    ):
        raise ValueError(
            f"the confirm record does not promote {strategy_id}: decision {state.decision!r}, "
            f"e-value {state.e_value:.4g} vs 1/alpha {threshold:.4g}"
        )
    return state


def _event(n: int, line: str) -> dict[str, Any]:
    try:
        event = json.loads(line)
    except ValueError as exc:
        raise _refused(f"line {n} is not JSON") from exc
    if not isinstance(event, dict) or canonical(event) != line:
        raise _refused(f"line {n} is not a canonical JSON event")
    keys = EVENT_KEYS.get(event.get("field"))
    if (
        keys is None
        or set(event) != keys
        or not all(isinstance(v, str) for k, v in event.items() if k != "confirm_record")
    ):
        raise _refused(f"line {n} is not a registered, state or role event")
    if event["field"] == "registered":  # a legacy contract caller may record no time
        required = ("id",) if event["module"] == MODULE else ("id", "at")
    else:
        required = ("id", "reason", "reviewer", "at")
    if not all(event[k] for k in required):
        raise _refused(f"line {n} lacks one of {required}")
    return event


def _replay(events) -> tuple[dict[str, str], dict[str, dict], set[str]]:
    """(state, registered event, promoted ids) after `events`; refuses an
    impossible sequence."""
    states: dict[str, str] = {}
    registered: dict[str, dict] = {}
    promoted: set[str] = set()
    for n, event in enumerate(events, start=1):
        sid, field = event["id"], event["field"]
        if field == "registered":
            if sid in registered or sid in states or sid in promoted:
                raise _refused(f"line {n}: {sid} is registered after events of its own")
            if (
                event["state"] != "active"
                or event["role"] not in ROLES
                or event["module"] not in MODULES
            ):
                raise _refused(f"line {n}: not a valid registration")
            registered[sid] = event
        elif field == "state":
            if event["from"] != states.get(sid, "active"):
                raise _refused(f"line {n}: {sid} was not {event['from']}")
            if (event["from"], event["to"]) not in TRANSITIONS:
                raise _refused(f"line {n}: {event['from']} -> {event['to']} is not allowed")
            states[sid] = event["to"]
        else:
            first_role = registered[sid]["role"] if sid in registered else INCUBATION_ONLY
            if (
                (event["from"], event["to"]) != (INCUBATION_ONLY, "deploy")
                or first_role != INCUBATION_ONLY
                or sid in promoted
            ):
                raise _refused(f"line {n}: a role event is one incubation-only -> deploy")
            try:
                check_confirm_record(event["confirm_record"], sid)
            except ValueError as exc:
                raise _refused(f"line {n}: {exc}") from exc
            promoted.add(sid)
    return states, registered, promoted


def _mismatches(entries, model) -> dict[str, str]:
    """Each strategy whose entry is not where the events leave it, and why."""
    states, registered, promoted = model
    bad: dict[str, str] = {}
    for entry in entries:
        reg = registered.get(entry.id)
        if reg is None and entry.module != MODULE:
            bad[entry.id] = "has no registered event (only a legacy contract entry may lack one)"
        elif reg is not None and (entry.module, entry.version) != (reg["module"], reg["version"]):
            bad[entry.id] = "has a module or version its registered event does not"
        role = "deploy" if entry.id in promoted else reg["role"] if reg else None
        if role is not None and entry.role != role:
            bad[entry.id] = f"says role {entry.role!r} but its events leave {role!r}"
        state = states.get(entry.id, "active")
        if entry.state != state:
            bad[entry.id] = (
                f"says {entry.state!r} but its events leave it {state!r}; "
                "a state changes only through set_state"
            )
    present = {entry.id for entry in entries}
    for sid in registered:
        if sid not in present:
            bad[sid] = "has a registered event but no entry"
    return bad


def verify_lifecycle(text: str, entries) -> tuple[dict[str, Any], ...]:
    """The events of a lifecycle file's text; refused unless every line is a
    canonical event, the hash chain is intact, each transition is allowed, and
    every entry's state, role, module and version are where its events leave
    them. Program and saved-portfolio entries need a `registered` event; a
    contract entry without one is a legacy entry (it predates the file), and
    its state still changes only through events.

    The one recoverable case, files with the last event written and its entry
    not yet, raises `LifecycleRewritePending` (`lifecycle_entry_rewrite_pending:<id>`)."""
    if text and not text.endswith("\n"):
        raise _refused("the file does not end with a newline")
    events: list[dict[str, Any]] = []
    previous = ""
    for n, line in enumerate(text.split("\n")[:-1], start=1):
        event = _event(n, line)
        if event["prev_hash"] != previous:
            raise _refused(f"line {n} does not follow the line before it (hash chain broken)")
        previous = digest(event)
        events.append(event)
    bad = _mismatches(entries, _replay(events))
    if not bad:
        return tuple(events)
    last = events[-1] if events else None
    present = {entry.id for entry in entries}
    if (
        last is not None
        and set(bad) == {last["id"]}
        and (last["field"] != "registered" or last["id"] not in present)
        and not _mismatches(entries, _replay(events[:-1]))
    ):
        raise LifecycleRewritePending(last)
    sid, why = next(iter(bad.items()))
    raise _refused(f"strategy {sid} {why}")


def _commit(root: Path, event: dict[str, Any], entry: RegistryEntry) -> RegistryEntry:
    """Append `event` (chained), then write `entry` into the registry.

    A crash between the two writes leaves `LifecycleRewritePending`; the same
    call again finds exactly its own event pending and only writes the entry."""
    text = _lifecycle_text(root)
    entries = parse_registry(_read(root, REGISTRY_FILE))
    try:
        events = verify_lifecycle(text, entries)
    except LifecycleRewritePending as pending:
        written = {k: v for k, v in pending.event.items() if k != "prev_hash"}
        if pending.strategy_id != event["id"] or written != event:
            raise ValueError(f"{pending}; only that same call completes it") from None
    else:
        event = {"prev_hash": digest(events[-1]) if events else "", **event}
        path = Path(root) / LIFECYCLE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((text + canonical(event) + "\n").encode("utf-8"))
        maybe_fault("registry:after_event")
    header, current = _parse(_read(root, REGISTRY_FILE))
    if any(e.id == entry.id for e in current):
        rewritten = [entry if e.id == entry.id else e for e in current]
    else:
        rewritten = [*current, entry]
    (Path(root) / REGISTRY_FILE).write_bytes(_render(header, rewritten).encode("utf-8"))
    return entry


def _entry(root: Path, strategy_id: str, reviewer: str, at: str) -> RegistryEntry:
    if not (isinstance(reviewer, str) and reviewer and isinstance(at, str) and at):
        raise ValueError("a lifecycle event needs a reviewer and a timestamp (`at`)")
    for entry in parse_registry(_read(root, REGISTRY_FILE)):
        if entry.id == strategy_id:
            return entry
    raise ValueError(f"strategy {strategy_id} is not registered")


def set_state(
    root: Path, strategy_id: str, state: State, *, reason: str, reviewer: str, at: str
) -> RegistryEntry:
    """Move an entry to `state` by appending a lifecycle event; return the new
    entry. Allowed: active <-> suspended and active/suspended -> retired. `at`
    is the event's timestamp, passed in (never read from the clock). This only
    records the decision a person (the strategy owner) confirmed; nothing here
    decides it."""
    if not (isinstance(reason, str) and reason):
        raise ValueError("a state change needs a reason")
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}: {state!r}")
    entry = _entry(root, strategy_id, reviewer, at)
    if entry.state == "retired":
        raise ValueError(f"strategy {strategy_id} is retired; retired is terminal")
    if entry.state == state:
        raise ValueError(f"strategy {strategy_id} is already {state}")
    event = {
        "field": "state",
        "id": strategy_id,
        "from": entry.state,
        "to": state,
        "reason": reason,
        "reviewer": reviewer,
        "at": at,
    }
    return _commit(root, event, replace(entry, state=state))


def promote(
    root: Path, strategy_id: str, *, confirm_record: Mapping, reviewer: str, at: str
) -> RegistryEntry:
    """Turn an active incubation-only entry into a deploy entry, only on a
    confirm record that promotes it (`check_confirm_record`); the record is
    kept in the lifecycle event. `enabled` is untouched."""
    entry = _entry(root, strategy_id, reviewer, at)
    if entry.role != INCUBATION_ONLY:
        raise ValueError(
            f"only an incubation-only entry is promoted; {strategy_id} is {entry.role}"
        )
    if entry.state != "active":
        raise ValueError(f"strategy {strategy_id} is {entry.state}; only an active one is promoted")
    check_confirm_record(confirm_record, strategy_id)
    event = {
        "field": "role",
        "id": strategy_id,
        "from": INCUBATION_ONLY,
        "to": "deploy",
        "reason": "confirmed",
        "reviewer": reviewer,
        "at": at,
        "confirm_record": json.loads(canonical(dict(confirm_record))),
    }
    return _commit(root, event, replace(entry, role="deploy"))
