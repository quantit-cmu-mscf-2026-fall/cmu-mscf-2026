"""Current authorizations by policy, as a function and a CLI.

This module never opens the research holdout and never fetches data. Without
`CAPSTONE_CURRENT_AUTHORITY` naming a store, every call refuses; an empty policy refuses;
a window touching 2021-2025 refuses (see `capstone.current_authority`). Anything current
(after 2020) is therefore refused by default.

`[daily] current_authorization` is the policy table `{vendor, fields, start?, end?}`:

- `vendor` (str): the `current_authority` vendor name (`[a-z0-9-]`, at most 40);
- `fields` (list of str): the authorized fields (for example the panel's fields,
  `funded_returns` for `decide`, `corporate_events` for carried events);
- `start` (str, default: the first session that needs it), `end` (str, default open).

`ensure(store, policy, session, authorized_by)` returns the record id that covers the
session, creating it through `current_authority.authorize_current` once and
remembering it in the run's `store` file (`<run>/autopilot/daily/current-authorization.json`),
so a rerun reuses the same record. Creation is write-ahead: the intent is recorded in
`current-authorization.intent.json` first, and a resumed attempt adopts the record a
killed one wrote to the global store (same arguments and signer, created after the
intent) instead of creating a second. A remembered record that no longer verifies
(revoked, changed, outside its window) is refused, never silently replaced. `authorize_current`
itself refuses any window touching the sealed 2021-2025 holdout.

CLI (the same policy, for an operator or a scheduler)::

    python -m capstone.pipeline.autopilot.authority --vendor V --field close
        [--field ...] --start YYYY-MM-DD [--end YYYY-MM-DD] --authorized-by NAME

prints `{"record_id": ...}`; a refusal is one line on stderr and exit 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from capstone import current_authority
from capstone.artifacts import atomic_json
from capstone.holdout_authority import HoldoutSealed
from capstone.pipeline.autopilot.intents import IntentFile, fault

KEYS = {"vendor", "fields", "start", "end"}
INTENT_KEY = "current_authorization"


class PolicyRefused(ValueError):
    """The policy cannot create or reuse an authorization for the session."""


def normalize(policy, session):
    """The authorization arguments for `session`, or PolicyRefused."""
    if not policy:
        raise PolicyRefused(
            f"session {session} is after 2020-12-31 and [daily] current_authorization is empty"
        )
    if unknown := sorted(set(policy) - KEYS):
        raise PolicyRefused(f"current_authorization has unknown keys {unknown}")
    vendor, fields = policy.get("vendor"), policy.get("fields")
    if not isinstance(vendor, str) or not vendor:
        raise PolicyRefused("current_authorization.vendor is required")
    if not isinstance(fields, list) or not fields or not all(isinstance(f, str) for f in fields):
        raise PolicyRefused("current_authorization.fields is a non-empty list of names")
    start, end = policy.get("start", session), policy.get("end")
    if not isinstance(start, str) or not (end is None or isinstance(end, str)):
        raise PolicyRefused("current_authorization.start / end are YYYY-MM-DD strings")
    if session < start or (end is not None and session > end):
        raise PolicyRefused(f"session {session} is outside the policy window [{start}, {end}]")
    return {"vendor": vendor, "fields": sorted(set(fields)), "start": start, "end": end}


def _window(arguments, session):
    if session < arguments["start"] or (
        arguments["end"] is not None and session > arguments["end"]
    ):
        window = f"[{arguments['start']}, {arguments['end']}]"
        raise PolicyRefused(f"session {session} is outside the authorized window {window}")


def intent_file(store):
    """The write-ahead intent beside the run's `store` file (`autopilot.intents`)."""
    store = Path(store)
    return IntentFile(store.with_name(store.stem + ".intent.json"))


def adopt(arguments, authorized_by, opened_at):
    """The record a killed attempt created for this intent, or None: same arguments and
    signer, created at or after the intent was opened, and still verifying."""
    folder = current_authority.store_root() / "authorizations"
    found = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        try:
            row = current_authority.authorization(path.stem)
        except (HoldoutSealed, OSError, ValueError):
            continue
        same = {k: row[k] for k in ("vendor", "start", "end", "fields", "authorized_by")} == {
            **{k: arguments[k] for k in ("vendor", "start", "end", "fields")},
            "authorized_by": authorized_by,
        }
        if same and row["at"] >= opened_at:
            found.append((row["at"], row["record_id"]))
    return min(found)[1] if found else None


def ensure(store, policy, session, authorized_by):
    """(record_id, created) for `session`; created is False when the store had it.

    Creation is write-ahead: the intent (arguments, signer, `opened_at`) is recorded before
    `authorize_current`. A resumed attempt with an open intent adopts the record the killed
    one wrote to the global store instead of creating a second (the record id hashes no
    clock, so `authorize_current` would also return that same unrevoked record).
    `created` is True for the attempt that finished the intent (the event's owner); the
    caller's event is keyed by the intent, so a repeated True is harmless."""
    store = Path(store)
    known = json.loads(store.read_text(encoding="utf-8")) if store.is_file() else None
    try:
        if known is not None:
            arguments = known["arguments"]
            if normalize(policy, arguments["start"]) != arguments:
                raise PolicyRefused(
                    "[daily] current_authorization changed after its record was created"
                )
            _window(arguments, session)
            current_authority.require_active(
                known["record_id"],
                vendor=arguments["vendor"],
                start=arguments["start"],
                end=session,
                fields=arguments["fields"],
            )
            return known["record_id"], False
        arguments = normalize(policy, session)
        intents = intent_file(store)
        opened = intents.get(INTENT_KEY)
        if opened is not None and (opened["arguments"], opened["authorized_by"]) != (
            arguments,
            authorized_by,
        ):
            # An earlier intent for other arguments: harmless unless it left a record.
            if adopt(opened["arguments"], opened["authorized_by"], opened["opened_at"]):
                raise PolicyRefused(
                    "[daily] current_authorization changed after a killed attempt created "
                    "a record for the earlier policy; a person reviews the store"
                )
            opened = None
        if opened is None:
            opened = intents.put(
                INTENT_KEY,
                arguments=arguments,
                authorized_by=authorized_by,
                opened_at=datetime.now(UTC).isoformat(),
            )
        record_id = adopt(arguments, authorized_by, opened["opened_at"])
        if record_id is None:
            record_id = current_authority.authorize_current(
                arguments["vendor"],
                start=arguments["start"],
                end=arguments["end"],
                fields=arguments["fields"],
                authorized_by=authorized_by,
            )
        fault("autopilot:daily:authorized")
    except HoldoutSealed as refused:
        raise PolicyRefused(str(refused)) from refused
    atomic_json(
        store, {"record_id": record_id, "arguments": arguments, "authorized_by": authorized_by}
    )
    intents.close(INTENT_KEY, record_id=record_id)
    return record_id, True


def parser():
    top = argparse.ArgumentParser(
        prog="python -m capstone.pipeline.autopilot.authority",
        description="Create a current-data authorization (validation.current_authority).",
    )
    top.add_argument("--vendor", required=True)
    top.add_argument("--field", action="append", required=True, dest="fields")
    top.add_argument("--start", required=True)
    top.add_argument("--end")
    top.add_argument("--authorized-by", required=True)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        record_id = current_authority.authorize_current(
            args.vendor,
            start=args.start,
            end=args.end,
            fields=args.fields,
            authorized_by=args.authorized_by,
        )
    except (HoldoutSealed, ValueError) as refused:
        print(f"authority: {refused}", file=sys.stderr)
        return 2
    print(json.dumps({"record_id": record_id, "store": str(current_authority.store_root())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
