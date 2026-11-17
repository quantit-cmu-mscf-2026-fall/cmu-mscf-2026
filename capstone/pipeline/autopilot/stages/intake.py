"""Intake stages: `sources` (freeze the inbox, ingest it) and `registration` (fill the template).

`sources` freezes the inbox once (`<run>/autopilot/inbox.json`, then
`<run>/autopilot/sources.json`, each atomic) and ingests it with
`source_batch.run_batch` (`resume` once the batch receipt exists), then verifies it with
`source_batch.status`. `registration` fills the template from the batch, the archive's
`data_version` and `[registration] authority_id`, writes
`<run>/autopilot/registration.json` once, and ends with the spend guard
`spend.guard(ctx, "research", <sum of call_limits>)`; its STOPs propagate.

No stage here calls a model. The template format is
`docs/examples/registration.example.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from capstone.factors.generation import ConstructorContext
from capstone.factors.idea_contracts import MethodsReview, ReviewedSourceInventory
from capstone.factors.idea_routes import catalogue_rows
from capstone.factors.research_artifacts import atomic_json
from capstone.factors.research_contracts import ResearchPolicy
from capstone.pipeline import source_batch
from capstone.pipeline.autopilot import ap2_inbox, intents, spend
from capstone.pipeline.autopilot.config import Key
from capstone.pipeline.autopilot.reasons import Blocked, ReasonSpec, Stop
from capstone.pipeline.autopilot.stage import Stage, StageResult
from capstone.strategy.program_schedule import load_schedule

UNIT = "intake"
INBOX = ("autopilot", "inbox.json")
SOURCES = ("autopilot", "sources.json")
REGISTRATION = spend.REGISTRATION
BATCH = "source-batch"
SOURCE_REASONS = ("source_missing", "pdf_requires_ocr", "pdf_partial_requires_ocr")
REQUIRED = ("catalogue", "policy", "constructor_context", "call_limits", "reviewed_methods")
OPTIONAL = ("qualified_inputs", "input_inventory_id", "max_prompt_chars")
INBOX_FILES = "inbox_files"  # The catalogue-entry selector; removed from the output


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- sources ------------------------------------------------------------------------


def _changes(frozen, current):
    added = sorted(current.keys() - frozen.keys())
    removed = sorted(frozen.keys() - current.keys())
    changed = sorted(n for n in frozen.keys() & current.keys() if frozen[n] != current[n])
    labelled = (("added", added), ("removed", removed), ("changed", changed))
    return "; ".join(f"{label} {names}" for label, names in labelled if names)


def _freeze_inbox(ctx):
    """The frozen {file name: sha256} and {file name: source_id}; written on first run."""
    inbox, listing = ctx.section["inbox"], ctx.path(*INBOX)
    if ctx.path(*SOURCES).is_file():
        frozen = _read(listing)
        try:
            current = ap2_inbox.fingerprint(inbox)
        except ap2_inbox.InboxError as error:
            raise Stop("inbox_changed_after_ingest", str(error)) from error
        if current != frozen["files"]:
            detail = f"{inbox}: {_changes(frozen['files'], current)}"
            raise Stop("inbox_changed_after_ingest", detail)
        _frozen_event(ctx, frozen)  # a kill after the freeze, before its event
        return frozen
    try:
        files = ap2_inbox.fingerprint(inbox)
        if not files:
            raise Blocked("inbox_empty", f"no source file in {inbox}")
        rows = ap2_inbox.source_rows(inbox, ctx.section["default_role"])
        requests = ap2_inbox.requests(rows)
        if ap2_inbox.fingerprint(inbox) != files:  # a file landed while we read
            raise Blocked("inbox_invalid", f"{inbox} changed while it was read; retrying")
    except ap2_inbox.InboxError as error:
        raise Blocked("inbox_invalid", str(error)) from error
    frozen = {
        "schema_version": 1,
        "inbox": str(inbox),
        "files": files,
        "source_ids": {name: request.source_id for name, request in requests.items()},
    }
    # inbox.json first: sources.json existing is what "frozen" means.
    atomic_json(listing, frozen)
    atomic_json(ctx.path(*SOURCES), {"sources": [row for _, row in rows]})
    intents.fault("autopilot:sources:frozen")
    _frozen_event(ctx, frozen)
    return frozen


def _frozen_event(ctx, frozen):
    """`sources_frozen`, once per frozen listing (emitted exactly once)."""
    rows = _read(ctx.path(*SOURCES))["sources"]
    key = intents.intent_id("sources", "sources_frozen", frozen["files"])
    intents.event_once(
        ctx, key, "sources_frozen", reviewer=ctx.reviewer, files=frozen["files"], sources=len(rows)
    )


def _batch(ctx):
    batch = ctx.path(BATCH)
    try:
        if (batch / "receipt.json").is_file():
            summary = source_batch.status(batch)
            if all(row["state"] != "pending" for row in summary["sources"]):
                # Every source has its recorded outcome: a resume would only advance the
                # history checkpoint, which `research_freeze` binds (a rerun after a
                # kill must freeze the same snapshot as an uninterrupted run).
                return summary
            source_batch.run_batch(batch)  # resume: never re-acquires a recorded outcome
        else:
            source_batch.run_batch(
                batch, sources=ctx.path(*SOURCES), max_bytes=ctx.section["max_bytes"]
            )
        return source_batch.status(batch)
    except ValueError as error:
        raise Stop("source_batch_refused", f"{batch}: {error}") from error


def _sources(ctx):
    frozen = _freeze_inbox(ctx)
    summary = _batch(ctx)
    names = {source_id: name for name, source_id in frozen["source_ids"].items()}
    not_ready = [row for row in summary["sources"] if row["state"] != "ready"]
    if not_ready:
        codes = [row["reason"] for row in not_ready if row["reason"] in SOURCE_REASONS]
        detail = ", ".join(
            f"{names.get(row['source_id'], row['source_id'][:12])}: "
            f"{row['state']} {row['reason']}".strip()
            for row in not_ready
        )
        raise Blocked(codes[0] if codes else "source_blocked", detail)
    revisions = {row["source_id"]: row["revision_id"] for row in summary["sources"]}
    if set(revisions) != set(names):
        detail = f"batch sources {sorted(revisions)} differ from {'/'.join(INBOX)}"
        raise Stop("source_batch_refused", detail)
    return StageResult(
        artifacts=[
            ctx.path(*INBOX),
            ctx.path(*SOURCES),
            ctx.path(BATCH, "manifest.json"),
            ctx.path(BATCH, "receipt.json"),
        ],
        data={"revisions": dict(sorted(revisions.items())), "states": summary["states"]},
    )


# --- registration -------------------------------------------------------------------


def _data_version(ctx):
    archive = ctx.config.core["input_archive"]
    if archive is None:
        raise Blocked("needs_data", "[autopilot] input_archive is not set")
    try:
        return ap2_inbox.archive_data_version(archive)
    except FileNotFoundError as error:
        raise Blocked("needs_data", str(error)) from error
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise Blocked("stage_refused", f"input archive refused: {error}") from error


def _revisions(ctx):
    """{inbox file name: revision_id}, checked against the batch's own status."""
    revisions = ctx.data("sources").get("revisions") or {}
    try:
        source_ids = _read(ctx.path(*INBOX))["source_ids"]
        summary = source_batch.status(ctx.path(BATCH))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise Blocked("stage_refused", f"source batch unreadable: {error}") from error
    ready = {
        row["source_id"]: row["revision_id"]
        for row in summary["sources"]
        if row["state"] == "ready" and row["revision_id"]
    }
    if revisions != ready or set(source_ids.values()) != set(ready):
        raise Blocked("stage_refused", "the source batch no longer matches `sources` data")
    return {name: ready[source_id] for name, source_id in source_ids.items()}


def _invalid(detail):
    return Blocked("registration_template_invalid", detail)


def _limits(call_limits, catalogue):
    if not isinstance(call_limits, dict) or not call_limits:
        raise _invalid("call_limits must be a nonempty table")
    for bucket, count in call_limits.items():
        if type(count) is not int or count < 0:
            raise _invalid(f"call_limits[{bucket}] = {count!r} is not a call count")
    buckets = {f"{row['brief']['family']}:{row['brief']['track']}" for row in catalogue}
    allowed = buckets | {f"{bucket}:alignment" for bucket in buckets}
    if not buckets <= set(call_limits) or set(call_limits) - allowed:
        raise _invalid(f"call_limits {sorted(call_limits)} must cover exactly {sorted(buckets)}")


def build_registration(template, *, revisions, data_version, authority_id):
    """The registration document the freeze reads; deterministic in its inputs.

    Every catalogue entry gets `source_revisions`: the revisions of its `inbox_files`
    when it names some, else of every ingested source."""
    if not isinstance(template, dict):
        raise _invalid("the template is not a JSON object")
    missing = [key for key in REQUIRED if key not in template]
    unknown = sorted(set(template) - set(REQUIRED) - set(OPTIONAL))
    if missing or unknown:
        raise _invalid(f"missing {missing}, unknown {unknown}")
    doc = json.loads(json.dumps(template))  # a deep copy
    try:
        catalogue = []
        for index, row in enumerate(doc["catalogue"]):
            entry = dict(row)
            files = entry.pop(INBOX_FILES, None)
            if files is None:
                chosen = set(revisions.values())
            elif not isinstance(files, list) or not files or not set(files) <= set(revisions):
                detail = f"catalogue[{index}].{INBOX_FILES} {files!r}: not files of the inbox"
                raise _invalid(f"{detail} {sorted(revisions)}")
            else:
                chosen = {revisions[name] for name in files}
            entry["source_revisions"] = sorted(chosen)
            catalogue.append(entry)
        doc["catalogue"] = catalogue_rows(catalogue)
        if not doc["catalogue"]:
            raise _invalid("the catalogue is empty")
        doc["constructor_context"] = {**doc["constructor_context"], "data_version": data_version}
        doc["reviewed_methods"] = {**doc["reviewed_methods"], "authority_id": authority_id}
        ResearchPolicy(**doc["policy"])
        context = dict(doc["constructor_context"])
        context["schedules"] = tuple(load_schedule(row) for row in context["schedules"])
        ConstructorContext(**context)
        ReviewedSourceInventory(
            authority_id, tuple(MethodsReview(**row) for row in doc["reviewed_methods"]["reviews"])
        )
        _limits(doc["call_limits"], doc["catalogue"])
    except Blocked:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise _invalid(f"{type(error).__name__}: {error}") from error
    return doc


def _registration(ctx):
    revisions = _revisions(ctx)
    data_version = _data_version(ctx)
    try:
        template = _read(Path(ctx.section["template"]))
    except (OSError, ValueError) as error:
        raise _invalid(f"{ctx.section['template']}: {error}") from error
    doc = build_registration(
        template,
        revisions=revisions,
        data_version=data_version,
        authority_id=ctx.section["authority_id"],
    )
    path = ctx.path(*REGISTRATION)
    if path.is_file():
        try:
            existing = _read(path)
        except (OSError, ValueError) as error:
            raise Blocked("stage_refused", f"{path}: {error}") from error
        if existing != doc:
            detail = f"{path} differs from the one regenerated from the template and batch"
            raise Blocked("stage_refused", detail)
    else:
        atomic_json(path, doc)
        intents.fault("autopilot:registration:written")
    intents.event_once(  # every pass: a kill after the write, before the event
        ctx,
        intents.intent_id("registration", "registration_written", doc),
        "registration_written",
        reviewer=ctx.reviewer,
        data_version=data_version,
        source_revisions=sorted(set(revisions.values())),
    )
    check = spend.guard(ctx, "research", spend.registration_ceiling(ctx))  # STOPs propagate
    return StageResult(
        artifacts=[path],
        data={
            "call_ceiling": check["ceiling"],
            "global_total": check["global_total"],
            "data_version": data_version,
        },
    )


STAGES = (
    Stage(
        name="sources",
        unit=UNIT,
        run=_sources,
        schema={
            "inbox": Key("path", help="folder of dropped source files"),
            "default_role": Key("str", "seed", "role of a file that has no sidecar"),
            "max_bytes": Key("int", 5_000_000, "per-source size limit (`ingest --max-bytes`)"),
        },
        reasons={
            "inbox_empty": ReasonSpec("Source", "drop at least one source file in the inbox"),
            "inbox_changed_after_ingest": ReasonSpec(
                "Source", "a source list is frozen per run; start a new run root for new files"
            ),
            "source_missing": ReasonSpec("Source"),
            "pdf_requires_ocr": ReasonSpec("Source"),
            "pdf_partial_requires_ocr": ReasonSpec("Source"),
            # inbox and batch refusals
            "inbox_invalid": ReasonSpec(
                "Source", "fix the named inbox file (see docs/SOURCE_INBOX.md)"
            ),
            "source_blocked": ReasonSpec(
                "Source", "read `detail`; the batch keeps outcomes, so start a new run root"
            ),
            "source_batch_refused": ReasonSpec(
                "Source", "read `detail`; restore `<run>/source-batch`, or start a new run root"
            ),
        },
    ),
    Stage(
        name="registration",
        unit=UNIT,
        run=_registration,
        depends_on=("sources",),
        schema={
            "template": Key("path", help="registration template (catalogue, policy, limits)"),
            "authority_id": Key("str", help="`reviewed_methods.authority_id`"),
        },
        reasons={
            "needs_data": ReasonSpec("Claim"),
            "stage_refused": ReasonSpec("Operator"),
            "registration_template_invalid": ReasonSpec(
                "Operator",
                "fix `[registration] template` (docs/examples/registration.example.json)",
            ),
        },
    ),
)
