"""`sources` (inbox -> source list -> batch) and `registration` (template + spend guard).

Synthetic only: inbox notes are written here, the input archive is a manufactured
`save_inputs` descriptor, and no stage reaches a model (none of these stages calls one).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.pipeline.autopilot import config as cfg
from capstone.pipeline.autopilot import runner, stages
from capstone.pipeline.autopilot.journal import Journal, sha256_file
from capstone.pipeline.autopilot.stages import intake
from capstone.pipeline.historical import HistoricalInputs
from capstone.pipeline.historical_bundle import save_inputs

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "docs/examples/registration.example.json"
ONLY = ("sources", "registration")
VERSION = "synthetic-intake-v1"
NOTE = "Synthetic note: high turnover relative to shares may revert next month."


def ap2_stages():
    """This unit's stages as discovered; other units' sections are not configured here."""
    return [stage for stage in stages.discover() if stage.unit == "intake"]


def archive(root):
    """A manufactured historical-inputs descriptor; no licensed data."""
    dates = pd.bdate_range("1996-01-02", "1996-01-31")
    values = pd.DataFrame({101: 1.0, 102: 2.0}, index=dates)
    names = ("returns", "volume", "shares")
    panel = Panel(
        {name: values for name in names},
        values.gt(0),
        values * 0,
        data_version=VERSION,
        calendar=dates,
        availability={name: pd.Series(dates, index=dates) for name in names},
    )
    qualification = {
        "source_hashes": {"daily": "4" * 64},
        "sources": {"calendar": {"qualified": True}, "events": {"qualified": True}},
        "event_return_identities": {"verified": True},
        "global_blockers": [],
    }
    events = pd.DataFrame(columns=["event_id", "event_date", "permno", "kind"])
    inputs = HistoricalInputs(
        panel, values, values.copy(), events, values * 0, values * 0, qualification
    )
    return save_inputs(root, inputs)


def template(**changes):
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc.update(changes)
    return doc


class Run:
    """One synthetic run: inbox, template, archive and an autopilot config under tmp."""

    def __init__(self, tmp, *, run_ceiling=24, archive_path="archive", accounting=(), doc=None):
        self.tmp = tmp
        self.inbox = tmp / "inbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.template = tmp / "registration.template.json"
        self.write_template(doc or template())
        if archive_path == "archive" and not (tmp / "archive").exists():
            archive(tmp / "archive")
        self.raw = {
            "autopilot": {"run_root": "run", "policy_id": "intake-test"},
            "spend": {
                "run_ceiling": run_ceiling,
                "global_ceiling": 300,
                "accounting_roots": list(accounting),
            },
            "sources": {"inbox": "inbox"},
            "registration": {"template": self.template.name, "authority_id": "synthetic-review"},
        }
        if archive_path is not None:
            self.raw["autopilot"]["input_archive"] = archive_path
        self.found = ap2_stages()
        self.config = cfg.validate(self.raw, self.found, path=tmp / "autopilot.toml")
        self.root = self.config.run_root

    def write_template(self, doc):
        self.template.write_text(json.dumps(doc), encoding="utf-8")

    def drop(self, name, text=NOTE):
        path = self.inbox / name
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text if isinstance(text, str) else json.dumps(text), encoding="utf-8")
        return path

    def tick(self, only=ONLY):
        return runner.tick(self.config, self.found, only=only)

    def stage(self, name):
        return Journal(self.root).stage(name)

    def read(self, *parts):
        return json.loads(self.root.joinpath(*parts).read_text(encoding="utf-8"))

    def toml(self):
        lines = []
        for section, values in self.raw.items():
            lines.append(f"[{section}]")
            lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
        path = self.tmp / "autopilot.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def reason(report, stage):
    return next(p for p in report.problems if p["stage"] == stage)


# --- sources ------------------------------------------------------------------------


def test_the_inbox_becomes_a_frozen_source_list_and_an_ingested_batch(tmp_path):
    run = Run(tmp_path)
    run.drop("b-note.md")
    run.drop("b-note.md.source.json", {"title": "Synthetic turnover note", "role": "reference"})
    run.drop("a-idea.source.json", {"text": "Original idea: attention precedes reversal."})
    run.drop(".hidden", "ignored")
    report = run.tick(only=("sources",))
    assert report.exit_code == 0, report.line()
    rec = run.stage("sources")
    assert rec["state"] == "done"
    listing = run.read("autopilot", "inbox.json")
    assert list(listing["files"]) == ["a-idea.source.json", "b-note.md", "b-note.md.source.json"]
    rows = run.read("autopilot", "sources.json")["sources"]
    assert [row["kind"] for row in rows] == ["inline_text", "local_text"]  # sorted by name
    assert rows[0]["role"] == "seed"  # [sources] default_role
    assert rows[1]["role"] == "reference" and rows[1]["title"] == "Synthetic turnover note"
    assert Path(rows[1]["path"]) == (run.inbox / "b-note.md").resolve()
    revisions = rec["data"]["revisions"]
    assert set(revisions) == set(listing["source_ids"].values()) and len(revisions) == 2
    assert rec["data"]["states"] == {"ready": 2}
    assert set(rec["artifacts"]) == {
        "autopilot/inbox.json",
        "autopilot/sources.json",
        "source-batch/manifest.json",
        "source-batch/receipt.json",
    }
    frozen = [e for e in Journal(run.root).events() if e["kind"] == "sources_frozen"]
    assert len(frozen) == 1 and frozen[0]["reviewer"] == "autopilot:intake-test"
    again = run.tick(only=("sources",))
    assert again.exit_code == 0 and again.ran == []  # done: not run again


def test_a_rerun_of_a_complete_batch_leaves_its_history_checkpoint_alone(tmp_path):
    """A kill after the batch completed must not advance the source history that
    `research_freeze` binds, or a resumed run freezes another snapshot than an
    uninterrupted one."""
    run = Run(tmp_path)
    run.drop("a-idea.source.json", {"text": "Original idea: attention precedes reversal."})
    assert run.tick(only=("sources",)).exit_code == 0
    history = run.root / "source-batch" / "archive" / "outcomes.jsonl"
    before = history.read_bytes()
    Journal(run.root).update("sources", state="running")  # killed before the journal update
    again = run.tick(only=("sources",))
    assert again.exit_code == 0 and again.ran == ["sources"]
    assert history.read_bytes() == before


def test_an_empty_inbox_waits_without_freezing_anything(tmp_path):
    run = Run(tmp_path)
    report = run.tick(only=("sources",))
    problem = reason(report, "sources")
    assert problem["status"] == "blocked" and problem["reason"]["code"] == "inbox_empty"
    assert problem["reason"]["class"] == "Source"
    assert not (run.root / "autopilot/sources.json").exists()
    run.drop("note.txt")
    assert run.tick(only=("sources",)).exit_code == 0
    assert run.stage("sources")["state"] == "done"


@pytest.mark.parametrize(
    ("name", "content", "detail"),
    [
        ("paper.docx", "x", "unsupported file type .docx"),
        ("idea.source.json", {"title": "no text or url"}, "exactly one of text, url"),
        ("idea.source.json", {"text": "t", "path": "C:/x.md"}, "unsupported fields"),
        ("note.md.source.json", {"role": "oracle"}, "unsupported evidence role"),
    ],
)
def test_an_inbox_the_grammar_refuses_is_a_typed_blocker_before_anything_is_frozen(
    tmp_path, name, content, detail
):
    run = Run(tmp_path)
    run.drop("note.md")
    run.drop(name, content)
    problem = reason(run.tick(only=("sources",)), "sources")
    assert problem["status"] == "blocked" and problem["reason"]["code"] == "inbox_invalid"
    assert detail in problem["reason"]["detail"]
    assert not (run.root / "autopilot").exists()


def test_two_files_that_name_one_source_are_refused(tmp_path):
    run = Run(tmp_path)
    for name in ("a.md", "b.md"):
        run.drop(name)
        run.drop(name + ".source.json", {"key": "same-work"})
    problem = reason(run.tick(only=("sources",)), "sources")
    assert (
        problem["reason"]["code"] == "inbox_invalid" and "one source" in problem["reason"]["detail"]
    )


def test_a_blocked_source_blocks_and_a_changed_inbox_after_freezing_is_a_stop(tmp_path):
    run = Run(tmp_path)
    run.drop("note.md")
    run.drop("scan.pdf", b"%PDF-1.4 not a readable document")
    first = reason(run.tick(only=("sources",)), "sources")
    assert first["status"] == "blocked"
    assert first["reason"]["code"] in (
        "pdf_requires_ocr",
        "pdf_partial_requires_ocr",
        "source_blocked",
    )
    assert "scan.pdf" in first["reason"]["detail"]
    assert (run.root / "autopilot/sources.json").is_file()
    retry = reason(run.tick(only=("sources",)), "sources")  # resumes; same outcome
    assert retry["reason"]["code"] == first["reason"]["code"]
    run.drop("late.md")
    stop = reason(run.tick(only=("sources",)), "sources")
    assert stop["status"] == "stopped" and stop["reason"]["code"] == "inbox_changed_after_ingest"
    assert "added ['late.md']" in stop["reason"]["detail"]
    assert run.stage("sources")["state"] == "stopped"


def test_an_edited_inbox_file_after_freezing_is_a_stop(tmp_path):
    run = Run(tmp_path)
    run.drop("note.md")
    run.drop("scan.pdf", b"%PDF-1.4 not a readable document")
    run.tick(only=("sources",))
    run.drop("note.md", NOTE + " edited")
    stop = reason(run.tick(only=("sources",)), "sources")
    assert stop["reason"]["code"] == "inbox_changed_after_ingest"
    assert "changed ['note.md']" in stop["reason"]["detail"]


def test_a_tampered_batch_is_a_typed_stop_not_a_traceback(tmp_path):
    run = Run(tmp_path)
    run.drop("note.md")
    run.drop("scan.pdf", b"%PDF-1.4 not a readable document")
    run.tick(only=("sources",))
    receipt = run.root / "source-batch/receipt.json"
    receipt.write_text(json.dumps({"manifest_id": "0" * 64}), encoding="utf-8")
    stop = reason(run.tick(only=("sources",)), "sources")
    assert stop["status"] == "stopped" and stop["reason"]["code"] == "source_batch_refused"
    assert "manifest changed" in stop["reason"]["detail"]


# --- registration -------------------------------------------------------------------


def ingested(tmp_path, **kwargs):
    run = Run(tmp_path, **kwargs)
    run.drop("note.md")
    assert run.tick(only=("sources",)).exit_code == 0
    return run


def test_registration_is_built_from_the_template_batch_and_archive_then_passes_the_guard(tmp_path):
    run = ingested(tmp_path)
    report = run.tick()
    assert report.exit_code == 0, report.line()
    assert report.ran == ["registration"]
    doc = run.read("autopilot", "registration.json")
    revision = next(iter(run.stage("sources")["data"]["revisions"].values()))
    assert doc["catalogue"][0]["source_revisions"] == [revision]
    assert doc["constructor_context"]["data_version"] == VERSION
    assert doc["reviewed_methods"]["authority_id"] == "synthetic-review"
    example = template()
    for key in ("policy", "call_limits", "max_prompt_chars"):
        assert doc[key] == example[key]
    rec = run.stage("registration")
    assert rec["data"]["call_ceiling"] == 4 == sum(example["call_limits"].values())
    assert rec["data"]["global_total"] == 4
    assert list(rec["artifacts"]) == ["autopilot/registration.json"]
    guard = run.read("autopilot", "spend-guard.json")
    assert guard["scopes"] == {"research": 4}
    events = Journal(run.root).events()
    assert [e["kind"] for e in events if e["kind"] in ("registration_written", "spend_guard")] == [
        "registration_written",
        "spend_guard",
    ]
    assert run.tick().ran == []  # idempotent


def test_inbox_files_choose_each_entrys_revisions(tmp_path):
    run = Run(tmp_path)
    run.drop("a.md")
    run.drop("b.md", NOTE + " A second synthetic note.")
    assert run.tick(only=("sources",)).exit_code == 0
    names = run.read("autopilot", "inbox.json")["source_ids"]
    revisions = run.stage("sources")["data"]["revisions"]
    doc = template()
    first = dict(doc["catalogue"][0], inbox_files=["b.md"])
    second = json.loads(json.dumps(doc["catalogue"][0]))
    second["brief"]["packet_id"] = "idea-1"
    run.write_template(dict(doc, catalogue=[first, second]))
    assert run.tick().exit_code == 0
    written = run.read("autopilot", "registration.json")["catalogue"]
    by_packet = {row["brief"]["packet_id"]: row for row in written}
    assert by_packet["idea-0"]["source_revisions"] == [revisions[names["b.md"]]]
    assert by_packet["idea-1"]["source_revisions"] == sorted(revisions.values())
    assert all("inbox_files" not in row for row in written)


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        ({"call_limits": {"liquidity_supply:discovery": 3}} | {"x": 1}, "unknown ['x']"),
        ({"call_limits": {"other:discovery": 3}}, "must cover exactly"),
        ({"call_limits": {"liquidity_supply:discovery": -1}}, "not a call count"),
        ({"catalogue": []}, "catalogue is empty"),
        ({"policy": {"version": 1}}, "TypeError"),
    ],
)
def test_an_invalid_template_is_a_typed_blocker_and_a_fixed_one_proceeds(tmp_path, change, detail):
    run = ingested(tmp_path)
    run.write_template(template(**change))
    problem = reason(run.tick(), "registration")
    assert problem["status"] == "blocked"
    assert problem["reason"]["code"] == "registration_template_invalid"
    assert detail in problem["reason"]["detail"]
    assert not (run.root / "autopilot/registration.json").exists()
    run.write_template(template())
    assert run.tick().exit_code == 0


def test_an_inbox_files_name_outside_the_inbox_is_refused(tmp_path):
    run = ingested(tmp_path)
    doc = template()
    doc["catalogue"][0]["inbox_files"] = ["absent.md"]
    run.write_template(doc)
    problem = reason(run.tick(), "registration")
    assert problem["reason"]["code"] == "registration_template_invalid"
    assert "absent.md" in problem["reason"]["detail"]


def test_without_a_readable_archive_registration_needs_data(tmp_path):
    run = ingested(tmp_path, archive_path=None)
    problem = reason(run.tick(), "registration")
    assert problem["reason"]["code"] == "needs_data" and problem["reason"]["class"] == "Claim"
    assert problem["reason"]["cleared_by"]  # the blocker glossary's row
    missing = ingested(tmp_path / "m", archive_path="nowhere")
    assert reason(missing.tick(), "registration")["reason"]["code"] == "needs_data"


def test_a_changed_archive_descriptor_is_refused(tmp_path):
    run = ingested(tmp_path)
    descriptor = tmp_path / "archive/historical-inputs.json"
    metadata = json.loads(descriptor.read_text())
    metadata["data_version"] = "edited"
    descriptor.write_text(json.dumps(metadata))
    problem = reason(run.tick(), "registration")
    assert (
        problem["reason"]["code"] == "stage_refused" and "input_hash" in problem["reason"]["detail"]
    )


def test_a_registration_that_differs_from_the_regenerated_one_is_refused(tmp_path):
    run = ingested(tmp_path, run_ceiling=0)
    stop = reason(run.tick(), "registration")  # written, then the guard stops it
    assert stop["reason"]["code"] == "spend_run_ceiling_exceeded"
    path = run.root / "autopilot/registration.json"
    doc = json.loads(path.read_text())
    doc["call_limits"]["liquidity_supply:discovery"] = 1
    path.write_text(json.dumps(doc))
    runner.clear(run.config, run.found, "registration", reviewer="person", reason="test")
    problem = reason(run.tick(), "registration")
    assert problem["status"] == "blocked" and problem["reason"]["code"] == "stage_refused"


# --- spend guard ---------------------------------------------------------------


def test_spend_guard_refuses_to_freeze_above_the_run_ceiling(tmp_path):
    run = ingested(tmp_path, run_ceiling=3)
    problem = reason(run.tick(), "registration")
    assert problem["status"] == "stopped"
    assert problem["reason"]["code"] == "spend_run_ceiling_exceeded"
    assert problem["reason"]["class"] == "Spend"
    assert "4 > run_ceiling 3" in problem["reason"]["detail"]
    assert not (run.root / "autopilot/spend-guard.json").exists()  # nothing accepted
    assert run.stage("registration")["state"] == "stopped"
    assert run.tick().ran == []  # a STOP holds until a person clears it


def accounted(root, reservations):
    path = root / "source-run/request-accounting.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outbound_reservations": reservations}), encoding="utf-8")
    return root


def test_spend_guard_sums_every_registered_runs_accounting_toward_the_global_300(tmp_path):
    accounted(tmp_path / "r1", 200)
    accounted(tmp_path / "r2", 97)
    run = ingested(tmp_path / "new", accounting=["../r1", "../r2"])
    problem = reason(run.tick(), "registration")
    assert problem["status"] == "stopped"
    assert problem["reason"]["code"] == "spend_global_ceiling_exceeded"
    assert "4 + other runs 297 > 300" in problem["reason"]["detail"]

    fits = ingested(tmp_path / "fits", accounting=["../r1"])
    assert fits.tick().exit_code == 0
    assert fits.stage("registration")["data"]["global_total"] == 204


def test_spend_guard_refuses_a_listed_run_without_accounting(tmp_path):
    (tmp_path / "empty-run").mkdir()
    run = ingested(tmp_path / "new", accounting=["../empty-run"])
    problem = reason(run.tick(), "registration")
    assert problem["status"] == "stopped"
    assert problem["reason"]["code"] == "spend_accounting_unreadable"


# --- kill and resume in a fresh process ----------------------------------------------

KILL = """
import os, sys
from pathlib import Path
from capstone.factors import source_archive
from capstone.pipeline.autopilot import config as cfg, runner, spend, stages

mode, marker = sys.argv[2], Path(sys.argv[1]) / ("killed-" + sys.argv[2])
if mode == "sources" and not marker.exists():
    intake = source_archive.SourceArchive.intake
    def killing(self, *args, **kwargs):
        outcome = intake(self, *args, **kwargs)
        marker.write_text("x")
        os._exit(9)  # killed after the first source is archived
    source_archive.SourceArchive.intake = killing
if mode == "registration" and not marker.exists():
    def killing(*args, **kwargs):
        marker.write_text("x")
        os._exit(9)  # killed after registration.json is written, before the guard passes
    spend.guard = killing
found = [s for s in stages.discover() if s.unit == "intake"]
config = cfg.load(Path(sys.argv[1]) / "autopilot.toml", found)
report = runner.tick(config, found, only=("sources", "registration"))
print(report.line())
sys.exit(report.exit_code)
"""


def fresh(tmp_path, mode):
    script = tmp_path / "kill.py"
    script.write_text(KILL, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    command = [sys.executable, str(script), str(tmp_path), mode]
    return subprocess.run(command, env=env, capture_output=True, text=True, cwd=tmp_path)


def test_sources_killed_mid_batch_resumes_in_a_fresh_process(tmp_path):
    run = Run(tmp_path)
    run.drop("a.md")
    run.drop("b.md", NOTE + " A second synthetic note.")
    run.toml()
    first = fresh(tmp_path, "sources")
    assert first.returncode == 9, first.stderr
    assert run.stage("sources")["state"] == "running"
    frozen = sha256_file(run.root / "autopilot/sources.json")
    second = fresh(tmp_path, "sources")
    assert second.returncode == 0, second.stdout + second.stderr
    rec = run.stage("sources")
    assert rec["state"] == "done" and rec["attempts"] == 2
    assert rec["data"]["states"] == {"ready": 2}
    assert sha256_file(run.root / "autopilot/sources.json") == frozen  # never regenerated
    assert run.stage("registration")["state"] == "done"
    starts = [e for e in Journal(run.root).events() if e["kind"] == "stage_started"]
    assert [(e["stage"], e["resumed"]) for e in starts] == [
        ("sources", False),
        ("sources", True),
        ("registration", False),
    ]
    assert len([e for e in Journal(run.root).events() if e["kind"] == "sources_frozen"]) == 1


def test_registration_killed_before_the_guard_resumes_in_a_fresh_process(tmp_path):
    run = Run(tmp_path)
    run.drop("note.md")
    run.toml()
    first = fresh(tmp_path, "registration")
    assert first.returncode == 9, first.stderr
    assert run.stage("registration")["state"] == "running"
    written = sha256_file(run.root / "autopilot/registration.json")
    assert not (run.root / "autopilot/spend-guard.json").exists()
    second = fresh(tmp_path, "registration")
    assert second.returncode == 0, second.stdout + second.stderr
    rec = run.stage("registration")
    assert rec["state"] == "done" and rec["attempts"] == 2
    assert rec["artifacts"] == {"autopilot/registration.json": written}  # same document
    assert run.read("autopilot", "spend-guard.json")["scopes"] == {"research": 4}
    events = Journal(run.root).events()
    assert len([e for e in events if e["kind"] == "registration_written"]) == 1
    assert Journal(run.root).verify_events() == len(events)


# --- the documented template ---------------------------------------------------------


def test_the_example_template_builds_with_only_the_documented_substitutions():
    doc = intake.build_registration(
        template(), revisions={"note.md": "ab" * 32}, data_version="v", authority_id="a"
    )
    example = template()
    assert doc["catalogue"][0]["source_revisions"] == ["ab" * 32]
    assert doc["constructor_context"] == {**example["constructor_context"], "data_version": "v"}
    assert doc["reviewed_methods"] == {"authority_id": "a", "reviews": []}
    rest = {k: v for k, v in doc.items() if k not in ("catalogue", "constructor_context")}
    assert rest == {
        k: v for k, v in example.items() if k not in ("catalogue", "constructor_context")
    } | {"reviewed_methods": {"authority_id": "a", "reviews": []}}
