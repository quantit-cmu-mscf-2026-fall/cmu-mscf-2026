"""The inbox grammar and the input archive's data version.

Synthetic notes written here and a manufactured `save_inputs` descriptor; no model, no
network and no data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.pipeline.autopilot import ap2_inbox
from capstone.pipeline.autopilot.ap2_inbox import InboxError
from capstone.pipeline.autopilot.journal import sha256_file
from capstone.pipeline.historical import HistoricalInputs
from capstone.pipeline.historical_bundle import save_inputs

VERSION = "synthetic-inbox-v1"
NOTE = "Synthetic note: high turnover relative to shares may revert next month."


def drop(inbox, name, text=NOTE):
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(text, encoding="utf-8")


def sidecar(inbox, name, **fields):
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(json.dumps(fields), encoding="utf-8")


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


def test_a_missing_inbox_has_no_fingerprint_and_no_rows(tmp_path):
    assert ap2_inbox.fingerprint(tmp_path / "absent") == {}
    assert ap2_inbox.source_rows(tmp_path / "absent", "seed") == []


def test_the_fingerprint_hashes_every_visible_file_and_skips_dot_names(tmp_path):
    inbox = tmp_path / "inbox"
    drop(inbox, "b.md")
    drop(inbox, "a.txt", "other")
    drop(inbox, ".hidden", "ignored")
    prints = ap2_inbox.fingerprint(inbox)
    assert list(prints) == ["a.txt", "b.md"]
    assert prints["b.md"] == sha256_file(inbox / "b.md")
    drop(inbox, "b.md", "changed")
    assert ap2_inbox.fingerprint(inbox)["b.md"] != prints["b.md"]


def test_the_inbox_is_flat(tmp_path):
    (tmp_path / "inbox" / "nested").mkdir(parents=True)
    with pytest.raises(InboxError, match="flat"):
        ap2_inbox.fingerprint(tmp_path / "inbox")


def test_local_files_become_rows_keyed_by_name_and_content_not_location(tmp_path):
    rows = []
    for place in ("one", "two"):
        inbox = tmp_path / place / "inbox"
        drop(inbox, "b-note.md")
        drop(inbox, "paper.pdf", "%PDF-synthetic")
        rows.append(dict(ap2_inbox.source_rows(inbox, "seed")))
    first, second = rows
    assert list(first) == ["b-note.md", "paper.pdf"]
    assert first["b-note.md"]["kind"] == "local_text" and first["paper.pdf"]["kind"] == "local_pdf"
    assert first["b-note.md"]["role"] == "seed"
    assert Path(first["b-note.md"]["path"]).is_absolute()
    assert first["b-note.md"]["key"] == second["b-note.md"]["key"]  # same name, same content
    assert first["b-note.md"]["key"].startswith("inbox:b-note.md@sha256:")
    assert first["b-note.md"]["path"] != second["b-note.md"]["path"]


def test_a_sidecar_gives_a_files_fields_and_may_set_its_key(tmp_path):
    inbox = tmp_path / "inbox"
    drop(inbox, "a.md")
    sidecar(inbox, "a.md.source.json", title="A note", role="reference", key="my-key")
    ((name, row),) = ap2_inbox.source_rows(inbox, "seed")
    assert name == "a.md"  # the sidecar is read with its file, not as a source of its own
    assert (row["title"], row["role"], row["key"]) == ("A note", "reference", "my-key")


def test_a_standalone_source_is_inline_text_or_a_url_and_never_both(tmp_path):
    inbox = tmp_path / "inbox"
    sidecar(inbox, "idea.source.json", text=NOTE)
    sidecar(inbox, "link.source.json", url="https://example.com/paper")
    rows = dict(ap2_inbox.source_rows(inbox, "seed"))
    assert rows["idea.source.json"]["kind"] == "inline_text"
    assert rows["link.source.json"]["kind"] == "url"
    sidecar(inbox, "both.source.json", text=NOTE, url="https://example.com/x")
    with pytest.raises(InboxError, match="exactly one of text, url"):
        ap2_inbox.source_rows(inbox, "seed")


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("data.csv", "x", "unsupported file type"),
        ("noext", "x", "unsupported file type"),
        ("bad.source.json", "{not json", "bad.source.json"),
        ("odd.source.json", json.dumps({"text": "x", "colour": "red"}), "unsupported fields"),
        ("list.source.json", "[1]", "not an object"),
    ],
)
def test_anything_the_grammar_refuses_is_an_error_not_a_skip(tmp_path, name, text, message):
    drop(tmp_path / "inbox", name, text)
    with pytest.raises(InboxError, match=message):
        ap2_inbox.source_rows(tmp_path / "inbox", "seed")


def test_requests_are_built_before_anything_is_frozen_and_duplicates_are_refused(tmp_path):
    inbox = tmp_path / "inbox"
    drop(inbox, "a.md")
    drop(inbox, "b.md", "different content")
    built = ap2_inbox.requests(ap2_inbox.source_rows(inbox, "seed"))
    assert list(built) == ["a.md", "b.md"]
    assert built["a.md"].source_id != built["b.md"].source_id
    sidecar(inbox, "a.md.source.json", key="shared")
    sidecar(inbox, "b.md.source.json", key="shared")
    with pytest.raises(InboxError, match="one source"):
        ap2_inbox.requests(ap2_inbox.source_rows(inbox, "seed"))


def test_a_row_the_batch_would_refuse_is_refused_here(tmp_path):
    inbox = tmp_path / "inbox"
    drop(inbox, "a.md")
    sidecar(inbox, "a.md.source.json", role="not-a-role")
    with pytest.raises(InboxError, match="a.md"):
        ap2_inbox.requests(ap2_inbox.source_rows(inbox, "seed"))


def test_the_archive_data_version_is_read_after_the_descriptor_checks(tmp_path):
    root = archive(tmp_path / "archive")
    assert ap2_inbox.archive_data_version(root) == VERSION


def test_a_missing_or_changed_archive_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        ap2_inbox.archive_data_version(tmp_path / "nowhere")
    root = archive(tmp_path / "archive")
    descriptor = root / "historical-inputs.json"
    metadata = json.loads(descriptor.read_text(encoding="utf-8"))
    descriptor.write_text(json.dumps({**metadata, "cutoff": "2022-01-01"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        ap2_inbox.archive_data_version(root)
    descriptor.write_text(json.dumps({**metadata, "data_version": "forged"}), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor changed"):
        ap2_inbox.archive_data_version(root)
    descriptor.write_text(json.dumps(metadata), encoding="utf-8")
    frame = root / next(iter(metadata["frames"].values()))["file"]
    frame.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="missing or changed"):
        ap2_inbox.archive_data_version(root)
