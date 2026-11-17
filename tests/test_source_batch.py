"""Mixed source files reach immutable evidence through a frozen, resumable batch."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from capstone.factors.source_acquisition import AcquisitionResult
from capstone.factors.source_archive import SourceArchive
from capstone.factors.source_intake import SourceRequest
from capstone.pipeline import source_batch
from capstone.pipeline.source_batch import run_batch, status


def batch_file(tmp_path):
    (tmp_path / "paper.md").write_text(
        "# A method\nUse completed observations.\n", encoding="utf-8"
    )
    source = tmp_path / "sources.json"
    source.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": "paper.md", "role": "reference"},
                    {
                        "kind": "inline_text",
                        "text": "Attention may precede reversal.",
                        "key": "attention",
                    },
                    {"path": "missing.pdf"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return source


def revisions_of(root):
    return {p.name: p.read_bytes() for p in (root / "archive/revisions").glob("*.json")}


def test_mixed_batch_real_extraction_and_resume_without_original_files(tmp_path):
    source = batch_file(tmp_path)
    root = tmp_path / "batch"
    first = run_batch(root, sources=source)
    assert first["states"] == {"blocked": 1, "ready": 2}
    assert not first["method_verified"] and not first["financial_admission"]
    assert first["model_requests"] == first["financial_evaluations"] == 0
    manifest = (root / "manifest.json").read_bytes()
    revisions = revisions_of(root)
    # Original inputs can disappear: resume uses verified archived evidence, not a new read.
    (tmp_path / "paper.md").unlink()
    second = run_batch(root)
    assert second["states"] == first["states"]
    assert second["sources"] == first["sources"]
    assert (root / "manifest.json").read_bytes() == manifest
    assert revisions_of(root) == revisions
    assert json.loads((root / "summary.json").read_text()) == second
    events = SourceArchive(root / "archive", create=False).history()
    assert len(events) == 6
    assert sorted(e["state"] for e in events[3:]) == ["blocked", "cached", "cached"]


def test_batch_order_and_duplicates_do_not_reacquire(tmp_path):
    source = tmp_path / "sources.json"
    urls = ["https://arxiv.org/abs/1", "https://arxiv.org/abs/2"]
    source.write_text(json.dumps(urls + urls[:1]))
    calls = []

    def blocked(request, **kwargs):
        calls.append(request.request_id)
        return AcquisitionResult("blocked", reason="http_403")

    first = run_batch(tmp_path / "batch", sources=source, acquire=blocked)
    assert len(calls) == 2 and first["input_rows"] == 3
    source.write_text(json.dumps(list(reversed(urls + urls[:1]))))
    second = run_batch(tmp_path / "batch", sources=source, acquire=blocked)
    assert second["sources"] == first["sources"] and len(calls) == 2
    run_batch(tmp_path / "batch", acquire=blocked, refresh=True)
    assert len(calls) == 4


def test_batch_rejects_changed_inputs_and_engine_without_touching_evidence(tmp_path, monkeypatch):
    source = batch_file(tmp_path)
    root = tmp_path / "batch"
    run_batch(root, sources=source)
    saved = (root / "archive/outcomes.jsonl").read_bytes()
    source.write_text(json.dumps(["https://arxiv.org/abs/changed"]))
    with pytest.raises(ValueError, match="inputs changed"):
        run_batch(root, sources=source)
    monkeypatch.setattr("capstone.pipeline.source_batch.engine_identity", lambda: {})
    with pytest.raises(ValueError, match="engine changed"):
        run_batch(root)
    assert (root / "archive/outcomes.jsonl").read_bytes() == saved


def test_engine_identity_names_the_code_and_runtime_it_froze():
    identity = source_batch.engine_identity()
    assert set(identity["code"]) == set(source_batch.ENGINE_FILES)
    assert all(len(digest) == 64 for digest in identity["code"].values())
    assert set(identity["packages"]) == {"pypdf", "requests", "urllib3"}
    assert identity == source_batch.engine_identity()


def test_corrupt_source_is_never_silently_refetched_or_reported_ready(tmp_path):
    root = tmp_path / "batch"
    source = batch_file(tmp_path)
    run_batch(root, sources=source)
    raw = next((root / "archive/objects").glob("*.bin"))
    raw.write_bytes(b"changed")
    before = (root / "archive/outcomes.jsonl").read_bytes()
    with pytest.raises(ValueError):
        run_batch(root)
    assert (root / "archive/outcomes.jsonl").read_bytes() == before


def test_incomplete_prepare_requires_original_inputs_before_acquisition(tmp_path):
    source = batch_file(tmp_path)
    root = tmp_path / "batch"
    root.mkdir()
    source_batch._prepare(root, source, 5_000_000)
    (root / "receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        run_batch(root)
    assert not (root / "archive").exists()
    assert run_batch(root, sources=source)["states"] == {"blocked": 1, "ready": 2}


def test_prepare_refuses_a_directory_that_is_not_an_empty_batch(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    (root / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not an empty source batch"):
        run_batch(root, sources=batch_file(tmp_path))
    assert (root / "notes.txt").read_text() == "keep me"
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("limit", [0, -1, 6_000_000, True])
def test_invalid_byte_limits_refused_before_freeze(tmp_path, limit):
    root = tmp_path / "batch"
    with pytest.raises(ValueError, match="max_bytes"):
        run_batch(root, sources=batch_file(tmp_path), max_bytes=limit)
    assert not (root / "manifest.json").exists()


def test_resume_cannot_silently_change_frozen_byte_limit(tmp_path):
    root = tmp_path / "batch"
    run_batch(root, sources=batch_file(tmp_path))
    with pytest.raises(ValueError, match="byte limit changed"):
        run_batch(root, max_bytes=1)


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_completed_history_cannot_be_lost_before_resume_acquisition(tmp_path, damage):
    root = tmp_path / "batch"
    run_batch(root, sources=batch_file(tmp_path))
    run_batch(root, refresh=True)
    journal = root / "archive/outcomes.jsonl"
    if damage == "missing":
        journal.unlink()
    else:
        journal.write_bytes(b"".join(journal.read_bytes().splitlines(keepends=True)[:-1]))

    def forbidden(*args, **kwargs):
        pytest.fail("damaged completed history must block before acquisition")

    with pytest.raises(ValueError, match="history"):
        run_batch(root, acquire=forbidden, refresh=True)


def test_resume_verifies_superseded_revisions_and_allows_uncheckpointed_crash_tail(tmp_path):
    root = tmp_path / "batch"
    source = batch_file(tmp_path)
    run_batch(root, sources=source)
    old_objects = set((root / "archive/objects").glob("*.bin"))
    (tmp_path / "paper.md").write_text("Changed source methods.", encoding="utf-8")
    run_batch(root, refresh=True)
    SourceArchive(root / "archive").intake(SourceRequest("local_text", str(tmp_path / "paper.md")))
    # A crash after a completed event but before summary publication keeps the old checkpoint.
    assert run_batch(root)["states"] == {"blocked": 1, "ready": 2}
    next(p for p in old_objects if b"A method" in p.read_bytes()).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="digest"):
        run_batch(root)


def test_resume_cli_in_a_fresh_process_and_readable_report(tmp_path):
    source = batch_file(tmp_path)
    root = tmp_path / "batch"
    first = run_batch(root, sources=source)
    revisions = revisions_of(root)
    (tmp_path / "paper.md").unlink()
    result = subprocess.run(
        [sys.executable, "-m", "capstone.pipeline.source_batch", "resume", "--out", str(root)],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    printed = json.loads(result.stdout)
    assert printed["states"] == first["states"] and "sources" not in printed
    assert {row["state"] for row in printed["revisions"]} == {"ready", "blocked"}
    assert status(root)["states"] == first["states"]
    assert revisions_of(root) == revisions
    assert "source acquisition and extraction" in (root / "SOURCE_INBOX.html").read_text()


def test_report_escapes_source_text(tmp_path):
    source = tmp_path / "sources.json"
    source.write_text(
        json.dumps(
            [{"kind": "inline_text", "key": "x", "text": "<script>alert(1)</script> & more"}]
        )
    )
    run_batch(tmp_path / "batch", sources=source)
    page = (tmp_path / "batch" / "SOURCE_INBOX.html").read_text()
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in page


def test_status_verifies_without_the_original_engine_and_without_writing(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    first = run_batch(root, sources=batch_file(tmp_path))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "batch.lock"}
    monkeypatch.setattr("capstone.pipeline.source_batch.engine_identity", lambda: {})
    assert status(root) == first
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "batch.lock"}
    assert after == before


def test_status_never_creates_nonexistent_output_or_missing_archive_directories(tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        status(root)
    assert not root.exists()
    root = tmp_path / "batch"
    run_batch(root, sources=batch_file(tmp_path))
    objects = root / "archive/objects"
    for file in objects.iterdir():
        file.unlink()
    objects.rmdir()
    with pytest.raises(ValueError):
        status(root)
    assert not objects.exists()


def test_corrupt_source_is_an_error_in_status(tmp_path):
    root = tmp_path / "batch"
    run_batch(root, sources=batch_file(tmp_path))
    next((root / "archive/objects").glob("*.bin")).write_bytes(b"changed")
    with pytest.raises(ValueError):
        status(root)


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_status_refuses_lost_completed_history(tmp_path, damage):
    root = tmp_path / "batch"
    run_batch(root, sources=batch_file(tmp_path))
    run_batch(root, refresh=True)
    journal = root / "archive/outcomes.jsonl"
    if damage == "missing":
        journal.unlink()
    else:
        journal.write_bytes(b"".join(journal.read_bytes().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError, match="history"):
        status(root)


def test_status_verifies_superseded_revisions(tmp_path):
    root = tmp_path / "batch"
    run_batch(root, sources=batch_file(tmp_path))
    old_objects = set((root / "archive/objects").glob("*.bin"))
    (tmp_path / "paper.md").write_text("Changed source methods.", encoding="utf-8")
    run_batch(root, refresh=True)
    assert status(root)["states"] == {"blocked": 1, "ready": 2}
    next(p for p in old_objects if b"A method" in p.read_bytes()).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="digest"):
        status(root)


@pytest.mark.parametrize(
    "argv",
    [
        ["ingest", "--out", "x"],
        ["resume", "--out", "x", "--sources", "s.json"],
        ["status", "--out", "x", "--max-bytes", "5"],
        ["status", "--out", "x", "--refresh"],
    ],
)
def test_cli_refuses_inconsistent_arguments(argv):
    with pytest.raises(SystemExit) as exit_info:
        source_batch.main(argv)
    assert exit_info.value.code == 2
