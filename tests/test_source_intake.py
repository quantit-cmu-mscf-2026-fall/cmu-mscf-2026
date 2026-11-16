"""Source intake contracts: immutable requests, exact evidence spans, bounded manifests."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from capstone.factors.source_intake import (
    MAX_MANIFEST_BYTES,
    EvidenceSpan,
    SourceArtifact,
    SourceRequest,
    load_source_requests,
    sha256,
)
from capstone.factors.sources import FileSource


@pytest.fixture
def source_request():
    return SourceRequest(
        "url", "https://example.org/paper", key="paper-a", title="Supplied title", role="seed"
    )


def test_contracts_are_immutable_and_bind_exact_span(source_request):
    with pytest.raises(FrozenInstanceError):
        source_request.title = "changed"
    span = EvidenceSpan.from_text("α\n beta", "page:2", start=7)
    assert (span.start, span.end) == (7, 14)
    assert span.text_sha256 == sha256("α\n beta".encode())
    with pytest.raises(ValueError, match="offsets"):
        replace(span, end=15)
    with pytest.raises(ValueError, match="hash"):
        replace(span, text_sha256="0" * 64)


@pytest.mark.parametrize(
    "url",
    [
        "file:///private.txt",
        "https://user:password@example.org/paper",
        "https://example.org/paper?access_token=secret",
        "https://example.org/paper?API_KEY=secret",
        "https://example.org/two words",
    ],
)
def test_reject_unsafe_source_addresses_before_persistence(url):
    with pytest.raises(ValueError):
        SourceRequest("url", url)


def test_parse_heterogeneous_json_and_preserve_identity_when_reordered(tmp_path):
    path = tmp_path / "sources.json"
    rows = [
        {"kind": "text", "text": "Economic mechanism", "id": "memo", "role": "adaptation"},
        {"url": "https://example.org/a", "author": "A", "date": "2020-01"},
        {"path": "inputs/paper.pdf", "title": "Paper"},
        {"type": "file", "path": "inputs/notes.md"},
    ]
    path.write_text(json.dumps({"sources": rows}), encoding="utf-8")
    requests = load_source_requests(path)
    assert [r.kind for r in requests] == ["inline_text", "url", "local_pdf", "local_text"]
    assert requests[2].locator == str((tmp_path / "inputs/paper.pdf").resolve())
    assert requests[1].published_date == "2020-01"
    path.write_text(json.dumps({"sources": list(reversed(rows))}), encoding="utf-8")
    assert load_source_requests(path) == tuple(reversed(requests))


def test_toml_and_plain_url_lists_keep_duplicates_and_legacy_untouched(tmp_path):
    toml = tmp_path / "source.toml"
    toml.write_text('[[sources]]\nkind="inline_text"\ntext="mechanism"\nkey="memo"\n')
    assert load_source_requests(toml)[0].key == "memo"
    urls = tmp_path / "sources.txt"
    urls.write_text("# public sources\nhttps://example.org/a\n\nhttps://example.org/a\n")
    first, second = load_source_requests(urls)
    assert first.source_id == second.source_id
    legacy = tmp_path / "papers.toml"
    legacy.write_text('[[papers]]\nkey="legacy"\ntitle="Legacy paper"\nabstract="Passed on"\n')
    assert FileSource(legacy).papers() == [
        {"key": "legacy", "title": "Legacy paper", "abstract": "Passed on"}
    ]
    with pytest.raises(ValueError, match="manifest"):
        load_source_requests(legacy)


def test_manifest_size_and_shape_are_bounded(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        load_source_requests(path)
    path.write_text('{"sources":[{"url":42}]}')
    with pytest.raises(ValueError, match="locator"):
        load_source_requests(path)
    path.write_text('{"sources":[]}')
    with pytest.raises(ValueError, match="sources"):
        load_source_requests(path)
    path.write_text('{"sources":[{"url":"https://example.org", "roles":"seed"}]}')
    with pytest.raises(ValueError, match="unknown"):
        load_source_requests(path)
    path.write_text('{"sources":["https://example.org"], "soruces": []}')
    with pytest.raises(ValueError, match="unknown"):
        load_source_requests(path)


def test_inline_http_locator_cannot_persist_credentials():
    with pytest.raises(ValueError, match="credentials"):
        SourceRequest("inline_text", "https://user:secret@example.org", text="snippet")


def test_work_identity_is_separate_from_request_revision_and_lineage():
    first = SourceRequest("inline_text", text="first version", key="work")
    changed = replace(first, text="changed version")
    other_work = replace(first, key="other-work")
    assert first.source_id == changed.source_id == first.work_id
    assert first.request_id != changed.request_id
    assert first.source_id != other_work.source_id
    assert replace(first, title="Attribution").source_id == first.source_id


def artifact(**changes):
    span = EvidenceSpan.from_text("Exact line one.", "chars:0")
    fields = {
        "source_id": "a" * 64,
        "request_id": "b" * 64,
        "original_locator": "https://example.org/paper",
        "resolved_locator": "https://example.org/paper.pdf",
        "media_type": "text/plain",
        "raw_sha256": "c" * 64,
        "raw_size": 15,
        "spans": [span],
        "extraction_version": "v1",
    }
    return SourceArtifact(**{**fields, **changes})


def test_artifact_identity_covers_every_field_and_round_trips():
    from dataclasses import asdict

    first = artifact()
    assert isinstance(first.spans, tuple)
    assert SourceArtifact.from_dict(asdict(first)) == first
    assert replace(first, extraction_version="v2").revision_id != first.revision_id
    assert artifact().revision_id == first.revision_id


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source_id": "XYZ"}, "identity"),
        ({"raw_size": 0}, "version or size"),
        ({"extraction_state": "maybe"}, "extraction state"),
        ({"spans": []}, "extracted evidence"),
        ({"extraction_state": "blocked", "spans": []}, "reason"),
        ({"role": "other"}, "role"),
    ],
)
def test_artifact_refuses_unqualified_records(changes, message):
    with pytest.raises(ValueError, match=message):
        artifact(**changes)
