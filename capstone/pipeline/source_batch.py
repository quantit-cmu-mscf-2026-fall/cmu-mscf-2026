"""Archive a mixed source list; this stage does not generate or financially admit strategies."""

from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from capstone.artifacts import atomic_json, canonical, digest
from capstone.factors.source_acquisition import MAX_BYTES
from capstone.factors.source_archive import SourceArchive
from capstone.factors.source_intake import SourceRequest, load_source_requests, sha256
from capstone.locking import exclusive_file

ENGINE_FILES = (
    "factors/source_intake.py",
    "factors/source_archive.py",
    "factors/source_acquisition.py",
    "factors/source_extract.py",
    "locking.py",
    "pipeline/source_batch.py",
)


def engine_identity():
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("pypdf", "requests", "urllib3"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "code": {p: sha256((root / p).read_bytes()) for p in ENGINE_FILES},
        "python": platform.python_version(),
        "packages": packages,
    }


def _read_manifest(root, *, require_engine):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if digest(manifest) != receipt["manifest_id"]:
        raise ValueError("source batch manifest changed")
    if manifest["schema_version"] != 1 or manifest["stage"] != "source_intake":
        raise ValueError("unsupported source batch")
    if require_engine and manifest["engine"] != engine_identity():
        raise ValueError(
            "source batch extraction engine changed; preserve it and create a new batch"
        )
    return manifest


def _prepare(root, source_list, max_bytes):
    requests = load_source_requests(source_list)
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_BYTES}")
    manifest = {
        "schema_version": 1,
        "stage": "source_intake",
        "requests": sorted((asdict(r) for r in requests), key=canonical),
        "max_bytes": max_bytes,
        "engine": engine_identity(),
        "model_requests": 0,
        "financial_evaluations": 0,
        "qualification": "source extraction only; no method, data or financial admission",
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("source batch inputs changed; create a new batch")
        if not (root / "receipt.json").exists():
            # Preparation cannot start acquisition before both immutable records exist.
            if (root / "archive").exists():
                raise ValueError("incomplete source receipt with existing archive")
            atomic_json(root / "receipt.json", {"manifest_id": digest(manifest)})
        return _read_manifest(root, require_engine=True)
    unexpected = [p for p in root.iterdir() if p.name != "batch.lock"]
    if unexpected:
        raise ValueError("output directory is not an empty source batch")
    atomic_json(manifest_path, manifest)
    atomic_json(root / "receipt.json", {"manifest_id": digest(manifest)})
    return manifest


def _verified_history(root, max_bytes):
    completed = root / "summary.json"
    if completed.exists() and not (root / "archive/outcomes.jsonl").is_file():
        raise ValueError("completed source outcome history is missing")
    archive = None
    if (root / "archive").exists():
        archive = SourceArchive(root / "archive", max_bytes=max_bytes, create=False)
    history = () if archive is None else archive.history()
    if completed.exists():
        checkpoint = json.loads(completed.read_text(encoding="utf-8"))["history_checkpoint"]
        sequence = checkpoint["sequence"]
        if (
            type(sequence) is not int
            or sequence < 1
            or sequence > len(history)
            or history[sequence - 1]["event_id"] != checkpoint["event_id"]
        ):
            raise ValueError("completed source history checkpoint is missing or changed")
    revisions = {}
    for event in history:
        revision = event["revision_id"]
        if revision is not None:
            if revision not in revisions:
                revisions[revision] = archive.load(revision)
            artifact = revisions[revision]
            if (
                artifact.request_id != event["request_id"]
                or artifact.source_id != event["source_id"]
            ):
                raise ValueError("source outcome identity mismatch")
    return archive, history


def _summary(root, manifest):
    archive, history = _verified_history(root, manifest["max_bytes"])
    latest = {event["request_id"]: event for event in history}
    rows = []
    unique = {digest(row): row for row in manifest["requests"]}
    for request_id, row in sorted(unique.items()):
        request = SourceRequest(**row)
        event = latest.get(request_id)
        item = {
            "request": row,
            "source_id": request.source_id,
            "request_id": request_id,
            "state": "pending" if event is None else event["state"],
            "reason": "" if event is None else event["reason"],
            "revision_id": None if event is None else event["revision_id"],
        }
        if item["revision_id"] is not None:
            if archive is None:
                raise ValueError("source archive is missing")
            artifact = archive.load(item["revision_id"])
            if artifact.request_id != request_id or artifact.source_id != request.source_id:
                raise ValueError("source outcome identity mismatch")
            item["state"] = artifact.extraction_state
            item["span_count"] = len(artifact.spans)
            item["excerpt"] = artifact.spans[0].text[:1600] if artifact.spans else ""
        if item["state"] == "cached":
            item["state"] = "ready"
        rows.append(item)
    return {
        "manifest_id": digest(manifest),
        "input_rows": len(manifest["requests"]),
        "unique_requests": len(rows),
        "unique_sources": len({r["source_id"] for r in rows}),
        "states": dict(sorted(Counter(r["state"] for r in rows).items())),
        "sources": rows,
        "model_requests": 0,
        "financial_evaluations": 0,
        "method_verified": False,
        "financial_admission": False,
        "history_checkpoint": (
            {key: history[-1][key] for key in ("sequence", "event_id")} if history else None
        ),
    }


def run_batch(output, *, sources=None, max_bytes=None, refresh=False, acquire=None):
    """Prepare/continue one frozen batch; refresh is explicit and retains old revisions."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with exclusive_file(root / "batch.lock", message="another source batch owns this directory"):
        manifest = (
            _prepare(root, sources, MAX_BYTES if max_bytes is None else max_bytes)
            if sources is not None
            else _read_manifest(root, require_engine=True)
        )
        if sources is None and max_bytes is not None and max_bytes != manifest["max_bytes"]:
            raise ValueError("source batch byte limit changed")
        _verified_history(root, manifest["max_bytes"])
        archive = SourceArchive(root / "archive", max_bytes=manifest["max_bytes"])
        requests = {digest(row): row for row in manifest["requests"]}
        for _, row in sorted(requests.items()):
            archive.intake(SourceRequest(**row), acquire=acquire, refresh=refresh)
        result = _summary(root, manifest)
        atomic_json(root / "summary.json", result)
        return result
