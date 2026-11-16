"""Private content-addressed source revisions and a durable intake outcome journal."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from capstone.artifacts import canonical, digest
from capstone.factors.source_intake import EvidenceSpan, SourceArtifact, SourceRequest, sha256
from capstone.faults import maybe_fault
from capstone.locking import exclusive_file

if TYPE_CHECKING:
    from capstone.factors.source_acquisition import AcquisitionResult


class ArchiveIntegrityError(ValueError):
    """A stored identity no longer matches its bytes; never repair by overwriting."""


@dataclass(frozen=True)
class IntakeOutcome:
    state: str
    source_id: str
    request_id: str
    revision_id: str | None = None
    reason: str = ""


class SourceArchive:
    """Intake serializes writers; keep the output directory private and outside Git.

    Resume verifies cached artifacts without network access. ``refresh=True`` explicitly
    reacquires a source; changed bytes produce a new immutable revision.
    """

    def __init__(self, root: str | Path, *, max_bytes: int = 5_000_000):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.root = Path(root).absolute()
        self.max_bytes = max_bytes
        for path in (self.root, self.root / "objects", self.root / "revisions"):
            path.mkdir(parents=path == self.root, exist_ok=True, mode=0o700)
        self.journal = self.root / "outcomes.jsonl"

    @staticmethod
    def _identifier(value: str) -> str:
        if not isinstance(value, str) or len(value) != 64:
            raise ArchiveIntegrityError("invalid archive identity")
        if any(char not in "0123456789abcdef" for char in value):
            raise ArchiveIntegrityError("invalid archive identity")
        return value

    def _object_path(self, raw_sha256: str) -> Path:
        return self.root / "objects" / (self._identifier(raw_sha256) + ".bin")

    def _revision_path(self, revision_id: str) -> Path:
        return self.root / "revisions" / (self._identifier(revision_id) + ".json")

    @staticmethod
    def _read(path: Path, limit: int) -> bytes:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ArchiveIntegrityError("stored artifact exceeds size limit")
        return raw

    @classmethod
    def _write_immutable(cls, path: Path, raw: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Publish a complete file exclusively; a crash cannot expose partial content.
                publish = os.rename if os.name == "nt" else os.link
                publish(temporary, path)
            except FileExistsError:
                if cls._read(path, len(raw)) != raw:
                    raise ArchiveIntegrityError("existing immutable artifact differs") from None
        finally:
            Path(temporary).unlink(missing_ok=True)

    def history(self) -> tuple[dict, ...]:
        """Read and validate the append-only hash chain, including failed attempts."""
        if not self.journal.exists():
            return ()
        events = []
        previous = ""
        with self.journal.open("rb") as handle:
            for number, raw in enumerate(handle, 1):
                try:
                    if not raw.endswith(b"\n"):
                        raise ValueError("incomplete event")
                    event = json.loads(raw)
                    event_id = event.pop("event_id")
                    if event["sequence"] != number or event["previous"] != previous:
                        raise ValueError("broken event sequence")
                    if digest(event) != event_id:
                        raise ValueError("event digest differs")
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise ArchiveIntegrityError("outcome journal integrity failure") from exc
                event["event_id"] = event_id
                events.append(event)
                previous = event_id
        return tuple(events)

    def _record(self, outcome: IntakeOutcome) -> IntakeOutcome:
        history = self.history()
        event = {
            **asdict(outcome),
            "sequence": len(history) + 1,
            "previous": history[-1]["event_id"] if history else "",
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        event["event_id"] = digest(event)
        previous_bytes = self.journal.read_bytes() if self.journal.exists() else b""
        descriptor, temporary = tempfile.mkstemp(prefix=".outcomes-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(previous_bytes)
                handle.write((canonical(event) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            # Publish the exact old prefix plus one whole event; never expose a torn append.
            os.replace(temporary, self.journal)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return outcome

    def load(self, revision_id: str) -> SourceArtifact:
        """Verify manifest identity, every exact span and the original acquired bytes."""
        try:
            raw = self._read(self._revision_path(revision_id), self.max_bytes * 10)
            artifact = SourceArtifact.from_dict(json.loads(raw))
            if artifact.revision_id != revision_id:
                raise ArchiveIntegrityError("source revision digest mismatch")
            content = self._read(self._object_path(artifact.raw_sha256), self.max_bytes)
            if len(content) != artifact.raw_size or sha256(content) != artifact.raw_sha256:
                raise ArchiveIntegrityError("source content digest mismatch")
            return artifact
        except ArchiveIntegrityError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ArchiveIntegrityError("source revision is missing or invalid") from exc

    def _artifact(self, request: SourceRequest, result: AcquisitionResult) -> SourceArtifact:
        if not isinstance(result.raw, bytes) or not result.raw:
            raise ValueError("empty_source")
        if len(result.raw) > self.max_bytes:
            raise ValueError("size_limit")
        if sum(len(segment.text.encode("utf-8")) for segment in result.segments) > self.max_bytes:
            raise ValueError("extracted_text_size_limit")
        return SourceArtifact(
            source_id=request.source_id,
            request_id=request.request_id,
            original_locator=request.locator,
            resolved_locator=result.resolved_locator or request.locator,
            media_type=result.media_type,
            raw_sha256=sha256(result.raw),
            raw_size=len(result.raw),
            spans=tuple(EvidenceSpan.from_text(s.text, s.locator) for s in result.segments),
            extraction_version=result.extraction_version,
            title=request.title,
            author=request.author,
            published_date=request.published_date,
            role=request.role,
            extraction_state=result.state,
            reason=result.reason,
        )

    def intake(
        self,
        request: SourceRequest,
        acquire: Callable[..., AcquisitionResult] | None = None,
        *,
        refresh: bool = False,
    ) -> IntakeOutcome:
        """Capture or verify one request and journal its terminal outcome before return."""
        lock = self.root / "archive.lock"
        with exclusive_file(lock, message="another intake owns this archive"):
            return self._intake(request, acquire, refresh=refresh)

    def _intake(self, request, acquire, *, refresh):
        if not isinstance(request, SourceRequest):
            raise TypeError("intake requires SourceRequest")
        if type(refresh) is not bool:
            raise ValueError("refresh must be a boolean")
        # A damaged journal has no safe append position; preserve it for inspection.
        history = self.history()
        prior = next((e for e in reversed(history) if e["request_id"] == request.request_id), None)
        if not refresh and prior is not None:
            try:
                if prior["revision_id"] is not None:
                    artifact = self.load(prior["revision_id"])
                    if (
                        artifact.request_id != request.request_id
                        or artifact.source_id != request.source_id
                    ):
                        raise ArchiveIntegrityError("cached request identity mismatch")
                state = "cached" if prior["state"] in ("ready", "cached") else prior["state"]
                return self._record(
                    IntakeOutcome(
                        state,
                        request.source_id,
                        request.request_id,
                        prior["revision_id"],
                        prior["reason"],
                    )
                )
            except ArchiveIntegrityError:
                return self._record(
                    IntakeOutcome(
                        "error", request.source_id, request.request_id, reason="cache_integrity"
                    )
                )
        if acquire is None:
            from capstone.factors.source_acquisition import acquire_source

            acquire = acquire_source
        try:
            result = acquire(request, max_bytes=self.max_bytes)
            if result.state not in ("ready", "blocked", "error"):
                raise ValueError("invalid_acquisition_state")
            if result.state != "ready" and not result.raw:
                outcome = IntakeOutcome(
                    result.state, request.source_id, request.request_id, reason=result.reason
                )
            else:
                artifact = self._artifact(request, result)
                self._write_immutable(self._object_path(artifact.raw_sha256), result.raw)
                self._write_immutable(
                    self._revision_path(artifact.revision_id),
                    canonical(asdict(artifact)).encode("utf-8"),
                )
                self.load(artifact.revision_id)
                outcome = IntakeOutcome(
                    result.state,
                    request.source_id,
                    request.request_id,
                    artifact.revision_id,
                    result.reason,
                )
        except ArchiveIntegrityError:
            outcome = IntakeOutcome(
                "error", request.source_id, request.request_id, reason="cache_integrity"
            )
        except Exception:
            # Acquisition errors may contain private addresses or credentials; persist only a code.
            outcome = IntakeOutcome(
                "error", request.source_id, request.request_id, reason="acquisition_error"
            )
        maybe_fault("sources:after_archive")  # revision published, outcome not yet journaled
        return self._record(outcome)
