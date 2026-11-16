"""Acquire one bounded source without models, retries, credentials or implicit file reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from capstone.factors.source_extract import EXTRACTION_VERSION, ExtractionError, extract_segments

if TYPE_CHECKING:
    from capstone.factors.source_intake import SourceRequest

MAX_BYTES = 5_000_000
MEDIA_TYPES = {"application/pdf", "text/html", "text/plain", "text/markdown"}


@dataclass(frozen=True)
class TextSegment:
    text: str
    locator: str


@dataclass(frozen=True)
class AcquisitionResult:
    state: str
    raw: bytes = b""
    media_type: str = ""
    segments: tuple[TextSegment, ...] = ()
    resolved_locator: str = ""
    reason: str = ""
    extraction_version: str = EXTRACTION_VERSION


def _extracted(raw, media_type, locator, encoding="utf-8-sig"):
    if raw.startswith(b"%PDF-"):
        media_type = "application/pdf"
    if media_type not in MEDIA_TYPES:
        return AcquisitionResult(
            "blocked", raw, media_type, resolved_locator=locator, reason="unsupported_media_type"
        )
    try:
        segments = tuple(
            TextSegment(text, path) for text, path in extract_segments(raw, media_type, encoding)
        )
    except ExtractionError as exc:
        return AcquisitionResult(
            "blocked", raw, media_type, resolved_locator=locator, reason=str(exc)
        )
    return AcquisitionResult("ready", raw, media_type, segments, locator)


def acquire_source(request: SourceRequest, *, transport=None, max_bytes=MAX_BYTES):
    """Acquire only the explicit input; a pasted fallback is a separate source request.

    Local and inline sources only for now: a URL request is refused as an unsupported kind
    until the HTTP adapter is added.
    """
    if type(max_bytes) is not int or max_bytes < 1 or max_bytes > MAX_BYTES:
        raise ValueError(f"max_bytes must be in 1..{MAX_BYTES}")
    if request.kind == "inline_text":
        raw = (request.text or "").encode("utf-8")
        locator = request.locator or "inline"
        if len(raw) > max_bytes:
            return AcquisitionResult(
                "blocked",
                raw[:max_bytes],
                "text/plain",
                resolved_locator=locator,
                reason="size_limit",
            )
        return _extracted(raw, "text/plain", locator)
    if request.kind not in {"local_pdf", "local_text"}:
        return AcquisitionResult("blocked", reason="unsupported_source_kind")
    if urlsplit(request.locator).scheme in {"http", "https", "file", "ftp"}:
        return AcquisitionResult("blocked", reason="explicit_local_path_required")
    path = Path(request.locator).expanduser()
    media_type = "application/pdf" if request.kind == "local_pdf" else "text/plain"
    if request.kind == "local_text" and path.suffix.lower() in {".md", ".markdown"}:
        media_type = "text/markdown"
    locator = ""
    try:
        locator = str(path.resolve())
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except FileNotFoundError:
        return AcquisitionResult(
            "blocked", media_type=media_type, resolved_locator=locator, reason="source_missing"
        )
    except PermissionError:
        return AcquisitionResult(
            "blocked",
            media_type=media_type,
            resolved_locator=locator,
            reason="source_access_denied",
        )
    except (OSError, ValueError):
        return AcquisitionResult("error", media_type=media_type, reason="local_read_error")
    if len(raw) > max_bytes:
        return AcquisitionResult(
            "blocked", raw[:max_bytes], media_type, resolved_locator=locator, reason="size_limit"
        )
    return _extracted(raw, media_type, locator)
