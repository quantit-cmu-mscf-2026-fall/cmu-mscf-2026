"""Immutable requests and exact evidence; independent of legacy paper sources."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from capstone.artifacts import digest

SOURCE_KINDS = ("url", "local_pdf", "local_text", "inline_text")
EVIDENCE_ROLES = ("seed", "reference", "adaptation")
MAX_MANIFEST_BYTES = 1_000_000
MAX_SOURCES = 1_000


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_url(value: str) -> None:
    """Never persist embedded credentials as a source address."""
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("source URL must use http or https and have a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URL must not contain credentials")
    sensitive = {"access_token", "api_key", "token", "password", "secret", "authorization"}
    if any(key.lower() in sensitive for key, _ in parse_qsl(parsed.query)):
        raise ValueError("source URL must not contain authentication parameters")
    if any(char.isspace() for char in value):
        raise ValueError("source URL must not contain whitespace")


@dataclass(frozen=True)
class SourceRequest:
    """A work identity survives revisions; a request identity binds the supplied inputs.

    Use an explicit key to preserve work identity when an inline source or its address
    changes. Optional bibliographic fields are supplied attribution, never model claims.
    """

    kind: str
    locator: str = ""
    text: str | None = None
    key: str | None = None
    title: str | None = None
    author: str | None = None
    published_date: str | None = None
    role: str | None = None

    def __post_init__(self):
        if self.kind not in SOURCE_KINDS:
            raise ValueError("unsupported source kind")
        if not isinstance(self.locator, str):
            raise ValueError("source locator must be text")
        for name in ("key", "title", "author", "published_date", "role"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be nonempty text when supplied")
        if self.role is not None and self.role not in EVIDENCE_ROLES:
            raise ValueError("unsupported evidence role")
        if self.kind == "inline_text":
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("inline source requires nonempty text")
        elif self.text is not None or not self.locator.strip():
            raise ValueError("addressed source requires a locator and no inline text")
        if self.kind == "url" or urlsplit(self.locator).scheme.lower() in ("http", "https"):
            validate_url(self.locator)

    @property
    def source_id(self) -> str:
        if self.key is not None:
            return digest({"source_key": self.key})
        return digest({"kind": self.kind, "locator": self.locator or self.text})

    @property
    def work_id(self) -> str:
        return self.source_id

    @property
    def request_id(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class EvidenceSpan:
    """Exact extracted text and offsets within its located, normalized text segment."""

    text: str
    locator: str
    start: int
    end: int
    text_sha256: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text or not self.locator:
            raise ValueError("evidence requires exact text and a locator")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("evidence offsets must be integers")
        if self.start < 0 or self.end - self.start != len(self.text):
            raise ValueError("evidence offsets do not match its text")
        if self.text_sha256 != sha256(self.text.encode("utf-8")):
            raise ValueError("evidence text hash mismatch")

    @classmethod
    def from_text(cls, text: str, locator: str, start: int = 0) -> EvidenceSpan:
        return cls(text, locator, start, start + len(text), sha256(text.encode("utf-8")))

    @property
    def span_id(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class SourceArtifact:
    """One immutable acquired revision, not an assertion of research qualification."""

    source_id: str
    request_id: str
    original_locator: str
    resolved_locator: str
    media_type: str
    raw_sha256: str
    raw_size: int
    spans: tuple[EvidenceSpan, ...]
    extraction_version: str
    title: str | None = None
    author: str | None = None
    published_date: str | None = None
    role: str | None = None
    extraction_state: str = "ready"
    reason: str = ""
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "spans", tuple(self.spans))
        for value in (self.source_id, self.request_id, self.raw_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("invalid artifact content identity")
        if self.version != 1 or type(self.raw_size) is not int or self.raw_size < 1:
            raise ValueError("invalid source artifact version or size")
        if self.extraction_state not in ("ready", "blocked", "error"):
            raise ValueError("invalid extraction state")
        if not isinstance(self.extraction_version, str) or not self.extraction_version:
            raise ValueError("extraction version is required")
        if not self.media_type or (self.extraction_state == "ready" and not self.spans):
            raise ValueError("ready source requires media type and extracted evidence")
        if self.extraction_state != "ready" and not self.reason:
            raise ValueError("unavailable extraction requires a reason")
        if not all(isinstance(span, EvidenceSpan) for span in self.spans):
            raise ValueError("invalid evidence span")
        if self.role is not None and self.role not in EVIDENCE_ROLES:
            raise ValueError("unsupported evidence role")

    @property
    def revision_id(self) -> str:
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value: dict) -> SourceArtifact:
        fields = dict(value)
        fields["spans"] = tuple(EvidenceSpan(**span) for span in fields["spans"])
        return cls(**fields)


def _request(row: dict | str, base: Path) -> SourceRequest:
    if isinstance(row, str):
        return SourceRequest("url", row)
    if not isinstance(row, dict):
        raise ValueError("each source must be a table or URL string")
    allowed = {
        "kind",
        "type",
        "locator",
        "url",
        "path",
        "text",
        "key",
        "id",
        "title",
        "author",
        "published_date",
        "date",
        "role",
    }
    if row.keys() - allowed:
        raise ValueError("unknown source manifest field")
    aliases = {"text": "inline_text", "file": "local_text", "pdf": "local_pdf"}
    kind = row.get("kind", row.get("type"))
    locator = row.get("locator", row.get("url", row.get("path", "")))
    if kind is None:
        kind = "inline_text" if "text" in row else "url" if "url" in row else None
        if kind is None and "path" in row:
            kind = "local_pdf" if str(locator).lower().endswith(".pdf") else "local_text"
    kind = aliases.get(kind, kind)
    if kind in ("local_text", "local_pdf"):
        if not isinstance(locator, str) or not locator.strip():
            raise ValueError("local source requires a path")
        path = Path(locator).expanduser()
        locator = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    return SourceRequest(
        kind=kind,
        locator=locator,
        text=row.get("text"),
        key=row.get("key", row.get("id")),
        title=row.get("title"),
        author=row.get("author"),
        published_date=row.get("published_date", row.get("date")),
        role=row.get("role"),
    )


def load_source_requests(path: str | Path) -> tuple[SourceRequest, ...]:
    """Read bounded JSON/TOML source tables or one HTTP(S) URL per noncomment line.

    Local paths are relative to the manifest. Row order does not enter an identity.
    Existing ``FileSource`` paper manifests keep their separate legacy contract.
    """
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("source manifest exceeds size limit")
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    elif path.suffix.lower() == ".toml":
        loaded = tomllib.loads(text)
    else:
        loaded = [line.strip() for line in text.splitlines() if line.strip()]
        loaded = [line for line in loaded if not line.startswith("#")]
    if isinstance(loaded, dict) and loaded.keys() - {"sources"}:
        raise ValueError("unknown source manifest field")
    rows = loaded.get("sources") if isinstance(loaded, dict) else loaded
    if not isinstance(rows, list) or not rows or len(rows) > MAX_SOURCES:
        raise ValueError("manifest requires between 1 and 1000 sources")
    return tuple(_request(row, path.resolve().parent) for row in rows)
