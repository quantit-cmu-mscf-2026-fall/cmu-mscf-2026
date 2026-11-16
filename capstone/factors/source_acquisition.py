"""Acquire one bounded source without models, retries, credentials or implicit file reads."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urljoin, urlsplit

from capstone.factors.source_extract import EXTRACTION_VERSION, ExtractionError, extract_segments

if TYPE_CHECKING:
    from capstone.factors.source_intake import SourceRequest

MAX_BYTES = 5_000_000
MAX_REDIRECTS = 3
TIMEOUT = (5, 15)
MEDIA_TYPES = {"application/pdf", "text/html", "text/plain", "text/markdown"}
# Only open-access hosts (and their subdomains) are ever fetched; a redirect off these is refused.
OPEN_ACCESS_DOMAINS = ("arxiv.org", "nber.org", "ssrn.com")
AUTH_PARAMETERS = {"access_token", "api_key", "token", "password", "secret", "authorization"}


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


def _url_allowed(url: str, *, resolve: bool):
    """Public, credential-free http(s) URL on an open-access host (arXiv, SSRN, NBER)."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or any(character.isspace() or ord(character) < 32 for character in url)
            or "\\" in url
            or port not in {None, 80, 443}
            or any(key.lower() in AUTH_PARAMETERS for key, _ in parse_qsl(parsed.query))
        ):
            return False
        host = host.rstrip(".").lower()
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return False
        if not any(host == d or host.endswith("." + d) for d in OPEN_ACCESS_DOMAINS):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            pass
        if all(character.isdigit() or character == "." for character in host):
            return False
        if "." not in host or host.endswith((".test", ".invalid")):
            return False
        if resolve:
            addresses = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80))
            return bool(addresses) and all(
                ipaddress.ip_address(item[4][0]).is_global for item in addresses
            )
        return True
    except (ValueError, OSError):
        return False


class _UnsafePeer(ValueError):
    pass


class _PublicSocket:
    def _new_conn(self):
        sock = super()._new_conn()
        try:
            address = ipaddress.ip_address(sock.getpeername()[0])
        except (AttributeError, IndexError, OSError, TypeError, ValueError) as exc:
            sock.close()
            raise _UnsafePeer("http_peer_unavailable") from exc
        if not address.is_global:
            sock.close()
            raise _UnsafePeer("http_private_peer")
        return sock


def _public_adapter():
    from requests.adapters import HTTPAdapter
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

    class PublicHTTPConnection(_PublicSocket, HTTPConnection):
        pass

    class PublicHTTPSConnection(_PublicSocket, HTTPSConnection):
        pass

    class PublicHTTPPool(HTTPConnectionPool):
        ConnectionCls = PublicHTTPConnection

    class PublicHTTPSPool(HTTPSConnectionPool):
        ConnectionCls = PublicHTTPSConnection

    class PublicAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            # urllib3 2.x: inspect the connected socket before HTTP or TLS sends any source data.
            self.poolmanager.pool_classes_by_scheme = {
                "http": PublicHTTPPool,
                "https": PublicHTTPSPool,
            }

    return PublicAdapter(max_retries=0)


def _default_transport(url, **kwargs):
    import requests

    # A fresh session never takes cookies, netrc credentials or proxy credentials from the machine.
    session = requests.Session()
    session.trust_env = False
    adapter = _public_adapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        response = session.get(url, **kwargs)
    except Exception:
        session.close()
        raise
    response._source_session = session
    return response


def _close(response):
    response.close()
    session = getattr(response, "_source_session", None)
    if session is not None:
        session.close()


def _target(url):
    import requests

    # Request.prepare canonicalizes host case, IDNs and escaped paths without session credentials.
    prepared = requests.Request("GET", url).prepare().url
    return urlsplit(prepared)._replace(fragment="").geturl()


def _http(request, transport, max_bytes):
    url = request.locator
    fetch = transport or _default_transport
    for hop in range(MAX_REDIRECTS + 1):
        if not _url_allowed(url, resolve=transport is None):
            return AcquisitionResult("blocked", resolved_locator="", reason="unsupported_url")
        raw = bytearray()
        media_type = ""
        try:
            response = fetch(
                url,
                timeout=TIMEOUT,
                stream=True,
                allow_redirects=False,
                headers={
                    "User-Agent": "capstone-source-intake/1",
                    "Accept": "text/html,application/pdf,text/plain,text/markdown",
                },
            )
            try:
                reported = getattr(response, "url", url) or url
                if _target(reported) != _target(url):
                    return AcquisitionResult("blocked", reason="unexpected_redirect")
                code = response.status_code
                if code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location", "")
                    if not location or hop == MAX_REDIRECTS:
                        return AcquisitionResult(
                            "blocked", resolved_locator=url, reason="redirect_limit"
                        )
                    url = urljoin(url, location)
                    continue
                content_type = Message()
                content_type["content-type"] = response.headers.get("Content-Type", "")
                media_type = content_type.get_content_type()
                encoding = content_type.get_content_charset() or "utf-8-sig"
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    remaining = max_bytes - len(raw)
                    raw.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        return AcquisitionResult(
                            "blocked",
                            bytes(raw),
                            media_type,
                            resolved_locator=url,
                            reason="size_limit",
                        )
                if code in {401, 407}:
                    reason = "http_auth_required"
                elif code in {403, 429, 451}:
                    reason = "http_denied"
                elif not 200 <= code < 300:
                    reason = "http_error"
                else:
                    return _extracted(bytes(raw), media_type, reported, encoding)
                return AcquisitionResult(
                    "blocked", bytes(raw), media_type, resolved_locator=url, reason=reason
                )
            finally:
                _close(response)
        except _UnsafePeer as exc:
            return AcquisitionResult("blocked", resolved_locator=url, reason=str(exc))
        except Exception:
            return AcquisitionResult(
                "error", bytes(raw), media_type, resolved_locator=url, reason="http_transport_error"
            )
    raise AssertionError("bounded redirect loop exhausted")


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
    """Acquire only the explicit input; a pasted fallback is a separate source request."""
    if type(max_bytes) is not int or max_bytes < 1 or max_bytes > MAX_BYTES:
        raise ValueError(f"max_bytes must be in 1..{MAX_BYTES}")
    if request.kind == "url":
        return _http(request, transport, max_bytes)
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
