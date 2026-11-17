"""Bounded, deterministic source text extraction; source contents are always data."""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser

EXTRACTION_VERSION = "source-text-v2"  # v2: partially blank PDFs block as pdf_partial_requires_ocr
MAX_TEXT_CHARS = 1_000_000
MAX_SEGMENTS = 10_000
MAX_HTML_DEPTH = 128
MAX_HTML_ELEMENTS = 50_000
MAX_PDF_PAGES = 200
MAX_PDF_STREAM = 8_000_000


class ExtractionError(ValueError):
    """A retained source needs human action before it supplies evidence."""


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _bounded(segments):
    result = tuple(segments)
    if len(result) > MAX_SEGMENTS or sum(len(text) for text, _ in result) > MAX_TEXT_CHARS:
        raise ExtractionError("extraction_limit")
    if not result:
        raise ExtractionError("no_extractable_text")
    return result


class _ArticleParser(HTMLParser):
    """Prefer article, then main, then visible body; never execute page instructions."""

    ignored = {"head", "script", "style", "nav", "aside", "footer", "form", "svg", "noscript"}
    blocks = {"p", "div", "section", "article", "main", "li", "blockquote", "pre", "tr"}
    blocks |= {f"h{level}" for level in range(1, 7)}
    voids = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
    voids |= {"param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.counts = [{}]
        self.buffer = []
        self.buffer_location = ""
        self.buffer_scope = 0
        self.segments = []
        self.characters = 0
        self.elements = 0

    def _flush(self):
        text = _normalized("".join(self.buffer))
        if text:
            self.characters += len(text)
            if self.characters > MAX_TEXT_CHARS or len(self.segments) >= MAX_SEGMENTS:
                raise ExtractionError("extraction_limit")
            self.segments.append((text, self.buffer_location, self.buffer_scope))
        self.buffer = []

    def handle_starttag(self, tag, attrs):
        self.elements += 1
        if self.elements > MAX_HTML_ELEMENTS or (
            tag not in self.voids and len(self.stack) >= MAX_HTML_DEPTH
        ):
            raise ExtractionError("html_structure_limit")
        if tag in self.blocks or tag == "hr":
            self._flush()
        if tag in ("br", "td", "th"):
            self.buffer.append("\n" if tag == "br" else " ")
        count = self.counts[-1].get(tag, 0) + 1
        self.counts[-1][tag] = count
        if tag not in self.voids:
            path = (self.stack[-1][1] if self.stack else "") + f"/{tag}[{count}]"
            self.stack.append((tag, path))
            self.counts.append({})

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.voids:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self.blocks:
            self._flush()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                del self.counts[index + 1 :]
                break

    def handle_data(self, data):
        tags = [tag for tag, _ in self.stack]
        if any(tag in self.ignored for tag in tags) or not data:
            return
        if not self.buffer:
            blocks = [path for tag, path in self.stack if tag in self.blocks]
            self.buffer_location = "html:" + (blocks[-1] if blocks else "/text")
            self.buffer_scope = 2 if "article" in tags else 1 if "main" in tags else 0
        self.buffer.append(data)

    def extracted(self):
        self.close()
        self._flush()
        scope = max((scope for _, _, scope in self.segments), default=0)
        return _bounded((text, path) for text, path, rank in self.segments if rank == scope)


def _paragraphs(text: str, prefix: str):
    for number, paragraph in enumerate(re.split(r"\n\s*\n", text), 1):
        normalized = _normalized(paragraph)
        if normalized:
            yield normalized, f"{prefix};paragraph={number}"


def _pdf(raw: bytes):
    try:
        from pypdf import PdfReader, apply_configuration
        from pypdf.errors import LimitReachedError
    except ImportError as exc:
        raise ExtractionError("needs_pdf_dependency") from exc
    try:
        with apply_configuration(
            maximum_declared_stream_length=MAX_PDF_STREAM,
            array_based_stream_maximum_output_length=MAX_PDF_STREAM,
            zlib_maximum_output_length=MAX_PDF_STREAM,
            lzw_maximum_output_length=MAX_PDF_STREAM,
            run_length_maximum_output_length=MAX_PDF_STREAM,
            page_tree_maximum_entries=MAX_PDF_PAGES * 2,
            xform_maximum_invocations_per_extraction=100,
            jbig2dec_binary=None,
        ):
            reader = PdfReader(io.BytesIO(raw), strict=True, root_object_recovery_limit=100)
            if reader.is_encrypted:
                raise ExtractionError("pdf_encrypted")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise ExtractionError("pdf_page_limit")
            segments = []
            decoded_size = textless_pages = 0
            for number, page in enumerate(reader.pages, 1):
                content = page.get_contents()
                decoded_size += len(content.get_data()) if content is not None else 0
                if decoded_size > MAX_PDF_STREAM:
                    raise ExtractionError("pdf_content_limit")
                text = page.extract_text() or ""
                found = tuple(_paragraphs(text, f"pdf:page={number}"))
                textless_pages += not found
                segments.extend(found)
                if (
                    len(segments) > MAX_SEGMENTS
                    or sum(len(s[0]) for s in segments) > MAX_TEXT_CHARS
                ):
                    raise ExtractionError("extraction_limit")
            if not segments:
                raise ExtractionError("pdf_requires_ocr")
            if textless_pages:
                # Some pages are image-only: a page subset is not the source's evidence.
                raise ExtractionError("pdf_partial_requires_ocr")
            return tuple(segments)
    except ExtractionError:
        raise
    except LimitReachedError as exc:
        raise ExtractionError("pdf_content_limit") from exc
    except Exception as exc:
        raise ExtractionError("pdf_unreadable") from exc


def extract_segments(raw: bytes, media_type: str, encoding: str = "utf-8-sig"):
    """Return exact normalized extracted segments, never a method interpretation."""
    if media_type == "application/pdf":
        return _pdf(raw)
    try:
        text = raw.decode(encoding)
    except (UnicodeError, LookupError) as exc:
        raise ExtractionError("unreadable_text") from exc
    if len(text) > MAX_TEXT_CHARS:
        raise ExtractionError("extraction_limit")
    if media_type == "text/html":
        parser = _ArticleParser()
        parser.feed(text)
        return parser.extracted()
    return _bounded(_paragraphs(text, "text"))
