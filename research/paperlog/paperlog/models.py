from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title or "").lower()
    t = re.sub(r"[\W_]+", " ", t)  # \w keeps CJK characters
    return re.sub(r"\s+", " ", t).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def normalize_arxiv(aid: str | None) -> str | None:
    if not aid:
        return None
    a = aid.strip().lower()
    a = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", "", a)
    a = re.sub(r"v\d+$", "", a).removesuffix(".pdf")
    return a or None


@dataclass
class Paper:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""  # YYYY-MM-DD
    url: str = ""
    sources: list[str] = field(default_factory=list)  # arxiv / openalex / s2
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    institutions: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    language: str = ""
    venue: str = ""
    found_via: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.title = re.sub(r"\s+", " ", (self.title or "")).strip()
        self.abstract = re.sub(r"\s+", " ", (self.abstract or "")).strip()
        self.doi = normalize_doi(self.doi)
        self.arxiv_id = normalize_arxiv(self.arxiv_id)
        if self.doi and self.doi.startswith("10.48550/arxiv."):
            self.arxiv_id = self.arxiv_id or normalize_arxiv(self.doi.split("arxiv.", 1)[1])
        self.countries = sorted({c.upper() for c in self.countries if c})

    @property
    def title_norm(self) -> str:
        return normalize_title(self.title)

    @property
    def key(self) -> str:
        # arXiv id first so preprint and arXiv-DOI records collapse to one key
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        return "title:" + hashlib.sha1(self.title_norm.encode()).hexdigest()[:16]

    def merge(self, other: Paper) -> None:
        """Fill gaps in self from another record describing the same work."""
        for f in (
            "abstract",
            "published",
            "url",
            "doi",
            "arxiv_id",
            "openalex_id",
            "s2_id",
            "language",
            "venue",
        ):
            if not getattr(self, f) and getattr(other, f):
                setattr(self, f, getattr(other, f))
        if len(other.abstract) > len(self.abstract):
            self.abstract = other.abstract
        if not self.authors:
            self.authors = other.authors
        for f in ("sources", "institutions", "countries", "found_via"):
            merged = list(dict.fromkeys(getattr(self, f) + getattr(other, f)))
            setattr(self, f, merged)
