"""Turn a thing a human (or an agent) pastes into a Paper: arXiv id or URL,
DOI, Semantic Scholar id or URL, OpenAlex id, or a plain title."""

import re

from .models import Paper, normalize_arxiv, normalize_doi
from .sources import arxiv, openalex
from .sources import semantic_scholar as s2

ARXIV = re.compile(r"arxiv\.org/(abs|pdf)/([^\s?#]+)|^(\d{4}\.\d{4,5})(v\d+)?$", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)


def resolve(identifier: str) -> tuple[Paper | None, str]:
    """Returns (paper, how_it_was_resolved)."""
    s = identifier.strip()

    m = ARXIV.search(s)
    if m:
        aid = normalize_arxiv(m.group(2) or m.group(3))
        p = s2.by_id(f"arXiv:{aid}")
        if p:
            p.arxiv_id = p.arxiv_id or aid
            return p, "arxiv id"
        # Semantic Scholar's anonymous pool is often rate-limited; arXiv itself isn't.
        try:
            p = arxiv.by_id(aid)
        except Exception:
            p = None
        if p:
            return p, "arxiv id (via arXiv)"
        return Paper(
            title=f"arXiv:{aid}",
            arxiv_id=aid,
            sources=["manual"],
            url=f"https://arxiv.org/abs/{aid}",
            found_via=["manual"],
        ), "arxiv id (metadata unavailable)"

    m = DOI.search(s)
    if m:
        doi = normalize_doi(m.group(1))
        try:
            p = openalex.by_doi(doi)
        except Exception:
            p = None
        if p:
            return p, "doi via openalex"
        p = s2.by_id(f"DOI:{doi}")
        if p:
            return p, "doi via semantic scholar"
        return None, f"could not resolve DOI {doi}"

    if "semanticscholar.org/paper/" in s:
        pid = s.rstrip("/").rsplit("/", 1)[-1]
        p = s2.by_id(pid)
        if p:
            return p, "semantic scholar id"

    if "openalex.org/W" in s:
        wid = s.rstrip("/").rsplit("/", 1)[-1]
        try:
            p = openalex.by_id(wid)
        except Exception:
            p = None
        if p:
            return p, "openalex id"

    # treat as a title
    pid = None
    try:
        pid = s2.resolve(s)
    except Exception:
        pass
    if pid:
        p = s2.by_id(pid)
        if p:
            return p, "title match (semantic scholar)"
    try:
        p = openalex.search_one(s)
    except Exception:
        p = None
    if p:
        return p, "title match (openalex)"
    return None, "no match found"
