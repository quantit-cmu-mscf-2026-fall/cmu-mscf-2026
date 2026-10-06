"""Semantic Scholar: citation expansion (who cites your anchors) and reference
expansion (what your adopted papers cite, which surfaces foundational work that
keyword search never finds). Free; S2_API_KEY gives a dedicated rate limit."""

import os
import time

from ..http import get
from ..models import Paper

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,authors,externalIds,publicationDate,year,venue,url"


def _headers():
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def resolve(title: str) -> str | None:
    data = (
        get(
            f"{API}/paper/search/match",
            params={"query": title, "fields": "paperId,title"},
            headers=_headers(),
        )
        .json()
        .get("data")
        or []
    )
    return data[0]["paperId"] if data else None


def by_id(pid: str) -> Paper | None:
    """pid may be a paperId, 'arXiv:2412.20138', 'DOI:10.1/abc', etc."""
    try:
        c = get(f"{API}/paper/{pid}", params={"fields": FIELDS}, headers=_headers()).json()
    except Exception:
        return None
    return to_paper(c, "manual") if c.get("title") else None


def to_paper(c: dict, found_via: str) -> Paper:
    ext = c.get("externalIds") or {}
    return Paper(
        title=c.get("title") or "",
        abstract=c.get("abstract") or "",
        authors=[a.get("name", "") for a in c.get("authors") or []],
        published=c.get("publicationDate") or (f"{c['year']}-01-01" if c.get("year") else ""),
        url=c.get("url") or "",
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        s2_id=c.get("paperId"),
        venue=c.get("venue") or "",
        sources=["s2"],
        found_via=[found_via],
    )


def _page(pid: str, edge: str, cap: int, log) -> list[dict]:
    items, offset = [], 0
    key = "citingPaper" if edge == "citations" else "citedPaper"
    while offset < cap:
        try:
            r = get(
                f"{API}/paper/{pid}/{edge}",
                headers=_headers(),
                params={"fields": FIELDS, "limit": 100, "offset": offset},
            ).json()
        except Exception as e:
            log(f"  s2: {edge} failed for {pid}: {e}")
            break
        items += [i.get(key) or {} for i in r.get("data") or []]
        if r.get("next") is None:
            break
        offset = r["next"]
        time.sleep(1.1)
    return items


def fetch_citations(anchors: list[tuple[str, str]], since: str, cap: int, log=print) -> list[Paper]:
    """anchors: (label, s2_paper_id) pairs."""
    out = []
    for label, pid in anchors:
        n = 0
        for c in _page(pid, "citations", cap, log):
            if not c.get("title"):
                continue
            p = to_paper(c, f"cites:{label[:60]}")
            if p.published and p.published[:4] < since[:4]:
                continue
            if p.published and len(p.published) == 10 and p.published < since:
                continue
            out.append(p)
            n += 1
        log(f"  s2: {n:>4} citing papers in window for '{label[:55]}'")
        time.sleep(1.1)
    return out


def fetch_references(anchors: list[tuple[str, str]], cap: int, log=print) -> list[Paper]:
    """References of adopted papers: no date filter, this is deliberate backfill."""
    out = []
    for label, pid in anchors:
        refs = [
            to_paper(c, f"cited-by:{label[:60]}")
            for c in _page(pid, "references", cap, log)
            if c.get("title")
        ]
        out += refs
        log(f"  s2: {len(refs):>4} references of '{label[:55]}'")
        time.sleep(1.1)
    return out
