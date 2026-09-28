"""arXiv API. Free, no key. Etiquette: one request every ~3 seconds.

Two passes per query: newest submissions, and newest *revisions* (so a paper
substantially revised inside the window resurfaces even if v1 is old).
"""

import time

import feedparser

from ..http import get
from ..models import Paper

API = "https://export.arxiv.org/api/query"


def _entries(query: str, sort_by: str, max_results: int, log):
    out, start, page = [], 0, 100
    while start < max_results:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(page, max_results - start),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        try:
            feed = feedparser.parse(get(API, params=params).text)
        except Exception as e:
            log(f"  arxiv query failed ({query[:50]}...): {e}")
            break
        if not feed.entries:
            break
        out += feed.entries
        if len(feed.entries) < params["max_results"]:
            break
        start += len(feed.entries)
        time.sleep(3.1)
    return out


def _to_paper(e, found_via: str) -> Paper:
    return Paper(
        title=e.get("title", ""),
        abstract=e.get("summary", ""),
        authors=[a.get("name", "") for a in e.get("authors", [])],
        published=(e.get("published") or "")[:10],
        url=e.get("id", ""),
        arxiv_id=e.get("id", ""),
        doi=e.get("arxiv_doi"),
        language="en",
        venue="arXiv",
        sources=["arxiv"],
        found_via=[found_via],
    )


def by_id(aid: str) -> Paper | None:
    """Metadata for one arXiv id, straight from arXiv (no key, no shared rate limit)."""
    feed = feedparser.parse(get(API, params={"id_list": aid, "max_results": 1}).text)
    entries = [e for e in feed.entries if e.get("title")]
    return _to_paper(entries[0], "manual") if entries else None


def fetch(
    cfg: dict, since: str, depth: int = 1, extra_queries: list[str] | None = None, log=print
) -> list[Paper]:
    papers: list[Paper] = []
    cap = cfg.get("max_results_per_query", 100) * depth
    for q in list(cfg.get("queries", [])) + list(extra_queries or []):
        seen_in_query = 0
        for sort_by, tag in (("submittedDate", "arxiv"), ("lastUpdatedDate", "arxiv-rev")):
            for e in _entries(q, sort_by, cap, log):
                published = (e.get("published") or "")[:10]
                updated = (e.get("updated") or "")[:10]
                if max(published, updated) < since:
                    continue
                papers.append(_to_paper(e, f"{tag}:{q}"))
                seen_in_query += 1
            time.sleep(3.1)
        log(f"  arxiv: {seen_in_query:>4} in window for {q[:66]}")
    return papers
