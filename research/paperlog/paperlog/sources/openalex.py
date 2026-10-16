"""OpenAlex API. Free API key required since Feb 2026 (openalex.org/settings/api).

Two passes per query: by publication date (new work) and by *record creation*
date (work OpenAlex indexed late, which the publication-date window would miss).
Paginates with a cursor so a query is never truncated by relevance ranking.
"""

import os

from ..http import get
from ..models import Paper, normalize_title

API = "https://api.openalex.org"
SELECT = (
    "id,doi,display_name,publication_date,abstract_inverted_index,authorships,"
    "language,primary_location,ids"
)
PAGE = 200


def _params(extra: dict) -> dict:
    p = dict(extra)
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        p["api_key"] = key
    return p


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def to_paper(w: dict, found_via: str) -> Paper:
    authors, insts, countries = [], [], []
    for a in w.get("authorships") or []:
        authors.append((a.get("author") or {}).get("display_name", ""))
        for inst in a.get("institutions") or []:
            insts.append(inst.get("display_name", ""))
            countries.append(inst.get("country_code") or "")
        countries += a.get("countries") or []
    loc = w.get("primary_location") or {}
    landing = loc.get("landing_page_url") or ""
    return Paper(
        title=w.get("display_name") or "",
        abstract=_abstract(w.get("abstract_inverted_index")),
        authors=authors,
        published=w.get("publication_date") or "",
        url=landing or w.get("doi") or w.get("id", ""),
        doi=w.get("doi"),
        arxiv_id=landing if "arxiv.org/abs/" in landing else None,
        openalex_id=w.get("id"),
        institutions=list(dict.fromkeys(i for i in insts if i)),
        countries=countries,
        language=w.get("language") or "",
        venue=((loc.get("source") or {}).get("display_name")) or "",
        sources=["openalex"],
        found_via=[found_via],
    )


def _paged(search: str | None, flt: str, max_pages: int, log) -> list[dict]:
    results, cursor, pages = [], "*", 0
    while cursor and pages < max_pages:
        params = {
            "filter": flt,
            "per_page": PAGE,
            "select": SELECT,
            "cursor": cursor,
            "sort": "publication_date:desc",
        }
        if search:
            params["search"] = search
        try:
            data = get(f"{API}/works", params=_params(params)).json()
        except Exception as e:
            log(f"  openalex page failed: {e}")
            break
        results += data.get("results", [])
        cursor = (data.get("meta") or {}).get("next_cursor")
        pages += 1
        if len(data.get("results", [])) < PAGE:
            break
    return results


def resolve_institutions(names: list[str], log=print) -> list[str]:
    ids = []
    for name in names:
        try:
            res = (
                get(f"{API}/institutions", params=_params({"search": name, "per_page": 1}))
                .json()
                .get("results", [])
            )
        except Exception as e:
            log(f"  openalex: could not resolve institution {name}: {e}")
            continue
        if res:
            ids.append(res[0]["id"].rsplit("/", 1)[-1])
    return ids


def by_doi(doi: str) -> Paper | None:
    res = (
        get(
            f"{API}/works",
            params=_params({"filter": f"doi:{doi}", "per_page": 1, "select": SELECT}),
        )
        .json()
        .get("results", [])
    )
    return to_paper(res[0], "manual") if res else None


def by_id(work_id: str) -> Paper | None:
    """An OpenAlex work id such as 'W2741809807'."""
    w = get(f"{API}/works/{work_id}", params=_params({"select": SELECT})).json()
    return to_paper(w, "manual") if w.get("display_name") else None


def search_one(title: str) -> Paper | None:
    res = (
        get(f"{API}/works", params=_params({"search": title, "per_page": 1, "select": SELECT}))
        .json()
        .get("results", [])
    )
    return to_paper(res[0], "manual") if res else None


def _with_references(**params) -> list[dict]:
    params = {"select": "id,display_name,referenced_works", "per_page": 10, **params}
    return get(f"{API}/works", params=_params(params)).json().get("results", [])


def references_of(title: str, doi: str | None, cap: int, log=print) -> list[Paper]:
    """What a paper cites, according to OpenAlex.

    The fallback for when Semantic Scholar has no reference list, which happens
    when the publisher elides it. One paper can have several OpenAlex records (the
    journal version, a working paper, an editorial duplicate that shares the DOI),
    so every record with this DOI or exactly this title is a candidate, and the one
    citing the most works is used.

    Returns [] unless the whole list arrived: a partial list would be marked done
    and its missing references never retried.
    """
    candidates = []
    if doi:  # each lookup on its own, so a failed DOI lookup still leaves the title search
        try:
            candidates += _with_references(filter=f"doi:{doi.split('doi.org/')[-1]}")
        except Exception as e:
            log(f"  openalex: DOI lookup failed for '{title[:55]}': {e}")
    want = normalize_title(title)
    try:
        candidates += [
            w
            for w in _with_references(search=title)
            if normalize_title(w.get("display_name") or "") == want
        ]
    except Exception as e:
        log(f"  openalex: title search failed for '{title[:55]}': {e}")
    best = max(candidates, key=lambda w: len(w.get("referenced_works") or []), default={})
    ids = [u.rsplit("/", 1)[-1] for u in best.get("referenced_works") or []][:cap]
    out = []
    for chunk in _chunks(ids, 50):
        params = {"filter": "openalex_id:" + "|".join(chunk), "per_page": 50, "select": SELECT}
        try:
            works = get(f"{API}/works", params=_params(params)).json().get("results", [])
        except Exception as e:
            log(f"  openalex: reference page failed for '{title[:55]}', will retry next run: {e}")
            return []
        out += [to_paper(w, f"cited-by:{title[:60]}") for w in works if w.get("display_name")]
    log(f"  openalex: {len(out):>4} references of '{title[:55]}'")
    return out


def fetch(
    cfg: dict, since: str, depth: int = 1, extra_queries: list[str] | None = None, log=print
) -> list[Paper]:
    out: list[Paper] = []
    max_pages = cfg.get("max_pages_per_query", 2) * depth
    queries = list(cfg.get("queries", [])) + list(extra_queries or [])

    jobs = [(q, f"from_publication_date:{since}", f"openalex:{q}") for q in queries]
    # late-indexed records: published earlier, added to OpenAlex inside the window
    jobs += [(q, f"from_created_date:{since}", f"openalex-new:{q}") for q in queries]

    inst_names = cfg.get("institutions", [])
    if inst_names and cfg.get("institution_query"):
        for i, chunk in enumerate(_chunks(resolve_institutions(inst_names, log), 25)):
            jobs.append(
                (
                    cfg["institution_query"],
                    f"from_publication_date:{since},authorships.institutions.lineage:{'|'.join(chunk)}",
                    f"openalex-labs{i}",
                )
            )

    for search, flt, tag in jobs:
        works = _paged(search, flt, max_pages, log)
        out += [to_paper(w, tag) for w in works]
        log(f"  openalex: {len(works):>4} results for {tag[:66]}")
    return out


def _chunks(seq, n):
    return [seq[i : i + n] for i in range(0, len(seq), n)]
