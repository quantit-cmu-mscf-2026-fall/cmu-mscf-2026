"""Collapse duplicates within a batch, then match the batch against the database."""

from rapidfuzz import fuzz, process

from .models import Paper


def _ids_conflict(a: Paper, b: Paper) -> bool:
    """Two records with different arXiv ids or DOIs are different papers, however
    similar their titles look."""
    for f in ("arxiv_id", "doi"):
        x, y = getattr(a, f), getattr(b, f)
        if x and y and x != y:
            return True
    return False


def dedupe_batch(papers: list[Paper], fuzzy_threshold: int = 95) -> list[Paper]:
    by_id: dict[str, Paper] = {}
    alias: dict[str, str] = {}  # any identifier -> canonical id

    def ids(p: Paper):
        out = [f"t:{p.title_norm}"]
        if p.arxiv_id:
            out.append(f"a:{p.arxiv_id}")
        if p.doi:
            out.append(f"d:{p.doi}")
        return out

    for p in papers:
        if not p.title_norm:
            continue
        hit = next((alias[i] for i in ids(p) if i in alias), None)
        if hit is None and by_id:
            # fuzzy title match catches punctuation/casing/subtitle variants
            pool = {k: v.title_norm for k, v in by_id.items() if not _ids_conflict(p, v)}
            m = process.extractOne(
                p.title_norm, pool, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold
            )
            if m:
                hit = m[2]
        if hit:
            by_id[hit].merge(p)
            for i in ids(by_id[hit]):
                alias[i] = hit
        else:
            cid = ids(p)[0]
            by_id[cid] = p
            for i in ids(p):
                alias[i] = cid
    return list(by_id.values())


def match_db(p: Paper, db_titles: list[tuple], fuzzy_threshold: int = 95) -> str | None:
    """db_titles: (key, title_norm[, arxiv_id, doi]) rows from the log."""
    if not db_titles:
        return None
    pool = {}
    for row in db_titles:
        key, title = row[0], row[1]
        aid = row[2] if len(row) > 2 else None
        doi = row[3] if len(row) > 3 else None
        if (p.arxiv_id and aid and p.arxiv_id != aid) or (p.doi and doi and p.doi != doi):
            continue
        pool[key] = title
    m = process.extractOne(p.title_norm, pool, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold)
    return m[2] if m else None
