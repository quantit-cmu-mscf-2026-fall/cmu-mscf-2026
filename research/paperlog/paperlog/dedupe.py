"""Collapse duplicates within a batch, then match the batch against the database."""

import re

from rapidfuzz import fuzz, process

from .models import Paper

_ARXIV_DOI = "10.48550/arxiv."
_ROMAN = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}


def _ids_conflict(a: Paper, b: Paper) -> bool:
    """Two records with different arXiv ids or DOIs are different papers, however
    similar their titles look. An arXiv DOI is the preprint's own identifier, so it
    never disagrees with the journal DOI of the same work."""
    if a.arxiv_id and b.arxiv_id and a.arxiv_id != b.arxiv_id:
        return True
    if a.doi and b.doi and a.doi != b.doi:
        return not (a.doi.startswith(_ARXIV_DOI) or b.doi.startswith(_ARXIV_DOI))
    return False


_MARKER_NUMBER = re.compile(
    r"\b(?:parts?|vol|volume|chapter|book|section|lecture)\s+(?P<roman>[ivx]+)\b|(?P<digits>\d+)"
)


def _numbering(title_norm: str) -> list[int]:
    """The numbers in a normalised title, in title order ("... part ii", "... 2"), so
    "part i"/"part 1", "3,000"/"3000", "s&p 500"/"s&p500" and "covid-19"/"covid19"
    agree but "volume 2, part i" and "volume 1, part ii" do not. A roman numeral counts
    only after a marker word (part, volume, chapter, ...); elsewhere "I" is a pronoun."""
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", title_norm)  # "3,000" normalises to "3 000"
    out = []
    for m in _MARKER_NUMBER.finditer(t):
        if m["digits"]:
            out.append(int(m["digits"]))
        elif m["roman"] in _ROMAN:
            out.append(_ROMAN[m["roman"]])
    return out


def _same_numbering(a: str, b: str) -> bool:
    """Titles that differ in a number or numeral ("part I" / "part II") are different works."""
    return _numbering(a) == _numbering(b)


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
        hit = next(
            (alias[i] for i in ids(p) if i in alias and not _ids_conflict(p, by_id[alias[i]])), None
        )
        if hit is None and by_id:
            # fuzzy title match catches punctuation/casing/subtitle variants
            pool = {
                k: v.title_norm
                for k, v in by_id.items()
                if not _ids_conflict(p, v) and _same_numbering(p.title_norm, v.title_norm)
            }
            m = process.extractOne(
                p.title_norm, pool, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold
            )
            if m:
                hit = m[2]
        if hit:
            by_id[hit].merge(p)
            for i in ids(by_id[hit]):
                alias.setdefault(i, hit)
        else:
            cid = ids(p)[0]
            if cid in by_id:  # same title, different paper: keep both records
                cid = f"{cid}#{len(by_id)}"
            by_id[cid] = p
            for i in ids(p):
                alias.setdefault(i, cid)
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
        if _ids_conflict(p, Paper(title=title, arxiv_id=aid, doi=doi)):
            continue
        if not _same_numbering(p.title_norm, title):
            continue
        pool[key] = title
    m = process.extractOne(p.title_norm, pool, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold)
    return m[2] if m else None
