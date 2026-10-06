"""Optional cross-language duplicate detection with a multilingual embedding model.

A Chinese journal version and an English arXiv preprint of the same work have
unrelated titles as strings, so ID and fuzzy-title matching can't link them.
Install with `pip install -e ".[embeddings]"` and set embeddings.enabled: true to use this.
"""

import numpy as np

_models: dict = {}


def _load(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer

        _models[name] = SentenceTransformer(name)
    return _models[name]


def encode(texts: list[str], model_name: str) -> np.ndarray:
    return _load(model_name).encode(texts, normalize_embeddings=True, batch_size=16)


def link_cross_language(con, new_keys: list[str], cfg: dict, log=print) -> int:
    rows = (
        con.execute(
            "SELECT key, title, abstract, language FROM papers "
            f"WHERE key IN ({','.join('?' * len(new_keys))})",
            new_keys,
        ).fetchall()
        if new_keys
        else []
    )
    if not rows:
        return 0
    vecs = encode([f"{r['title']}\n{(r['abstract'] or '')[:1500]}" for r in rows], cfg["model"])
    for r, v in zip(rows, vecs, strict=True):
        con.execute(
            "UPDATE papers SET embedding=? WHERE key=?", (v.astype(np.float32).tobytes(), r["key"])
        )

    old = con.execute(
        "SELECT key, language, embedding FROM papers "
        "WHERE embedding IS NOT NULL AND duplicate_of IS NULL"
    ).fetchall()
    keys = [o["key"] for o in old]
    langs = [o["language"] or "" for o in old]
    mat = np.stack([np.frombuffer(o["embedding"], dtype=np.float32) for o in old])
    linked = 0
    new_set = set(new_keys)
    for r, v in zip(rows, vecs, strict=True):
        sims = mat @ v.astype(np.float32)
        for idx in np.argsort(-sims)[:5]:
            k = keys[idx]
            if k == r["key"] or k in new_set and k > r["key"]:
                continue
            if sims[idx] < cfg.get("cross_language_dup_threshold", 0.9):
                break
            if (langs[idx] or "en") != (r["language"] or "en"):
                con.execute(
                    "UPDATE papers SET duplicate_of=?, status='duplicate' WHERE key=?",
                    (k, r["key"]),
                )
                linked += 1
                break
    log(f"  embeddings: linked {linked} cross-language duplicates")
    return linked
