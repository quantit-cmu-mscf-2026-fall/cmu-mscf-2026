"""Tag every paper in an export with Jev, apply a frozen tuning, and make the labelling queue.

    python research/jev_tag.py --papers papers.jsonl.gz --tuning tuning.json \
        --cache-from teacher1/cache.jsonl --cache-from trial1/cache.jsonl \
        --teacher teacher1/teacher_cache.jsonl --max-calls 4500 --out full1

Jev answers are cached in OUT/cache.jsonl (the `--cache-from` files are copied in first, so
papers already answered are not asked again). The tuning must match the current Jev question
design or the run stops before any call. Writes OUT/tagged_all.csv, OUT/label_queue.csv and
OUT/LABELING.md. The run is logged once to the ledger before the first call; no Claude call is
made, and the Jev key (TYPESAFE_API_KEY) is never written anywhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jev_paper_screen as jps
import jev_queue as jq
import jev_tuning as jt
import pandas as pd

from capstone.jev import http_post, jev_key

META = ["key", "title", "link", "published", "status", "haiku_relevance", "haiku_tags"]


def tagged_columns(tags: list[str]) -> list[str]:
    return [
        *META,
        "jev_relevance_raw",
        "jev_relevance",
        "jev_high",
        *[f"p_{t}" for t in tags],
        "jev_tags",
        "sonnet_relevance",
        "sonnet_tags",
    ]


def add_sonnet(df: pd.DataFrame, sonnet: dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    df["sonnet_relevance"] = df.key.map(lambda k: sonnet[k]["relevance"] if k in sonnet else None)
    df["sonnet_tags"] = df.key.map(lambda k: ";".join(sonnet[k]["tags"]) if k in sonnet else "")
    return df


def finish(df: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """The published column set, rounded for reading."""
    df = df.copy()
    df["link"] = df.key.map(jq.link_for)
    df["haiku_tags"] = df.haiku_tags.map(lambda t: ";".join(t) if isinstance(t, list) else t)
    for col in ["jev_relevance_raw", "jev_relevance", *[f"p_{t}" for t in tags]]:
        df[col] = df[col].round(3)
    return df[tagged_columns(tags)]


def tag_frame(
    papers: list[dict], rows: list[dict], tuning: jt.Tuning, sonnet: dict[str, dict]
) -> pd.DataFrame:
    """Tuned, Sonnet-annotated frame with abstracts (for the queue) for every cached paper."""
    base = jps.results_frame(papers, rows, tuning.tags)
    return add_sonnet(tuning.apply(base), sonnet)


def merge_caches(sources: list[Path], dest: Path) -> int:
    """Append each source's lines to dest (created if absent); returns lines copied."""
    n = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "a", encoding="utf-8") as out:
        for src in sources:
            for line in jt._lines(src):
                out.write(line + "\n")
                n += 1
    return n


def main(argv: list[str] | None = None, post=http_post, sleep=time.sleep) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--papers", type=Path, required=True)
    ap.add_argument("--tuning", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-from", type=Path, action="append", default=[])
    ap.add_argument("--teacher", type=Path, help="teacher_cache.jsonl, for the sonnet columns")
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = jps.load_config()
    tags = list(cfg["taxonomy"])
    defs = jps.load_definitions(tags)
    questions = jps.build_questions(cfg, defs)
    tuning = jt.load_tuning(a.tuning, jps.design_hash(questions))  # refuses a stale design
    papers = sorted(jps.load_papers_jsonl(a.papers), key=lambda p: p["key"])
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cache_from and not (a.out / "cache.jsonl").exists():
        merge_caches(a.cache_from, a.out / "cache.jsonl")
    params = {"mode": "all-tuned", "source": a.papers.name, "tuning_created": tuning.created}
    rows = jps.run(papers, questions, tags, a.out, jev_key(), a.max_calls, post, sleep, params)
    sonnet = jt.load_teacher_labels(a.teacher) if a.teacher else {}
    df = tag_frame(papers, rows, tuning, sonnet)
    finish(df, tags).to_csv(a.out / "tagged_all.csv", index=False, encoding="utf-8-sig")
    queue = jq.build_queue(df, tuning, a.seed)
    queue.to_csv(a.out / "label_queue.csv", index=False, encoding="utf-8-sig")
    (a.out / "LABELING.md").write_text(jq.labeling_md(cfg, defs), encoding="utf-8")
    print(f"tagged {len(rows)} of {len(papers)} papers; queue {len(queue)} rows; wrote {a.out}")
    return a.out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
