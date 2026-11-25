"""A seeded queue of papers for a human to label, and the one-page instructions for it.

Three kinds of pick, in this order and with no paper picked twice:

- random: a seeded sample of the screened, new and filtered papers. Because every paper in the
  pool is equally likely, accuracy measured on these rows is unbiased;
- uncertain: the papers whose tuned relevance is nearest the "high" cut, then those where some
  tag's raw probability is nearest its tuned threshold;
- disagree: Sonnet-labelled papers where tuned Jev relevance and Sonnet's differ most.

No model call is made.
"""

from __future__ import annotations

import random
import re

import jev_paper_screen as jps
import numpy as np
import pandas as pd

POOL = ("screened", "new", "filtered")
SNIPPET = 600
COLUMNS = [
    "queue_id",
    "pick_reason",
    "key",
    "link",
    "title",
    "abstract_snippet",
    "jev_relevance",
    "jev_tags",
    "sonnet_relevance",
    "human_relevance",
    "human_tags",
    "human_note",
]


def link_for(key: str) -> str:
    """A web link for a paper key: doi.org for doi: keys, arxiv.org/abs for arxiv: keys."""
    scheme, _, rest = key.partition(":")
    if scheme == "doi" and rest:
        return f"https://doi.org/{rest}"
    if scheme == "arxiv" and rest:
        return f"https://arxiv.org/abs/{rest}"
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?|[a-z.-]+/\d{7}", key):
        return f"https://arxiv.org/abs/{key}"
    return ""


def build_queue(
    df: pd.DataFrame,
    tuning,
    seed: int = 0,
    n_random: int = 40,
    n_uncertain: int = 40,
    n_disagree: int = 20,
) -> pd.DataFrame:
    """`df`: tuned frame (Tuning.apply output) with abstract, status, sonnet_relevance."""
    pool = df[df.status.isin(POOL)].sort_values("key").reset_index(drop=True)
    picked: list[tuple[str, str]] = []
    taken: set[str] = set()

    def take(frame: pd.DataFrame, n: int, reason: str) -> None:
        added = 0
        for key in frame.key:
            if added >= n:
                break
            if key not in taken:
                taken.add(key)
                picked.append((key, reason))
                added += 1

    keys = list(pool.key)
    chosen = random.Random(f"{seed}:queue:random").sample(keys, min(n_random, len(keys)))
    take(pool.set_index("key").loc[chosen].reset_index(), n_random, "random")

    cut = tuning.relevance["high_cut"]
    by_rel = pool.assign(d=(pool.jev_relevance - cut).abs()).sort_values(["d", "key"])
    thr = np.array([tuning.thresholds[t] for t in tuning.tags])
    raw = pool[[f"raw_p_{t}" for t in tuning.tags]].to_numpy(float)
    by_tag = pool.assign(d=np.abs(raw - thr).min(axis=1)).sort_values(["d", "key"])
    half = n_uncertain // 2
    take(by_rel, half, "uncertain")
    take(by_tag, n_uncertain - half, "uncertain")

    sonnet = pool[pool.sonnet_relevance.notna()]
    gap = sonnet.assign(d=(sonnet.jev_relevance - sonnet.sonnet_relevance).abs())
    take(gap.sort_values(["d", "key"], ascending=[False, True]), n_disagree, "disagree")

    reasons = dict(picked)
    order = [k for k, _ in picked]
    random.Random(f"{seed}:queue:order").shuffle(order)
    rows = pool.set_index("key").loc[order].reset_index()
    queue = pd.DataFrame(
        {
            "queue_id": [f"Q{i:03d}" for i in range(1, len(rows) + 1)],
            "pick_reason": [reasons[k] for k in rows.key],
            "key": rows.key,
            "link": rows.key.map(link_for),
            "title": rows.title,
            "abstract_snippet": rows.abstract.map(lambda a: jps.snippet(a, SNIPPET)),
            "jev_relevance": rows.jev_relevance.round(2),
            "jev_tags": rows.jev_tags,
            "sonnet_relevance": rows.sonnet_relevance.map(lambda v: "" if pd.isna(v) else int(v)),
            "human_relevance": "",
            "human_tags": "",
            "human_note": "",
        }
    )
    return queue[COLUMNS]


def labeling_md(cfg: dict, definitions: dict[str, str]) -> str:
    levels = [ln.strip() for ln in cfg["rubric"].strip().splitlines() if ln.strip()]
    tag_lines = [f"- {t}: {d}" for t, d in definitions.items()]
    return "\n".join(
        [
            "# Labelling the Jev queue",
            "",
            "Open `label_queue.csv` in Excel. For each paper you want to label, read the title and "
            "abstract and fill the three blank columns. Label before you look at the jev_ and "
            "sonnet_ columns, which are only there to show what the models said.",
            "Partial labelling is fine: rows you leave blank are ignored. Save as CSV (UTF-8).",
            "",
            "- human_relevance: a whole number 0 to 5 (rubric below); required for a row to count.",
            "- human_tags: zero or more of the 18 tag names, separated by semicolons "
            "(e.g. `multiple_testing;sharpe_inference`). Blank means no tag applies.",
            "- human_note: anything you like; it is not used.",
            "",
            "Relevance rubric (the reader is the research team; a paper scores the higher of its "
            "METHODS value and its FINDINGS value):",
            *[f"- {ln}" for ln in levels],
            "",
            "Tags (draft definitions, each a yes/no question about the title and abstract):",
            *tag_lines,
            "",
            "Refit with your labels, from the capstone branch vincal848/jev-papers (no new Jev "
            "calls; it uses the saved answers):",
            "",
            "    python research/jev_refit.py --labels label_queue.csv "
            "--jev-cache jev_cache.jsonl.gz "
            "--tagged tagged_all.csv --prev-tuning tuning.json --out refit1",
            "",
        ]
    )
