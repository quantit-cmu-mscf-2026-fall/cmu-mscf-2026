"""Refit the Jev tuning with human labels, from saved Jev answers only (no model calls).

    python research/jev_refit.py --labels label_queue.csv --jev-cache jev_cache.jsonl.gz \
        --tagged tagged_all.csv --prev-tuning tuning.json --out refit1

Human labels (human_relevance 0-5, human_tags separated by semicolons) are read from the
label CSVs; rows with nothing filled in are ignored and malformed rows are reported, not
used. The training table is the Sonnet labels, where a human label for the same paper
REPLACES Sonnet's and one for a paper Sonnet never saw is added; human rows get sample weight
`--human-weight` (default 3, applied by repeating the row). The tuning is refitted, and every
paper is re-tagged from the Jev cache; if any paper needed is not in the cache the refit
stops rather than calling the API.

Scores against HUMAN labels, on the random-pick rows only (the unbiased sample) and only when
at least 10 of those are labelled: tag macro-F1 and relevance Spearman for raw Jev, Jev tuned
on Sonnet (the previous tuning), Jev tuned with human labels (out-of-fold, so no label grades
the fit that used it), Haiku and Sonnet. The saved tuning is fitted on all human labels.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import jev_paper_screen as jps
import jev_queue as jq
import jev_tag as jtag
import jev_teacher_trial as jtt
import jev_tuning as jt
import numpy as np
import pandas as pd

from capstone import runlog
from capstone.analysis.outputs import Stat, Statement, write_outputs

MIN_RANDOM = 10
FOLDS = 5
SYSTEMS = ("raw", "tuned_on_sonnet", "tuned_with_human", "haiku", "sonnet")


# ---------------------------------------------------------------------------
# Reading human labels


def read_human_labels(paths: list[Path], tags: list[str]) -> tuple[dict[str, dict], list[str], int]:
    """({key: {relevance, tags, pick_reason}}, problems, rows ignored as empty).

    A row counts when human_relevance is a whole number 0-5; tags may be empty. Blank rows
    (no relevance, no tags) are ignored; tags without a relevance, a bad number, an unknown
    tag or a missing key are problems and the row is not used. A later row for the same key
    replaces an earlier one.
    """
    allowed = set(tags)
    labels: dict[str, dict] = {}
    problems: list[str] = []
    ignored = 0
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for n, row in enumerate(csv.DictReader(handle), 2):
                where = f"{Path(path).name} line {n}"
                rel = (row.get("human_relevance") or "").strip()
                raw_tags = [t.strip() for t in (row.get("human_tags") or "").split(";")]
                given = [t for t in raw_tags if t]
                if not rel and not given:
                    ignored += 1
                    continue
                key = (row.get("key") or "").strip()
                if not key:
                    problems.append(f"{where}: no key")
                    continue
                try:
                    value = float(rel)
                    ok = value == int(value) and 0 <= value <= 5
                except ValueError:
                    ok = False
                if not rel:
                    problems.append(f"{where} ({key}): tags given but no human_relevance")
                    continue
                if not ok:
                    problems.append(f"{where} ({key}): human_relevance {rel!r} is not 0-5")
                    continue
                unknown = sorted(set(given) - allowed)
                if unknown:
                    problems.append(f"{where} ({key}): unknown tags {unknown}")
                    continue
                labels[key] = {
                    "relevance": int(value),
                    "tags": sorted(set(given)),
                    "pick_reason": (row.get("pick_reason") or "").strip(),
                }
    return labels, problems, ignored


def sonnet_from_tagged(tagged: pd.DataFrame) -> dict[str, dict]:
    sub = tagged[tagged.sonnet_relevance.notna()]
    return {
        r.key: {
            "relevance": int(r.sonnet_relevance),
            "tags": [
                t for t in str(r.sonnet_tags if pd.notna(r.sonnet_tags) else "").split(";") if t
            ],
        }
        for r in sub.itertuples()
    }


def combine(sonnet: dict[str, dict], human: dict[str, dict]) -> tuple[dict, dict]:
    """Sonnet labels with humans overriding; returns (labels by key, counts)."""
    overridden = sorted(set(sonnet) & set(human))
    added = sorted(set(human) - set(sonnet))
    kept = {k: v for k, v in sonnet.items() if k not in human}
    return kept, {"overridden": len(overridden), "added": len(added)}


# ---------------------------------------------------------------------------
# Scoring against human labels


def tags_of(cell) -> list[str]:
    return [t for t in str(cell if pd.notna(cell) else "").split(";") if t]


def score_system(rel, pred_tags, human_rel, human_tags, tags: list[str]) -> dict:
    """n, relevance Spearman and tag macro-F1 on the rows where the system has a value."""
    rel = np.asarray(rel, float)
    have = np.isfinite(rel)
    n = int(have.sum())
    if n < 3:
        return {"n": n, "spearman": None, "macro_f1": None}
    idx = np.flatnonzero(have)
    pred = jtt.tag_matrix(pd.DataFrame({"t": [pred_tags[i] for i in idx]}), "t", tags)
    truth = jtt.tag_matrix(pd.DataFrame({"t": [human_tags[i] for i in idx]}), "t", tags)
    sp = jps.spearman(rel[idx], np.asarray(human_rel, float)[idx])
    f1 = jtt.macro_f1(pred, truth)
    return {
        "n": n,
        "spearman": None if not np.isfinite(sp) else sp,
        "macro_f1": None if not np.isfinite(f1) else f1,
    }


def out_of_fold(
    table: pd.DataFrame, human_keys: list[str], tags: list[str], design: str, seed: int
) -> pd.DataFrame:
    """Tuned-with-human relevance and tags for each human paper from a fit that excluded it."""
    keys = sorted(human_keys)
    random.Random(f"{seed}:refit:folds").shuffle(keys)
    k = min(FOLDS, len(keys))
    recs = []
    for i in range(k):
        held = set(keys[i::k])
        tuning = jt.Tuning.fit(table[~table.key.isin(held)], tags, design)
        applied = tuning.apply(table[table.key.isin(held)])
        recs.append(applied[["key", "jev_relevance", "jev_tags"]])
    return pd.concat(recs).set_index("key")


def evaluate(
    table: pd.DataFrame,
    human: dict[str, dict],
    sonnet: dict[str, dict],
    tagged: pd.DataFrame | None,
    prev: jt.Tuning,
    tags: list[str],
    design: str,
    seed: int,
) -> tuple[dict, pd.DataFrame, int]:
    """Metrics per system on the random-pick human rows ({} when fewer than MIN_RANDOM)."""
    random_keys = sorted(k for k, v in human.items() if v["pick_reason"] == "random")
    if len(random_keys) < MIN_RANDOM:
        return {}, pd.DataFrame(), len(random_keys)
    feats = table.set_index("key")
    ev = feats.loc[random_keys].reset_index()
    prev_applied = prev.apply(ev).set_index("key")
    oof = out_of_fold(table, sorted(human), tags, design, seed)
    haiku = tagged.set_index("key") if tagged is not None else pd.DataFrame()
    rows = []
    for key in random_keys:
        h = haiku.loc[key] if key in haiku.index else None
        rows.append(
            {
                "key": key,
                "human_relevance": human[key]["relevance"],
                "human_tags": human[key]["tags"],
                "raw_rel": ev.set_index("key").loc[key, "jev_relevance"],
                "raw_tags": [t for t in tags if ev.set_index("key").loc[key, f"p_{t}"] >= 0.5],
                "prev_rel": prev_applied.loc[key, "jev_relevance"],
                "prev_tags": tags_of(prev_applied.loc[key, "jev_tags"]),
                "new_rel": oof.loc[key, "jev_relevance"],
                "new_tags": tags_of(oof.loc[key, "jev_tags"]),
                "haiku_rel": np.nan
                if h is None or pd.isna(h.haiku_relevance)
                else h.haiku_relevance,
                "haiku_tags": [] if h is None else tags_of(h.haiku_tags),
                "sonnet_rel": sonnet[key]["relevance"] if key in sonnet else np.nan,
                "sonnet_tags": sonnet[key]["tags"] if key in sonnet else [],
            }
        )
    frame = pd.DataFrame(rows)
    pairs = {
        "raw": ("raw_rel", "raw_tags"),
        "tuned_on_sonnet": ("prev_rel", "prev_tags"),
        "tuned_with_human": ("new_rel", "new_tags"),
        "haiku": ("haiku_rel", "haiku_tags"),
        "sonnet": ("sonnet_rel", "sonnet_tags"),
    }
    metrics = {
        name: score_system(
            frame[r], list(frame[t]), frame.human_relevance, list(frame.human_tags), tags
        )
        for name, (r, t) in pairs.items()
    }
    return metrics, frame, len(random_keys)


# ---------------------------------------------------------------------------
# Report


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def build_report(
    out: Path, args: dict, tuning: jt.Tuning, metrics: dict, n_random: int, summary: dict, frame
) -> Path:
    stats: dict[str, Stat] = {}
    statements: list[Statement] = []
    stats["run_params"] = Stat(args, "arguments of this refit", ())
    stats["labels"] = Stat(
        summary,
        "human label rows read (labelled / ignored as empty / problems), how many replaced a "
        "Sonnet label or were added where Sonnet had none, and training rows by source",
        ("label csvs",),
    )
    stats["tuning.fitted_on"] = Stat(
        tuning.fitted_on,
        "labelled papers per source behind the saved tuning.json",
        ("tuning.json",),
    )
    stats["tuning.thresholds"] = Stat(
        tuning.thresholds, "per-tag F1-best thresholds of the new tuning", ("tuning.json",)
    )
    stats["tuning.high_cut"] = Stat(
        tuning.relevance["high_cut"],
        "tuned-relevance cut for jev_high of the new tuning",
        ("tuning.json",),
    )
    stats["eval.n_random_labelled"] = Stat(
        n_random,
        "human-labelled rows whose pick_reason is random (the unbiased sample)",
        ("label csvs",),
    )
    h = tuning.fitted_on["human"]
    statements.append(
        Statement(
            f"The new tuning was fitted on {tuning.fitted_on['sonnet']} Sonnet and {h} human "
            f"labels (human weight {tuning.human_weight}); {summary['overridden']} human labels "
            f"replaced a Sonnet label and {summary['added']} were added.",
            "tuning.fitted_on",
            "Human rows count several times in the fit by repetition, so a few labels can move "
            "the thresholds; per-tag fits with few positives are noisy.",
        )
    )
    if summary["problems"]:
        stats["labels.problems"] = Stat(
            summary["problems"], "label rows that could not be used, and why", ("label csvs",)
        )
        statements.append(
            Statement(
                f"{len(summary['problems'])} label rows were not used because they were malformed.",
                "labels.problems",
                "Fix the rows and refit; the listed rows are not in the training table.",
            )
        )
    tables: dict[str, pd.DataFrame] = {}
    if not metrics:
        statements.append(
            Statement(
                f"Only {n_random} random-pick rows have human labels, fewer than {MIN_RANDOM}; "
                "no unbiased accuracy estimate is reported yet.",
                "eval.n_random_labelled",
                "Rows picked as uncertain or as disagreements are not a fair sample, so they are "
                "used to fit but never to score.",
            )
        )
    else:
        stats["metrics"] = Stat(
            metrics,
            "tag macro-F1 and relevance Spearman against human labels on the random-pick rows, "
            "per system (n = rows the system has a value for); tuned_with_human is out-of-fold",
            ("label csvs", "cache.jsonl", "tagged csv"),
        )
        tables["human_random_rows"] = frame.assign(
            human_tags=frame.human_tags.map(";".join),
            raw_tags=frame.raw_tags.map(";".join),
            prev_tags=frame.prev_tags.map(";".join),
            new_tags=frame.new_tags.map(";".join),
            haiku_tags=frame.haiku_tags.map(";".join),
            sonnet_tags=frame.sonnet_tags.map(";".join),
        )
        text = "; ".join(
            f"{name} F1 {fmt(m['macro_f1'])} / rho {fmt(m['spearman'])} (n={m['n']})"
            for name, m in metrics.items()
        )
        statements.append(
            Statement(
                f"Against {n_random} randomly picked human-labelled papers: {text}.",
                "metrics",
                f"Only {n_random} papers: differences of a few hundredths are noise. The Jev "
                "tuned-with-human score is out-of-fold; Haiku has no score for unscreened (new) "
                "papers and Sonnet only for its labelled sample, so n differs by system and the "
                "systems are not scored on identical papers. Human labels are one person's reading "
                "of draft tag definitions.",
            )
        )
    return write_outputs(
        out,
        title="Jev tuning refitted with human labels",
        tables=tables,
        stats=stats,
        statements=statements,
    )


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--jev-cache", type=Path, required=True, help="cache.jsonl or .jsonl.gz")
    ap.add_argument("--teacher", type=Path, help="teacher_cache.jsonl (else from --tagged)")
    ap.add_argument("--tagged", type=Path, help="tagged_all.csv: metadata, Haiku, Sonnet columns")
    ap.add_argument("--prev-tuning", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--human-weight", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.human_weight < 1:
        ap.error("--human-weight must be at least 1")

    tags, design = jt.current_design()
    prev = jt.load_tuning(a.prev_tuning, design)  # refuses a stale design
    cache = jt.load_jev_cache(a.jev_cache, design)
    human, problems, ignored = read_human_labels(a.labels, tags)
    tagged = pd.read_csv(a.tagged, dtype={"key": str}, encoding="utf-8-sig") if a.tagged else None
    if a.teacher:
        sonnet = jt.load_teacher_labels(a.teacher)
    elif tagged is not None:
        sonnet = sonnet_from_tagged(tagged)
    else:
        raise SystemExit("need --teacher or --tagged for the Sonnet labels")

    universe = sorted(tagged.key) if tagged is not None else sorted(cache)
    missing = sorted((set(universe) | set(human)) - set(cache))
    if missing:
        raise SystemExit(
            f"refusing to refit: {len(missing)} papers are not in the Jev cache "
            f"(first: {missing[:3]}); this script never calls the API"
        )
    if not human:
        raise SystemExit("no usable human labels found" + (f": {problems[:3]}" if problems else ""))

    features = jt.jev_features(cache, tags, sorted(set(universe) | set(human)))
    kept_sonnet, counts = combine(sonnet, human)
    table = pd.concat(
        [
            jt.training_table(features, kept_sonnet, "sonnet"),
            jt.training_table(features, human, "human", a.human_weight),
        ],
        ignore_index=True,
    )
    summary = {
        "rows_labelled": len(human),
        "rows_ignored_empty": ignored,
        "problems": problems,
        **counts,
        "training_rows": table.groupby("source").size().to_dict(),
    }
    runlog.log_run(
        name="jev-refit",
        tags=["methodology", "jev"],
        params={
            "human_labels": len(human),
            "human_weight": a.human_weight,
            "n_papers_retagged": len(universe),
            "design": design,
            "seed": a.seed,
            "label_files": [p.name for p in a.labels],
        },
        notes="Refit Jev tuning with human labels overriding Sonnet; saved Jev answers only, "
        "no model calls.",
    )

    tuning = jt.Tuning.fit(table, tags, design)
    a.out.mkdir(parents=True, exist_ok=True)
    tuning.save(a.out / "tuning.json")

    metrics, frame, n_random = evaluate(table, human, sonnet, tagged, prev, tags, design, a.seed)

    meta = (
        tagged[[c for c in jtag.META + ["sonnet_relevance", "sonnet_tags"] if c in tagged]]
        if tagged is not None
        else pd.DataFrame({"key": universe})
    )
    base = features[features.key.isin(universe)].merge(meta, on="key", how="left")
    for col in ("title", "published", "status", "haiku_tags", "sonnet_tags"):
        if col not in base:
            base[col] = ""
    for col in ("haiku_relevance", "sonnet_relevance"):
        if col not in base:
            base[col] = np.nan
    if tagged is None:
        base = jtag.add_sonnet(base, sonnet)
    base["haiku_tags"] = base.haiku_tags.fillna("")
    applied = tuning.apply(base)
    applied["link"] = applied.key.map(jq.link_for)
    for col in ["jev_relevance_raw", "jev_relevance", *[f"p_{t}" for t in tags]]:
        applied[col] = applied[col].round(3)
    applied[jtag.tagged_columns(tags)].to_csv(
        a.out / "tagged_all.csv", index=False, encoding="utf-8-sig"
    )

    args = {
        k: [str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
    }
    path = build_report(a.out, args, tuning, metrics, n_random, summary, frame)
    print(f"refitted on {tuning.fitted_on}; {len(problems)} bad rows; wrote {path}")
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
