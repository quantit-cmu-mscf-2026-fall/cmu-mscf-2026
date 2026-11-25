"""Compare Jev with the Claude Haiku paper screen on relevance and tagging.

Jev (TypeSafe's decision model) is documented to fit short-text classification, so
this tests it on paperlog: for a stratified sample of screened papers plus the
adopted ones, Jev reads title and abstract only and answers

- two Score questions (METHODS value and FINDINGS value) over the rubric levels
  from research/paperlog/config.yaml; relevance = the larger expected score;
- one Noul per taxonomy tag, defined in research/jev_paper_tags.toml (drafts).

The Haiku screen already stored in paperlog is the reference. It is another model,
not ground truth: agreement with it is not accuracy. The only human signal is the
adopted papers. The disagreement sheet is for a human to adjudicate.

    python research/jev_paper_screen.py --db papers.db --n-per-level 25 --seed 0 \
        --out experiments/analysis/jev-papers/trial1

Every Jev answer is appended to cache.jsonl in the output folder, so a rerun
resumes without paying twice. The key comes from TYPESAFE_API_KEY, never a file.
The run is logged once to the ledger before any call; no Claude call is made.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from capstone import runlog
from capstone.analysis.outputs import Stat, Statement, write_outputs
from capstone.jev import JEV_URL, http_post, jev_key, post_with_retry

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "research" / "paperlog" / "config.yaml"
TAG_FILE = REPO / "research" / "jev_paper_tags.toml"
MODEL = "jev-1.14.0"
PRICE_PER_M_INPUT = 0.042
THRESHOLDS = tuple(round(0.1 * i, 1) for i in range(1, 10))
HIGH = 4  # "high" relevance: rubric level 4 or 5
ABSTRACT_CAP = 6000
DISAGREEMENTS = 30


# ---------------------------------------------------------------------------
# Questions


def load_config(path: Path = CONFIG) -> dict:
    """`team_description`, `rubric` and `taxonomy` from the `screening:` block of config.yaml.

    The shared environment has no YAML library, so this reads just those three keys
    (a folded string, a literal block and a list) and fails loudly if they are absent.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.rstrip() == "screening:"), None)
    if start is None:
        raise ValueError(f"{path.name} has no screening: block")
    block = []
    for ln in lines[start + 1 :]:
        if ln.strip() and not ln.startswith(" ") and not ln.startswith("#"):
            break
        block.append(ln)
    out: dict = {}
    for key in ("team_description", "rubric", "taxonomy"):
        head = next((i for i, ln in enumerate(block) if re.match(rf"  {key}:", ln)), None)
        if head is None:
            raise ValueError(f"{path.name} screening block has no {key}")
        body = []
        for ln in block[head + 1 :]:
            if ln.strip() and len(ln) - len(ln.lstrip()) <= 2 and not ln.lstrip().startswith("- "):
                break
            if ln.strip() or body:
                body.append(ln)
        body = [ln for ln in body if not ln.lstrip().startswith("#")]
        if key == "taxonomy":
            out[key] = [ln.strip()[2:].strip() for ln in body if ln.strip().startswith("- ")]
        else:
            text = [ln.strip() for ln in body] if key == "team_description" else body
            if key == "team_description":
                out[key] = " ".join(t for t in text if t)
            else:
                indent = min(len(ln) - len(ln.lstrip()) for ln in text if ln.strip())
                out[key] = "\n".join(ln[indent:] for ln in text).strip()
    return out


def load_definitions(taxonomy: list[str], path: Path = TAG_FILE) -> dict[str, str]:
    import tomllib

    tags = tomllib.loads(path.read_text(encoding="utf-8"))["tags"]
    if set(tags) != set(taxonomy):
        raise ValueError(
            f"{path.name} tags differ from taxonomy: missing {sorted(set(taxonomy) - set(tags))}, "
            f"extra {sorted(set(tags) - set(taxonomy))}"
        )
    return {t: tags[t] for t in taxonomy}


def rubric_levels(rubric: str) -> list[tuple[str, str]]:
    """(methods text, findings text) for rubric levels 0..5 in order.

    A level written as `METHODS: ... FINDINGS: ...` is split. A level with no such
    markers is not separable without rewriting it, so both questions get its full
    text verbatim.
    """
    levels: dict[int, str] = {}
    for line in rubric.strip().splitlines():
        match = re.match(r"\s*(\d+)\s*=\s*(.*\S)", line)
        if match:
            levels[int(match.group(1))] = match.group(2)
    if sorted(levels) != list(range(len(levels))) or len(levels) < 2:
        raise ValueError(f"rubric levels must be 0..n in order, got {sorted(levels)}")
    out = []
    for n in range(len(levels)):
        text = levels[n]
        split = re.match(r"\s*METHODS:\s*(.*?)\s*FINDINGS:\s*(.*)", text, re.S)
        if split:
            out.append((split.group(1).strip(" .;"), split.group(2).strip(" .;")))
        else:
            out.append((text, text))
    return out


def build_questions(cfg: dict, definitions: dict[str, str]) -> dict:
    """The Jev questions: methods and findings Scores, then one Noul per tag."""
    team = " ".join(cfg["team_description"].split())
    levels = rubric_levels(cfg["rubric"])
    context = f"The reader is {team}\n\n"
    questions: dict = {
        "methods_relevance": {
            "type": "score",
            "instructions": context + "How useful are the METHODS of the paper (`title` and "
            "`abstract`) to this team? Ignore what the paper finds about stock returns.",
            "criteria": [m for m, _ in levels],
        },
        "findings_relevance": {
            "type": "score",
            "instructions": context + "How useful are the FINDINGS of the paper (`title` and "
            "`abstract`) to this team as a source of testable hypotheses about stock returns? "
            "Ignore the methods.",
            "criteria": [f for _, f in levels],
        },
    }
    for tag, definition in definitions.items():
        questions[f"tag_{tag}"] = {"type": "noul", "instructions": definition}
    return questions


def design_hash(questions: dict) -> str:
    return hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:12]


def build_state(paper: dict) -> dict:
    abstract = (paper.get("abstract") or "").strip() or "(no abstract available)"
    return {"title": (paper.get("title") or "").strip(), "abstract": abstract[:ABSTRACT_CAP]}


# ---------------------------------------------------------------------------
# Parsing


class JevParseError(ValueError):
    """The response lacks a well-formed answer to a question."""


def _score(answer: dict, name: str, top: int) -> dict:
    try:
        expected = answer["score"]
        probs = {int(k): float(v) for k, v in answer["probabilities"].items()}
    except (KeyError, TypeError, ValueError, AttributeError):
        raise JevParseError(f"{name}: malformed score answer") from None
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise JevParseError(f"{name}: score {expected!r} is not a number")
    if not 0 <= expected <= top:
        raise JevParseError(f"{name}: score {expected!r} outside 0..{top}")
    return {
        "expected": float(expected),
        "p_high": sum(p for lvl, p in probs.items() if lvl >= HIGH),
    }


def parse_response(response: dict, tags: list[str], top: int = 5) -> dict:
    """Methods and findings score summaries plus P(yes) per tag, or raise `JevParseError`."""
    try:
        answers = response["answers"]
        out = {
            "methods": _score(answers["methods_relevance"], "methods_relevance", top),
            "findings": _score(answers["findings_relevance"], "findings_relevance", top),
        }
        tag_p = {tag: answers[f"tag_{tag}"]["noul"] for tag in tags}
    except (KeyError, TypeError) as exc:
        raise JevParseError(f"response lacks an answer: {exc!r}") from None
    for tag, p in tag_p.items():
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise JevParseError(f"tag_{tag} probability {p!r} is not a number in [0, 1]")
    out["tags"] = {t: float(p) for t, p in tag_p.items()}
    return out


# ---------------------------------------------------------------------------
# Data and sample


def load_papers(db: Path, *, screened_only: bool = True) -> list[dict]:
    """Read-only rows of papers; by default only those with a Haiku relevance score."""
    con = sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    where = " WHERE relevance IS NOT NULL" if screened_only else ""
    try:
        rows = con.execute(
            "SELECT key, title, abstract, published, status, relevance, tags FROM papers" + where
        ).fetchall()
    finally:
        con.close()
    papers = []
    for r in rows:
        paper = dict(r)
        paper["haiku_tags"] = json.loads(paper.pop("tags") or "[]")
        papers.append(paper)
    return papers


EXPORT_FIELDS = ("key", "title", "abstract", "published", "status", "relevance", "tags")


def export_papers(db: Path, path: Path) -> int:
    """Write every paper's EXPORT_FIELDS as gzipped JSON lines (for a machine without the db)."""
    papers = load_papers(db, screened_only=False)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for p in sorted(papers, key=lambda p: p["key"]):
            row = {k: p.get(k) for k in EXPORT_FIELDS if k != "tags"}
            line = json.dumps(row | {"tags": p["haiku_tags"]}, ensure_ascii=False)
            handle.write(line + "\n")
    return len(papers)


def load_papers_jsonl(path: Path) -> list[dict]:
    """Papers from an `export_papers` file (.jsonl or .jsonl.gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    papers = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["haiku_tags"] = list(row.pop("tags", None) or [])
                papers.append(row)
    return papers


def tag_all(
    rows: list[dict], papers: list[dict], tags: list[str], threshold: float
) -> pd.DataFrame:
    """Jev's relevance and tag probabilities per paper, with tags at `threshold`."""
    df = results_frame(papers, rows, tags)
    probs = df[[f"p_{t}" for t in tags]]
    df["jev_tags"] = [
        ";".join(t for t in tags if row[f"p_{t}"] >= threshold) for _, row in probs.iterrows()
    ]
    return df.drop(columns=["abstract"])


def sample_papers(papers: list[dict], n_per_level: int, seed: int) -> list[dict]:
    """All adopted papers plus `n_per_level` screened ones per Haiku relevance level."""
    adopted = sorted((p for p in papers if p["status"] == "adopted"), key=lambda p: p["key"])
    screened = [p for p in papers if p["status"] == "screened"]
    chosen = list(adopted)
    for level in sorted({p["relevance"] for p in screened}):
        pool = sorted((p for p in screened if p["relevance"] == level), key=lambda p: p["key"])
        rng = random.Random(f"{seed}:{level}")
        chosen += rng.sample(pool, min(n_per_level, len(pool)))
    return chosen


# ---------------------------------------------------------------------------
# Running with a resumable cache


def _load_cache(path: Path, design: str) -> dict:
    cache: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("design") == design and row.get("model") == MODEL and not row["error"]:
                    cache[row["key"]] = row
    return cache


def run(
    papers: list[dict],
    questions: dict,
    tags: list[str],
    out: Path,
    key: str,
    max_calls: int,
    post=http_post,
    sleep=time.sleep,
    params: dict | None = None,
    log: bool = True,
) -> list[dict]:
    """Ask Jev about every paper not yet cached; one row per paper, in sample order.

    `log=False` is for a caller that has already logged the whole run to the ledger.
    """
    out.mkdir(parents=True, exist_ok=True)
    design = design_hash(questions)
    path = out / "cache.jsonl"
    cache = _load_cache(path, design)
    pending = [p for p in papers if p["key"] not in cache]
    if len(pending) > max_calls:
        raise SystemExit(
            f"refusing to start: {len(pending)} Jev calls needed, --max-calls {max_calls}"
        )
    if log:
        runlog.log_run(
            name="jev-paper-screen",
            tags=["methodology", "jev"],
            params={
                "model": MODEL,
                "n_papers": len(papers),
                "pending_calls": len(pending),
                "max_calls": max_calls,
                "design": design,
                **(params or {}),
            },
            notes="Jev vs Haiku on paperlog relevance and tags; Haiku is the reference, not truth.",
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "jev-paper-screen/1",
    }
    with open(path, "a", encoding="utf-8") as handle:
        for paper in pending:
            body = {"model": MODEL, "state": build_state(paper), "questions": questions}
            start = time.perf_counter()
            response = post_with_retry(post, headers, body, sleep, url=JEV_URL)
            seconds = time.perf_counter() - start
            row = {"key": paper["key"], "model": MODEL, "design": design, "error": ""}
            try:
                row |= parse_response(response, tags)
            except JevParseError as exc:
                row["error"] = str(exc)
            row["usage"] = response.get("usage", {}) if isinstance(response, dict) else {}
            row["seconds"] = seconds
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if not row["error"]:
                cache[paper["key"]] = row
    return [cache[p["key"]] for p in papers if p["key"] in cache]


# ---------------------------------------------------------------------------
# Metrics


def spearman(x, y) -> float:
    """Pearson correlation of average ranks; nan when either side is constant."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if len(rx) < 2 or rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x, y) -> float:
    """Kendall tau-b; nan when either side is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1)
    dx, dy = dx[iu], dy[iu]
    denom = math.sqrt(np.abs(dx).sum() * np.abs(dy).sum())
    return float((dx * dy).sum() / denom) if denom else float("nan")


def auroc(scores, truth) -> float:
    """P(a random positive scores above a random negative), ties half; nan if one class is empty."""
    s, t = np.asarray(scores, float), np.asarray(truth, bool)
    pos, neg = s[t], s[~t]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def rank_of(values: pd.Series, key: str, keys: pd.Series) -> tuple[int, int]:
    """(1 + papers scoring strictly higher, papers tied with it including itself)."""
    mine = values[keys == key].iloc[0]
    return int((values > mine).sum()) + 1, int((values == mine).sum())


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1; F1 is nan when the reference has no positives."""
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else float("nan")
    return precision, recall, f1


def tag_sweep(p: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per tag and threshold: counts, precision, recall and F1 of (p >= threshold) vs truth."""
    recs = []
    for tag in truth.columns:
        t = truth[tag].to_numpy(bool)
        for thr in THRESHOLDS:
            d = p[tag].to_numpy() >= thr
            tp, fp, fn = int((d & t).sum()), int((d & ~t).sum()), int((~d & t).sum())
            pr, rc, f1 = prf(tp, fp, fn)
            recs.append(
                {
                    "tag": tag,
                    "threshold": thr,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": pr,
                    "recall": rc,
                    "f1": f1,
                }
            )
    return pd.DataFrame(recs)


def tag_summary(sweep: pd.DataFrame, p: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for tag in truth.columns:
        s = sweep[sweep.tag == tag].set_index("threshold")
        best = s.f1.fillna(-1).idxmax()
        at = s.loc[0.5]
        recs.append(
            {
                "tag": tag,
                "n_haiku": int(truth[tag].sum()),
                "n_jev_at_0.5": int((p[tag] >= 0.5).sum()),
                "auroc": auroc(p[tag], truth[tag]),
                "precision_0.5": at.precision,
                "recall_0.5": at.recall,
                "f1_0.5": at.f1,
                "disagree_0.5": int(at.fp + at.fn),
                "best_threshold": best,
                "precision_best": s.loc[best].precision,
                "recall_best": s.loc[best].recall,
                "f1_best": s.loc[best].f1,
            }
        )
    return pd.DataFrame(recs)


def macro(values: pd.Series) -> float:
    """Mean over tags where the value is defined (tags Haiku never used have no F1)."""
    return float(values.dropna().mean()) if values.notna().any() else float("nan")


# ---------------------------------------------------------------------------
# Outputs


def snippet(text: str | None, n: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + "..."


def results_frame(papers: list[dict], rows: list[dict], tags: list[str]) -> pd.DataFrame:
    by_key = {r["key"]: r for r in rows}
    recs = []
    for p in papers:
        r = by_key.get(p["key"])
        if r is None:
            continue
        recs.append(
            {
                "key": p["key"],
                "title": p["title"],
                "abstract": p["abstract"],
                "published": p["published"],
                "status": p["status"],
                "haiku_relevance": p["relevance"],
                "haiku_tags": p["haiku_tags"],
                "jev_methods": r["methods"]["expected"],
                "jev_findings": r["findings"]["expected"],
                "jev_relevance": max(r["methods"]["expected"], r["findings"]["expected"]),
                "jev_p_high": max(r["methods"]["p_high"], r["findings"]["p_high"]),
                **{f"p_{t}": r["tags"][t] for t in tags},
                "seconds": r["seconds"],
                "input_tokens": r["usage"].get("input_tokens", 0),
            }
        )
    return pd.DataFrame(recs)


def disagreement_sheet(df: pd.DataFrame, tags: list[str], n: int = DISAGREEMENTS) -> pd.DataFrame:
    sheet = df.assign(abs_diff=(df.jev_relevance - df.haiku_relevance).abs())
    sheet = sheet.sort_values(["abs_diff", "key"], ascending=[False, True]).head(n)
    jev_tags = (
        sheet[[f"p_{t}" for t in tags]]
        .ge(0.5)
        .apply(lambda row: ";".join(t for t, v in zip(tags, row, strict=True) if v), axis=1)
    )
    return pd.DataFrame(
        {
            "key": sheet.key,
            "title": sheet.title,
            "published": sheet.published,
            "abstract_snippet": sheet.abstract.map(snippet),
            "haiku_relevance": sheet.haiku_relevance,
            "jev_relevance": sheet.jev_relevance.round(2),
            "jev_methods": sheet.jev_methods.round(2),
            "jev_findings": sheet.jev_findings.round(2),
            "jev_minus_haiku": (sheet.jev_relevance - sheet.haiku_relevance).round(2),
            "haiku_tags": sheet.haiku_tags.map(";".join),
            "jev_tags_at_0.5": jev_tags,
            "human_relevance": "",
        }
    )


def report(
    papers: list[dict],
    rows: list[dict],
    tags: list[str],
    out: Path,
    args: dict,
    skipped_errors: int,
) -> Path:
    df = results_frame(papers, rows, tags)
    if len(df) < 3:
        raise SystemExit("too few answered papers to report")
    stats: dict[str, Stat] = {}
    statements: list[Statement] = []
    inputs = ("results.csv",)

    def stat(key: str, value, definition: str, inp: tuple[str, ...] = inputs) -> None:
        stats[key] = Stat(value, definition, inp)

    n = len(df)
    stat("n_papers", n, "papers in the sample that Jev answered", inputs)
    stat(
        "n_by_haiku_level",
        {str(k): int(v) for k, v in df.haiku_relevance.value_counts().sort_index().items()},
        "answered papers per Haiku relevance level (stratified, not the natural mix)",
    )
    stat("n_parse_errors", skipped_errors, "papers whose response could not be parsed", ())
    stat("run_params", args, "arguments of this run", ())

    caveat = (
        "Haiku 4.5 is the reference, not ground truth, so agreement is not accuracy; no human "
        f"ratings exist beyond 3 adopted papers. Stratified sample of {n} papers; rubric levels "
        "that mix methods and findings were sent verbatim to both Jev score questions."
    )
    rho = spearman(df.jev_relevance, df.haiku_relevance)
    tau = kendall(df.jev_relevance, df.haiku_relevance)
    stat(
        "relevance.spearman",
        rho,
        "Spearman: Jev max(methods, findings) expected score vs Haiku relevance",
    )
    stat("relevance.kendall", tau, "Kendall tau-b: same two scores")
    statements.append(
        Statement(
            f"Jev's relevance score tracks Haiku's with Spearman {rho:.2f} and Kendall {tau:.2f} "
            f"over {n} papers.",
            "relevance.spearman",
            caveat,
        )
    )
    haiku_high = df.haiku_relevance >= HIGH
    for label, jev_high in (
        ("expected", df.jev_relevance >= HIGH),
        ("p_high", df.jev_p_high >= 0.5),
    ):
        tp = int((jev_high & haiku_high).sum())
        fp = int((jev_high & ~haiku_high).sum())
        fn = int((~jev_high & haiku_high).sum())
        tn = int((~jev_high & ~haiku_high).sum())
        stat(
            f"high.{label}",
            {"both": tp, "jev_only": fp, "haiku_only": fn, "neither": tn},
            f"confusion of Jev-high vs Haiku-high (relevance >= {HIGH}); Jev-high is "
            + ("max expected score >= 4" if label == "expected" else "max P(level >= 4) >= 0.5"),
        )
        statements.append(
            Statement(
                f"Jev-high ({'expected score' if label == 'expected' else 'P(level>=4) >= 0.5'}) "
                f"and Haiku-high agree on {tp} papers, Jev alone flags {fp}, "
                f"Haiku alone flags {fn}, "
                f"neither {tn}.",
                f"high.{label}",
                caveat
                + " The sample holds equal numbers per Haiku level, so Haiku-high is over-sampled.",
            )
        )
    adopted = df[df.status == "adopted"]
    ranks = {}
    for k in adopted.key:
        jr, jt = rank_of(df.jev_relevance, k, df.key)
        hr, ht = rank_of(df.haiku_relevance, k, df.key)
        ranks[k] = {"jev_rank": jr, "jev_tied": jt, "haiku_rank": hr, "haiku_tied": ht}
    stat(
        "adopted.ranks",
        ranks,
        f"rank (1 = top, 1 + papers strictly higher) of each adopted paper among {n}, "
        "with tie counts",
    )
    if ranks:
        text = "; ".join(
            f"Jev {v['jev_rank']} (tied {v['jev_tied']}) vs "
            f"Haiku {v['haiku_rank']} (tied {v['haiku_tied']})"
            for v in ranks.values()
        )
        statements.append(
            Statement(
                f"The {len(ranks)} adopted papers rank, in key order, as: {text}, out of {n}.",
                "adopted.ranks",
                "Three papers are the only human signal and cannot separate the systems; the "
                "sample is stratified, so Haiku ranks are partly fixed by construction.",
            )
        )

    truth = pd.DataFrame({t: df.haiku_tags.map(lambda ts, t=t: t in ts) for t in tags})
    probs = df[[f"p_{t}" for t in tags]].set_axis(tags, axis=1)
    sweep = tag_sweep(probs, truth)
    summary = tag_summary(sweep, probs, truth)
    macro05 = macro(summary["f1_0.5"])
    macro_best = macro(summary.f1_best)
    n_used = int((summary.n_haiku > 0).sum())
    stat(
        "tags.macro_f1_0.5",
        macro05,
        "macro-F1 of Jev tags at threshold 0.5 vs Haiku tags, over tags Haiku used",
        ("tag_summary.csv",),
    )
    stat(
        "tags.macro_f1_best",
        macro_best,
        "macro-F1 with each tag's threshold chosen for best F1 on this same sample "
        "(optimistic, in-sample)",
        ("tag_summary.csv", "tag_sweep.csv"),
    )
    stat(
        "tags.n_used",
        n_used,
        "taxonomy tags Haiku applied at least once in the sample",
        ("tag_summary.csv",),
    )
    worst = summary[summary.n_haiku > 0].sort_values(["f1_0.5", "tag"]).head(5)
    stat(
        "tags.worst",
        {
            r.tag: {"f1_0.5": r["f1_0.5"], "n_haiku": r.n_haiku, "n_jev": r["n_jev_at_0.5"]}
            for _, r in worst.iterrows()
        },
        "five tags with the lowest F1 at 0.5 against Haiku",
        ("tag_summary.csv",),
    )
    most = summary.sort_values(["disagree_0.5", "tag"], ascending=[False, True]).head(5)
    stat(
        "tags.most_disagreed",
        {r.tag: int(r["disagree_0.5"]) for _, r in most.iterrows()},
        "five tags with the most papers where Jev (>= 0.5) and Haiku differ",
        ("tag_summary.csv",),
    )
    tag_caveat = (
        "Haiku tags are the reference, not truth; the tag definitions given to Jev are drafts "
        f"awaiting team review and Haiku saw only tag names; sample of {n} papers with Haiku "
        "relevance levels equally represented; best-threshold F1 is tuned in-sample."
    )
    statements.append(
        Statement(
            f"Jev's tags reach macro-F1 {macro05:.2f} against Haiku's at threshold 0.5 over "
            f"{n_used} tags Haiku used.",
            "tags.macro_f1_0.5",
            tag_caveat,
        )
    )
    statements.append(
        Statement(
            f"With each tag's threshold chosen for best F1 on this same sample, macro-F1 is "
            f"{macro_best:.2f}.",
            "tags.macro_f1_best",
            tag_caveat,
        )
    )
    statements.append(
        Statement(
            "Lowest agreement at 0.5 by F1: "
            + ", ".join(f"{t} ({v['f1_0.5']:.2f})" for t, v in stats["tags.worst"].value.items())
            + ".",
            "tags.worst",
            tag_caveat,
        )
    )

    secs = df.seconds
    tokens = int(df.input_tokens.sum())
    stat("cost.calls", n, "Jev calls behind the answered rows (cache.jsonl)", ("cache.jsonl",))
    stat("cost.input_tokens", tokens, "billable input tokens over answered rows", ("cache.jsonl",))
    stat(
        "cost.usd",
        tokens * PRICE_PER_M_INPUT / 1e6,
        f"input tokens x ${PRICE_PER_M_INPUT} per million (output is free)",
        ("cache.jsonl",),
    )
    stat(
        "latency.median_s",
        float(secs.median()),
        "median seconds per call including retries",
        ("cache.jsonl",),
    )
    stat(
        "latency.p95_s",
        float(secs.quantile(0.95)),
        "95th percentile seconds per call",
        ("cache.jsonl",),
    )
    statements.append(
        Statement(
            f"The run took {n} Jev calls, {tokens} input tokens (about "
            f"${tokens * PRICE_PER_M_INPUT / 1e6:.3f}), median {secs.median():.2f}s per call.",
            "cost.usd",
            "Answered rows only; cost is from reported input tokens at the documented price.",
        )
    )
    sheet = disagreement_sheet(df, tags)
    stat(
        "disagreements.mean_abs_diff",
        float(sheet.jev_minus_haiku.abs().mean()),
        f"mean |Jev - Haiku| relevance over the {len(sheet)} sheet papers",
        ("disagreements.csv",),
    )

    flat = df.assign(haiku_tags=df.haiku_tags.map(";".join))
    sheet.to_csv(out / "disagreements.csv", index=False, encoding="utf-8-sig")
    tables = {"results": flat, "tag_sweep": sweep, "tag_summary": summary, "disagreements": sheet}
    return write_outputs(
        out,
        title="Jev vs Haiku on paperlog screening",
        tables=tables,
        stats=stats,
        statements=statements,
    )


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None, post=http_post, sleep=time.sleep) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", type=Path, help="paperlog papers.db (opened read-only)")
    src.add_argument("--papers", type=Path, help="an export_papers file (.jsonl or .jsonl.gz)")
    ap.add_argument("--export", type=Path, help="with --db: write the papers export here and stop")
    ap.add_argument("--all", action="store_true", help="tag every paper (no Haiku comparison)")
    ap.add_argument("--status", help="with --all: comma-separated statuses to tag (default all)")
    ap.add_argument("--threshold", type=float, default=0.5, help="with --all: tag cut-off")
    ap.add_argument("--n-per-level", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.export:
        if not a.db:
            ap.error("--export needs --db")
        n = export_papers(a.db, a.export)
        print(f"wrote {n} papers to {a.export}")
        return a.export
    cfg = load_config()
    tags = list(cfg["taxonomy"])
    questions = build_questions(cfg, load_definitions(tags))
    source = a.db or a.papers
    if a.all:
        papers = load_papers(a.db, screened_only=False) if a.db else load_papers_jsonl(a.papers)
        if a.status:
            keep = {s.strip() for s in a.status.split(",")}
            papers = [p for p in papers if p.get("status") in keep]
        papers = sorted(papers, key=lambda p: p["key"])
        params = {"mode": "all", "status": a.status, "source": source.name}
        rows = run(papers, questions, tags, a.out, jev_key(), a.max_calls, post, sleep, params)
        out = a.out / "tagged.csv"
        tag_all(rows, papers, tags, a.threshold).to_csv(out, index=False)
        print(f"tagged {len(rows)} of {len(papers)} papers; wrote {out}")
        return out
    papers = load_papers(a.db) if a.db else load_papers_jsonl(a.papers)
    papers = sample_papers(
        [p for p in papers if p.get("relevance") is not None], a.n_per_level, a.seed
    )
    params = {"n_per_level": a.n_per_level, "seed": a.seed, "db": source.name}
    rows = run(papers, questions, tags, a.out, jev_key(), a.max_calls, post, sleep, params)
    path = report(
        papers, rows, tags, a.out, params | {"max_calls": a.max_calls}, len(papers) - len(rows)
    )
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
