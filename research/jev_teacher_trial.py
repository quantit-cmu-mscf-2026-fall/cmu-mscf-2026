"""Can a stronger teacher model be used to tune Jev's paper tagging?

Claude Sonnet 5.5 labels a stratified sample of paperlog papers once (research/teacher_label.py).
The sample is split 50/50 (seeded, stratified by Haiku relevance) into TRAIN and TEST. Jev still
reads only title and abstract; the teacher's labels are never put into its input. They are used
only to fit, on TRAIN:

- per-tag thresholds: the grid value (0.1 .. 0.9) with the best F1 against the teacher;
- a relevance mapping: a ridge regression (alpha 1, standardised inputs) from Jev's methods and
  findings expected scores and P(level >= 4) to the teacher's relevance, clipped to 0..5, and a
  "high" cut-off (steps of 0.25) with the best F1 for teacher relevance >= 4;
- a Platt calibration of each tag probability (logistic on the logit; identity when TRAIN has
  fewer than 5 positives or 5 negatives for that tag).

On TEST, raw Jev (tags at 0.5, relevance = max expected score), tuned Jev and the stored Haiku
screen are all scored against the teacher. The teacher is a stronger model, not ground truth.

    python research/jev_teacher_trial.py --papers papers.jsonl.gz --seed 0 --n-per-level 100 \
        --jev-cache-from trial1/cache.jsonl --out experiments/analysis/jev-papers/teacher1

The run is logged once to the ledger before the first Sonnet or Jev call.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import sys
import time
from pathlib import Path

import jev_paper_screen as jps
import numpy as np
import pandas as pd
import teacher_label as tl

from capstone import runlog
from capstone.analysis.outputs import Stat, Statement, write_outputs
from capstone.jev import http_post, jev_key

SYSTEMS = ("raw", "tuned", "haiku")
HIGH_GRID = tuple(np.round(np.arange(0.5, 5.01, 0.25), 2))
MIN_CLASS = 5
RIDGE = 1.0
ECE_BINS = 10
SPOTCHECK_SNIPPET = 400


# ---------------------------------------------------------------------------
# Sample, split, weights


def population_counts(papers: list[dict]) -> dict[int, int]:
    """Papers per Haiku level among the screened and filtered ones (the real mix)."""
    counts: dict[int, int] = {}
    for p in papers:
        if p.get("relevance") is not None and p.get("status") in ("screened", "filtered"):
            counts[int(p["relevance"])] = counts.get(int(p["relevance"]), 0) + 1
    return dict(sorted(counts.items()))


def split_halves(df: pd.DataFrame, seed: int) -> pd.Series:
    """'train' or 'test' per row: each Haiku level is shuffled with `seed` and halved."""
    out = pd.Series("test", index=df.index)
    for level in sorted(df.haiku_relevance.unique()):
        keys = sorted(df.loc[df.haiku_relevance == level, "key"])
        random.Random(f"{seed}:split:{level}").shuffle(keys)
        train = set(keys[: (len(keys) + 1) // 2])
        out[(df.haiku_relevance == level) & df.key.isin(train)] = "train"
    return out


def population_weights(df: pd.DataFrame, pop: dict[int, int]) -> pd.Series:
    """Weight so the sampled papers of each Haiku level stand for that level's real count.

    Adopted papers are outside the screened population and get weight 0.
    """
    screened = df[df.status == "screened"]
    per_level = screened.groupby("haiku_relevance").size()
    w = df.haiku_relevance.map(
        lambda lvl: pop.get(int(lvl), 0) / per_level[lvl] if lvl in per_level.index else 0.0
    )
    return w.where(df.status == "screened", 0.0).astype(float)


# ---------------------------------------------------------------------------
# Metrics


def wrank(x, w) -> np.ndarray:
    """Weighted mid-rank in [0, 1]: weight strictly below plus half the weight tied."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    below = (w[None, :] * (x[None, :] < x[:, None])).sum(1)
    tied = (w[None, :] * (x[None, :] == x[:, None])).sum(1)
    return (below + 0.5 * tied) / w.sum()


def wspearman(x, y, w) -> float:
    """Weighted Pearson correlation of weighted mid-ranks; nan when degenerate."""
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return float("nan")
    rx, ry = wrank(x, w), wrank(y, w)
    mx, my = (w * rx).sum() / w.sum(), (w * ry).sum() / w.sum()
    cov = (w * (rx - mx) * (ry - my)).sum()
    vx, vy = (w * (rx - mx) ** 2).sum(), (w * (ry - my) ** 2).sum()
    return float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def f1_vector(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-column F1 of boolean matrices; nan for columns with no true positives."""
    tp = (pred & truth).sum(0)
    fp = (pred & ~truth).sum(0)
    fn = (~pred & truth).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
    return np.where(truth.sum(0) > 0, f1, np.nan)


def macro_f1(pred: np.ndarray, truth: np.ndarray) -> float:
    f1 = f1_vector(pred, truth)
    return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


def high_scores(pred: np.ndarray, truth: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Precision, recall, F1 of a boolean 'high' call vs the truth, optionally weighted."""
    w = np.ones(len(pred)) if w is None else w
    tp, fp, fn = (
        (w * (pred & truth)).sum(),
        (w * (pred & ~truth)).sum(),
        (w * (~pred & truth)).sum(),
    )
    pr, rc, f1 = jps.prf(tp, fp, fn)
    return {"precision": pr, "recall": rc, "f1": f1}


def brier(p: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((p - truth.astype(float)) ** 2))


def ece(p: np.ndarray, truth: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-width bins over all probabilities given."""
    p, t = np.ravel(p), np.ravel(truth).astype(float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(p) * abs(p[m].mean() - t[m].mean())
    return float(total)


# ---------------------------------------------------------------------------
# Tuning (fitted on TRAIN only)


def fit_thresholds(p: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Per tag, the grid threshold with the best F1; ties go to the one nearest 0.5."""
    out = {}
    for tag in truth.columns:
        t = truth[tag].to_numpy(bool)
        if not t.any():
            out[tag] = 0.5
            continue
        best = max(
            jps.THRESHOLDS,
            key=lambda thr: (
                jps.prf(
                    int(((p[tag].to_numpy() >= thr) & t).sum()),
                    int(((p[tag].to_numpy() >= thr) & ~t).sum()),
                    int((~(p[tag].to_numpy() >= thr) & t).sum()),
                )[2],
                -abs(thr - 0.5),
            ),
        )
        out[tag] = best
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit_platt(p, truth) -> tuple[float, float]:
    """(slope, intercept) of a logistic fit on logit(p); identity when a class is too small."""
    t = np.asarray(truth, float)
    if t.sum() < MIN_CLASS or (1 - t).sum() < MIN_CLASS:
        return 1.0, 0.0
    z = _logit(p)
    theta = np.array([1.0, 0.0])
    x = np.column_stack([z, np.ones_like(z)])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(x @ theta)))
        grad = x.T @ (mu - t) + 1e-3 * theta
        hess = (x.T * (mu * (1 - mu))) @ x + 1e-3 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(theta[0]), float(theta[1])


def apply_platt(p, params: tuple[float, float]) -> np.ndarray:
    a, b = params
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


FEATURES = ("jev_methods", "jev_findings", "jev_p_high")


def fit_relevance_map(train: pd.DataFrame) -> dict:
    """Ridge map from Jev's (methods, findings, p_high) to the teacher's relevance, plus a cut."""
    x = train[list(FEATURES)].to_numpy(float)
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    y = train.sonnet_relevance.to_numpy(float)
    ybar = y.mean()
    coef = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ (y - ybar))
    model = {"mean": mean.tolist(), "std": std.tolist(), "coef": coef.tolist(), "intercept": ybar}
    score = apply_relevance_map(model, train)
    truth = train.sonnet_relevance.to_numpy() >= jps.HIGH
    cut = 3.5
    if truth.any():
        cut = max(
            HIGH_GRID,
            key=lambda c: (high_scores(score >= c, truth)["f1"], -abs(c - 3.5)),
        )
    model["high_cut"] = float(cut)
    return model


def apply_relevance_map(model: dict, df: pd.DataFrame) -> np.ndarray:
    z = (df[list(FEATURES)].to_numpy(float) - model["mean"]) / model["std"]
    return np.clip(z @ np.array(model["coef"]) + model["intercept"], 0.0, 5.0)


# ---------------------------------------------------------------------------
# Frame


def build_frame(
    papers: list[dict], jev_rows: list[dict], teacher_rows: list[dict], tags: list[str]
) -> pd.DataFrame:
    df = jps.results_frame(papers, jev_rows, tags)
    by_key = {r["key"]: r for r in teacher_rows}
    df = df[df.key.isin(by_key)].reset_index(drop=True)
    df["sonnet_relevance"] = df.key.map(lambda k: by_key[k]["relevance"])
    df["sonnet_tags"] = df.key.map(lambda k: by_key[k]["tags"])
    df["sonnet_reason"] = df.key.map(lambda k: by_key[k]["reason"])
    return df


def tag_matrix(df: pd.DataFrame, column: str, tags: list[str]) -> np.ndarray:
    return np.array([[t in ts for t in tags] for ts in df[column]], dtype=bool).reshape(
        len(df), len(tags)
    )


def prob_matrix(df: pd.DataFrame, tags: list[str]) -> np.ndarray:
    return df[[f"p_{t}" for t in tags]].to_numpy(float)


class Tuning:
    """Everything fitted on TRAIN."""

    def __init__(self, train: pd.DataFrame, tags: list[str]):
        p = pd.DataFrame(prob_matrix(train, tags), columns=tags)
        truth = pd.DataFrame(tag_matrix(train, "sonnet_tags", tags), columns=tags)
        self.tags = tags
        self.thresholds = fit_thresholds(p, truth)
        self.platt = {t: fit_platt(p[t], truth[t]) for t in tags}
        self.relevance = fit_relevance_map(train)

    def systems(self, df: pd.DataFrame) -> dict:
        """Relevance scores, high calls and tag predictions for raw, tuned and Haiku."""
        probs = prob_matrix(df, self.tags)
        thr = np.array([self.thresholds[t] for t in self.tags])
        tuned = apply_relevance_map(self.relevance, df)
        return {
            "raw": {
                "score": df.jev_relevance.to_numpy(float),
                "high": df.jev_relevance.to_numpy(float) >= jps.HIGH,
                "tags": probs >= 0.5,
            },
            "tuned": {
                "score": tuned,
                "high": tuned >= self.relevance["high_cut"],
                "tags": probs >= thr,
            },
            "haiku": {
                "score": df.haiku_relevance.to_numpy(float),
                "high": df.haiku_relevance.to_numpy(float) >= jps.HIGH,
                "tags": tag_matrix(df, "haiku_tags", self.tags),
            },
        }

    def calibrated(self, df: pd.DataFrame) -> np.ndarray:
        probs = prob_matrix(df, self.tags)
        return np.column_stack(
            [apply_platt(probs[:, i], self.platt[t]) for i, t in enumerate(self.tags)]
        )


# ---------------------------------------------------------------------------
# Bootstrap


def bootstrap(
    systems: dict,
    truth_rel: np.ndarray,
    truth_tags: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Stratified bootstrap of the TEST papers (tuning is held fixed at its TRAIN fit)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out: dict[str, list[float]] = {}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals: dict[str, float] = {}
        for name in SYSTEMS:
            s = systems[name]
            vals[f"macro_f1.{name}"] = macro_f1(s["tags"][idx], truth_tags[idx])
            vals[f"spearman.{name}"] = jps.spearman(s["score"][idx], truth_rel[idx])
            vals[f"wspearman.{name}"] = wspearman(s["score"][idx], truth_rel[idx], weights[idx])
        for metric in ("macro_f1", "spearman", "wspearman"):
            for ref in ("raw", "haiku"):
                vals[f"{metric}.tuned_minus_{ref}"] = (
                    vals[f"{metric}.tuned"] - vals[f"{metric}.{ref}"]
                )
        for k, v in vals.items():
            out.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def ci(values: np.ndarray) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if not len(v):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Output


def fmt(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"


def spotcheck_sheet(test: pd.DataFrame, systems: dict, tags: list[str]) -> pd.DataFrame:
    def names(mat: np.ndarray) -> list[str]:
        return [";".join(t for t, on in zip(tags, row, strict=True) if on) for row in mat]

    return pd.DataFrame(
        {
            "key": test.key.to_numpy(),
            "title": test.title.to_numpy(),
            "published": test.published.to_numpy(),
            "abstract_snippet": test.abstract.map(lambda a: jps.snippet(a, SPOTCHECK_SNIPPET)),
            "sonnet_relevance": test.sonnet_relevance.to_numpy(),
            "sonnet_tags": test.sonnet_tags.map(";".join).to_numpy(),
            "sonnet_reason": test.sonnet_reason.to_numpy(),
            "haiku_relevance": test.haiku_relevance.to_numpy(),
            "haiku_tags": test.haiku_tags.map(";".join).to_numpy(),
            "jev_raw_relevance": systems["raw"]["score"].round(2),
            "jev_tuned_relevance": systems["tuned"]["score"].round(2),
            "jev_raw_tags": names(systems["raw"]["tags"]),
            "jev_tuned_tags": names(systems["tuned"]["tags"]),
            "human_relevance": "",
            "human_tags": "",
        }
    )


def report(
    df: pd.DataFrame,
    tuning: Tuning,
    pop: dict[int, int],
    out: Path,
    args: dict,
    costs: dict,
    reps: int,
    seed: int,
) -> Path:
    tags = tuning.tags
    train, test = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    if len(test) < 10 or len(train) < 10:
        raise SystemExit("too few labelled papers to tune and test")
    systems = tuning.systems(test)
    truth_rel = test.sonnet_relevance.to_numpy(float)
    truth_high = truth_rel >= jps.HIGH
    truth_tags = tag_matrix(test, "sonnet_tags", tags)
    w = test.weight.to_numpy(float)

    stats: dict[str, Stat] = {}
    statements: list[Statement] = []

    def stat(key: str, value, definition: str, inputs: tuple[str, ...] = ("results.csv",)) -> None:
        stats[key] = Stat(value, definition, inputs)

    stat("n.sample", len(df), "papers labelled by both the teacher and Jev")
    stat("n.split", {"train": len(train), "test": len(test)}, "papers per half")
    stat(
        "n.by_haiku_level",
        {str(k): int(v) for k, v in test.haiku_relevance.value_counts().sort_index().items()},
        "TEST papers per Haiku relevance level (stratified sample, not the natural mix)",
    )
    stat(
        "population.by_haiku_level",
        {str(k): v for k, v in pop.items()},
        "screened + filtered papers per Haiku level in the full export; the reweighting target",
        (),
    )
    stat("run_params", args, "arguments of this run", ())
    dist = test.sonnet_relevance.value_counts().sort_index()
    stat(
        "teacher.test_relevance_counts",
        {str(k): int(v) for k, v in dist.items()},
        "TEST papers per Sonnet relevance level",
    )
    stat(
        "tuning.thresholds",
        tuning.thresholds,
        "per-tag threshold with the best F1 against Sonnet on TRAIN",
        ("tuning.csv",),
    )
    stat(
        "tuning.relevance_map",
        {
            "features": list(FEATURES),
            "coef_per_std": tuning.relevance["coef"],
            "intercept": tuning.relevance["intercept"],
            "high_cut": tuning.relevance["high_cut"],
        },
        "ridge map from Jev scores to Sonnet relevance, fitted on TRAIN, and its high cut-off",
        ("results.csv",),
    )

    boot = bootstrap(systems, truth_rel, truth_tags, w, test.haiku_relevance.to_numpy(), reps, seed)
    stat("bootstrap.reps", reps, "stratified (by Haiku level) bootstrap resamples of TEST", ())

    base = (
        "Sonnet 5.5 is a stronger model, not ground truth; agreement with it is not accuracy "
        "and human labels are still needed. Tag definitions are drafts awaiting team review. "
        f"TEST holds {len(test)} papers stratified by Haiku level; Jev was tuned on the other "
        f"{len(train)}. Bootstrap CIs hold the TRAIN fit fixed and resample TEST papers. "
        "Sampling is stratified by Haiku level, which flatters Haiku in unweighted comparisons."
    )

    # relevance
    rel = {}
    for name in SYSTEMS:
        s = systems[name]["score"]
        rel[name] = {
            "spearman": jps.spearman(s, truth_rel),
            "weighted_spearman": wspearman(s, truth_rel, w),
            "kendall": jps.kendall(s, truth_rel),
        }
    for name in SYSTEMS:
        for metric in ("spearman", "weighted_spearman"):
            key = "wspearman" if metric.startswith("weighted") else "spearman"
            rel[name][f"{metric}_ci"] = ci(boot[f"{key}.{name}"])
    stat("relevance.spearman", rel, "Spearman of each system's relevance vs Sonnet, on TEST")
    diffs = {}
    for metric in ("spearman", "wspearman", "macro_f1"):
        for ref in ("raw", "haiku"):
            k = f"{metric}.tuned_minus_{ref}"
            point = {
                "spearman": rel["tuned"]["spearman"] - rel[ref]["spearman"],
                "wspearman": rel["tuned"]["weighted_spearman"] - rel[ref]["weighted_spearman"],
                "macro_f1": None,
            }[metric]
            diffs[k] = {"point": point, "ci95": ci(boot[k])}
    # macro-F1 points filled after the tag section

    def rel_text(name: str, label: str) -> str:
        r = rel[name]
        lo, hi = r["spearman_ci"]
        return (
            f"{label} {fmt(r['spearman'])} (95% CI {fmt(lo)} to {fmt(hi)}; "
            f"{fmt(r['weighted_spearman'])} reweighted to the real mix)"
        )

    statements.append(
        Statement(
            "Spearman correlation of relevance with Sonnet's on TEST: "
            + "; ".join(
                rel_text(n, lab)
                for n, lab in (("raw", "raw Jev"), ("tuned", "tuned Jev"), ("haiku", "Haiku"))
            )
            + f" ({len(test)} papers).",
            "relevance.spearman",
            base,
        )
    )

    # high relevance
    high = {}
    for name in SYSTEMS:
        h = systems[name]["high"]
        high[name] = {
            **high_scores(h, truth_high),
            **{f"weighted_{k}": v for k, v in high_scores(h, truth_high, w).items() if k != "f1"},
            "n_called": int(h.sum()),
        }
    stat(
        "high.scores",
        {"n_sonnet_high": int(truth_high.sum()), **high},
        f"precision and recall of each system's high call (relevance >= {jps.HIGH}; Jev raw: max "
        "expected score; tuned: mapped score >= TRAIN cut) vs Sonnet >= 4, on TEST; weighted_* "
        "reweight to the real Haiku-level mix",
    )
    statements.append(
        Statement(
            f"Against Sonnet-high ({int(truth_high.sum())} TEST papers), recall/precision is "
            + ", ".join(
                f"{lab} {fmt(high[n]['recall'])}/{fmt(high[n]['precision'])}"
                for n, lab in (("raw", "raw Jev"), ("tuned", "tuned Jev"), ("haiku", "Haiku"))
            )
            + "; reweighted to the real mix "
            + ", ".join(
                f"{lab} {fmt(high[n]['weighted_recall'])}/{fmt(high[n]['weighted_precision'])}"
                for n, lab in (("raw", "raw Jev"), ("tuned", "tuned Jev"), ("haiku", "Haiku"))
            )
            + ".",
            "high.scores",
            base + " The stratified sample over-represents high papers, so unweighted precision "
            "is optimistic.",
        )
    )

    # tags
    per_tag = pd.DataFrame(
        {
            "tag": tags,
            "n_sonnet_test": truth_tags.sum(0),
            "threshold": [tuning.thresholds[t] for t in tags],
            **{f"f1_{n}": f1_vector(systems[n]["tags"], truth_tags) for n in SYSTEMS},
        }
    )
    macro = {n: macro_f1(systems[n]["tags"], truth_tags) for n in SYSTEMS}
    macro_ci = {n: ci(boot[f"macro_f1.{n}"]) for n in SYSTEMS}
    stat(
        "tags.macro_f1",
        {n: {"value": macro[n], "ci95": macro_ci[n]} for n in SYSTEMS},
        "macro-F1 vs Sonnet tags on TEST over tags Sonnet used there (raw Jev at 0.5, tuned Jev "
        "at TRAIN thresholds, Haiku's stored tags)",
        ("tag_f1_test.csv",),
    )
    for ref in ("raw", "haiku"):
        diffs[f"macro_f1.tuned_minus_{ref}"]["point"] = macro["tuned"] - macro[ref]
    stat(
        "differences",
        diffs,
        "tuned minus raw Jev and tuned Jev minus Haiku, each with a stratified bootstrap 95% CI",
        (),
    )
    n_used = int((truth_tags.sum(0) > 0).sum())
    stat("tags.n_used", n_used, "taxonomy tags Sonnet applied at least once on TEST")
    statements.append(
        Statement(
            f"Tag macro-F1 against Sonnet over {n_used} tags: raw Jev at 0.5 {fmt(macro['raw'])} "
            f"(95% CI {fmt(macro_ci['raw'][0])} to {fmt(macro_ci['raw'][1])}), tuned Jev "
            f"{fmt(macro['tuned'])} ({fmt(macro_ci['tuned'][0])} to {fmt(macro_ci['tuned'][1])}), "
            f"Haiku {fmt(macro['haiku'])} ({fmt(macro_ci['haiku'][0])} to "
            f"{fmt(macro_ci['haiku'][1])}).",
            "tags.macro_f1",
            base,
        )
    )

    def diff_text(metric: str) -> str:
        parts = []
        for ref, name in (("raw", "raw Jev"), ("haiku", "Haiku")):
            d = diffs[f"{metric}.tuned_minus_{ref}"]
            lo, hi = d["ci95"]
            parts.append(f"{fmt(d['point'])} (95% CI {fmt(lo)} to {fmt(hi)}) over {name}")
        return " and ".join(parts)

    statements.append(
        Statement(
            "Tuned Jev's tag macro-F1 advantage is " + diff_text("macro_f1") + ".",
            "differences",
            base + " A CI that includes 0 means the gain is not distinguishable from noise here.",
        )
    )
    statements.append(
        Statement(
            "Tuned Jev's relevance Spearman advantage is "
            + diff_text("spearman")
            + "; reweighted to the real mix, "
            + diff_text("wspearman")
            + ".",
            "differences",
            base,
        )
    )

    # calibration
    p_test = prob_matrix(test, tags)
    cal_test = tuning.calibrated(test)
    calib = {
        "brier_raw": brier(p_test, truth_tags),
        "brier_calibrated": brier(cal_test, truth_tags),
        "ece_raw": ece(p_test, truth_tags),
        "ece_calibrated": ece(cal_test, truth_tags),
        "n_tags_calibrated": sum(tuning.platt[t] != (1.0, 0.0) for t in tags),
    }
    stat(
        "calibration",
        calib,
        "Brier score and 10-bin ECE of Jev tag probabilities vs Sonnet tags on TEST, pooled over "
        "papers and tags, before and after per-tag Platt scaling fitted on TRAIN",
    )
    statements.append(
        Statement(
            f"Platt calibration moves pooled tag Brier from {calib['brier_raw']:.3f} to "
            f"{calib['brier_calibrated']:.3f} and ECE from {calib['ece_raw']:.3f} to "
            f"{calib['ece_calibrated']:.3f} ({calib['n_tags_calibrated']} of {len(tags)} tags "
            "had enough TRAIN positives to be fitted).",
            "calibration",
            base + " Pooled over many rare tags, so Brier is dominated by the negative class.",
        )
    )

    # Haiku on its own
    hs = systems["haiku"]["score"]
    haiku_alone = {
        "exact_agreement": float((hs == truth_rel).mean()),
        "within_one": float((np.abs(hs - truth_rel) <= 1).mean()),
        "mean_abs_diff": float(np.abs(hs - truth_rel).mean()),
        "mean_signed_diff": float((hs - truth_rel).mean()),
        "spearman": rel["haiku"]["spearman"],
        "weighted_spearman": rel["haiku"]["weighted_spearman"],
        "high_recall": high["haiku"]["recall"],
        "high_precision": high["haiku"]["precision"],
        "macro_f1": macro["haiku"],
    }
    stat(
        "haiku.vs_sonnet",
        haiku_alone,
        "the stored Haiku 4.5 screen scored against Sonnet on TEST: exact and within-one level "
        "agreement, signed difference (Haiku minus Sonnet), rank correlation, high-call "
        "recall/precision and tag macro-F1",
    )
    statements.append(
        Statement(
            f"By Sonnet's standard the current Haiku screen agrees on the exact level for "
            f"{haiku_alone['exact_agreement']:.0%} of TEST papers and within one level for "
            f"{haiku_alone['within_one']:.0%}, with Spearman {fmt(haiku_alone['spearman'])} and "
            f"tag macro-F1 {fmt(haiku_alone['macro_f1'])}; its mean level is "
            f"{haiku_alone['mean_signed_diff']:+.2f} versus Sonnet's.",
            "haiku.vs_sonnet",
            base + " Sonnet and Haiku saw different tag instructions (Haiku saw names only).",
        )
    )

    # cost
    stat(
        "cost.teacher",
        costs["teacher"],
        "Sonnet 5.5 calls, tokens and list-price cost ($2/M input, $10/M output, cache reads 0.1x, "
        "cache writes 1.25x) over cached rows",
        ("teacher_cache.jsonl",),
    )
    stat(
        "cost.jev",
        costs["jev"],
        f"Jev calls, input tokens and cost at ${jps.PRICE_PER_M_INPUT} per million input tokens",
        ("cache.jsonl",),
    )
    t = costs["teacher"]
    statements.append(
        Statement(
            f"Sonnet labelled {t['calls']} papers using {t['input_tokens']} uncached and "
            f"{t['cache_read_input_tokens']} cached input tokens and {t['output_tokens']} output "
            f"tokens, about ${t['usd']:.2f} at list price; Jev's {costs['jev']['calls']} calls "
            f"cost about ${costs['jev']['usd']:.3f}.",
            "cost.teacher",
            "Cost is reconstructed from reported usage at list prices, not from an invoice.",
        )
    )

    tuning_table = pd.DataFrame(
        {
            "tag": tags,
            "threshold": [tuning.thresholds[t] for t in tags],
            "platt_slope": [tuning.platt[t][0] for t in tags],
            "platt_intercept": [tuning.platt[t][1] for t in tags],
            "n_sonnet_train": tag_matrix(train, "sonnet_tags", tags).sum(0),
        }
    )
    flat = df.assign(
        haiku_tags=df.haiku_tags.map(";".join), sonnet_tags=df.sonnet_tags.map(";".join)
    ).drop(columns=["abstract"])
    sheet = spotcheck_sheet(test, systems, tags)
    sheet.to_csv(out / "test_spotcheck.csv", index=False, encoding="utf-8-sig")
    return write_outputs(
        out,
        title="Can a Sonnet 5.5 teacher improve Jev's paper tagging?",
        tables={
            "results": flat,
            "tag_f1_test": per_tag,
            "tuning": tuning_table,
            "test_spotcheck": sheet,
        },
        stats=stats,
        statements=statements,
    )


# ---------------------------------------------------------------------------
# CLI


def summarize_costs(teacher_rows: list[dict], jev_rows: list[dict]) -> dict:
    jev_tokens = sum(r["usage"].get("input_tokens", 0) for r in jev_rows)
    return {
        "teacher": {"calls": len(teacher_rows), **tl.cost_usd(teacher_rows)},
        "jev": {
            "calls": len(jev_rows),
            "input_tokens": jev_tokens,
            "usd": jev_tokens * jps.PRICE_PER_M_INPUT / 1e6,
        },
    }


def main(argv: list[str] | None = None, post=http_post, sleep=time.sleep, client=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--papers", type=Path, required=True, help="an export_papers file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-level", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jev-cache-from", type=Path, help="a Jev cache.jsonl to copy in first")
    ap.add_argument("--max-teacher-calls", type=int, default=700)
    ap.add_argument("--max-jev-calls", type=int, default=700)
    ap.add_argument("--workers", type=int, default=tl.WORKERS)
    ap.add_argument("--max-tokens", type=int, default=tl.MAX_TOKENS)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args(argv)

    cfg = jps.load_config()
    tags = list(cfg["taxonomy"])
    defs = jps.load_definitions(tags)
    questions = jps.build_questions(cfg, defs)
    system = tl.build_system(cfg, defs)
    schema = tl.build_schema(tags)
    t_design = tl.design_hash(system, schema, tl.MODEL, a.max_tokens)
    j_design = jps.design_hash(questions)

    everything = jps.load_papers_jsonl(a.papers)
    pop = population_counts(everything)
    papers = jps.sample_papers(
        [p for p in everything if p.get("relevance") is not None], a.n_per_level, a.seed
    )
    a.out.mkdir(parents=True, exist_ok=True)
    if a.jev_cache_from and not (a.out / "cache.jsonl").exists():
        shutil.copy(a.jev_cache_from, a.out / "cache.jsonl")

    t_needed = tl.check_cap(papers, a.out, t_design, a.max_teacher_calls)
    j_cache = jps._load_cache(a.out / "cache.jsonl", j_design)
    j_needed = sum(p["key"] not in j_cache for p in papers)
    if j_needed > a.max_jev_calls:
        raise SystemExit(
            f"refusing to start: {j_needed} Jev calls needed, --max-jev-calls {a.max_jev_calls}"
        )
    params = {
        "teacher_model": tl.MODEL,
        "jev_model": jps.MODEL,
        "n_sample": len(papers),
        "n_per_level": a.n_per_level,
        "seed": a.seed,
        "teacher_pending": t_needed,
        "jev_pending": j_needed,
        "max_teacher_calls": a.max_teacher_calls,
        "max_jev_calls": a.max_jev_calls,
        "teacher_design": t_design,
        "jev_design": j_design,
        "papers_sha": hashlib.sha256(a.papers.read_bytes()).hexdigest()[:12],
    }
    runlog.log_run(
        name="jev-teacher-trial",
        tags=["methodology", "jev", "teacher"],
        params=params,
        notes="Sonnet 5.5 labels a sample; Jev tuned on half against those labels, scored on the "
        "other half with raw Jev and Haiku; Sonnet is a stronger model, not ground truth.",
    )

    teacher_rows, teacher_errors = tl.run_teacher(
        papers,
        system,
        tags,
        a.out,
        client if client is not None else tl.make_client(),
        a.max_teacher_calls,
        a.workers,
        a.max_tokens,
    )
    jev_rows = jps.run(
        papers, questions, tags, a.out, jev_key(), a.max_jev_calls, post, sleep, params, log=False
    )
    df = build_frame(papers, jev_rows, teacher_rows, tags)
    df["split"] = split_halves(df, a.seed)
    df["weight"] = population_weights(df, pop)
    tuning = Tuning(df[df.split == "train"], tags)
    costs = summarize_costs(teacher_rows, jev_rows)
    args = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("papers", "out", "jev_cache_from")
    } | {"teacher_errors_this_run": teacher_errors, "papers_file": a.papers.name}
    path = report(df, tuning, pop, a.out, args, costs, a.bootstrap, a.seed)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
