"""A frozen, saved tuning for Jev's paper tagging, fitted on a table of labels.

Same methods as research/jev_teacher_trial.py: per-tag F1-best thresholds, a Platt
calibration per tag (identity when a class is too small), and a ridge map from Jev's
methods / findings / P(level >= 4) to the label relevance with a "high" cut. The fit is
saved as JSON (`tuning.json`) with the model id and the hash of the Jev question design it
was fitted for; `load_tuning` refuses a tuning whose hash differs from the current questions.

Labels come from a table with one row per paper: label relevance (0..5), label tags and a
`source` ('sonnet' or 'human'). A row's `weight` is applied by repeating it (integers), so
human labels can count for more than Sonnet's. Thresholds apply to the RAW Jev tag
probabilities (as in the trial); the Platt-calibrated probabilities are for reading.

    python research/jev_tuning.py --teacher teacher_cache.jsonl --jev-cache cache.jsonl \
        --out tuning.json

No model call is made.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import jev_paper_screen as jps
import jev_teacher_trial as jtt
import numpy as np
import pandas as pd

from capstone.artifacts import atomic_json


class TuningError(ValueError):
    """A tuning cannot be used for the current questions or model."""


# ---------------------------------------------------------------------------
# Reading caches and labels (plain or gzipped)


def _lines(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [ln for ln in handle.read().splitlines() if ln.strip()]


def load_jev_cache(path: Path, design: str) -> dict[str, dict]:
    """Error-free Jev answers for this question design, by paper key (.jsonl or .jsonl.gz)."""
    cache: dict[str, dict] = {}
    for line in _lines(Path(path)):
        row = json.loads(line)
        if row.get("design") == design and row.get("model") == jps.MODEL and not row["error"]:
            cache[row["key"]] = row
    return cache


def load_teacher_labels(path: Path) -> dict[str, dict]:
    """Sonnet labels by key from a teacher_cache.jsonl; refuses a mix of prompt designs."""
    rows = [json.loads(ln) for ln in _lines(Path(path))]
    rows = [r for r in rows if not r["error"]]
    designs = {r["design"] for r in rows}
    if len(designs) > 1:
        raise TuningError(f"{path} mixes teacher designs {sorted(designs)}")
    return {r["key"]: {"relevance": int(r["relevance"]), "tags": list(r["tags"])} for r in rows}


def jev_features(cache: dict[str, dict], tags: list[str], keys=None) -> pd.DataFrame:
    """One row per cached paper: Jev's scores and raw tag probabilities (no paper text needed)."""
    recs = []
    for key in sorted(cache if keys is None else keys):
        r = cache[key]
        recs.append(
            {
                "key": key,
                "jev_methods": r["methods"]["expected"],
                "jev_findings": r["findings"]["expected"],
                "jev_relevance": max(r["methods"]["expected"], r["findings"]["expected"]),
                "jev_p_high": max(r["methods"]["p_high"], r["findings"]["p_high"]),
                **{f"p_{t}": r["tags"][t] for t in tags},
            }
        )
    return pd.DataFrame(recs)


def training_table(
    features: pd.DataFrame, labels: dict[str, dict], source: str, weight: int = 1
) -> pd.DataFrame:
    """Join Jev features to labels ({key: {relevance, tags}}); keys lacking features are dropped."""
    f = features.set_index("key")
    keep = [k for k in labels if k in f.index]
    t = f.loc[keep].reset_index()
    t["label_relevance"] = [labels[k]["relevance"] for k in keep]
    t["label_tags"] = [list(labels[k]["tags"]) for k in keep]
    t["source"] = source
    t["weight"] = weight
    return t


# ---------------------------------------------------------------------------
# The tuning

MAX_PLATT = 10.0  # a slope or intercept beyond this means the fit separated the classes


def stable_platt(p, truth) -> tuple[float, float]:
    """Platt fit from the trial, but identity if it diverged (perfectly separable tag)."""
    with np.errstate(over="ignore"):
        a, b = jtt.fit_platt(p, truth)
    if not (abs(a) <= MAX_PLATT and abs(b) <= MAX_PLATT):
        return 1.0, 0.0
    return a, b


class Tuning:
    def __init__(
        self,
        *,
        model: str,
        design: str,
        tags: list[str],
        thresholds: dict[str, float],
        platt: dict[str, tuple[float, float]],
        relevance: dict,
        fitted_on: dict[str, int],
        human_weight: int = 1,
        created: str = "",
    ):
        self.model, self.design, self.tags = model, design, list(tags)
        self.thresholds, self.platt, self.relevance = thresholds, platt, relevance
        self.fitted_on, self.human_weight, self.created = fitted_on, human_weight, created

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls, table: pd.DataFrame, tags: list[str], design: str, model: str = jps.MODEL
    ) -> Tuning:
        """Fit on `table` (see `training_table`); each row is repeated `weight` times."""
        if len(table) < 10:
            raise TuningError(f"only {len(table)} labelled papers; need at least 10 to tune")
        rep = table.loc[table.index.repeat(table.weight.astype(int))].reset_index(drop=True)
        p = pd.DataFrame(rep[[f"p_{t}" for t in tags]].to_numpy(float), columns=tags)
        truth = pd.DataFrame(jtt.tag_matrix(rep.assign(x=rep.label_tags), "x", tags), columns=tags)
        relevance = jtt.fit_relevance_map(rep.assign(sonnet_relevance=rep.label_relevance))
        by_source = table.groupby("source").size().to_dict()
        human = table[table.source == "human"]
        return cls(
            model=model,
            design=design,
            tags=tags,
            thresholds=jtt.fit_thresholds(p, truth),
            platt={t: stable_platt(p[t], truth[t]) for t in tags},
            relevance=relevance,
            fitted_on={"sonnet": int(by_source.get("sonnet", 0)), "human": int(len(human))},
            human_weight=int(human.weight.iloc[0]) if len(human) else 1,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    # -- saving -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "design_hash": self.design,
            "created": self.created,
            "fitted_on": self.fitted_on,
            "human_weight": self.human_weight,
            "tags": {
                t: {
                    "threshold": self.thresholds[t],
                    "platt_a": self.platt[t][0],
                    "platt_b": self.platt[t][1],
                }
                for t in self.tags
            },
            "relevance": {"features": list(jtt.FEATURES)} | self.relevance,
        }

    def save(self, path: Path) -> None:
        atomic_json(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> Tuning:
        tags = list(d["tags"])
        rel = {k: v for k, v in d["relevance"].items() if k != "features"}
        if list(d["relevance"]["features"]) != list(jtt.FEATURES):
            raise TuningError("tuning was fitted on different relevance features")
        return cls(
            model=d["model"],
            design=d["design_hash"],
            tags=tags,
            thresholds={t: float(d["tags"][t]["threshold"]) for t in tags},
            platt={
                t: (float(d["tags"][t]["platt_a"]), float(d["tags"][t]["platt_b"])) for t in tags
            },
            relevance=rel,
            fitted_on=d["fitted_on"],
            human_weight=d.get("human_weight", 1),
            created=d["created"],
        )

    # -- applying -----------------------------------------------------------

    def check(self, design: str, model: str = jps.MODEL) -> None:
        """Refuse to apply this tuning to a different question design or Jev model."""
        if self.design != design:
            raise TuningError(
                f"tuning was fitted for Jev question design {self.design}, current is {design}; "
                "refit before applying"
            )
        if self.model != model:
            raise TuningError(f"tuning was fitted for model {self.model}, current is {model}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """New columns: jev_relevance_raw, jev_relevance (tuned), jev_high, p_<tag> calibrated,
        jev_tags (raw probabilities at the tuned thresholds). `df` needs the `jev_features`
        columns; its own `p_<tag>` are replaced by calibrated ones.
        """
        raw = df[[f"p_{t}" for t in self.tags]].to_numpy(float)
        tuned = jtt.apply_relevance_map(self.relevance, df)
        thr = np.array([self.thresholds[t] for t in self.tags])
        on = raw >= thr
        out = df.copy()
        out["jev_relevance_raw"] = df.jev_relevance.to_numpy(float)
        out["jev_relevance"] = tuned
        out["jev_high"] = tuned >= self.relevance["high_cut"]
        for i, t in enumerate(self.tags):
            out[f"raw_p_{t}"] = raw[:, i]
            out[f"p_{t}"] = jtt.apply_platt(raw[:, i], self.platt[t])
        out["jev_tags"] = [
            ";".join(t for t, v in zip(self.tags, row, strict=True) if v) for row in on
        ]
        return out


def load_tuning(path: Path, design: str, model: str = jps.MODEL) -> Tuning:
    """Read tuning.json and refuse it unless its design hash and model match the current ones."""
    tuning = Tuning.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    tuning.check(design, model)
    return tuning


# ---------------------------------------------------------------------------
# CLI


def current_design(tags: list[str] | None = None) -> tuple[list[str], str]:
    cfg = jps.load_config()
    tags = list(cfg["taxonomy"])
    return tags, jps.design_hash(jps.build_questions(cfg, jps.load_definitions(tags)))


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--teacher", type=Path, required=True, help="teacher_cache.jsonl")
    ap.add_argument("--jev-cache", type=Path, required=True, help="Jev cache.jsonl(.gz)")
    ap.add_argument("--out", type=Path, required=True, help="tuning.json to write")
    a = ap.parse_args(argv)
    tags, design = current_design()
    cache = load_jev_cache(a.jev_cache, design)
    labels = load_teacher_labels(a.teacher)
    table = training_table(jev_features(cache, tags), labels, "sonnet")
    if len(table) < len(labels):
        print(f"note: {len(labels) - len(table)} labelled papers have no Jev answer; skipped")
    tuning = Tuning.fit(table, tags, design)
    tuning.save(a.out)
    print(f"fitted on {tuning.fitted_on}; wrote {a.out}")
    return a.out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
