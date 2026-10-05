"""Calibrate the alignment judge on hand labels before it rejects anything.

QUANTIT-83: the judge's threshold (`min_alignment` in config/factors.toml) is
fixed from hand-labelled proposals, never tuned on factor performance.

    python scripts/alignment_calibration.py score [--db PATH] [--out CSV]
    python scripts/alignment_calibration.py report [--sheet CSV]

`score` runs the judge (`alignment_model`) on every proposal in a factor
store that parsed, and writes a labelling sheet. The store is opened
read-only and never changed. It calls the model API, so it costs money (Haiku:
a few cents for a few dozen proposals); credentials come from the
environment, as for the pipeline.

Then fill in the `label` column by hand for at least 30 rows: 1 if the formula
implements its hypothesis (right variables, direction and horizon), 0 if not.
Label from the hypothesis and the formula alone, before looking at the judge's
columns.

`report` compares the judge with the labels at each possible threshold and
prints how many misaligned formulas each would catch and how many aligned
ones it would wrongly reject. Choose `min_alignment` from that table.

The sheet holds working data and is gitignored. Neither step sees market data
or performance, so neither is a ledger trial.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from capstone.factors import alignment
from capstone.factors.hypotheses import _FIELDS
from capstone.factors.llm import FactorConfig, load_config, make_client

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SHEET = REPO / "experiments" / "alignment_labels.csv"
COLUMNS = [
    "proposal_id",
    "paper_key",
    *_FIELDS,
    "expression",
    "rationale",
    "status",
    "label",
    "label_note",
    *alignment.CHECKS,
    "score",
    "judge_reason",
    "judge_model",
]
N_CHECKS = len(alignment.CHECKS)
MIN_LABELS = 30


def _proposals(db: Path) -> list[dict]:
    """Every proposal that parsed, with its hypothesis, read without changing the store."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    fields = ", ".join(f"h.{name}" for name in _FIELDS)
    rows = con.execute(
        f"SELECT p.id AS proposal_id, h.paper_key, {fields}, p.expression, p.rationale, "
        "p.status FROM proposals p JOIN hypotheses h ON h.id = p.hypothesis_id "
        "WHERE h.kind = 'market' AND (p.status != 'rejected' OR p.reason NOT LIKE 'parse error%') "
        "ORDER BY p.id"
    ).fetchall()
    con.close()
    return [dict(row) for row in rows]


def score(db: Path, out: Path, client, config: FactorConfig) -> Counter:
    """Judge every parsed proposal in `db` and write the labelling sheet to `out`."""
    if not config.alignment_model:
        raise SystemExit("alignment_model is empty in the config; nothing to calibrate")
    usage: Counter = Counter()
    rows = []
    for proposal in _proposals(db):
        verdict = alignment.judge(
            client,
            config,
            proposal,
            proposal["expression"],
            proposal["rationale"] or "",
            usage,
        )
        rows.append(
            {
                **proposal,
                "label": "",
                "label_note": "",
                **{check: int(verdict.checks[check]) for check in alignment.CHECKS},
                "score": round(verdict.score, 4),
                "judge_reason": verdict.reason,
                "judge_model": verdict.model,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return usage


def report(sheet: Path) -> tuple[list[dict], int]:
    """Agreement between the judge and the hand labels at each threshold, and the label count."""
    with open(sheet, newline="", encoding="utf-8") as handle:
        labelled = [row for row in csv.DictReader(handle) if row["label"].strip() in {"0", "1"}]

    # Scores are a count of passed checks out of N_CHECKS; compare counts, not
    # the rounded decimals in the sheet.
    def passed(row: dict) -> int:
        return round(float(row["score"]) * N_CHECKS)

    aligned = [passed(r) for r in labelled if r["label"].strip() == "1"]
    misaligned = [passed(r) for r in labelled if r["label"].strip() == "0"]
    table = []
    for needed in range(1, N_CHECKS + 1):
        caught = sum(p < needed for p in misaligned)
        wrongly = sum(p < needed for p in aligned)
        table.append(
            {
                "min_alignment": needed / N_CHECKS,
                "misaligned_caught": f"{caught}/{len(misaligned)}",
                "aligned_rejected": f"{wrongly}/{len(aligned)}",
                "agreement": (caught + len(aligned) - wrongly) / len(labelled)
                if labelled
                else float("nan"),
            }
        )
    return table, len(labelled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="judge every proposal and write the labelling sheet")
    s.add_argument("--db", type=Path, help="factor store (default: the configured store)")
    s.add_argument("--out", type=Path, default=DEFAULT_SHEET)
    s.add_argument("--config", type=Path, help="parameters file (default: config/factors.toml)")
    r = sub.add_parser("report", help="compare the judge with the hand labels")
    r.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    args = parser.parse_args(argv)

    if args.command == "score":
        from capstone.factors import store

        db = args.db or store.default_path()
        usage = score(db, args.out, make_client(), load_config(args.config))
        print(
            f"wrote {args.out}; judge calls {usage['alignment_calls']}, tokens in "
            f"{usage['alignment_input_tokens']}, out {usage['alignment_output_tokens']}"
        )
        return 0

    table, labelled = report(args.sheet)
    if labelled < MIN_LABELS:
        print(f"only {labelled} labelled rows; label at least {MIN_LABELS} first", file=sys.stderr)
    print("min_alignment  misaligned caught  aligned rejected  agreement")
    for row in table:
        print(
            f"{row['min_alignment']:<14.2f} {row['misaligned_caught']:<18} "
            f"{row['aligned_rejected']:<17} {row['agreement']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
