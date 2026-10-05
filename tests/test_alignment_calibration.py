"""The alignment calibration kit: score a store into a labelling sheet, then report.

A scripted fake model stands in for the judge; no API calls, no market data.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from capstone.factors import store
from capstone.factors.llm import FactorConfig
from capstone.factors.tree import parse

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "alignment_calibration.py"
_spec = importlib.util.spec_from_file_location("alignment_calibration", SCRIPT)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

CONFIG = FactorConfig(alignment_model="judge-model")


class Judge:
    """Answers each judge call with the next scripted verdict."""

    def __init__(self, *verdicts: dict):
        self.verdicts = list(verdicts)
        self.messages = self

    def create(self, **kwargs):
        tool = kwargs["tools"][0]["name"]
        block = SimpleNamespace(type="tool_use", name=tool, input=self.verdicts.pop(0))
        return SimpleNamespace(content=[block])


def _verdict(**fails):
    checks = {check: check not in fails for check in ("variables", "direction", "horizon")}
    return {**checks, "reason": "r"}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "factors.db"
    con = store.connect(path)
    store.add_paper(con, "p", "Paper")
    hid = store.add_hypothesis(
        con,
        store.Hypothesis(
            paper_key="p",
            observation="Losers rebound.",
            knowledge="k",
            justification="j",
            specification="Minus the 5-day return.",
            falsification_condition="f",
            kind="market",
        ),
    )
    store.add_factor(con, parse("-ts_sum(returns, 5)"), hid, rationale="reversal")
    store.add_factor(con, parse("ts_sum(returns, 5) * log(cap)"), hid, rationale="sign?")
    store.reject(con, hid, "ts_sum(returns,", "parse error: unexpected end")
    con.close()
    return path


def test_score_writes_one_row_per_parsed_proposal_and_leaves_the_store_alone(db, tmp_path):
    before = db.read_bytes()
    out = tmp_path / "sheet.csv"
    usage = cal.score(db, out, Judge(_verdict(), _verdict(direction=True)), CONFIG)
    assert db.read_bytes() == before
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and "log(cap)" in rows[1]["expression"]  # parse error left out
    assert rows[0]["specification"] == "Minus the 5-day return."
    assert (rows[0]["score"], rows[1]["score"], rows[1]["direction"]) == ("1.0", "0.6667", "0")
    assert {r["label"] for r in rows} == {""}
    assert {r["judge_model"] for r in rows} == {"judge-model"}
    assert usage["alignment_calls"] == 2


def test_score_needs_a_judge_model(db, tmp_path):
    with pytest.raises(SystemExit, match="alignment_model"):
        cal.score(db, tmp_path / "s.csv", Judge(), FactorConfig())


def _sheet(tmp_path, rows):
    path = tmp_path / "labelled.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cal.COLUMNS)
        writer.writeheader()
        for score, label in rows:
            writer.writerow({"score": score, "label": label})
    return path


def test_report_counts_catches_and_false_rejections_at_each_threshold(tmp_path):
    # Two misaligned (scores 1/3, 2/3), three aligned (one wrongly at 2/3), one unlabelled.
    sheet = _sheet(
        tmp_path,
        [(0.3333, "0"), (0.6667, "0"), (1.0, "1"), (1.0, "1"), (0.6667, "1"), (1.0, "")],
    )
    table, labelled = cal.report(sheet)
    assert labelled == 5
    by_threshold = {round(row["min_alignment"], 2): row for row in table}
    assert by_threshold[0.33]["misaligned_caught"] == "0/2"  # 1/3 passes at 1/3
    assert by_threshold[0.67]["misaligned_caught"] == "1/2"
    assert by_threshold[1.0]["misaligned_caught"] == "2/2"
    assert by_threshold[1.0]["aligned_rejected"] == "1/3"
    assert by_threshold[1.0]["agreement"] == pytest.approx(4 / 5)


def test_report_warns_below_thirty_labels(tmp_path, capsys):
    sheet = _sheet(tmp_path, [(1.0, "1"), (0.3333, "0")])
    assert cal.main(["report", "--sheet", str(sheet)]) == 0
    captured = capsys.readouterr()
    assert "label at least 30" in captured.err and "misaligned caught" in captured.out
