"""The optional `[paper_monitor]` section: off by default, read-only when set.

Synthetic folders only; nothing here talks to a broker or an account.
"""

import json
from types import SimpleNamespace

from test_autopilot_ops import SECTIONS, make_config, tick, toy

from capstone.pipeline.autopilot import ops
from capstone.pipeline.autopilot.ops import common, paper_monitor

ALERT = {"severity": "stopped", "at": "2026-01-02T00:00:00Z", "line": "paper: halted"}


def test_the_section_is_declared_and_discovered_and_defaults_to_nothing(tmp_path):
    sections, _ = ops.discover()
    assert list(sections["paper_monitor"]) == ["root"]
    assert "paper_monitor" in SECTIONS
    toys = [toy("a")]
    config = make_config(tmp_path, toys)
    assert paper_monitor.status_fields(config) == {}
    assert paper_monitor.root(SimpleNamespace(config=config)) is None


def test_status_is_unchanged_without_a_root_and_shows_the_paper_alert_with_one(tmp_path):
    toys = [toy("a")]
    plain = make_config(tmp_path / "plain", toys)
    tick(plain, toys)
    plain_doc = json.loads((common.ops_dir(plain.run_root) / "status.json").read_text("utf-8"))
    assert "paper_alert" not in plain_doc

    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "ALERT.json").write_text(json.dumps(ALERT), encoding="utf-8")
    config = make_config(tmp_path / "with", toys, paper_monitor={"root": str(paper)})
    tick(config, toys)
    folder = common.ops_dir(config.run_root)
    doc = json.loads((folder / "status.json").read_text("utf-8"))
    assert doc["paper_alert"] == ALERT
    assert "Last paper alert (stopped, 2026-01-02T00:00:00Z): paper: halted" in (
        folder / "status.md"
    ).read_text("utf-8")
    assert doc["ok"] is True  # a paper alert is shown, never counted as this run's attention


def test_a_missing_or_unreadable_paper_alert_is_none_not_an_error(tmp_path):
    toys = [toy("a")]
    paper = tmp_path / "paper"
    paper.mkdir()
    config = make_config(tmp_path, toys, paper_monitor={"root": str(paper)})
    assert paper_monitor.status_fields(config) == {"paper_alert": None}
    (paper / "ALERT.json").write_text("{torn", encoding="utf-8")
    assert paper_monitor.status_fields(config) == {"paper_alert": None}


def test_reading_the_paper_folder_never_writes_into_it(tmp_path):
    toys = [toy("a")]
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "ALERT.json").write_text(json.dumps(ALERT), encoding="utf-8")
    config = make_config(tmp_path, toys, paper_monitor={"root": str(paper)})
    tick(config, toys)
    assert sorted(p.name for p in paper.iterdir()) == ["ALERT.json"]
