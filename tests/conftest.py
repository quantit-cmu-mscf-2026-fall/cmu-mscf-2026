"""Checkpoint reports: tests marked `report` run only with --report.

They write what a checkpoint shows for review to experiments/reports/, which
is gitignored. The normal run, ``pytest -q -m "not network"``, skips them.

    pytest -m report --report
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "reports"


def pytest_addoption(parser):
    parser.addoption("--report", action="store_true", help="run the checkpoint reports")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--report"):
        return
    skip = pytest.mark.skip(reason="checkpoint report: run with --report")
    for item in items:
        if "report" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR
