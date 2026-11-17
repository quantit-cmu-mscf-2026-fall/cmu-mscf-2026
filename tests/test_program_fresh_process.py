"""A program's targets reproduce in a fresh process.

Synthetic seeded panels only; no data, no model, no financial claim.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import capstone
from capstone.factors.evaluation import Panel
from capstone.strategy.contract import save_spec
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.schedule_dispatch import decision_due

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fresh_process_targets.py"


def script():
    loader = importlib.util.spec_from_file_location("fresh_process_targets", SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def synthetic_panel(seed, dates, names, data_version):
    rng = np.random.default_rng(seed)
    columns = pd.Index(names, name="permno")
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(dates), len(names))), axis=0)),
        index=dates,
        columns=columns,
    )
    cap = close * rng.uniform(1e3, 1e4, len(names))
    fields = {"close": close, "cap": cap, "returns": close.pct_change(fill_method=None)}
    return Panel(
        fields,
        pd.DataFrame(True, index=dates, columns=columns),
        cap * 0,
        data_version=data_version,
        calendar=dates,
        availability={name: pd.Series(dates, index=dates) for name in fields},
    )


def program_case(tmp_path):
    dates = pd.bdate_range("2020-03-02", periods=60)
    panel = synthetic_panel(11, dates, [101, 102, 103, 104, 105, 106, 107, 108], "fresh-v1")
    spec = ProgramStrategySpec(
        expression="rank(delta(close, 5))",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="fresh/program",
        data_version=panel.data_version,
        anchor=str(dates[10].date()),
        side="long_short",
        tail=0.25,
        weighting="capitalization",
        schedule=ProgramSchedule(interval=10),
    )
    path = tmp_path / "program.toml"
    save_spec(spec, path)
    # Formations at rows 10, 20, 30, 40; row 45 holds the row-40 book; 2020-04-04 is a Saturday.
    return spec, path, panel, dates[40], dates[45], "2020-04-04"


@pytest.fixture
def case(tmp_path):
    return program_case(tmp_path)


def weights(text):
    return json.loads(text)["weights"]


def test_fresh_process_targets_are_byte_identical(case, tmp_path):
    spec, path, panel, at, *_ = case
    tool = script()
    expected = tool.render(spec, panel, at)
    assert weights(expected), "a nonempty book makes the identity meaningful"
    panel_path = tmp_path / "panel.json"
    tool.write_panel(panel, panel_path)
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(capstone.__file__).resolve().parents[1]),
        "CAPSTONE_LEDGER_DIR": str(tmp_path / "ledger"),
    }
    # The fresh process must import the same capstone, not another installed copy.
    imported = subprocess.run(
        [sys.executable, "-c", "import capstone; print(capstone.__file__)"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert Path(imported.stdout.strip()).resolve() == Path(capstone.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), str(panel_path), str(at.date())],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    printed, checksum = result.stdout.splitlines()
    assert printed == expected
    assert checksum == hashlib.sha256(expected.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("axis", ["columns", "rows"])
def test_row_and_column_permutations_leave_targets_identical(case, axis):
    spec, _, panel, at, *_ = case
    tool = script()
    expected = tool.render(spec, panel, at)
    rng = np.random.default_rng(3)
    rows = np.arange(len(panel.universe.index))
    columns = np.arange(len(panel.universe.columns))
    if axis == "rows":
        rows = rng.permutation(rows)
    else:
        columns = rng.permutation(columns)

    def shuffle(frame):
        return frame.iloc[rows, columns] if frame.ndim == 2 else frame.iloc[rows]

    panel.fields = {name: shuffle(frame) for name, frame in panel.fields.items()}
    panel.availability = {name: values.iloc[rows] for name, values in panel.availability.items()}
    panel.universe = shuffle(panel.universe)
    panel.half_spread = shuffle(panel.half_spread)
    panel.calendar = panel.calendar[rows]
    assert tool.render(spec, panel, at) == expected


def test_future_perturbation_leaves_targets_identical(case):
    spec, _, panel, at, *_ = case
    tool = script()
    expected = tool.render(spec, panel, at)
    rng = np.random.default_rng(9)
    for frame in panel.fields.values():
        later = frame.index > at
        frame.loc[later] = rng.normal(0, 1e6, frame.loc[later].shape)
    panel.universe.loc[panel.universe.index > at] = False
    assert tool.render(spec, panel, at) == expected


def test_off_schedule_decision_holds_last_formation_book(case):
    spec, _, panel, at, held, weekend = case
    tool = script()
    assert decision_due(spec, panel, at) and not decision_due(spec, panel, held)
    assert weights(tool.render(spec, panel, held)) == weights(tool.render(spec, panel, at))
    with pytest.raises(ValueError, match="calendar must contain"):
        tool.render(spec, panel, weekend)
