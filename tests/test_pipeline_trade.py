"""`pipeline.trade.accept`: accepted families to registered strategies, books and monitors.

Synthetic CRSP-shaped rows and a hand-built gate result; the ledger goes to a temp
folder. Nothing is enabled, deployed or read from shared data.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_strategy_contract import EXPRESSION, fake_rows

from capstone import runlog
from capstone.factors.evaluation import build_panel
from capstone.pipeline import trade
from capstone.pipeline.config import IncubationSettings
from capstone.scorecard import GateResult
from capstone.strategy.registry import INCUBATION_ONLY, read_lifecycle, read_registry

REPORT = "reports/gate.json"


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    (root / "reports").mkdir(parents=True)
    trade.prepare_registry(root)
    (root / REPORT).write_text("{}", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def panel():
    return build_panel(fake_rows(n_stocks=20, n_days=80), "synthetic-v1")


@pytest.fixture
def inputs(panel):
    dates = panel.universe.index
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame({"v1": rng.normal(0.001, 0.01, len(dates))}, index=dates)
    summary = pd.DataFrame(
        {"expression": [EXPRESSION], "hold": [5], "neutral": ["sector"]}, index=["v1"]
    )
    card = pd.DataFrame(
        {"accepted": [True], "best": ["v1"], "expected_mean": [0.001], "scale": [0.01]},
        index=["fam"],
    )
    return matrix, summary, card


def config(**incubation):
    return SimpleNamespace(
        run_id="r1",
        seed=7,
        gross=1.0,
        min_trade=0.0,
        incubation=IncubationSettings(horizon=40, confirm_horizon=60, block=5, **incubation),
    )


def run(workdir, panel, inputs, *, result=None, **incubation):
    matrix, summary, card = inputs
    result = result or GateResult(card, None, ("fam",))
    return trade.accept(
        config(**incubation),
        workdir,
        result,
        matrix,
        summary,
        panel,
        workdir / REPORT,
        "synthetic-v1",
    )


def test_an_accepted_family_gets_a_disabled_entry_a_book_and_a_logged_monitor(
    workdir, panel, inputs
):
    strategies, incubations = run(workdir, panel, inputs)
    (outcome,) = strategies
    assert outcome.family == "fam" and outcome.role == "deploy"
    assert outcome.entry.enabled is False and outcome.entry.gate_report == REPORT
    assert read_registry(workdir) == (outcome.entry,)
    assert outcome.as_of == panel.universe.index[-1]
    assert outcome.weights.abs().sum() == pytest.approx(1.0)
    # from an empty book the trades are the book itself
    assert outcome.orders.trades.round(12).equals(outcome.weights.round(12))
    assert outcome.digest == trade.book_digest(outcome.weights, outcome.orders)
    (inc,) = incubations
    assert inc.confirm is None  # decay monitor only, at the search horizon
    assert inc.config.horizon == 40
    (entry,) = runlog.read_entries()
    assert entry["name"] == trade.INCUBATION_NAME
    assert set(entry["tags"]) == {"incubation", "synthetic"}
    assert entry["params"]["data_version"] == "synthetic-v1"
    assert "role" not in entry["params"]


def test_everything_is_written_under_the_workdir(workdir, panel, inputs, tmp_path):
    run(workdir, panel, inputs)
    written = {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"work", "ledger"}


def test_a_gate_result_without_incubate_families_yields_only_accepted_ones(workdir, panel, inputs):
    matrix, summary, card = inputs
    result = GateResult(card, None, ("fam",))
    assert not hasattr(result, "incubate")  # the gate labels no family incubate here
    strategies, _ = run(workdir, panel, inputs, result=result)
    assert [s.role for s in strategies] == ["deploy"]
    assert {e.role for e in read_registry(workdir)} == {"deploy"}
    assert all(e["params"].get("role") is None for e in runlog.read_entries())


def test_an_empty_incubate_field_is_inert_too(workdir, panel, inputs):
    matrix, summary, card = inputs
    result = SimpleNamespace(scorecard=card, accepted=("fam",), incubate=())
    strategies, incubations = run(workdir, panel, inputs, result=result)
    assert [s.role for s in strategies] == ["deploy"] and len(incubations) == 1


def test_nothing_accepted_registers_nothing(workdir, panel, inputs):
    matrix, summary, card = inputs
    result = SimpleNamespace(scorecard=card, accepted=(), incubate=())
    assert run(workdir, panel, inputs, result=result) == ((), ())
    assert read_registry(workdir) == ()
    assert runlog.read_entries() == []


def test_a_populated_incubate_family_is_incubation_only_disabled_and_confirming(
    workdir, panel, inputs
):
    matrix, summary, card = inputs
    result = SimpleNamespace(scorecard=card, accepted=(), incubate=("fam",))
    (outcome,), (inc,) = run(workdir, panel, inputs, result=result)
    assert outcome.role == INCUBATION_ONLY and outcome.entry.role == INCUBATION_ONLY
    assert outcome.entry.enabled is False
    assert inc.confirm is not None and inc.config.horizon == 60
    (event,) = read_lifecycle(workdir)
    assert (event["field"], event["role"]) == ("registered", INCUBATION_ONLY)
    (entry,) = runlog.read_entries()
    assert entry["params"]["role"] == INCUBATION_ONLY
    assert "confirm_e_value" in entry["metrics"]


def test_confirms_only_for_incubation_only_or_when_asked():
    default, asked = IncubationSettings(), IncubationSettings(confirm_accepted=True)
    assert not trade.confirms(default, "deploy")
    assert trade.confirms(asked, "deploy")
    assert trade.confirms(default, INCUBATION_ONLY)


def test_the_digest_ignores_float_noise_and_negative_zero():
    a = pd.Series({1: 0.5, 2: -0.5})
    book = SimpleNamespace(trades=pd.Series({1: 0.5, 2: -0.0}))
    noisy = SimpleNamespace(trades=pd.Series({1: 0.5 + 1e-15, 2: 0.0}))
    assert trade.book_digest(a, book) == trade.book_digest(a + 1e-15, noisy)
    assert trade.book_digest(a, book) != trade.book_digest(a * 1.1, book)


def test_prepare_registry_keeps_an_existing_file(workdir):
    path = workdir / "deploy" / "strategies.yml"
    before = path.read_bytes()
    trade.prepare_registry(workdir)
    assert path.read_bytes() == before


def test_the_module_does_not_import_stage_three_or_spanning():
    tree = ast.parse(Path(trade.__file__).read_text(encoding="utf-8"))
    names = [
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (getattr(n, "names", None) or [None])
    ]
    assert not [m for m in names if m and ("stage3" in m or "spanning" in m)]
