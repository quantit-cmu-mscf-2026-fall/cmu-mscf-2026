"""The variant grid on worker processes (`capstone.pipeline.panel_pool`).

`evaluate_factors` at one worker and at four writes the same ledger, store rows and frames,
a failing worker surfaces as a typed error naming the factor, and `[resources]` sizes the
pool without entering the config digest. Synthetic rows only.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_factor_evaluation import fake_rows

from capstone import runlog
from capstone.factors import evaluation as ev
from capstone.factors import store
from capstone.factors.evaluation import (
    FactorOutcome,
    TradingSpec,
    build_panel,
    evaluate_factors,
    factor_outcome,
)
from capstone.factors.tree import Field_, parse
from capstone.pipeline import executor, panel_pool
from capstone.pipeline.config import load_config
from capstone.pipeline.rows import REPO
from capstone.pipeline.search_identity import config_digest

MANY = 4
SPECS = (TradingSpec(hold=1), TradingSpec(hold=5), TradingSpec(hold=1, neutral="sector"))
EXPRESSIONS = ["volume / shares", "-ts_sum(returns, 5)", "rank(volume) - rank(ts_mean(volume, 10))"]
VOLATILE = ("ts_utc", "git_sha", "session_id", "user")


def _panel(n_stocks=6):
    return build_panel(fake_rows(n_stocks=n_stocks, n_days=120), "fixture")


def _dump(path: Path) -> dict:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {
            t: [
                {k: r[k] for k in r.keys() if k not in ("created_at", "added_at")}
                for r in con.execute(f'SELECT * FROM "{t}" ORDER BY rowid')
            ]
            for t in sorted(tables)
        }
    finally:
        con.close()


def _evaluate(base: Path, workers: int):
    """The grid with its ledger, store and frames, on a pool of `workers`."""
    panel = _panel()
    base.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CAPSTONE_LEDGER_DIR", str(base / "ledger"))
        con = store.connect(base / "factors.db")
        store.add_paper(con, "p", "Paper")
        hid = store.add_hypothesis(con, store.Hypothesis("p", "o", "k", "j", "s", "f"))
        trees = {store.add_factor(con, parse(e), hid)[0]: parse(e) for e in EXPRESSIONS}
        try:
            with panel_pool.PanelPool(panel, workers, base) as pool:
                assert pool.parallel == (workers > executor.INLINE)
                matrix, summary = evaluate_factors(
                    trees, panel, specs=SPECS, con=con, outcomes=pool.outcomes(SPECS)
                )
        finally:
            con.close()
        entries = [{k: v for k, v in e.items() if k not in VOLATILE} for e in runlog.read_entries()]
    assert not (base / panel_pool.STAGING).exists()  # the staged panel is removed
    matrix.to_parquet(base / "matrix.parquet")
    summary.to_parquet(base / "summary.parquet")
    return entries, _dump(base / "factors.db"), base


def test_the_grid_is_byte_identical_at_one_and_four_workers(tmp_path):
    one, store_one, dir_one = _evaluate(tmp_path / "one", executor.INLINE)
    many, store_many, dir_many = _evaluate(tmp_path / "many", MANY)
    assert len(one) == len(EXPRESSIONS) * len(SPECS)  # every variant is a logged trial
    assert one == many
    assert store_one == store_many and store_one["variants"]
    for name in ("matrix.parquet", "summary.parquet"):
        assert (dir_one / name).read_bytes() == (dir_many / name).read_bytes()


def test_the_pool_gives_the_sequential_loops_frames(tmp_path, monkeypatch):
    panel = _panel()
    trees = {f"f{i}": parse(e) for i, e in enumerate(EXPRESSIONS)}
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "l1"))
    m1, s1 = evaluate_factors(trees, panel, specs=SPECS)
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "l2"))
    with panel_pool.PanelPool(panel, MANY, tmp_path) as pool:
        m2, s2 = evaluate_factors(trees, panel, specs=SPECS, outcomes=pool.outcomes(SPECS))
    assert m1.equals(m2) and s1.equals(s2)


def test_factor_outcome_is_pure_and_matches_the_logged_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = _panel()
    tree = parse("volume / shares")
    outcome = factor_outcome(tree, panel, SPECS)
    assert isinstance(outcome, FactorOutcome) and outcome.error is None
    assert len(outcome.results) == len(SPECS) and outcome.refused is None
    assert runlog.read_entries() == []  # nothing was logged by computing it
    evaluate_factors({"a": tree}, panel, specs=SPECS)
    logged = [e["metrics"] for e in runlog.read_entries()]
    assert logged == [metrics for metrics, _ in outcome.results]


def test_a_backtest_refusal_is_logged_then_raised_as_the_sequential_loop_did(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    real = ev._backtest
    calls = []

    def refusing(traded, panel, spec):
        calls.append(spec)
        if len(calls) == 2:
            raise ValueError("no cost rate on a trading day")
        return real(traded, panel, spec)

    monkeypatch.setattr(ev, "_backtest", refusing)
    with pytest.raises(ValueError, match="no cost rate"):
        evaluate_factors({"a": parse("volume / shares")}, _panel(), specs=SPECS)
    entries = runlog.read_entries()
    assert [("error" in e["metrics"]) for e in entries] == [False, True]  # the refused spec counts


def test_resources_are_read_from_their_section_and_never_enter_the_digest(tmp_path):
    for name in ("e2e_synthetic.toml", "e2e_crsp.toml"):
        config = load_config(REPO / "config" / name)
        assert config.resources == executor.Resources()
        digest = config_digest(config)
        assert config_digest(replace(config, resources=executor.Resources(3, 512))) == digest
    text = (REPO / "config" / "e2e_synthetic.toml").read_text(encoding="utf-8")
    path = tmp_path / "c.toml"
    path.write_text(text + "\n[resources]\nmax_workers = 3\nmemory_budget_mb = 512\n", "utf-8")
    assert load_config(path).resources == executor.Resources(3, 512)
    path.write_text(text + "\n[resources]\nworkers = 3\n", "utf-8")
    with pytest.raises(ValueError, match="resources"):
        load_config(path)


def test_the_worker_count_follows_resources_and_the_environment_cap(monkeypatch):
    panel = _panel()
    monkeypatch.delenv(executor.ENV_CAP, raising=False)
    assert panel_pool.workers(1, panel, executor.Resources(max_workers=MANY)) == executor.INLINE
    tasks = os.cpu_count() or MANY
    cap = min(MANY, executor.detect().cores)
    assert panel_pool.workers(tasks + MANY, panel, executor.Resources(max_workers=MANY)) == cap
    monkeypatch.setenv(executor.ENV_CAP, str(executor.INLINE))
    assert panel_pool.workers(tasks + MANY, panel, executor.Resources()) == executor.INLINE


# ---------------------------------------------------------------------------
# A failing worker


class NotATree:
    """Not a factor tree: evaluating it raises TypeError (not a domain error)."""


class WorkerFault(Field_):
    """A valid tree here; in a worker it unpickles as `NotATree`, so only the worker's
    evaluation fails (with an error the sequential loop would raise, too)."""

    def __reduce__(self):
        return (NotATree, ())


class _Crash:
    """Kills the worker process that unpickles it (a crashed worker)."""

    def __reduce__(self):
        return (os._exit, (3,))


def test_an_error_in_a_worker_surfaces_as_a_typed_error_naming_the_factor(tmp_path):
    panel = _panel()
    trees = {"good": parse("volume / shares"), "bad": WorkerFault("volume")}
    with panel_pool.PanelPool(panel, 2, tmp_path) as pool:
        assert pool.parallel
        with pytest.raises(panel_pool.EvaluationWorkerError, match="bad") as info:
            evaluate_factors(trees, panel, specs=SPECS[:2], outcomes=pool.outcomes(SPECS[:2]))
    assert info.value.__cause__ is not None
    assert not (tmp_path / panel_pool.STAGING).exists()


def test_a_crashed_worker_surfaces_as_a_typed_error(tmp_path):
    panel = _panel()
    trees = {"a": parse("volume / shares"), "b": _Crash()}
    with panel_pool.PanelPool(panel, 2, tmp_path) as pool:
        with pytest.raises(panel_pool.EvaluationWorkerError):
            evaluate_factors(trees, panel, specs=SPECS[:2], outcomes=pool.outcomes(SPECS[:2]))


def test_outcomes_out_of_order_are_refused():
    panel = _panel()
    trees = {"a": parse("volume / shares"), "b": parse("-ts_sum(returns, 5)")}

    def shuffled(items):
        return reversed([(fid, factor_outcome(tree, panel, SPECS)) for fid, tree in items])

    with pytest.raises(RuntimeError, match="out of order"):
        evaluate_factors(trees, panel, specs=SPECS, outcomes=shuffled)
