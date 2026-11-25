"""The driver's search steps (`capstone.pipeline.e2e.search`): papers, deepen and evaluation.

A scripted model client, a scripted deepen editor and synthetic rows; a temporary
ledger. No model, no network, no market data.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_deepen import ScriptedEditor
from test_factor_evaluation import fake_rows

from capstone import runlog
from capstone.factors import store
from capstone.factors.evaluation import build_panel
from capstone.pipeline import e2e
from capstone.pipeline.budget import BudgetedClient, CallBudgetExceeded
from capstone.pipeline.config import load_config
from capstone.pipeline.search_identity import config_digest

VOLATILE = ("ts_utc", "git_sha", "session_id", "user")
FACTORS = {
    "alpha": ["rank(volume) - rank(ts_mean(volume, 10))", "-ts_sum(returns, 5)"],
    "beta": ["ts_mean(close, 7) / close", "ts_std(returns, 20)"],
}
EDITS = [[f"ts_mean(close, {3 + 2 * i})", f"rank(volume) * {i + 2}"] for i in range(10)]


def _hypothesis(title: str) -> dict:
    return {
        "observation": f"{title}: observation",
        "knowledge": "k",
        "justification": "j",
        "specification": f"spec {title}",
        "falsification_condition": "f",
        "kind": "market",
    }


class ScriptedClient:
    """One hypothesis per paper and two factors per hypothesis, keyed by the paper's title."""

    def __init__(self):
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        tool = kwargs["tools"][0]["name"]
        user = kwargs["messages"][0]["content"]
        if tool == "record_hypotheses":
            title = json.loads(user)["title"]
            if "FAIL" in title:
                raise RuntimeError("model unavailable")
            answer = {"hypotheses": [_hypothesis(title)]}
        else:
            title = next(t for t in FACTORS if f"spec {t}" in user)
            answer = {"factors": [{"expression": e, "rationale": "r"} for e in FACTORS[title]]}
        block = SimpleNamespace(type="tool_use", name=tool, input=answer)
        return SimpleNamespace(
            content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=1)
        )


def _papers(tmp_path, titles=tuple(FACTORS)):
    lines = []
    for i, title in enumerate(titles):
        lines += ["[[papers]]", f'key = "k:{i}"', f'title = "{title}"', 'abstract = "a"', ""]
    path = tmp_path / "papers.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _config(tmp_path, **changes):
    config = load_config()
    return replace(
        config,
        papers=str(_papers(tmp_path)),
        deepen_rounds=3,
        specs=config.specs[:2],
        factors=replace(config.factors, hypotheses_per_paper=1, factors_per_hypothesis=2),
        **changes,
    )


@pytest.fixture(scope="module")
def panel():
    return build_panel(fake_rows(n_stocks=12, n_days=150), "fixture")


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def _run(tmp_path, panel, config, *, client=None, editor="scripted", name="run"):
    workdir = tmp_path / name
    workdir.mkdir()
    editor = ScriptedEditor(*EDITS) if editor == "scripted" else editor
    budget = BudgetedClient(client or ScriptedClient(), config.max_model_calls)
    return e2e.search(config, panel, budget, editor, workdir), workdir


def _entries():
    return [{k: v for k, v in e.items() if k not in VOLATILE} for e in runlog.read_entries()]


def test_search_runs_papers_deepen_and_evaluation_on_one_panel(tmp_path, panel, ledger):
    config = _config(tmp_path)
    found, workdir = _run(tmp_path, panel, config)

    assert 2 <= len(found.factor_ids) <= 4  # two papers, two factors each, some screened out
    assert found.deepen and found.deepen_rounds == 3 and found.deepen_stopped == ""
    assert set(found.trees) > set(found.factor_ids)  # plus the admitted children
    assert len(found.summary) == len(found.trees) * len(config.specs)
    assert set(found.matrix.columns) <= set(found.summary.index)
    con = store.connect(workdir / "factors.db")
    assert {f["id"] for f in store.factors(con)} == set(found.trees)
    con.close()


def test_every_deepen_score_and_variant_carries_the_run_and_data_tags(tmp_path, panel, ledger):
    config = _config(tmp_path)
    found, _ = _run(tmp_path, panel, config)

    entries = runlog.read_entries()
    assert entries and all({config.run_tag, config.data_tag} <= set(e["tags"]) for e in entries)
    evaluated = [e for e in entries if "evaluation" in e["tags"]]
    deepened = [e for e in entries if "deepen" in e["tags"]]
    assert len(evaluated) == len(found.summary)  # every variant is a logged trial
    assert deepened and len(evaluated) + len(deepened) == len(entries)
    assert all(e["params"]["run_id"] == config.run_id for e in evaluated)
    assert config.data_tag == "e2e-data:synthetic-crsp:start:2005-12-30"


def test_the_result_is_the_same_at_one_worker_and_many(tmp_path, panel, monkeypatch):
    results = {}
    for name, workers in (("one", 1), ("many", 3)):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / f"ledger-{name}"))
        monkeypatch.setattr(e2e.panel_pool, "workers", lambda *args, n=workers: n)
        found, _ = _run(tmp_path, panel, _config(tmp_path), name=name)
        results[name] = (found, _entries())
    (one, ledger_one), (many, ledger_many) = results.values()
    pd.testing.assert_frame_equal(one.matrix, many.matrix)
    pd.testing.assert_frame_equal(one.summary, many.summary)
    assert one.factor_ids == many.factor_ids and list(one.trees) == list(many.trees)
    assert ledger_one == ledger_many and ledger_one


def test_isolated_papers_store_the_same_factors_as_sequential_ones(tmp_path, panel, ledger):
    sequential, _ = _run(tmp_path, panel, _config(tmp_path), name="seq")
    isolated, _ = _run(
        tmp_path, panel, _config(tmp_path, paper_mode="isolated", concurrency=2), name="iso"
    )
    assert set(isolated.factor_ids) == set(sequential.factor_ids)


def test_a_failed_paper_stops_the_search_before_deepen_or_evaluation(tmp_path, panel, ledger):
    config = replace(_config(tmp_path), papers=str(_papers(tmp_path, ("alpha", "FAIL"))))
    with pytest.raises(RuntimeError, match="papers failed"):
        _run(tmp_path, panel, config)
    assert runlog.read_entries() == []


def test_a_used_up_budget_in_the_paper_phase_raises_before_deepen(tmp_path, panel, ledger):
    config = _config(tmp_path, max_model_calls=3)  # the papers need four calls
    with pytest.raises(CallBudgetExceeded):
        _run(tmp_path, panel, config)
    assert runlog.read_entries() == []


@pytest.mark.parametrize(("on_budget", "stops"), [("stop", True), ("raise", False)])
def test_a_used_up_budget_during_deepen_stops_or_raises(tmp_path, panel, ledger, on_budget, stops):
    config = _config(tmp_path, max_model_calls=4, deepen_on_budget=on_budget)  # papers use all 4
    if not stops:
        with pytest.raises(CallBudgetExceeded):
            _run(tmp_path, panel, config, editor=None)
        return
    found, _ = _run(tmp_path, panel, config, editor=None)  # the editor's first call is refused
    assert found.deepen_rounds == 0 and "model calls are used up" in found.deepen_stopped
    assert len(found.summary) == len(found.factor_ids) * len(
        config.specs
    )  # the seeds are still evaluated


def test_the_per_family_cap_ends_deepen_when_every_family_has_used_it(tmp_path, panel, ledger):
    uncapped, _ = _run(tmp_path, panel, _config(tmp_path), name="uncapped")
    capped, _ = _run(
        tmp_path, panel, _config(tmp_path, deepen_max_children_per_family=2), name="capped"
    )
    assert uncapped.deepen_stopped == "" and uncapped.deepen_rounds == 3
    assert "max_children_per_family=2" in capped.deepen_stopped
    assert capped.deepen_rounds < 3  # two families at two children each: the cap ends the rounds
    assert len(capped.deepen) <= 4
    assert len(capped.summary) == len(capped.trees) * 2


def test_the_search_leaves_no_staged_panel_behind(tmp_path, panel, ledger, monkeypatch):
    monkeypatch.setattr(e2e.panel_pool, "workers", lambda *args: 2)
    _, workdir = _run(tmp_path, panel, _config(tmp_path))
    assert not (workdir / e2e.panel_pool.STAGING).exists()


def test_the_generator_names_the_agent_selector(tmp_path):
    config = _config(tmp_path)
    assert e2e.generator_name(config) == (
        f"{config.generator}/{config.factors.model}/{config.factors.prompt_version}"
    )
    assert e2e.generator_name(replace(config, deepen_selector="agent")).endswith("/agent-deepen")
    assert config_digest(config) != config_digest(replace(config, deepen_selector="agent"))
