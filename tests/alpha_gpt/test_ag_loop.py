"""End-to-end tests of the Seed + Analyst loop with a scripted LLM."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from capstone.alpha_gpt.loop import (
    LoopConfig,
    finalize_on_test,
    run_seed_analyst_loop,
    select_best,
)
from capstone.alpha_gpt.panel import OHLCVPanel
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel
from capstone.alpha_gpt.trials import LedgerContext
from capstone.evaluate import benjamini_hochberg

CONTEXT = LedgerContext(ledger_name="alpha_gpt_loop_test", run_id="run-test", seed=0)

JUNK = ["ts_mean(volume, 20)", "minus(high, low)", "normed_rank(close)"]
NULL_ALPHAS = [
    "neg(returns)",
    "ts_mean(returns, 5)",
    "normed_rank(volume)",
    "ts_delta(close, 10)",
    "div(close, vwap)",
    "neg(ts_zscore_scale(close, 20))",
    "ts_corr(close, volume, 10)",
    "minus(high, low)",
]


class ScriptedLLM:
    """Replies in order; a callable reply is called with the prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        reply = self.replies.pop(0)
        return reply(prompt) if callable(reply) else reply


def _seed(*expressions: str) -> str:
    alphas = [{"expression": e, "rationale": "scripted"} for e in expressions]
    return json.dumps({"type": "seed_proposal", "alphas": alphas})


def _review(revised_idea: str, summary: str = "scripted summary") -> str:
    return json.dumps(
        {
            "type": "analyst_review",
            "summary": summary,
            "diagnoses": [],
            "revised_idea": revised_idea,
        }
    )


@pytest.fixture(scope="module")
def planted():
    panel = make_ohlcv_panel(n_dates=800, n_assets=100, pattern="reversal", strength=0.06, seed=3)
    return panel, SplitSpec.from_fractions(panel.dates)


def _run(panel, splits, llm, tmp_path, **cfg):
    config = LoopConfig(**cfg)
    result = run_seed_analyst_loop(
        "stocks that fell sharply recently tend to bounce",
        panel,
        splits,
        config,
        llm=llm,
        context=CONTEXT,
        run_dir=tmp_path,
    )
    return config, result


def test_planted_signal_is_selected_and_confirmed_on_test(planted, tmp_path, ledger):
    panel, splits = planted
    llm = ScriptedLLM([_seed(*JUNK, "ts_std(returns, 10)", "neg(returns)")])
    config, result = _run(panel, splits, llm, tmp_path, n_rounds=1, n_alphas=5)

    assert len(llm.prompts) == 1  # one round: no Analyst call
    assert [r.expression for r in result.selected] == ["neg(returns)"]

    report = finalize_on_test(result, panel, splits, config, context=CONTEXT, run_dir=tmp_path)
    assert report.entries[0].test.ic_tstat > 4

    entries = ledger()
    assert len(entries) == 5 + 1  # every search trial, plus the one TEST evaluation
    assert [e["tags"][-1] for e in entries] == ["search"] * 5 + ["test"]

    written = json.loads((tmp_path / "final_test.json").read_text())
    assert written["n_search_trials"] == 5
    assert written["selected"][0]["expression"] == "neg(returns)"
    assert len((tmp_path / "trials.jsonl").read_text().splitlines()) == 5


def test_signal_found_through_analyst_feedback(planted, tmp_path, ledger):
    panel, splits = planted
    revised = "short-term reversal: yesterday's biggest losers outperform tomorrow"
    llm = ScriptedLLM(
        [
            _seed(*JUNK),
            _review(revised, summary="None of the volume or level alphas show a clear effect."),
            _seed("neg(returns)", "neg(ts_delta(close, 2))", "ts_std(returns, 10)"),
        ]
    )
    _, result = _run(panel, splits, llm, tmp_path, n_rounds=2, n_alphas=3)

    analyst_prompt, second_qd_prompt = llm.prompts[1], llm.prompts[2]
    assert "| r1a1 | 1 | `ts_mean(volume, 20)` |" in analyst_prompt
    assert revised in second_qd_prompt
    assert "None of the volume or level alphas" in second_qd_prompt
    assert "- ts_mean(volume, 20)" in second_qd_prompt
    assert result.rounds[1].idea == revised
    assert result.selected[0].round == 2
    assert result.selected[0].expression in {"neg(returns)", "neg(ts_delta(close, 2))"}


def test_every_trial_is_in_the_ledger_before_the_analyst_is_asked(planted, tmp_path, ledger):
    panel, splits = planted

    def analyst(prompt):
        assert len(ledger()) == 3
        return _review("anything")

    llm = ScriptedLLM([_seed(*JUNK), analyst, _seed("neg(returns)", "returns", "neg(volume)")])
    _run(panel, splits, llm, tmp_path, n_rounds=2, n_alphas=3)
    assert len(ledger()) == 6


def test_resubmitted_expression_is_retried_then_not_recounted(planted, tmp_path, ledger):
    panel, splits = planted
    llm = ScriptedLLM(
        [
            _seed("neg(returns)", "returns"),
            _review("same again"),
            _seed("neg(returns)", "neg(volume)"),
            _seed("neg(returns)", "neg(volume)"),
        ]
    )
    _, result = _run(panel, splits, llm, tmp_path, n_rounds=2, n_alphas=2, max_attempts=2)

    assert "already evaluated in an earlier round" in llm.prompts[3]
    assert [d.of for d in result.rounds[1].duplicates] == ["r1a1"]
    assert len(result.trials) == 3
    assert len(ledger()) == 3


def test_trial_budget_stops_the_loop(planted, tmp_path, ledger):
    panel, splits = planted
    llm = ScriptedLLM([_seed(*JUNK, "neg(returns)", "returns")])  # no Analyst reply scripted
    _, result = _run(panel, splits, llm, tmp_path, n_rounds=3, n_alphas=5, max_trials=3)

    assert result.budget_exhausted
    assert len(result.trials) == 3
    assert len(ledger()) == 3


def test_test_period_data_never_reaches_the_llm_or_selection(planted, tmp_path, ledger):
    panel, splits = planted
    test_start = splits.test[0]
    rng = np.random.default_rng(11)
    scrambled = {}
    for name, frame in panel.fields.items():
        changed = frame.copy()
        mask = changed.index >= test_start
        changed.loc[mask] = rng.permutation(changed.loc[mask].to_numpy().ravel()).reshape(
            changed.loc[mask].shape
        )
        scrambled[name] = changed
    other = OHLCVPanel(fields=scrambled, groups=panel.groups, descriptor=panel.descriptor)

    def replies():
        return [_seed(*JUNK), _review("reversal"), _seed("neg(returns)", "returns", "neg(close)")]

    runs = []
    for i, p in enumerate([panel, other]):
        llm = ScriptedLLM(replies())
        _, result = _run(p, splits, llm, tmp_path / f"run{i}", n_rounds=2, n_alphas=3)
        runs.append((llm.prompts, [r.as_dict() for r in result.trials]))

    assert runs[0][0] == runs[1][0]  # byte-identical prompts
    for a, b in zip(runs[0][1], runs[1][1], strict=True):
        assert a.keys() == b.keys()
        for key, value in a.items():
            if isinstance(value, float):
                # The search panels are identical; only pandas' summation order can
                # differ with memory layout (~1e-15), so compare to that tolerance.
                assert value == pytest.approx(b[key], rel=1e-9, abs=1e-12), key
            else:
                assert value == b[key], key


def test_test_is_evaluated_only_once_per_run(planted, tmp_path, ledger):
    panel, splits = planted
    llm = ScriptedLLM([_seed("neg(returns)", "returns")])
    config, result = _run(panel, splits, llm, tmp_path, n_rounds=1, n_alphas=2)

    finalize_on_test(result, panel, splits, config, context=CONTEXT, run_dir=tmp_path)
    with pytest.raises(FileExistsError, match="already evaluated"):
        finalize_on_test(result, panel, splits, config, context=CONTEXT, run_dir=tmp_path)
    assert sum(e["tags"][-1] == "test" for e in ledger()) == 1


def test_selection_is_signed_and_puts_nan_last(planted, tmp_path, ledger):
    panel, splits = planted
    llm = ScriptedLLM([_seed("returns", "neg(returns)", "ts_mean(returns, 5)")])
    _, result = _run(panel, splits, llm, tmp_path, n_rounds=1, n_alphas=3)
    ordered = [r.expression for r in select_best(result.trials, 3)]
    assert ordered[0] == "neg(returns)"
    assert ordered[-1] == "returns"


def test_on_null_data_the_pipeline_can_say_nothing_here(tmp_path, ledger):
    """Same pipeline, same scripted alphas, ten signal-free panels.

    Tolerances fixed before running: BH on VALIDATION p-values finds nothing in at
    least 8 of 10 seeds, and the selected alpha is significant on TEST (p < 0.01)
    in at most 1 of 10.
    """
    bh_empty = 0
    test_significant = 0
    for seed in range(10):
        panel = make_ohlcv_panel(n_dates=800, n_assets=100, pattern="none", seed=seed)
        splits = SplitSpec.from_fractions(panel.dates)
        llm = ScriptedLLM([_seed(*NULL_ALPHAS)])
        run_dir = tmp_path / f"seed{seed}"
        config, result = _run(panel, splits, llm, run_dir, n_rounds=1, n_alphas=8)

        pvalues = pd.Series({r.trial_id: r.valid.ic_pvalue for r in result.trials})
        bh_empty += not benjamini_hochberg(pvalues, alpha=0.05).any()

        report = finalize_on_test(result, panel, splits, config, context=CONTEXT, run_dir=run_dir)
        test_significant += report.entries[0].test.ic_pvalue < 0.01

    assert bh_empty >= 8
    assert test_significant <= 1
