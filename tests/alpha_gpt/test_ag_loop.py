"""End-to-end tests: the discovery procedure, its cross-validation, and TEST (scripted LLM)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from capstone.alpha_gpt.loop import (
    LoopConfig,
    TestAlreadyEvaluated,
    cross_validate_procedure,
    experiment_summary,
    finalize_on_test,
    previous_test_runs,
    run_final_procedure,
    run_procedure,
    select_best,
)
from capstone.alpha_gpt.panel import OHLCVPanel
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel
from capstone.alpha_gpt.trials import LedgerContext
from capstone.cv import PurgedKFold

CONTEXT = LedgerContext(ledger_name="alpha_gpt_loop_test", run_id="run-test", seed=0)
IDEA = "stocks that fell sharply recently tend to bounce"

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


class RoleLLM:
    """Answers by role, for runs with many procedures; proposals can depend on the round."""

    def __init__(self, *, rounds, review=None):
        self.rounds = list(rounds)
        self.review = review
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("# Role\nYou are the Analyst"):
            if self.review is None:
                raise AssertionError("Analyst was not scripted")
            return self.review
        for k in range(len(self.rounds) - 1, 0, -1):
            if f"after round {k}." in prompt:
                return self.rounds[k]
        return self.rounds[0]


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


def _final(panel, splits, llm, run_dir, **cfg):
    return run_final_procedure(
        IDEA, panel, splits, LoopConfig(**cfg), llm=llm, context=CONTEXT, run_dir=run_dir
    )


def _cv(panel, splits, llm, run_dir, n_splits=3, **cfg):
    return cross_validate_procedure(
        IDEA,
        panel,
        splits,
        LoopConfig(**cfg),
        PurgedKFold(n_splits, horizon=1, embargo_pct=0.01),
        llm=llm,
        context=CONTEXT,
        run_dir=run_dir,
    )


def _assert_records_close(first, second):
    """Same records; floats to ~1e-9, since pandas' summation order can vary (~1e-15)."""
    for record_a, record_b in zip(first, second, strict=True):
        a, b = record_a.as_dict(), record_b.as_dict()
        assert a.keys() == b.keys()
        for key, value in a.items():
            if isinstance(value, float):
                assert value == pytest.approx(b[key], rel=1e-9, abs=1e-12), key
            else:
                assert value == b[key], key


@pytest.fixture(scope="module")
def planted():
    panel = make_ohlcv_panel(n_dates=800, n_assets=100, pattern="reversal", strength=0.06, seed=3)
    return panel, SplitSpec.from_fractions(panel.dates)


class TestProcedure:
    def test_planted_signal_is_selected_and_confirmed_on_test(self, planted, tmp_path, ledger):
        panel, splits = planted
        llm = ScriptedLLM([_seed(*JUNK, "ts_std(returns, 10)", "neg(returns)")])
        final = _final(panel, splits, llm, tmp_path, n_rounds=1, n_alphas=5)

        assert len(llm.prompts) == 1  # one round: no Analyst call
        assert [r.expression for r in final.selected] == ["neg(returns)"]

        report = finalize_on_test(final, panel, splits, context=CONTEXT, run_dir=tmp_path)
        assert report.entries[0].held_out.ic_tstat > 4

        entries = ledger()
        assert [e["tags"][-1] for e in entries] == ["search"] * 5 + ["test"]
        assert entries[-1]["params"]["splits"] == splits.as_dict()
        written = json.loads((tmp_path / "final_test.json").read_text())
        assert written["selected"][0]["expression"] == "neg(returns)"
        assert len((tmp_path / "final" / "trials.jsonl").read_text().splitlines()) == 5

    def test_signal_found_through_analyst_feedback(self, planted, tmp_path, ledger):
        panel, splits = planted
        revised = "short-term reversal: yesterday's biggest losers outperform tomorrow"
        llm = ScriptedLLM(
            [
                _seed(*JUNK),
                _review(revised, summary="None of the volume or level alphas show an effect."),
                _seed("neg(returns)", "neg(ts_delta(close, 2))", "ts_std(returns, 10)"),
            ]
        )
        final = _final(panel, splits, llm, tmp_path, n_rounds=2, n_alphas=3)

        analyst_prompt, second_qd_prompt = llm.prompts[1], llm.prompts[2]
        assert "| r1a1 | 1 | `ts_mean(volume, 20)` |" in analyst_prompt
        assert "VALIDATION" not in analyst_prompt
        assert revised in second_qd_prompt
        assert "None of the volume or level alphas" in second_qd_prompt
        assert "- ts_mean(volume, 20)" in second_qd_prompt
        assert final.rounds[1].idea == revised
        assert final.selected[0].round == 2
        assert final.selected[0].expression in {"neg(returns)", "neg(ts_delta(close, 2))"}

    def test_every_trial_is_in_the_ledger_before_the_analyst_is_asked(
        self, planted, tmp_path, ledger
    ):
        panel, splits = planted

        def analyst(prompt):
            assert len(ledger()) == 3
            return _review("anything")

        llm = ScriptedLLM([_seed(*JUNK), analyst, _seed("neg(returns)", "returns", "neg(volume)")])
        _final(panel, splits, llm, tmp_path, n_rounds=2, n_alphas=3)
        assert len(ledger()) == 6

    def test_resubmitted_expression_is_retried_then_not_recounted(self, planted, tmp_path, ledger):
        panel, splits = planted
        llm = ScriptedLLM(
            [
                _seed("neg(returns)", "returns"),
                _review("same again"),
                _seed("neg(returns)", "neg(volume)"),
                _seed("neg(returns)", "neg(volume)"),
            ]
        )
        final = _final(panel, splits, llm, tmp_path, n_rounds=2, n_alphas=2, max_attempts=2)

        assert "already evaluated in an earlier round" in llm.prompts[3]
        assert [d.of for d in final.rounds[1].duplicates] == ["r1a1"]
        assert len(final.trials) == 3
        assert len(ledger()) == 3

    def test_trial_budget_stops_the_procedure(self, planted, tmp_path, ledger):
        panel, splits = planted
        llm = ScriptedLLM([_seed(*JUNK, "neg(returns)", "returns")])  # no Analyst reply scripted
        final = _final(panel, splits, llm, tmp_path, n_rounds=3, n_alphas=5, max_trials=3)

        assert final.budget_exhausted
        assert len(final.trials) == 3
        assert len(ledger()) == 3

    def test_refuses_a_panel_that_contains_test_dates(self, planted, tmp_path):
        panel, splits = planted
        with pytest.raises(ValueError, match="TEST dates"):
            run_procedure(
                IDEA,
                panel,
                splits.dates(panel.dates, "development"),
                LoopConfig(),
                llm=ScriptedLLM([]),
                context=CONTEXT,
                test_start=splits.test[0],
                run_dir=tmp_path,
            )

    def test_selection_is_signed_and_puts_nan_last(self, planted, tmp_path, ledger):
        panel, splits = planted
        llm = ScriptedLLM([_seed("returns", "neg(returns)", "ts_mean(returns, 5)")])
        final = _final(panel, splits, llm, tmp_path, n_rounds=1, n_alphas=3)
        ordered = [r.expression for r in select_best(final.trials, 3)]
        assert ordered[0] == "neg(returns)"
        assert ordered[-1] == "returns"

    def test_test_period_data_never_reaches_the_llm_or_selection(self, planted, tmp_path, ledger):
        panel, splits = planted
        rng = np.random.default_rng(11)
        scrambled = {}
        for name, frame in panel.fields.items():
            changed = frame.copy()
            rows = changed.index >= splits.test[0]
            block = changed.loc[rows]
            changed.loc[rows] = rng.permutation(block.to_numpy().ravel()).reshape(block.shape)
            scrambled[name] = changed
        other = OHLCVPanel(fields=scrambled, groups=panel.groups, descriptor=panel.descriptor)

        def replies():
            return [
                _seed(*JUNK),
                _review("reversal"),
                _seed("neg(returns)", "returns", "neg(close)"),
            ]

        runs = []
        for i, p in enumerate([panel, other]):
            llm = ScriptedLLM(replies())
            final = _final(p, splits, llm, tmp_path / f"run{i}", n_rounds=2, n_alphas=3)
            runs.append((llm.prompts, final.trials))

        assert runs[0][0] == runs[1][0]  # byte-identical prompts
        _assert_records_close(runs[0][1], runs[1][1])


class TestCrossValidation:
    def test_scores_the_whole_procedure_on_each_held_out_fold(self, planted, tmp_path, ledger):
        panel, splits = planted
        llm = RoleLLM(rounds=[_seed(*JUNK, "ts_std(returns, 10)", "neg(returns)")])
        result = _cv(panel, splits, llm, tmp_path, n_splits=3, n_rounds=1, n_alphas=5)

        assert len(result.folds) == 3 and len(llm.prompts) == 3
        for fold in result.folds:
            (score,) = fold.held_out
            assert score.expression == "neg(returns)"
            assert score.held_out.ic_tstat > 2
            assert (tmp_path / f"fold{fold.fold}" / "llm_calls.jsonl").exists()
        assert result.summary()["share_of_folds_positive"] == 1.0

        entries = ledger()
        tags = [e["tags"][-1] for e in entries]
        assert tags.count("search") == 15 and tags.count("cv_valid") == 3
        assert [e["params"]["fold"] for e in entries if e["tags"][-1] == "cv_valid"] == [1, 2, 3]
        assert len((tmp_path / "cv_folds.jsonl").read_text().splitlines()) == 3

    def test_a_held_out_fold_never_reaches_the_procedure_trained_around_it(
        self, planted, tmp_path, ledger
    ):
        panel, splits = planted
        search = splits.search_panel(panel)
        cv = PurgedKFold(3, horizon=1, embargo_pct=0.01)
        _, valid = list(cv.split(splits.dates(search.dates, "development")))[1]
        label_end = search.dates[search.dates.get_loc(valid[-1]) + 1]
        next_return = search.dates[search.dates.get_loc(label_end) + 1]

        rng = np.random.default_rng(5)
        scrambled = {}
        for name, frame in panel.fields.items():
            changed = frame.copy()
            rows = (changed.index >= valid[0]) & (changed.index <= label_end)
            if name == "returns":
                rows |= changed.index == next_return
            block = changed.loc[rows]
            changed.loc[rows] = rng.permutation(block.to_numpy().ravel()).reshape(block.shape)
            scrambled[name] = changed
        other = OHLCVPanel(fields=scrambled, groups=panel.groups, descriptor=panel.descriptor)

        def llm():
            return RoleLLM(
                rounds=[_seed(*JUNK), _seed("neg(returns)", "returns", "neg(close)")],
                review=_review("reversal"),
            )

        results = [
            _cv(p, splits, llm(), tmp_path / f"run{i}", n_splits=3, n_rounds=2, n_alphas=3)
            for i, p in enumerate([panel, other])
        ]

        def fold_two_prompts(i):
            path = tmp_path / f"run{i}" / "fold2" / "llm_calls.jsonl"
            return [json.loads(line)["prompt"] for line in path.read_text().splitlines()]

        assert fold_two_prompts(0) == fold_two_prompts(1)
        fold_two = [result.folds[1] for result in results]
        _assert_records_close(fold_two[0].procedure.trials, fold_two[1].procedure.trials)
        # The scramble did change what the fold itself scores.
        first, second = (fold.held_out[0].held_out.ic_mean for fold in fold_two)
        assert first != pytest.approx(second, abs=1e-6)

    def test_on_null_data_it_exposes_selection_bias_and_finds_nothing(self, tmp_path, ledger):
        """Six signal-free panels, three folds each, the same eight scripted alphas.

        Tolerances fixed before running: the selected alphas' in-sample t exceeds their
        held-out t by more than 0.5 on average, the held-out mean |t| stays below 1, and
        at most 2 of 18 held-out scores reach p < 0.01.
        """
        in_sample, held_out, significant = [], [], 0
        for seed in range(6):
            panel = make_ohlcv_panel(n_dates=600, n_assets=60, pattern="none", seed=seed)
            splits = SplitSpec.from_fractions(panel.dates)
            llm = RoleLLM(rounds=[_seed(*NULL_ALPHAS)])
            result = _cv(panel, splits, llm, tmp_path / f"seed{seed}", n_rounds=1, n_alphas=8)
            for fold in result.folds:
                (score,) = fold.held_out
                in_sample.append(score.train.ic_tstat)
                held_out.append(score.held_out.ic_tstat)
                significant += score.held_out.ic_pvalue < 0.01

        assert np.mean(in_sample) - np.mean(held_out) > 0.5
        assert abs(np.mean(held_out)) < 1.0
        assert significant <= 2

    def test_summary_counts_the_whole_family(self, planted, tmp_path, ledger):
        panel, splits = planted
        llm = RoleLLM(rounds=[_seed("neg(returns)", "returns")])
        cv_result = _cv(panel, splits, llm, tmp_path, n_splits=2, n_rounds=1, n_alphas=2)
        final = _final(panel, splits, llm, tmp_path, n_rounds=1, n_alphas=2)
        report = finalize_on_test(final, panel, splits, context=CONTEXT, run_dir=tmp_path)
        summary = experiment_summary(cv_result, final, report)

        tags = [e["tags"][-1] for e in ledger()]
        assert summary["family_size"] == tags.count("search") == 6
        assert tags.count("cv_valid") == 2 and tags.count("test") == 1
        assert summary["in_sample"]["expression"] == "neg(returns)"
        assert 0.0 <= summary["in_sample"]["deflated_sharpe_ratio"] <= 1.0
        assert summary["test"]["test_ic_tstat"] > 4
        json.dumps(summary)


class TestTestLock:
    def test_test_is_evaluated_only_once_per_run(self, planted, tmp_path, ledger):
        panel, splits = planted
        final = _final(
            panel,
            splits,
            ScriptedLLM([_seed("neg(returns)", "returns")]),
            tmp_path,
            n_rounds=1,
            n_alphas=2,
        )

        finalize_on_test(final, panel, splits, context=CONTEXT, run_dir=tmp_path)
        with pytest.raises(FileExistsError, match="already evaluated"):
            finalize_on_test(final, panel, splits, context=CONTEXT, run_dir=tmp_path)
        assert sum(e["tags"][-1] == "test" for e in ledger()) == 1

    def test_locked_across_runs_unless_the_reuse_is_recorded(self, planted, tmp_path, ledger):
        panel, splits = planted

        def search(run_dir):
            return _final(
                panel,
                splits,
                ScriptedLLM([_seed("neg(returns)", "returns")]),
                run_dir,
                n_rounds=1,
                n_alphas=2,
            )

        first = search(tmp_path / "a")
        finalize_on_test(first, panel, splits, context=CONTEXT, run_dir=tmp_path / "a")

        second = search(tmp_path / "b")
        with pytest.raises(TestAlreadyEvaluated, match="run-test"):
            finalize_on_test(second, panel, splits, context=CONTEXT, run_dir=tmp_path / "b")
        assert not (tmp_path / "b" / "final_test.json").exists()

        finalize_on_test(
            second, panel, splits, context=CONTEXT, run_dir=tmp_path / "b", reuse_test=True
        )
        test_entries = [e for e in ledger() if e["tags"][-1] == "test"]
        assert len(test_entries) == 2
        assert "test_reuse" not in test_entries[0]["tags"]
        assert "test_reuse" in test_entries[1]["tags"]

    def test_lock_is_specific_to_the_data_and_split(self, planted, tmp_path, ledger):
        panel, splits = planted
        final = _final(
            panel,
            splits,
            ScriptedLLM([_seed("neg(returns)", "returns")]),
            tmp_path,
            n_rounds=1,
            n_alphas=2,
        )
        assert previous_test_runs(panel, splits) == []
        finalize_on_test(final, panel, splits, context=CONTEXT, run_dir=tmp_path)

        assert previous_test_runs(panel, splits) == ["run-test"]
        other_split = SplitSpec.from_fractions(panel.dates, test_frac=0.25)
        assert previous_test_runs(panel, other_split) == []
        other_data = make_ohlcv_panel(
            n_dates=800, n_assets=100, pattern="reversal", strength=0.06, seed=4
        )
        assert previous_test_runs(other_data, splits) == []
