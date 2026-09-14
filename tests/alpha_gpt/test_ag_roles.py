"""Tests for capstone.alpha_gpt.roles. Claude is never called."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from capstone.alpha_gpt.roles import (
    AnalystReview,
    RoleFailedError,
    SeedProposal,
    call_role,
    check_analyst_review,
    check_seed_proposal,
    claude_llm,
)

FIELDS = ["close", "volume", "returns"]
REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        return self.replies.pop(0)


def _seed(*expressions: str) -> str:
    alphas = [{"expression": e, "rationale": "because"} for e in expressions]
    return json.dumps({"type": "seed_proposal", "alphas": alphas})


def _seed_check(n_alphas=2, already_evaluated=()):
    return lambda reply: check_seed_proposal(
        reply,
        fields=FIELDS,
        groups=[],
        n_alphas=n_alphas,
        already_evaluated=already_evaluated,
    )


class TestCallRole:
    def test_valid_reply_first_time(self):
        llm = ScriptedLLM([_seed("neg(returns)", "ts_mean(volume, 5)")])
        records = []
        result = call_role(
            "PROMPT", SeedProposal, llm=llm, role="qd", check=_seed_check(), record=records.append
        )

        assert result.attempts == 1
        assert result.problems == []
        assert [a.expression for a in result.reply.alphas] == ["neg(returns)", "ts_mean(volume, 5)"]
        (entry,) = records
        assert entry["role"] == "qd" and entry["prompt"] == "PROMPT" and entry["problems"] == []
        assert len(entry["prompt_sha256"]) == 64

    def test_fenced_json_is_rejected_then_retried_with_the_error(self):
        fenced = "```json\n" + _seed("neg(returns)", "returns") + "\n```"
        llm = ScriptedLLM([fenced, _seed("neg(returns)", "returns")])
        records = []
        result = call_role(
            "PROMPT", SeedProposal, llm=llm, role="qd", check=_seed_check(), record=records.append
        )

        assert result.attempts == 2
        retry = llm.prompts[1]
        assert retry.startswith("PROMPT\n\n# Your previous reply was rejected")
        assert "not valid JSON" in retry
        assert fenced in retry
        assert [r["raw"] for r in records] == [fenced, _seed("neg(returns)", "returns")]

    def test_invalid_expression_is_retried(self):
        llm = ScriptedLLM(
            [_seed("neg(returns)", "close / volume"), _seed("neg(returns)", "div(close, volume)")]
        )
        result = call_role("P", SeedProposal, llm=llm, role="qd", check=_seed_check())
        assert result.attempts == 2
        assert "infix arithmetic" in llm.prompts[1]

    def test_schema_violation_every_time_raises_with_all_attempts(self):
        bad = json.dumps({"type": "seed_proposal", "alphas": [], "extra": 1})
        llm = ScriptedLLM([bad, bad, bad])
        with pytest.raises(RoleFailedError, match="no parseable reply in 3 attempts") as info:
            call_role("P", SeedProposal, llm=llm, role="qd", check=_seed_check())
        assert [a["raw"] for a in info.value.attempts] == [bad] * 3

    def test_check_problems_left_after_last_attempt_are_returned(self):
        reply = _seed("neg(returns)", "foo(close)")
        llm = ScriptedLLM([reply, reply])
        result = call_role(
            "P", SeedProposal, llm=llm, role="qd", check=_seed_check(), max_attempts=2
        )
        assert result.attempts == 2
        assert len(result.problems) == 1
        assert "unknown operator 'foo'" in result.problems[0]

    def test_analyst_unknown_alpha_id_is_retried(self):
        def review(alpha_id):
            return json.dumps(
                {
                    "type": "analyst_review",
                    "summary": "s",
                    "diagnoses": [{"alpha_id": alpha_id, "note": "n"}],
                    "revised_idea": "idea",
                }
            )

        llm = ScriptedLLM([review("r9a9"), review("r1a1")])
        result = call_role(
            "P",
            AnalystReview,
            llm=llm,
            role="analyst",
            check=lambda r: check_analyst_review(r, known_ids={"r1a1"}),
        )
        assert result.attempts == 2
        assert "'r9a9' is not in the results table" in llm.prompts[1]

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError):
            call_role("P", SeedProposal, llm=ScriptedLLM([]), role="qd", max_attempts=0)


class TestCheckSeedProposal:
    def _reply(self, *expressions):
        return SeedProposal.model_validate_json(_seed(*expressions))

    def test_count_mismatch(self):
        problems = _seed_check(n_alphas=3)(self._reply("neg(returns)", "returns"))
        assert problems == ["expected exactly 3 alphas, got 2"]

    def test_duplicate_within_reply_uses_canonical_form(self):
        problems = _seed_check()(self._reply("add(close, volume)", "add(volume, close)"))
        assert problems == ["'add(volume, close)' duplicates another alpha in this reply"]

    def test_already_evaluated(self):
        check = _seed_check(already_evaluated={"neg(returns)"})
        assert "already evaluated" in check(self._reply("neg(returns)", "returns"))[0]


def test_claude_llm_runs_isolated_from_the_repository():
    seen = {}

    def runner(args, **kwargs):
        seen["args"] = args
        seen["cwd"] = kwargs["cwd"]
        seen["cwd_existed"] = os.path.isdir(kwargs["cwd"])
        seen["cwd_empty"] = os.listdir(kwargs["cwd"]) == []
        payload = {"type": "result", "subtype": "success", "is_error": False, "result": "{}"}
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    llm = claude_llm(timeout=10, model="sonnet", runner=runner)
    assert llm("hello") == "{}"

    args = seen["args"]
    assert args[args.index("--tools") + 1] == ""
    assert "--no-session-persistence" in args
    assert args[args.index("--max-turns") + 1] == "1"
    assert args[args.index("--model") + 1] == "sonnet"
    assert seen["cwd_existed"] and seen["cwd_empty"]
    assert not Path(seen["cwd"]).resolve().is_relative_to(REPO_ROOT)
    assert not os.path.exists(seen["cwd"])
