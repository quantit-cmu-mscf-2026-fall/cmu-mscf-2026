"""Tests for capstone.agent.claude_cli.

Real Claude is never called: a fake runner stands in for `subprocess.run`,
records how it was invoked, and hands back a CompletedProcess we built. The
JSON payloads mirror the shape of `claude -p --output-format json`.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from capstone.agent import claude_cli
from capstone.agent.claude_cli import ClaudeCLIError, build_command, run_claude


def _payload(**overrides) -> str:
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "result": "hello from claude",
        "session_id": "00000000-0000-0000-0000-000000000000",
    }
    base.update(overrides)
    return json.dumps(base)


class FakeRunner:
    """Stands in for subprocess.run, recording each call."""

    def __init__(self, *, stdout="", stderr="", returncode=0, error=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


class TestInvocation:
    def test_base_command_and_subprocess_options(self):
        runner = FakeRunner(stdout=_payload())
        run_claude("say hi", runner=runner)

        ((args, kwargs),) = runner.calls
        assert args == ["claude", "-p", "--output-format", "json"]
        assert kwargs["input"] == "say hi"
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False

    def test_prompt_is_not_in_argv(self):
        runner = FakeRunner(stdout=_payload())
        run_claude("secret-ish prompt text", runner=runner)

        args, _ = runner.calls[0]
        assert "secret-ish prompt text" not in args

    def test_max_turns_is_forwarded(self):
        runner = FakeRunner(stdout=_payload())
        run_claude("say hi", max_turns=3, runner=runner)

        args, _ = runner.calls[0]
        assert args[-2:] == ["--max-turns", "3"]

    def test_max_turns_none_omits_the_flag(self):
        assert "--max-turns" not in build_command(max_turns=None)

    def test_timeout_and_executable_are_forwarded(self):
        runner = FakeRunner(stdout=_payload())
        run_claude("say hi", timeout=12.5, executable="/opt/claude", runner=runner)

        args, kwargs = runner.calls[0]
        assert args[0] == "/opt/claude"
        assert kwargs["timeout"] == 12.5

    def test_extra_args_follow_base_flags_and_precede_max_turns(self):
        runner = FakeRunner(stdout=_payload())
        run_claude("say hi", max_turns=1, extra_args=["--tools", ""], runner=runner)

        args, _ = runner.calls[0]
        assert args == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--max-turns",
            "1",
        ]

    def test_cwd_is_forwarded_and_defaults_to_none(self, tmp_path):
        runner = FakeRunner(stdout=_payload())
        run_claude("say hi", cwd=tmp_path, runner=runner)
        run_claude("say hi", runner=runner)

        assert runner.calls[0][1]["cwd"] == tmp_path
        assert runner.calls[1][1]["cwd"] is None

    def test_default_runner_is_subprocess_run(self, monkeypatch):
        fake = FakeRunner(stdout=_payload())
        monkeypatch.setattr(claude_cli.subprocess, "run", fake)

        assert run_claude("say hi") == "hello from claude"
        assert len(fake.calls) == 1


class TestInputValidation:
    @pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3"])
    def test_rejects_non_positive_int_max_turns(self, bad):
        runner = FakeRunner(stdout=_payload())
        with pytest.raises(ValueError, match="max_turns"):
            run_claude("say hi", max_turns=bad, runner=runner)
        assert runner.calls == []

    @pytest.mark.parametrize("bad", ["--tools", [1], ["--tools", None]])
    def test_rejects_extra_args_that_are_not_a_sequence_of_str(self, bad):
        runner = FakeRunner(stdout=_payload())
        with pytest.raises(ValueError, match="extra_args"):
            run_claude("say hi", extra_args=bad, runner=runner)
        assert runner.calls == []

    @pytest.mark.parametrize("blank", ["", "   \n"])
    def test_rejects_blank_prompt(self, blank):
        runner = FakeRunner(stdout=_payload())
        with pytest.raises(ValueError, match="prompt"):
            run_claude(blank, runner=runner)
        assert runner.calls == []


class TestSuccess:
    def test_returns_result_text(self):
        runner = FakeRunner(stdout=_payload(result="the final answer"))
        assert run_claude("q", runner=runner) == "the final answer"

    def test_empty_result_string_is_still_a_result(self):
        runner = FakeRunner(stdout=_payload(result=""))
        assert run_claude("q", runner=runner) == ""


class TestProcessFailures:
    def test_nonzero_exit_raises_with_stderr_and_captured_output(self):
        runner = FakeRunner(stdout="partial", stderr="auth failed", returncode=1)

        with pytest.raises(ClaudeCLIError, match=r"exited with code 1: auth failed") as info:
            run_claude("q", runner=runner)

        assert info.value.returncode == 1
        assert info.value.stdout == "partial"
        assert info.value.stderr == "auth failed"

    def test_nonzero_exit_falls_back_to_stdout_when_stderr_empty(self):
        runner = FakeRunner(stdout=_payload(is_error=True, result="API Error"), returncode=1)
        with pytest.raises(ClaudeCLIError, match="API Error"):
            run_claude("q", runner=runner)

    def test_missing_executable_raises_clearly(self):
        runner = FakeRunner(error=FileNotFoundError("claude"))
        with pytest.raises(ClaudeCLIError, match="installed and on PATH"):
            run_claude("q", runner=runner)

    def test_timeout_raises_with_partial_output(self):
        error = subprocess.TimeoutExpired(["claude"], 5, output=b"half", stderr=None)
        runner = FakeRunner(error=error)

        with pytest.raises(ClaudeCLIError, match="timed out after 5") as info:
            run_claude("q", timeout=5, runner=runner)

        assert info.value.stdout == "half"
        assert info.value.stderr == ""


class TestOutputFailures:
    def test_non_json_stdout_raises(self):
        runner = FakeRunner(stdout="not json at all")
        with pytest.raises(ClaudeCLIError, match="did not print JSON"):
            run_claude("q", runner=runner)

    def test_empty_stdout_raises(self):
        runner = FakeRunner(stdout="")
        with pytest.raises(ClaudeCLIError, match="did not print JSON"):
            run_claude("q", runner=runner)

    def test_json_that_is_not_an_object_raises(self):
        runner = FakeRunner(stdout="[1, 2, 3]")
        with pytest.raises(ClaudeCLIError, match="JSON object"):
            run_claude("q", runner=runner)

    def test_is_error_with_zero_exit_raises(self):
        runner = FakeRunner(stdout=_payload(is_error=True, result="something broke"))
        with pytest.raises(ClaudeCLIError, match="something broke"):
            run_claude("q", runner=runner)

    def test_max_turns_reached_raises_even_when_is_error_is_false(self):
        stdout = json.dumps(
            {"type": "result", "subtype": "error_max_turns", "is_error": False, "num_turns": 2}
        )
        runner = FakeRunner(stdout=stdout)
        with pytest.raises(ClaudeCLIError, match="error_max_turns"):
            run_claude("q", max_turns=2, runner=runner)

    def test_missing_result_field_raises(self):
        stdout = json.dumps({"type": "result", "subtype": "success", "is_error": False})
        runner = FakeRunner(stdout=stdout)
        with pytest.raises(ClaudeCLIError, match="'result'"):
            run_claude("q", runner=runner)

    def test_non_string_result_raises(self):
        runner = FakeRunner(stdout=_payload(result={"nested": "object"}))
        with pytest.raises(ClaudeCLIError, match="'result'"):
            run_claude("q", runner=runner)
