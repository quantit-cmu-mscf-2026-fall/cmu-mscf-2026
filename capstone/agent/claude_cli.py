"""Claude, reached only through the installed `claude -p` CLI.

The experiment rules (docs/experiments/alpha_gpt.md) route every model call
through `claude -p` rather than the Anthropic Python API. This module is that
one door: it builds the command, runs it with `subprocess.run`, and turns
Claude Code's JSON output into the final result text — or into an exception.

Every failure path raises `ClaudeCLIError`. A caller running unattended must
never mistake a crashed subprocess, a hit turn limit, or unparseable output for
an answer, so there is no path that returns an empty string instead of failing.

The subprocess dependency is injectable (`runner=`) so tests never call Claude.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

CLAUDE_EXECUTABLE = "claude"

#: Anything with `subprocess.run`'s calling convention.
Runner = Callable[..., subprocess.CompletedProcess]


class ClaudeCLIError(RuntimeError):
    """`claude -p` failed, or its output could not be trusted as an answer.

    Carries the raw process output so the caller can log or inspect it; the
    message alone is not always enough to tell an auth failure from a bad flag.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def build_command(
    *, max_turns: int | None = None, executable: str = CLAUDE_EXECUTABLE
) -> list[str]:
    """The argv for one non-interactive call. The prompt goes on stdin, not here.

    Keeping the prompt out of argv avoids the OS argument-length limit and keeps
    prompt text out of process listings.
    """
    command = [executable, "-p", "--output-format", "json"]
    if max_turns is not None:
        command += ["--max-turns", str(max_turns)]
    return command


def run_claude(
    prompt: str,
    *,
    max_turns: int | None = None,
    timeout: float | None = None,
    executable: str = CLAUDE_EXECUTABLE,
    runner: Runner | None = None,
) -> str:
    """Send `prompt` to `claude -p` and return the final result text.

    Args:
        prompt: the full prompt, passed on stdin.
        max_turns: forwarded as `--max-turns`; None leaves the CLI default.
        timeout: seconds before the subprocess is killed; None waits forever.
        executable: the CLI to invoke, for installs not named `claude` on PATH.
        runner: stand-in for `subprocess.run`. Resolved at call time, so
            monkeypatching `subprocess.run` in this module also works.

    Returns:
        The `result` field of Claude Code's JSON output.

    Raises:
        ValueError: if `prompt` is blank or `max_turns` is not a positive int.
        ClaudeCLIError: if the CLI is missing, times out, exits non-zero,
            prints output that is not the expected JSON object, or reports an
            unsuccessful run (including hitting `max_turns`).
    """
    if not prompt.strip():
        raise ValueError("prompt is empty")
    # bool is an int subclass; --max-turns True would reach the CLI as "True".
    if max_turns is not None and (
        isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1
    ):
        raise ValueError(f"max_turns must be a positive int, got {max_turns!r}")

    run = runner or subprocess.run
    command = build_command(max_turns=max_turns, executable=executable)

    try:
        completed = run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ClaudeCLIError(
            f"could not start {executable!r}: is Claude Code installed and on PATH?"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCLIError(
            f"{executable} -p timed out after {timeout} s",
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    def fail(message: str) -> ClaudeCLIError:
        return ClaudeCLIError(
            message, returncode=completed.returncode, stdout=stdout, stderr=stderr
        )

    if completed.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "(no output)"
        raise fail(f"{executable} -p exited with code {completed.returncode}: {detail}")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise fail(f"{executable} -p did not print JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise fail(f"expected a JSON object, got {type(payload).__name__}")

    # A run that stops at --max-turns reports subtype "error_max_turns" and can
    # still carry is_error=false, so the subtype has to be checked too.
    subtype = payload.get("subtype")
    if payload.get("is_error") or subtype != "success":
        reason = payload.get("result")
        suffix = f": {reason}" if isinstance(reason, str) and reason else ""
        raise fail(f"claude reported an unsuccessful run (subtype={subtype!r}){suffix}")

    result = payload.get("result")
    if not isinstance(result, str):
        raise fail("claude output has no string 'result' field")
    return result


def _as_text(value: str | bytes | None) -> str:
    """TimeoutExpired carries whatever was captured, possibly bytes, possibly None."""
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
