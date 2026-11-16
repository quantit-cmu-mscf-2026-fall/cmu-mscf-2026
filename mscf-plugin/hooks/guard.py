"""PreToolUse/PostToolUse guard for the mscf-pipeline plugin (stdlib only).

Reads the hook JSON on stdin. Deny: exit 2 with the reason on stderr. Warn or
report: exit 0 with a hook JSON object on stdout. Everything else: exit 0, silent.
The guard only denies or warns, never grants: the permission system stays
authoritative, and the rules in the pipeline-boundaries skill are the real line.
This is the backstop.

Two rules are opt-in through the environment, so nothing here assumes where a
teammate keeps their clone or which email policy they use:

- MSCF_GUARD_DENY_PATHS: path prefixes (separated by the OS path separator) that
  shell commands and edits must not touch. Empty by default; the team repository
  itself is never denied.
- MSCF_GUARD_NOREPLY_REMOTE: a substring of a remote URL (e.g. the team repo's
  name). A `git push` to a matching remote is denied when a commit it would send
  has an author or committer email that is not a users.noreply.github.com address.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SHELLS = {"Bash", "PowerShell"}
EDITS = {"Edit", "MultiEdit", "Write"}
TAG = "mscf-pipeline guard"
DENY_PATHS_ENV = "MSCF_GUARD_DENY_PATHS"
NOREPLY_REMOTE_ENV = "MSCF_GUARD_NOREPLY_REMOTE"

HOLDOUT_FLAG = re.compile(r"--(?:end|as-of|start|before)(?:=|\s+)[\"']?202[1-5](?!\d)")
HOLDOUT_KWARG = re.compile(r"\b(?:end|start)\s*=\s*[\"']202[1-5]-")
FORCE_PUSH = re.compile(r"\bgit\s+push\b[^;&|\n]*?\s(?:--force(?![-\w])|-[a-zA-Z]*f[a-zA-Z]*\b)")
SECRET = re.compile(r"\$\{?(?:env:)?(?:ANTHROPIC|WRDS|BROKER|GITHUB_TOKEN|GH_TOKEN)\w*", re.I)
PRINT_VERB = re.compile(r"(?:^|[;&|]\s*|\s)(?:echo|printf|print|Write-Host|Write-Output)\b", re.I)
ENV_DUMP = re.compile(r"(?:^|[;&|]\s*)(?:env|printenv|export\s+-p|set)\s*(?:$|[;&|])", re.I)
WORKING_DATA = re.compile(
    r"(?:runs\.jsonl|\.parquet\b|data_cache/|\.sessions/|factors\.db|\.ipynb_checkpoints)"
)
GIT_ADD = re.compile(r"\bgit\s+add\b([^;&|\n]*)")
RUNS_PYTHON_SCRIPT = re.compile(
    r"\bpython3?\b[^;&|\n]*?(?:\s-m\s+(?:experiments|scripts)\.|\s(?:experiments|scripts)[\\/]\S+\.py)"
)
NOREPLY = re.compile(r"@users\.noreply\.github\.com$", re.I)
PUSH_VALUE_OPTS = {"-o", "--push-option", "--repo", "--receive-pack", "--exec"}
PUSH_UNKNOWN_OPTS = {"--all", "--branches", "--mirror", "--tags", "--prune"}
CD = re.compile(r"^(?:cd|chdir|pushd|Set-Location|sl)\s+(.+)$", re.I)


# --- configurable guarded paths ---------------------------------------------------------------


def guarded_paths() -> list[str]:
    """The configured path prefixes, slash-normalised and lower-cased; empty when unset."""
    raw = os.environ.get(DENY_PATHS_ENV, "")
    prefixes = (
        p.strip().strip("\"'").replace("\\", "/").rstrip("/").lower() for p in raw.split(os.pathsep)
    )
    return [p for p in prefixes if p]


def guarded_path_in(text: str) -> str | None:
    """The configured prefix that `text` names (as a whole path component), else None."""
    normal = text.replace("\\", "/").lower()
    for prefix in guarded_paths():
        if re.search(r"(?<![\w.-])" + re.escape(prefix) + r"(?![\w-])", normal):
            return prefix
    return None


def path_denial(text: str) -> str | None:
    prefix = guarded_path_in(text)
    if prefix is None:
        return None
    return f"{prefix} is listed in {DENY_PATHS_ENV}; this work does not touch it"


# --- shell commands ---------------------------------------------------------------------------


def bash_denial(command: str) -> str | None:
    reason = path_denial(command)
    if reason:
        return reason
    runs_python = re.search(r"\bpython", command, re.I) is not None
    if HOLDOUT_FLAG.search(command) or (runs_python and HOLDOUT_KWARG.search(command)):
        return (
            "holdout: 2021-2025 stays sealed for discovery and stages 1-3; only the "
            "pre-registered stage-4 run reads it, once (pipeline-boundaries rule 1)"
        )
    if FORCE_PUSH.search(command):
        return (
            "git push --force rewrites shared history; use --force-with-lease on your own PR branch"
        )
    # a presence check ("test -n $KEY && echo set") is fine; printing the value is not
    if any(
        SECRET.search(seg) and PRINT_VERB.search(seg)
        for seg in re.split(r"&&|\|\||[;|\n]", command)
    ):
        return "credentials: never print ANTHROPIC_*/WRDS_*/broker/GitHub token variables"
    if ENV_DUMP.search(command):
        return (
            "credentials: dumping the environment can print tokens; "
            "test one variable's presence instead"
        )
    for match in GIT_ADD.finditer(command):
        if WORKING_DATA.search(match.group(1)):
            return (
                "public repo: the run ledger, parquet data, data_cache, .sessions and notebook "
                "checkpoints are never committed; results go in the PR description"
            )
    return None


# --- edits ------------------------------------------------------------------------------------


def edit_verdict(payload: dict) -> tuple[str | None, str | None]:
    raw = str(payload.get("file_path", ""))
    reason = path_denial(raw)
    if reason:
        return "deny", reason
    path = raw.replace("\\", "/")
    if path.endswith("experiments/runs.jsonl") or path.endswith("/runs.jsonl"):
        return "deny", "the ledger is append-only through capstone.runlog.log_run; never edit it"
    if "/data_cache/" in path or path.endswith(".parquet"):
        return "deny", "shared data changes only through the data maintainer's WRDS refresh"
    new = payload.get("content") or payload.get("new_string") or ""
    for edit in payload.get("edits") or []:
        new += "\n" + str(edit.get("new_string", ""))
    if path.endswith(".py") and HOLDOUT_KWARG.search(new):
        return "warn", (
            "this code passes a 2021-2025 date. Only the pre-registered stage-4 evaluation may "
            "read the holdout, once; discovery and stages 1-3 end at 2020-12-31"
        )
    return None, None


# --- after an experiment script ---------------------------------------------------------------


def ledger_count(cwd: str) -> int | None:
    root = os.environ.get("CAPSTONE_LEDGER_DIR")
    path = Path(root) / "runs.jsonl" if root else Path(cwd) / "experiments" / "runs.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def post_bash(command: str, cwd: str) -> str | None:
    if not RUNS_PYTHON_SCRIPT.search(command):
        return None
    count = ledger_count(cwd)
    if count is None:
        return "no ledger found after this run: was every trial logged with runlog.log_run?"
    return f"ledger now holds {count} runs (python -m capstone.runlog stats for the breakdown)"


# --- opt-in: no-reply emails on pushes to a named remote ---------------------------------------


def _segments(command):
    return [s.strip() for s in re.split(r"\|\||&&|[|;\n]", command) if s.strip()]


def _tokens(segment):
    return [t.strip("\"'") for t in re.findall(r"\"[^\"]*\"|'[^']*'|\S+", segment)]


def _git(where, *args):
    try:
        done = subprocess.run(
            ["git", "-C", str(where), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _push_args(tokens):
    """(the -C folder or None, the arguments after `push`), or None if not a git push."""
    if "git" not in tokens:
        return None
    i, where = tokens.index("git") + 1, None
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] == "-C" and i + 1 < len(tokens):
            where = tokens[i + 1]
        i += 2 if tokens[i] in ("-C", "-c") else 1
    if i < len(tokens) and tokens[i] == "push":
        return where, tokens[i + 1 :]
    return None


def _pushed_commits(where, args, remote_match):
    """[(sha, author, committer)] the push would send to a matching remote; [] when the push
    does not go there; None when the range cannot be determined."""
    positional, i = [], 0
    while i < len(args):
        arg = args[i]
        if arg in PUSH_UNKNOWN_OPTS:
            return None
        if arg in ("-d", "--delete"):
            return []
        if arg in PUSH_VALUE_OPTS:
            i += 1
        elif not arg.startswith("-"):
            positional.append(arg)
        i += 1
    branch = _git(where, "symbolic-ref", "--quiet", "--short", "HEAD")
    remote = positional[0] if positional else None
    if remote is None and branch:
        remote = _git(where, "config", f"branch.{branch}.remote")
    remote = remote or "origin"
    is_url = re.search(r"[/:\\]", remote) is not None
    url = remote if is_url else _git(where, "remote", "get-url", "--push", remote)
    if url is None:
        return None if remote_match in remote else []
    if remote_match not in url:
        return []
    specs = positional[1:] or ([branch] if branch else [])
    if not specs:
        return None
    commits = []
    for spec in specs:
        src, _, dst = spec.lstrip("+").partition(":")
        if "*" in spec:
            return None
        if not src:
            continue  # ":branch" deletes; nothing is sent
        dst = (dst or (branch if src == "HEAD" else src) or "").removeprefix("refs/heads/")
        if is_url:
            return None
        tracking = f"refs/remotes/{remote}/{dst}"
        if dst and _git(where, "rev-parse", "--verify", "--quiet", tracking):
            span = [f"{tracking}..{src}"]
        else:
            span = [src, "--not", f"--remotes={remote}"]
        out = _git(where, "log", "--format=%H %ae %ce", *span)
        if out is None:
            return None
        commits += [tuple(line.split(" ", 2)) for line in out.splitlines() if line.strip()]
    return commits


def _noreply_push(where, args, remote_match):
    commits = _pushed_commits(where, args, remote_match)
    if commits is None:
        return None, (
            "push to a guarded remote: could not work out which commits it sends; check every "
            "author and committer email is a users.noreply.github.com address"
        )
    bad = [c for c in commits if len(c) < 3 or not all(NOREPLY.search(e) for e in c[1:])]
    if not bad:
        return None, None
    shown = ", ".join(f"{c[0][:9]} ({' / '.join(c[1:])})" for c in bad[:5])
    return (
        f"push to a guarded remote: {len(bad)} commit(s) carry an email that is not a "
        f"users.noreply.github.com address: {shown}. Set git config user.email to your "
        "no-reply address and recommit (merge, never force-push)"
    ), None


def _resolve(where, raw):
    path = Path(raw.strip("\"'"))
    return path if path.is_absolute() else Path(where) / path


def push_rules(command, cwd):
    """(denial or None, [warnings]) for the opt-in no-reply rule; silent when it is not set."""
    remote_match = os.environ.get(NOREPLY_REMOTE_ENV, "").strip()
    where, warnings = Path(cwd), []
    if not remote_match:
        return None, warnings
    for segment in _segments(command):
        moved = CD.match(segment)
        if moved:
            where = _resolve(where, moved.group(1).strip())
            continue
        push = _push_args(_tokens(segment))
        if push is not None:
            folder = _resolve(where, push[0]) if push[0] else where
            deny, warn = _noreply_push(folder, push[1], remote_match)
            if deny:
                return deny, warnings
            warnings += [warn] if warn else []
    return None, warnings


# --- dispatch ---------------------------------------------------------------------------------


def verdict(data: dict) -> tuple[str | None, str | None]:
    event, tool = data.get("hook_event_name"), data.get("tool_name")
    payload = data.get("tool_input") or {}
    cwd = data.get("cwd") or os.getcwd()
    if event == "PostToolUse":
        if tool in SHELLS:
            return None, post_bash(payload.get("command", ""), cwd)
        return None, None
    if tool in SHELLS:
        command = payload.get("command", "")
        reason = bash_denial(command)
        if reason:
            return "deny", reason
        reason, notes = push_rules(command, cwd)
        if reason:
            return "deny", reason
        return ("warn", "\n".join(notes)) if notes else (None, None)
    if tool in EDITS:
        return edit_verdict(payload)
    return None, None


def main() -> int:
    data = json.loads(sys.stdin.read() or "{}")
    kind, message = verdict(data)
    if kind == "deny":
        print(f"{TAG}: {message}", file=sys.stderr)
        return 2
    if message:
        event = data.get("hook_event_name", "PreToolUse")
        text = f"{TAG}: {message}"
        output = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}
        print(json.dumps(output | {"systemMessage": text}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
