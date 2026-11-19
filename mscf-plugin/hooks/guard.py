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

# Discovery ends 2020-12-31: a start inside 2021-2025 or an end/as-of/before from 2021 on
# touches the holdout. Dates may be dashed or compact (2023-06-30, 20230630).
_DATE_TAIL = r"(?:-?\d{2}){0,2}(?!\d)"
_OPEN = r"(?:=|\s+)[\"']?"
_ANY_YEAR = r"20(?:2[1-9]|[3-9]\d)"
HOLDOUT_FLAG = re.compile(
    rf"--(?:(?:end|as-of|before){_OPEN}{_ANY_YEAR}|start{_OPEN}202[1-5]){_DATE_TAIL}", re.I
)
HOLDOUT_KWARG = re.compile(
    rf"\b(?:(?:end|as_of|before)\s*=\s*[\"']{_ANY_YEAR}|start\s*=\s*[\"']202[1-5]){_DATE_TAIL}",
    re.I,
)
_SECRET_NAME = r"(?:ANTHROPIC|WRDS|BROKER|GITHUB_TOKEN|GH_TOKEN)\w*"
SECRET = re.compile(r"\$\{?(?:env:)?" + _SECRET_NAME, re.I)
PRINT_VERB = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:echo|printf|print|Write-Host|Write-Output)\b|<<<", re.I
)
# reading one secret without a `$`: printenv NAME, a PowerShell env: drive listing or item,
# or a segment that is nothing but the $env:NAME expression
SECRET_READ = re.compile(
    r"^\s*(?:\$\()?\$env:" + _SECRET_NAME + r"\)?\s*$"
    r"|\bprintenv\s+[\"']?"
    + _SECRET_NAME
    + r"|\b(?:gci|get-childitem|dir|ls|gi|get-item|gc|get-content|type|cat)\b[^;&|\n]*?"
    r"\benv:(?:\\)?(?:\*?\s*$|" + _SECRET_NAME + r")",
    re.I,
)
LEDGER_WRITE = re.compile(
    r">|\b(?:rm|del|erase|mv|move|ren|rename|cp|copy|truncate|tee|sed\s+-\w*i|remove-item|ri"
    r"|move-item|set-content|add-content|clear-content|out-file)\b",
    re.I,
)
ENV_DUMP = re.compile(r"(?:^|[;&|]\s*)(?:env|printenv|export\s+-p|set)\s*(?:$|[;&|])", re.I)
WORKING_DATA = re.compile(
    r"runs\.jsonl|\.parquet\b|(?:^|/)data_cache(?:/|$)|(?:^|/)\.sessions(?:/|$)"
    r"|factors\.db|\.ipynb_checkpoints",
    re.I,
)
PROTECTED_BRANCH = re.compile(r"(?:^|[:/])(?:main|master|trunk)$")
WRAPPERS = {"sudo", "command", "env", "time", "exec", "nohup", "nice"}
GLOBAL_VALUE_OPTS = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--super-prefix",
}
SHELL_C = {"-c", "-lc", "-command"}
REDIR_ALONE = re.compile(r"^(?:\d*|&)(?:>>?|<)$")
REDIR_ATTACHED = re.compile(r"^(?:\d*|&)(?:>>?|<)&?\S+$")
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
    if runs_python and (HOLDOUT_FLAG.search(command) or HOLDOUT_KWARG.search(command)):
        return (
            "holdout: 2021-2025 stays sealed for discovery and stages 1-3; only the "
            "pre-registered stage-4 run reads it, once (pipeline-boundaries rule 1)"
        )
    for call in _git_calls(command):
        reason = _git_denial(call)
        if reason:
            return reason
    # a presence check ("test -n $KEY && echo set") is fine; printing the value is not
    segments = re.split(r"&&|\|\||[;|\n]", command)
    if any(
        SECRET_READ.search(seg) or (SECRET.search(seg) and PRINT_VERB.search(seg))
        for seg in segments
    ):
        return "credentials: never print ANTHROPIC_*/WRDS_*/broker/GitHub token variables"
    if ENV_DUMP.search(command):
        return (
            "credentials: dumping the environment can print tokens; "
            "test one variable's presence instead"
        )
    for seg in segments:
        if "runs.jsonl" in seg.lower() and LEDGER_WRITE.search(re.sub(r"\d*>&\d+", "", seg)):
            return "the ledger is append-only through capstone.runlog.log_run; never edit it"
    return None


# --- edits ------------------------------------------------------------------------------------


def edit_verdict(payload: dict, cwd: str = ".") -> tuple[str | None, str | None]:
    raw = str(payload.get("file_path", ""))
    reason = path_denial(raw)
    if reason:
        return "deny", reason
    # resolve against the hook's folder and fold case: macOS and Windows disks ignore it
    path = os.path.normpath(os.path.join(cwd, raw)).replace("\\", "/").rstrip(" .").lower()
    if path.endswith("/runs.jsonl"):
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
    return (
        f"ledger file now holds {count} entries, corrections included; the trial count is "
        "what `python -m capstone.runlog stats` reports"
    )


# --- opt-in: no-reply emails on pushes to a named remote ---------------------------------------


def _segments(command):
    return [s.strip() for s in re.split(r"\|\||&&|[|;\n]", command) if s.strip()]


def _tokens(segment):
    """The words of one shell segment, quotes stripped, redirections (and targets) gone."""
    words = [t.strip("\"'") for t in re.findall(r"\"[^\"]*\"|'[^']*'|\S+", segment)]
    kept, skip = [], False
    for word in words:
        if skip:
            skip = False
        elif REDIR_ALONE.match(word):
            skip = True
        elif not REDIR_ATTACHED.match(word):
            kept.append(word)
    return kept


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


def _git_call(tokens):
    """(the -C folder or None, the subcommand, its arguments) for a git command, else None.
    Skips env assignments, wrappers like sudo, and git's global options (-C, -c, --git-dir)."""
    i = 0
    while i < len(tokens) and (re.match(r"^\w+=", tokens[i]) or tokens[i].lower() in WRAPPERS):
        i += 1
    if i >= len(tokens):
        return None
    if Path(tokens[i].replace("\\", "/")).name.lower() not in ("git", "git.exe"):
        return None
    i, where = i + 1, None
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] == "-C" and i + 1 < len(tokens):
            where = tokens[i + 1]
        i += 2 if tokens[i] in GLOBAL_VALUE_OPTS else 1
    if i >= len(tokens):
        return None
    return where, tokens[i].lower(), tokens[i + 1 :]


def _git_calls(command):
    """Every git call in a command line, including ones inside `bash -c "..."` strings."""
    calls = []
    for segment in _segments(command):
        tokens = _tokens(segment)
        call = _git_call(tokens)
        if call:
            calls.append(call)
        for before, word in zip(tokens, tokens[1:], strict=False):
            if before.lower() in SHELL_C and re.search(r"\s", word):
                calls += _git_calls(word)
    return calls


def _push_args(tokens):
    """(the -C folder or None, the arguments after `push`), or None if not a git push."""
    call = _git_call(tokens)
    return (call[0], call[2]) if call and call[1] == "push" else None


def _forced(args):
    return any(
        a in ("--force", "--force-if-includes")
        or (a.startswith("-") and not a.startswith("--") and "f" in a[1:])
        for a in args
    )


def _git_denial(call):
    _, sub, args = call
    if sub == "push":
        plain = [a for a in args if not a.startswith("-")]
        if _forced(args) or any(a.startswith("+") for a in plain):
            return (
                "git push --force (or a +refspec) rewrites shared history; merge instead, "
                "and use --force-with-lease only on your own PR branch"
            )
        if any(a.startswith("--force-with-lease") for a in args) and any(
            PROTECTED_BRANCH.search(a) for a in plain
        ):
            return "git push --force-with-lease to main rewrites shared history"
    if sub in ("add", "stage"):
        for arg in args:
            if not arg.startswith("-") and WORKING_DATA.search(arg.replace("\\", "/")):
                return (
                    "public repo: the run ledger, parquet data, data_cache, .sessions and "
                    "notebook checkpoints are never committed; results go in the PR description"
                )
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
        return edit_verdict(payload, cwd)
    return None, None


def main() -> int:
    # fail closed: only exit 2 blocks, so unreadable input or an internal error must exit 2
    try:
        data = json.loads(sys.stdin.read())
        if not isinstance(data, dict):
            raise ValueError("hook input is not a JSON object")
        kind, message = verdict(data)
    except Exception as exc:  # noqa: BLE001
        print(
            f"{TAG}: could not evaluate the hook input ({type(exc).__name__}); denied",
            file=sys.stderr,
        )
        return 2
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
