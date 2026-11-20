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

import fnmatch
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

_DATE_TAIL = r"(?:-?\d{2}){0,2}(?!\d)"
_OPEN = r"(?:=|\s+)[\"']?"
_ANY_YEAR = r"20(?:2[1-9]|[3-9]\d)"
HOLDOUT_FLAG = re.compile(
    rf"--(?:(?:test[-_])?(?:end|as[-_]?of|before)(?:[-_]date)?{_OPEN}{_ANY_YEAR}"
    rf"|(?:test[-_])?start(?:[-_]date)?{_OPEN}202[1-5]){_DATE_TAIL}",
    re.I,
)
HOLDOUT_KWARG = re.compile(
    rf"\b(?:(?:end|as_of|before)\s*=\s*[\"']{_ANY_YEAR}|start\s*=\s*[\"']202[1-5]"
    rf"|Timestamp\(\s*[\"']{_ANY_YEAR}){_DATE_TAIL}",
    re.I,
)
_SECRET_NAME = r"(?:ANTHROPIC|WRDS|BROKER|GITHUB_TOKEN|GH_TOKEN)\w*"
SECRET = re.compile(r"\$\{?(?:env:)?" + _SECRET_NAME, re.I)
PRINT_VERB = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:echo|printf|print|Write-Host|Write-Output)\b|<<<", re.I
)
SECRET_READ = re.compile(
    r"^\s*(?:\$\()?\$env:" + _SECRET_NAME + r"\)?\s*$"
    r"|\bprintenv\s+[\"']?"
    + _SECRET_NAME
    + r"|\b(?:gci|get-childitem|dir|ls|gi|get-item|gc|get-content|type|cat)\b[^;&|\n]*?"
    r"\benv:(?:\\)?(?:\*?\s*$|" + _SECRET_NAME + r")"
    r"|\b(?:cat|less|more|head|tail|type|bat|strings|grep|sed|awk|cp|copy|get-content|gc)\b"
    r"[^;&|\n]*\.(?:pgpass|netrc)\b"
    r"|\bgh\s+auth\s+(?:token\b|status\b[^;&|\n]*--show-token)",
    re.I,
)
PY_ENV_READ = re.compile(
    r"(?:environ\s*\[|environ\.get\(|getenv\()\s*[\"']"
    + _SECRET_NAME
    + r"|\((?:dict\()?os\.environ\)|\benviron\.(?:items|copy|values)\(",
    re.I,
)
ENV_DUMP = re.compile(
    r"(?:^|[;&|]\s*)(?:env(?:\s+-\w+)*|printenv(?:\s+-\w+)*|export(?:\s+-p)?"
    r"|declare\s+-\w*[xp]\w*|typeset\s+-\w*x\w*|set)"
    r"(?:\s+\d*>>?&?\S*)*\s*(?:$|[;&|])",
    re.I,
)
LEDGER = "runs.jsonl"
LEDGER_WRITERS = set(
    "rm del erase mv move ren rename truncate tee shred remove-item ri move-item mi set-content "
    "sc add-content ac clear-content clc out-file ln".split()
)
LEDGER_COPIERS = {"cp", "copy", "copy-item", "cpi"}
IN_PLACE = re.compile(r"^-[a-zA-Z]*i[a-zA-Z0-9.]*$|^--in-place\b")
PY_LEDGER_WRITE = re.compile(
    r"\bopen\([^)]*,\s*(?:mode\s*=\s*)?[\"'][^\"']*[wax+][^\"']*[\"']"
    r"|\.(?:write_text|write_bytes|unlink|rename|replace|touch|truncate)\("
    r"|\bos\.(?:remove|unlink|rename|replace|truncate)\b|\bshutil\.",
)
WORKING_DATA = re.compile(
    r"runs\.jsonl|\.parquet\b|(?:^|/)data_cache(?:/|$)|(?:^|/)\.sessions(?:/|$)"
    r"|factors\.db|\.ipynb_checkpoints",
    re.I,
)
PROTECTED_BRANCH = re.compile(r"(?:^|[:/])(?:main|master|trunk)$")
WRAPPERS = {"sudo", "command", "env", "exec", "nohup", "nice", "xargs"}
WRAPPER_VALUE_OPTS = {"-u", "-g", "-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a", "-C", "-S"}
KEYWORDS = {
    "if",
    "then",
    "else",
    "elif",
    "do",
    "while",
    "until",
    "fi",
    "done",
    "!",
    "time",
    "{",
    "}",
}
PYTHON = re.compile(r"^python(?:\d+(?:\.\d+)*)?(?:\.exe)?$")
SHELL_C_CLUSTER = re.compile(r"^-[a-z]*c$")
NESTED = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
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


def bash_denial(command: str, cwd: str = ".") -> str | None:
    reason = path_denial(command)
    if reason:
        return reason
    where = Path(cwd)
    walked = list(_walk(command, where))
    python_runs = False
    for part, tokens, folder in walked:
        words = _command(tokens)
        if not words:
            continue
        runs_python = _runs_python(words)
        python_runs = python_runs or runs_python
        if runs_python and (HOLDOUT_FLAG.search(part) or HOLDOUT_KWARG.search(part)):
            return HOLDOUT_REASON
        if runs_python and PY_ENV_READ.search(part):
            return CREDENTIAL_REASON
        if _ledger_write(part, words, runs_python):
            return LEDGER_REASON
        call = _git_call(tokens)
        if call:
            reason = _git_denial(_resolve(folder, call[0]) if call[0] else folder, call[1], call[2])
            if reason:
                return reason
    if python_runs and re.search(r"<<-?\s*[\"']?\w", command):
        if HOLDOUT_FLAG.search(command) or HOLDOUT_KWARG.search(command):
            return HOLDOUT_REASON
    segments = re.split(r"&&|\|\||[;|\n]", command)
    if any(
        SECRET_READ.search(seg) or (SECRET.search(seg) and PRINT_VERB.search(seg))
        for seg in segments
    ):
        return CREDENTIAL_REASON
    if ENV_DUMP.search(command):
        return (
            "credentials: dumping the environment can print tokens; "
            "test one variable's presence instead"
        )
    return None


HOLDOUT_REASON = (
    "holdout: 2021-2025 stays sealed for discovery and stages 1-3; only the "
    "pre-registered stage-4 run reads it, once (pipeline-boundaries rule 1)"
)
CREDENTIAL_REASON = "credentials: never print ANTHROPIC_*/WRDS_*/broker/GitHub token variables"
LEDGER_REASON = "the ledger is append-only through capstone.runlog.log_run; never edit it"


def _is_ledger(word: str) -> bool:
    name = word.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].rsplit("=", 1)[-1]
    name = name.strip("\"'").lower()
    if any(c in name for c in "?*["):
        return fnmatch.fnmatchcase(LEDGER, name)
    return name == LEDGER


def _ledger_write(part: str, words: list[str], runs_python: bool) -> bool:
    """True when this segment writes, moves or deletes the ledger; reads and copies out are fine."""
    cleaned = re.sub(r"\d*>&\d*-?|<&\d*", " ", part)
    if any(_is_ledger(m.group(1)) for m in re.finditer(r">>?\|?\s*(\S+)", cleaned)):
        return True
    if runs_python:
        return LEDGER in part.lower() and PY_LEDGER_WRITE.search(part) is not None
    named = [w for w in words[1:] if _is_ledger(w)]
    if not named:
        return False
    cmd = Path(words[0].replace("\\", "/")).name.lower()
    rest = [w.lower() for w in words[1:]]
    if cmd in LEDGER_COPIERS:
        dest = [words[i + 2] for i, w in enumerate(rest[:-1]) if w in ("-destination", "-dest")]
        return _is_ledger((dest or words[-1:])[-1])
    if cmd in LEDGER_WRITERS or any(w in LEDGER_WRITERS - {"ln"} for w in rest):
        return True
    if cmd in ("sed", "gsed", "perl", "awk", "gawk"):
        return any(IN_PLACE.match(w) for w in words[1:]) or (
            "inplace" in rest and cmd in ("awk", "gawk")
        )
    return cmd == "find" and "-delete" in rest


def edit_verdict(payload: dict, cwd: str = ".") -> tuple[str | None, str | None]:
    raw = str(payload.get("file_path", ""))
    reason = path_denial(raw)
    if reason:
        return "deny", reason
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


def _split(command, aware=True):
    """Split a command line at shell operators outside quotes: ; | & && || newline ( ) { }
    and backticks. Redirections (2>&1, &>f) stay whole. An unterminated quote falls back to
    splitting everywhere, so a stray apostrophe cannot hide a command."""
    parts, buf, quote, i, n = [], [], None, 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 1
            elif ch == quote:
                quote = None
        elif aware and ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "&" and (
            (i + 1 < n and command[i + 1] == ">") or (i and command[i - 1] in "<>")
        ):
            buf.append(ch)
        elif ch == "{" and i and command[i - 1] == "$":
            buf.append(ch)
        elif ch in ";|&\n(){}`":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if quote and aware:
        return _split(command, aware=False)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


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


def _command(tokens):
    """The tokens from the command word on: env assignments, shell keywords and wrappers
    (sudo, env, xargs ... with their options) skipped."""
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if re.match(r"^\w+=", word) or word.lower() in KEYWORDS:
            i += 1
        elif word.lower() in WRAPPERS:
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 2 if tokens[i] in WRAPPER_VALUE_OPTS else 1
        else:
            break
    return tokens[i:]


def _runs_python(words):
    if not words:
        return False
    first = Path(words[0].replace("\\", "/")).name.lower()
    return bool(PYTHON.match(first)) or (first == "uv" and len(words) > 1 and words[1] == "run")


def _walk(command, where, depth=0):
    """(segment text, its tokens, the working folder) for every segment of a command line,
    following `cd`, `eval`, `bash -c "..."` strings and $(...) or backtick substitutions."""
    if depth > 4:
        return
    for part in _split(command):
        moved = CD.match(part)
        if moved:
            where = _resolve(where, moved.group(1).strip())
            continue
        tokens = _tokens(part)
        yield part, tokens, where
        for inner, ticked in NESTED.findall(part):
            yield from _walk(inner or ticked, where, depth + 1)
        words = _command(tokens)
        if words and words[0].lower() == "eval":
            yield from _walk(" ".join(words[1:]), where, depth + 1)
        for before, word in zip(words, words[1:], strict=False):
            is_shell_c = before.lower() in SHELL_C or SHELL_C_CLUSTER.match(before.lower())
            if is_shell_c and re.search(r"\s", word):
                yield from _walk(word, where, depth + 1)


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
    Skips env assignments, wrappers like sudo and xargs, and git's global options."""
    words = _command(tokens)
    if not words or Path(words[0].replace("\\", "/")).name.lower() not in ("git", "git.exe"):
        return None
    i, where = 1, None
    while i < len(words) and words[i].startswith("-"):
        if words[i] == "-C" and i + 1 < len(words):
            where = words[i + 1]
        i += 2 if words[i] in GLOBAL_VALUE_OPTS else 1
    if i >= len(words):
        return None
    return where, words[i].lower(), words[i + 1 :]


def _forced(args):
    return any(
        a in ("--force", "--force-if-includes", "--mirror")
        or (a.startswith("--for") and "--force".startswith(a))
        or (a.startswith("-") and not a.startswith("--") and "f" in a[1:])
        for a in args
    )


def _forced_add(args):
    for a in args:
        if a == "--":
            return False
        if a == "--force" or (a.startswith("--for") and "--force".startswith(a)):
            return True
        if a.startswith("-") and not a.startswith("--") and "f" in a[1:]:
            return True
    return False


def _git_denial(where, sub, args):
    if sub == "push":
        plain = [a for a in args if not a.startswith("-")]
        if _forced(args) or any(a.startswith("+") for a in plain):
            return (
                "git push --force, --mirror or a +refspec rewrites shared history; merge "
                "instead, and use --force-with-lease only on your own PR branch"
            )
        if any(a.startswith("--force-with-lease") for a in args):
            refspecs = plain[1:]
            if not refspecs or refspecs == ["HEAD"]:
                branch = _git(where, "symbolic-ref", "--quiet", "--short", "HEAD") or ""
                refspecs = [branch]
            if any(PROTECTED_BRANCH.search(a) for a in refspecs):
                return "git push --force-with-lease to main rewrites shared history"
    if sub in ("add", "stage"):
        if _forced_add(args):
            return (
                "git add --force stages what .gitignore keeps out of this public repo (the run "
                "ledger, .sessions, data); add the files you mean without -f"
            )
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
            continue
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
    warnings = []
    if not remote_match:
        return None, warnings
    for _, tokens, folder in _walk(command, Path(cwd)):
        call = _git_call(tokens)
        if call and call[1] == "push":
            where = _resolve(folder, call[0]) if call[0] else folder
            deny, warn = _noreply_push(where, call[2], remote_match)
            if deny:
                return deny, warnings
            warnings += [warn] if warn else []
    return None, warnings


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
        reason = bash_denial(command, cwd)
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
