"""The mscf-pipeline guard hook (mscf-plugin/hooks/guard.py): what it denies, what it leaves
alone, and its two opt-in rules (guarded paths, no-reply pushes).

Every case runs the hook as the plugin does, JSON on stdin. Git repos are local temp folders;
nothing is pushed and no network is used (a remote's URL is only read, never contacted)."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / "mscf-plugin" / "hooks" / "guard.py"
OPT_IN = ("MSCF_GUARD_DENY_PATHS", "MSCF_GUARD_NOREPLY_REMOTE", "CAPSTONE_LEDGER_DIR")
NOREPLY = "1+someone@users.noreply.github.com"
TEAM_URL = "https://github.com/example-org/cmu-mscf-2026.git"


def guard(data: dict, env: dict | None = None, cwd: Path | None = None):
    base = {k: v for k, v in os.environ.items() if k not in OPT_IN and not k.startswith("GIT_")}
    return subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
        env={**base, **(env or {})},
    )


def bash(command: str, event: str = "PreToolUse", cwd: Path | None = None) -> dict:
    return {
        "hook_event_name": event,
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "cwd": str(cwd or ROOT),
    }


def edit(tool: str, path: Path, **fields) -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "cwd": str(ROOT),
        "tool_input": {"file_path": str(path), **fields},
    }


def denied(done) -> bool:
    return done.returncode == 2 and "mscf-pipeline guard" in done.stderr


def note(done) -> str:
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["systemMessage"] if done.stdout else ""


@pytest.mark.parametrize(
    "command",
    [
        "python -m experiments.momentum --end 2023-06-30",
        "python -c \"sd.load('sp500_daily_1990_2025', end='2022-01-01')\"",
        "git push --force origin main",
        "git push -f",
        "echo $ANTHROPIC_API_KEY",
        "Write-Host $env:WRDS_PASSWORD",
        "printenv",
        "git add experiments/runs.jsonl",
        "git add data_cache/wrds/sp500_daily_1990_2025.parquet",
        "git -C . push --force origin main",
        "git -c push.default=simple push -f origin main",
        "git --git-dir=.git push --force origin main",
        "git --git-dir .git push origin main -f",
        "git push origin +main",
        "git push origin +HEAD:main",
        "git push -fu origin topic",
        "git push --force-with-lease origin main",
        "git push --force-with-lease origin HEAD:main",
        "sudo git push -f",
        'bash -c "git push --force origin main"',
        "git add -f .sessions",
        "git add -f .sessions/2026-11-19.jsonl",
        "git stage -f experiments/runs.jsonl",
        "git stage -f .sessions",
        "git -C . add -f data_cache",
        "git add -f experiments/RUNS.JSONL",
        "python -m experiments.momentum --end 2026-06-30",
        "python -m experiments.momentum --end 20230630",
        "python -m experiments.momentum --as-of=2024-01-01",
        "python -c \"sd.load('x', end='2026-01-01')\"",
        "python -c \"sd.load('x', end='20260101')\"",
        "python -m experiments.momentum --start 20210104",
        "printenv ANTHROPIC_API_KEY",
        "cat <<< $ANTHROPIC_API_KEY",
        "$env:ANTHROPIC_API_KEY",
        "gci env:ANTHROPIC_API_KEY",
        "Get-ChildItem env:",
        "Get-Item env:WRDS_PASSWORD",
        "echo x >> experiments/runs.jsonl",
        "rm experiments/runs.jsonl",
        "sed -i '$d' experiments/runs.jsonl",
        "Remove-Item experiments/RUNS.jsonl",
        "(git push -f origin main)",
        "{ git push -f origin main; }",
        "if true; then git push -f origin main; fi",
        "! git push --force origin main",
        "true & git push -f origin main",
        "eval 'git push -f origin main'",
        "eval git push -f origin main",
        "xargs git push --force",
        "xargs -n1 git push -f",
        "git push --mirror",
        "git push --mirror origin",
        "echo $(git push -f origin main)",
        "echo `git push -f origin main`",
        'echo "$(git push -f origin main)"',
        "env -i git push -f origin main",
        "bash -ec 'git push -f origin main'",
        "git push --for origin topic",
        "git add -f .",
        "git add -f -A",
        "git add -fA",
        "git add --force experiments",
        "git add -f experiments/",
        "git add -f ':/'",
        "git add -f 'exp*'",
        "git add -f mscf-plugin",
        "git -C . add --force .",
        "python -m experiments.momentum --end-date 2026-06-30",
        "python -m experiments.momentum --end_date=2026-06-30",
        "python -m experiments.momentum --test-start 2022-01-01",
        "python -m experiments.momentum --start-date 2021-01-04",
        "python3 -m experiments.momentum --end 2023-01-01",
        "uv run python s.py --end 2023-01-01",
        "uv run s.py --end 2023-01-01",
        "cd experiments && python s.py --end 2023-01-01",
        'bash -c "python s.py --end 2023-01-01"',
        "python -c \"import pandas as pd; pd.Timestamp('2023-01-01')\"",
        "cat ~/.pgpass",
        "cat ~/.netrc",
        "gh auth token",
        "export",
        "export -p",
        "declare -x",
        "env -0",
        "env -0 2>&1",
        "python -c \"import os; print(os.environ['ANTHROPIC_API_KEY'])\"",
        'python -c "import os; print(dict(os.environ))"',
        "python -c \"import os; print(os.getenv('WRDS_PASSWORD'))\"",
        "sed --in-place '$d' experiments/runs.jsonl",
        "sed -i.bak '$d' experiments/runs.jsonl",
        "perl -pi -e 's/a/b/' experiments/runs.jsonl",
        "awk -i inplace '1' experiments/runs.jsonl",
        "python -c \"open('experiments/runs.jsonl', 'w').write('')\"",
        "python -c \"open('experiments/runs.jsonl', mode='a')\"",
        "python -c \"import pathlib; pathlib.Path('experiments/runs.jsonl').write_text('')\"",
        "rm experiments/run?.jsonl",
        "rm experiments/*.jsonl",
        "mv experiments/runs.jsonl /tmp/gone.jsonl",
        "cp /tmp/x.jsonl experiments/runs.jsonl",
        "echo x | tee -a experiments/runs.jsonl",
        "git rm experiments/runs.jsonl",
        "find experiments -name runs.jsonl -delete",
        "Copy-Item -Destination experiments/runs.jsonl -Path x.jsonl",
        "bash -c 'rm experiments/runs.jsonl'",
        "> experiments/runs.jsonl",
        ">experiments/runs.jsonl",
        "cd experiments && > runs.jsonl",
        "git push origin :main",
        "git push origin :master",
        "git push origin :refs/heads/main",
        "git push --delete origin main",
        "git push origin --delete main",
        "git push -d origin main",
    ],
)
def test_denies(command):
    assert denied(guard(bash(command)))


@pytest.mark.parametrize(
    "command",
    [
        "python -m experiments.momentum --end 2020-12-31",
        'git commit -m "keep 2021-2025 sealed"',
        "git push --force-with-lease origin cal/topic",
        "git add mscf-plugin tests",
        "pytest -q -m 'not network'",
        'test -n "$ANTHROPIC_API_KEY" && echo set',
        "git -C ~/code/cmu-mscf-2026 status",
        r"git -C D:\work\cmu-mscf-2026 status",
        "git log --before=2024-01-01 --oneline",
        "python -m experiments.momentum --end 2020-12-31 --start 2010-01-01",
        "git -C . push origin topic",
        "git push origin topic 2>&1 | tail -5",
        "git add -u",
        "git stage mscf-plugin",
        "Get-ChildItem Env:PATH",
        'if ($env:ANTHROPIC_API_KEY) { "set" }',
        "tail -n 3 experiments/runs.jsonl 2>&1",
        "python -m capstone.runlog stats",
        'git commit -m "run python s.py --end 2022-12-31 later"',
        'gh pr create --body "we ran python -m scripts.x --start 2021-01-01"',
        'git commit -m "never git push -f; open a PR"',
        "echo 'git push --force is denied'",
        "git add -n .",
        "git add -- mscf-plugin",
        "cp experiments/runs.jsonl /tmp/backup.jsonl",
        "cp experiments/runs.jsonl /tmp/backup.jsonl 2>&1",
        "wc -l experiments/runs.jsonl",
        "cat experiments/runs.jsonl > /tmp/copy.jsonl",
        "grep rm experiments/notes.txt",
        "export FOO=bar",
        "env FOO=1 pytest -q",
        "gh auth status",
        "cat ~/.gitconfig",
        "python -c \"import os; print('ANTHROPIC_API_KEY' in os.environ)\"",
        "python -c \"import os; print(os.environ.get('HOME'))\"",
        "python -c \"import pandas as pd; pd.Timestamp('2019-01-01')\"",
        "python -c \"print(open('experiments/runs.jsonl').read())\"",
    ],
)
def test_allows_including_a_clone_of_the_team_repo(command):
    done = guard(bash(command))
    assert done.returncode == 0 and not done.stderr


def test_a_forced_add_is_denied_in_a_scratch_repo_where_it_would_stage_working_data(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    git(scratch, "init", "-q", "-b", "topic")
    (scratch / ".gitignore").write_text(".sessions/\nexperiments/runs.jsonl\n", encoding="utf-8")
    (scratch / ".sessions").mkdir()
    (scratch / ".sessions" / "s.jsonl").write_text("{}", encoding="utf-8")
    (scratch / "experiments").mkdir()
    (scratch / "experiments" / "runs.jsonl").write_text("{}", encoding="utf-8")
    (scratch / "ok.py").write_text("x = 1", encoding="utf-8")

    git(scratch, "add", "-f", ".")
    staged = git(scratch, "ls-files").split()
    assert ".sessions/s.jsonl" in staged and "experiments/runs.jsonl" in staged
    git(scratch, "rm", "-r", "-q", "--cached", ".")

    for command in ["git add -f .", "git add -f experiments", "git add -f -A", "git add -f :/"]:
        assert denied(guard(bash(command, cwd=scratch), cwd=scratch)), command
    for command in ["git add .", "git add ok.py", "git add -A"]:
        done = guard(bash(command, cwd=scratch), cwd=scratch)
        assert done.returncode == 0 and not done.stderr, command


def test_force_with_lease_without_a_refspec_is_denied_only_on_main(tmp_path):
    scratch = tmp_path / "lease"
    scratch.mkdir()
    git(scratch, "init", "-q", "-b", "main")
    commit(scratch, "a.txt")
    for command in [
        "git push --force-with-lease",
        "git push --force-with-lease origin",
        "git push --force-with-lease origin HEAD",
        "git push origin --force-with-lease",
    ]:
        assert denied(guard(bash(command, cwd=scratch), cwd=scratch)), command
    git(scratch, "checkout", "-q", "-b", "cal/topic")
    for command in ["git push --force-with-lease", "git push --force-with-lease origin"]:
        done = guard(bash(command, cwd=scratch), cwd=scratch)
        assert done.returncode == 0 and not done.stderr, command


def test_denies_editing_the_ledger_and_data_and_warns_on_holdout_dates_in_code():
    assert denied(guard(edit("Write", ROOT / "experiments" / "runs.jsonl", content="{}")))
    assert denied(guard(edit("Write", ROOT / "data_cache" / "x.csv", content="a")))
    warned = guard(
        edit("Edit", ROOT / "experiments" / "x.py", old_string="a", new_string="end='2023-01-01'")
    )
    assert "holdout" in note(warned)
    multi = guard(
        edit(
            "MultiEdit",
            ROOT / "x.py",
            edits=[{"old_string": "a", "new_string": 'start="2022-01-01"'}],
        )
    )
    assert "holdout" in note(multi)
    assert (
        guard(edit("Edit", ROOT / "x.py", old_string="a", new_string="end='2020-12-31'")).stdout
        == ""
    )


@pytest.mark.parametrize(
    "path",
    [
        "experiments/RUNS.jsonl",
        "experiments/Runs.JSONL",
        "experiments/../experiments/runs.jsonl",
        "EXPERIMENTS\\Runs.jsonl",
        "data_cache/X.PARQUET",
    ],
)
def test_denies_editing_the_ledger_and_data_whatever_the_case_or_dots(path):
    for tool in ("Edit", "Write", "MultiEdit"):
        assert denied(guard(edit(tool, ROOT / path, content="x")))
    assert denied(guard(edit("Edit", Path(path), new_string="x")))


@pytest.mark.parametrize(
    "raw", ["", "not json", "[1, 2]", "null", '{"tool_name": "Bash", "tool_input": 5}']
)
def test_unreadable_or_malformed_hook_input_fails_closed(raw):
    done = subprocess.run(
        [sys.executable, str(GUARD)], input=raw, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 2 and "Traceback" not in done.stderr and "denied" in done.stderr


def test_reports_the_ledger_count_after_an_experiment_script(tmp_path):
    (tmp_path / "runs.jsonl").write_text('{"name": "a"}\n{"name": "b"}\n', encoding="utf-8")
    post = bash("python experiments/momentum.py", "PostToolUse")
    done = guard(post, {"CAPSTONE_LEDGER_DIR": str(tmp_path)})
    assert "ledger file now holds 2 entries" in note(done)
    empty = guard(post, {"CAPSTONE_LEDGER_DIR": str(tmp_path / "none")})
    assert "no ledger found" in note(empty)
    assert guard(bash("ls", "PostToolUse")).stdout == ""


def test_no_path_is_guarded_unless_configured():
    for command in [r"git -C D:\other\clone status", "ls /srv/shared/clone"]:
        assert guard(bash(command)).returncode == 0


def test_guarded_paths_deny_commands_and_edits_that_name_them(tmp_path):
    sep = os.pathsep
    env = {"MSCF_GUARD_DENY_PATHS": sep.join(["/srv/shared/clone", r"D:\other\clone"])}
    for command in [
        "ls /srv/shared/clone/data",
        r"git -C D:\other\clone status",
        "cd /SRV/shared/clone",
    ]:
        done = guard(bash(command), env)
        assert denied(done) and "MSCF_GUARD_DENY_PATHS" in done.stderr, command
    assert denied(guard(edit("Write", Path("/srv/shared/clone/a.py"), content="x"), env))
    for command in [
        "ls /srv/shared/clone-two",
        "ls /srv/shared/cloned",
        "git -C ~/cmu-mscf-2026 log",
    ]:
        assert guard(bash(command), env).returncode == 0, command
    assert guard(edit("Write", tmp_path / "ok.py", content="x"), env).returncode == 0


def git(repo: Path, *args: str, email: str = NOREPLY) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": email,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(repo.parent),
        "PATH": os.environ["PATH"],
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }
    done = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return done.stdout.strip()


def commit(repo: Path, name: str, email: str = NOREPLY) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name, email=email)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    """A repo on branch `topic` whose `origin` is a team-repo URL and whose origin/topic is the
    first commit (a remote-tracking ref only; nothing is fetched)."""
    path = tmp_path / "team"
    path.mkdir()
    git(path, "init", "-q", "-b", "topic")
    git(path, "remote", "add", "origin", TEAM_URL)
    git(path, "remote", "add", "other", "https://github.com/example-org/elsewhere.git")
    git(path, "update-ref", "refs/remotes/origin/topic", commit(path, "a.txt", "me@example.com"))
    return path


ON = {"MSCF_GUARD_NOREPLY_REMOTE": "cmu-mscf-2026"}


def test_the_noreply_rule_is_off_unless_configured(repo):
    commit(repo, "b.txt", "me@example.com")
    done = guard(bash("git push origin topic", cwd=repo), cwd=repo)
    assert done.returncode == 0 and not done.stderr


def test_denies_a_push_carrying_a_non_noreply_email(repo):
    commit(repo, "b.txt", "me@example.com")
    for command in [
        "git push",
        "git push origin topic",
        "git push -u origin HEAD:topic",
        "git push > push.log",
        "git push origin topic 2>&1",
        "git push origin topic 2>&1 | tail -5",
        "git push origin topic >push.log 2>&1",
        "git -C . push origin topic",
    ]:
        done = guard(bash(command, cwd=repo), ON, repo)
        assert denied(done) and "users.noreply.github.com" in done.stderr, (command, done)


def test_allows_a_push_whose_commits_are_all_noreply(repo):
    commit(repo, "b.txt")
    for command in ["git push", "git push origin topic", f"git -C {repo} push origin topic"]:
        done = guard(bash(command, cwd=repo.parent if "-C" in command else repo), ON, repo)
        assert done.returncode == 0 and not done.stderr, (command, done)


def test_a_new_branch_push_checks_commits_not_on_any_remote_ref(repo):
    git(repo, "checkout", "-q", "-b", "fresh")
    commit(repo, "c.txt", "me@example.com")
    assert denied(guard(bash("git push -u origin fresh", cwd=repo), ON, repo))
    assert denied(guard(bash(f"cd {repo} && git push -u origin fresh", cwd=repo.parent), ON))


def test_pushes_elsewhere_and_deletes_are_not_checked(repo):
    commit(repo, "b.txt", "me@example.com")
    assert guard(bash("git push other topic", cwd=repo), ON, repo).returncode == 0
    assert guard(bash("git push origin --delete old", cwd=repo), ON, repo).returncode == 0
    assert guard(bash("git push origin :old", cwd=repo), ON, repo).returncode == 0


def test_an_undeterminable_push_warns_rather_than_denies(repo):
    for command in ["git push --all origin", f"git push {TEAM_URL} topic"]:
        done = guard(bash(command, cwd=repo), ON, repo)
        assert "could not work out" in note(done), command


def test_guard_source_has_no_python_310_syntax():
    source = (ROOT / "mscf-plugin" / "hooks" / "guard.py").read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    assert not any(isinstance(n, ast.Match) for n in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "zip":
            assert all(k.arg != "strict" for k in node.keywords)


@pytest.mark.parametrize(
    "command", ["git push origin :feature", "git push --delete origin cal/topic"]
)
def test_deleting_a_non_protected_remote_branch_is_allowed(command):
    assert not denied(guard(bash(command)))
