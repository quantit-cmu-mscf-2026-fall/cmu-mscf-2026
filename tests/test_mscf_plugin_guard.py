"""The mscf-pipeline guard hook (mscf-plugin/hooks/guard.py): what it denies, what it leaves
alone, and its two opt-in rules (guarded paths, no-reply pushes).

Every case runs the hook as the plugin does, JSON on stdin. Git repos are local temp folders;
nothing is pushed and no network is used (a remote's URL is only read, never contacted)."""

from __future__ import annotations

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


# --- the always-on rules ----------------------------------------------------------------------


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
    ],
)
def test_allows_including_a_clone_of_the_team_repo(command):
    done = guard(bash(command))
    assert done.returncode == 0 and not done.stderr


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


def test_reports_the_ledger_count_after_an_experiment_script(tmp_path):
    (tmp_path / "runs.jsonl").write_text('{"name": "a"}\n{"name": "b"}\n', encoding="utf-8")
    post = bash("python experiments/momentum.py", "PostToolUse")
    done = guard(post, {"CAPSTONE_LEDGER_DIR": str(tmp_path)})
    assert "ledger now holds 2 runs" in note(done)
    empty = guard(post, {"CAPSTONE_LEDGER_DIR": str(tmp_path / "none")})
    assert "no ledger found" in note(empty)
    assert guard(bash("ls", "PostToolUse")).stdout == ""


# --- opt-in: guarded paths --------------------------------------------------------------------


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
    # a sibling with the same stem, and everything else (the team repo included), is untouched
    for command in [
        "ls /srv/shared/clone-two",
        "ls /srv/shared/cloned",
        "git -C ~/cmu-mscf-2026 log",
    ]:
        assert guard(bash(command), env).returncode == 0, command
    assert guard(edit("Write", tmp_path / "ok.py", content="x"), env).returncode == 0


# --- opt-in: no-reply emails on pushes --------------------------------------------------------


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
    for command in ["git push", "git push origin topic", "git push -u origin HEAD:topic"]:
        done = guard(bash(command, cwd=repo), ON, repo)
        assert denied(done) and "users.noreply.github.com" in done.stderr, (command, done)


def test_allows_a_push_whose_commits_are_all_noreply(repo):
    commit(repo, "b.txt")  # the bad first commit is already on origin/topic
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
