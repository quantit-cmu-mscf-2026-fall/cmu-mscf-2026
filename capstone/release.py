"""Record the release and runtime versions an artifact was produced under."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

RELEASE_STAMP = "RELEASE"  # written into an exported tree: commit, tag and file hashes
REPO = Path(__file__).resolve().parents[1]


def runtime_versions():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "pyarrow", "statsmodels", "requests")
        },
    }


def _check_tag(repo, deployment_tag, commit, resolve_tag):
    if deployment_tag:
        if resolve_tag is None:
            raise ValueError("a deployment tag needs a tag checker")
        if resolve_tag(repo, deployment_tag) != commit:
            raise ValueError("deployment tag does not identify the running code")


def stamped_identity(repo, deployment_tag=None, *, resolve_tag=None):
    """`release_identity` of an exported tree: commit and tag from its RELEASE stamp; dirty
    when any exported file changed or is missing (files outside the export are ignored).
    A deployment tag also requires the independent `resolve_tag` checker.
    """
    repo = Path(repo)
    path = Path(repo) / RELEASE_STAMP
    if not path.is_file():
        raise ValueError(
            f"{repo} is neither a git checkout nor an exported release with a {RELEASE_STAMP} stamp"
        )
    stamp = json.loads(path.read_text(encoding="utf-8"))
    files = stamp.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("release stamp needs a nonempty file manifest")
    dirty = False
    for name, expected in files.items():
        item = Path(repo) / name
        if not item.is_file() or hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            dirty = True
            break
    if deployment_tag and deployment_tag != stamp.get("tag"):
        raise ValueError("deployment tag does not identify the running code")
    _check_tag(repo, deployment_tag, stamp["commit"], resolve_tag)
    return {"commit": stamp["commit"], "dirty": dirty, "deployment_tag": deployment_tag}


def release_identity(deployment_tag=None, *, repo=None, resolve_tag=None):
    """The commit, dirty flag and deployment tag of the running code.

    `resolve_tag(repo, tag) -> commit` checks a deployment tag; it is injected because tag
    policy belongs to the deploy layer. Naming a tag without a checker is refused.
    """
    repo = Path(repo) if repo is not None else REPO
    if not (repo / ".git").exists():  # a worktree's .git is a file; an export has neither
        return stamped_identity(repo, deployment_tag, resolve_tag=resolve_tag)

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True
        ).stdout.strip()

    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    _check_tag(repo, deployment_tag, sha, resolve_tag)
    return {"commit": sha, "dirty": dirty, "deployment_tag": deployment_tag}
