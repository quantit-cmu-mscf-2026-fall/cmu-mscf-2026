"""scripts/export_release.py: a byte-reproducible archive with a RELEASE stamp.

Uses a throwaway git repository built in `tmp_path` (synthetic files only); nothing is
pushed and no network is touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_release.py"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="exporting needs git")


def load():
    spec = importlib.util.spec_from_file_location("export_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(repo, *args):
    config = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "tag.gpgsign=false",
        "-c",
        "core.autocrlf=false",
    ]
    dates = {"GIT_COMMITTER_DATE": "2024-01-02T03:04:05+00:00"}
    done = subprocess.run(
        ["git", *config, *args],
        cwd=repo,
        env={**os.environ, **dates},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def run_hash(repo, text_path):
    # a blob holding the link target, as git stores a symlink
    done = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=text_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    run(path, "init", "-q")
    (path / "a.txt").write_bytes(b"alpha\n")
    (path / "sub").mkdir()
    (path / "sub" / "b.txt").write_bytes(b"beta\n")
    run(path, "add", "-A")
    run(path, "commit", "-q", "-m", "first")
    return path


def test_stamp_names_the_commit_and_hashes_every_file(repo, tmp_path):
    module = load()
    sha = run(repo, "rev-parse", "HEAD")
    result = module.export(tmp_path / "out" / "r.tar.gz", repo=repo)
    assert result["commit"] == sha and result["tag"] is None and result["files"] == 2
    with tarfile.open(tmp_path / "out" / "r.tar.gz") as tar:
        # the filter argument needs Python 3.11.4+; requires-python allows 3.11.0
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(tmp_path / "x", **kwargs)
    (tree,) = (tmp_path / "x").iterdir()
    assert tree.name == f"cmu-mscf-2026-{sha[:12]}"
    stamp = json.loads((tree / "RELEASE").read_text(encoding="utf-8"))
    assert stamp["commit"] == sha and stamp["tag"] is None and stamp["date"]
    assert stamp["files"] == {
        "a.txt": hashlib.sha256(b"alpha\n").hexdigest(),
        "sub/b.txt": hashlib.sha256(b"beta\n").hexdigest(),
    }
    assert not (tree / ".git").exists()
    assert str(tmp_path) not in (tree / "RELEASE").read_text(encoding="utf-8")


def test_a_symlink_is_recorded_with_its_target(repo, tmp_path):
    run(repo, "update-index", "--add", "--cacheinfo", "120000," + run_hash(repo, "a.txt") + ",link")
    run(repo, "commit", "-q", "-m", "link")
    module = load()
    module.export(tmp_path / "r.tar.gz", repo=repo)
    with tarfile.open(tmp_path / "r.tar.gz") as tar:
        release = next(m for m in tar if m.name.endswith("/RELEASE"))
        stamp = json.loads(tar.extractfile(release).read())
    assert stamp["files"]["link"] == "symlink:a.txt"
    assert stamp["files"]["a.txt"] == hashlib.sha256(b"alpha\n").hexdigest()


def test_an_exact_match_tag_is_recorded(repo, tmp_path):
    run(repo, "tag", "v1")
    result = load().export(tmp_path / "r.tar.gz", repo=repo)
    assert result["tag"] == "v1"
    (repo / "a.txt").write_bytes(b"alpha2\n")
    run(repo, "commit", "-q", "-am", "second")
    assert load().export(tmp_path / "s.tar.gz", repo=repo)["tag"] is None
    assert load().export(tmp_path / "t.tar.gz", commit="v1", repo=repo)["tag"] == "v1"


def test_the_archive_is_byte_reproducible(repo, tmp_path):
    module = load()
    module.export(tmp_path / "a.tar.gz", repo=repo)
    module.export(tmp_path / "b.tar.gz", repo=repo)
    assert (tmp_path / "a.tar.gz").read_bytes() == (tmp_path / "b.tar.gz").read_bytes()


def test_an_unknown_commit_fails_without_writing(repo, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        load().export(tmp_path / "r.tar.gz", commit="no-such-ref", repo=repo)
    assert not (tmp_path / "r.tar.gz").exists()


def test_the_command_line_prints_one_json_line(tmp_path):
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path / "head.tar.gz")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 and not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert done.returncode == 0, done.stderr
    printed = json.loads(done.stdout)
    assert len(printed["commit"]) == 40 and (tmp_path / "head.tar.gz").is_file()
