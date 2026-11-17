"""Release identity from a git checkout or an exported RELEASE stamp; runtime versions."""

import hashlib
import json
import platform
import subprocess

import pytest

from capstone.release import RELEASE_STAMP, release_identity, runtime_versions, stamped_identity


def git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q")
    (root / "a.txt").write_text("one\n")
    git(root, "add", "a.txt")
    git(root, "commit", "-q", "-m", "first")
    return root


def stamp(root, files, commit="c" * 40, tag="deploy-2026.01.02"):
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in files}
    (root / RELEASE_STAMP).write_text(json.dumps({"commit": commit, "tag": tag, "files": hashes}))


def test_runtime_versions_name_python_and_the_scientific_stack():
    versions = runtime_versions()
    assert versions["python"] == platform.python_version()
    assert set(versions) == {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "pyarrow",
        "statsmodels",
        "requests",
    }
    assert all(isinstance(v, str) and v for v in versions.values())


def test_git_checkout_reports_head_and_dirty_flag(repo):
    head = git(repo, "rev-parse", "HEAD")
    assert release_identity(repo=repo) == {"commit": head, "dirty": False, "deployment_tag": None}
    (repo / "a.txt").write_text("two\n")
    assert release_identity(repo=repo)["dirty"] is True


def test_deployment_tag_goes_through_the_injected_checker(repo):
    head = git(repo, "rev-parse", "HEAD")
    seen = []

    def resolve(path, tag):
        seen.append((path, tag))
        return head

    identity = release_identity("deploy-x", repo=repo, resolve_tag=resolve)
    assert identity["deployment_tag"] == "deploy-x" and seen == [(repo, "deploy-x")]
    with pytest.raises(ValueError, match="does not identify"):
        release_identity("deploy-x", repo=repo, resolve_tag=lambda r, t: "0" * 40)
    with pytest.raises(ValueError, match="needs a tag checker"):
        release_identity("deploy-x", repo=repo)


def test_exported_tree_reads_its_stamp_and_detects_edits(tmp_path):
    (tmp_path / "a.txt").write_text("one\n")
    stamp(tmp_path, ["a.txt"])
    assert release_identity(repo=tmp_path) == {
        "commit": "c" * 40,
        "dirty": False,
        "deployment_tag": None,
    }
    assert (
        stamped_identity(tmp_path, "deploy-2026.01.02", resolve_tag=lambda r, t: "c" * 40)[
            "deployment_tag"
        ]
        == "deploy-2026.01.02"
    )
    (tmp_path / "a.txt").write_text("edited\n")
    assert stamped_identity(tmp_path)["dirty"] is True
    (tmp_path / "a.txt").unlink()
    assert stamped_identity(tmp_path)["dirty"] is True


def test_exported_tree_refuses_a_different_tag_and_a_missing_stamp(tmp_path):
    (tmp_path / "a.txt").write_text("one\n")
    with pytest.raises(ValueError, match="neither a git checkout"):
        stamped_identity(tmp_path)
    stamp(tmp_path, ["a.txt"])
    with pytest.raises(ValueError, match="does not identify"):
        stamped_identity(tmp_path, "deploy-2030.01.01")


@pytest.mark.parametrize("identity", [release_identity, stamped_identity])
def test_exported_deployment_tag_requires_an_independent_checker(tmp_path, identity):
    (tmp_path / "a.txt").write_text("one\n")
    stamp(tmp_path, ["a.txt"])
    with pytest.raises(ValueError, match="needs a tag checker"):
        identity(repo=tmp_path, deployment_tag="deploy-2026.01.02")


def test_exported_tag_is_checked_against_the_recorded_commit(tmp_path):
    (tmp_path / "a.txt").write_text("one\n")
    stamp(tmp_path, ["a.txt"])
    seen = []

    def resolve(path, tag):
        seen.append((path, tag))
        return "c" * 40

    assert (
        release_identity("deploy-2026.01.02", repo=tmp_path, resolve_tag=resolve)["dirty"] is False
    )
    assert seen == [(tmp_path, "deploy-2026.01.02")]
    stamp(tmp_path, ["a.txt"], commit="d" * 40)
    with pytest.raises(ValueError, match="does not identify"):
        release_identity("deploy-2026.01.02", repo=tmp_path, resolve_tag=resolve)


def test_untracked_files_remain_dirty_when_git_configuration_hides_them(repo):
    (repo / ".gitignore").write_text("data_cache/\nexperiments/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore generated files")
    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "data_cache").mkdir()
    (repo / "data_cache" / "derived.txt").write_text("synthetic\n")
    (repo / "experiments").mkdir()
    (repo / "experiments" / "runs.jsonl").write_text("synthetic\n")
    assert release_identity(repo=repo)["dirty"] is False
    (repo / "new.py").write_text("x = 1\n")
    assert release_identity(repo=repo)["dirty"] is True


def test_git_status_uses_utf8_for_unquoted_non_ascii_names(repo, monkeypatch):
    git(repo, "config", "core.quotePath", "false")
    (repo / "\u03b1.py").write_text("x = 1\n")
    run = subprocess.run

    def require_utf8(*args, **kwargs):
        assert kwargs.get("encoding") == "utf-8"
        return run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", require_utf8)
    assert release_identity(repo=repo)["dirty"] is True


@pytest.mark.parametrize("files", [None, {}])
def test_export_stamp_requires_a_nonempty_file_manifest(tmp_path, files):
    content = {"commit": "c" * 40, "tag": "deploy-x"}
    if files is not None:
        content["files"] = files
    (tmp_path / RELEASE_STAMP).write_text(json.dumps(content))
    with pytest.raises(ValueError, match="nonempty.*manifest"):
        stamped_identity(tmp_path)
