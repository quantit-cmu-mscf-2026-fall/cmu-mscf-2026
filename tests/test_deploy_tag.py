"""Reading a deploy at its tag: tag-only reads, no-gate refusal, disabled default.

Each test builds its own git repo and tag; never the real repo's tags.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest
from test_execution_universe import universe_text
from test_strategy_contract import make_spec

from capstone.execution.deploy import (
    SPEC_DIR,
    Refused,
    deployed_entries,
    deployed_specs,
    deployed_universe,
    resolve_tag,
)
from capstone.strategy.contract import save_spec
from capstone.strategy.registry import RegistryEntry, _parse, _render

TAG = "deploy-2026.08.24"


def git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def new_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "deploy" / "strategies").mkdir(parents=True)
    (root / "deploy" / "strategies.yml").write_text("# registry\nstrategies: []\n")
    (root / "deploy" / "universe.csv").write_text(universe_text())
    git(root, "init", "-q", "-b", "main")
    return root


def register(root: Path, spec, *, enabled=True, report=None) -> RegistryEntry:
    """Write the spec file, its gate report and a registry entry for it."""
    save_spec(spec, root / SPEC_DIR / f"{spec.id}.toml")
    report_path = root / spec.gate_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report if report is not None else {"gated": True}))
    registry = root / "deploy" / "strategies.yml"
    header, entries = _parse(registry.read_text())
    entry = RegistryEntry(
        id=spec.id,
        module="capstone.strategy.contract",
        gate_report=spec.gate_report,
        enabled=enabled,
    )
    registry.write_text(_render(header, [*entries, entry]))
    return entry


def tag(root: Path, name: str = TAG) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", "deploy")
    git(root, "tag", "-f", name)
    return resolve_tag(root, name)


def test_only_a_deploy_tag_is_resolved(tmp_path):
    root = new_root(tmp_path)
    commit = tag(root)
    assert len(commit) == 40
    for bad in ("main", "HEAD", "deploy-2026.8.24", "deploy-2026.08.25"):
        with pytest.raises(Refused, match="tag"):
            resolve_tag(root, bad)


def test_the_deploy_is_read_at_the_tag_not_the_working_tree_or_tip(tmp_path):
    root = new_root(tmp_path)
    spec = make_spec()
    register(root, spec)
    commit = tag(root)
    # later edits and a later commit never change what the tag reads
    (root / "deploy" / "strategies.yml").write_text("# registry\nstrategies: []\n")
    git(root, "commit", "-qam", "tip moves on")
    assert deployed_specs(root, commit) == (spec,)
    with pytest.raises(Refused, match="nothing runs"):
        deployed_entries(root, tag(root, "deploy-2026.08.25"))


def test_a_disabled_entry_is_skipped_and_nothing_enabled_is_refused(tmp_path, caplog):
    root = new_root(tmp_path)
    on, off = make_spec(hypothesis_id="H-001"), make_spec(hypothesis_id="H-002")
    register(root, on)
    register(root, off, enabled=False)
    with caplog.at_level(logging.INFO, logger="capstone.execution.deploy"):
        entries = deployed_entries(root, tag(root))
    assert [spec for _, spec in entries] == [on]
    assert f"skipping disabled strategy {off.id}" in caplog.text
    only_off = new_root(tmp_path / "b")
    register(only_off, off, enabled=False)
    with pytest.raises(Refused, match="no strategy is enabled"):
        deployed_entries(only_off, tag(only_off))


def test_a_report_that_says_not_gated_is_refused_at_the_tag(tmp_path):
    root = new_root(tmp_path)
    register(root, make_spec(), report={"gated": False})
    with pytest.raises(Refused, match="not gated"):
        deployed_entries(root, tag(root))


def test_an_enabled_entry_with_no_spec_file_or_a_wrong_one_is_refused(tmp_path):
    root = new_root(tmp_path)
    spec = make_spec()
    register(root, spec)
    path = root / SPEC_DIR / f"{spec.id}.toml"
    path.write_text(path.read_text().replace("H-001", "H-999"))
    with pytest.raises(Refused, match="refused"):
        deployed_entries(root, tag(root))
    path.unlink()
    with pytest.raises(Refused, match="no spec file"):
        deployed_entries(root, tag(root))
    save_spec(make_spec(hypothesis_id="H-002"), path)  # another strategy's file under this id
    with pytest.raises(Refused, match="holds spec id"):
        deployed_entries(root, tag(root))


def test_the_registry_must_exist_and_its_gate_report_must_be_the_specs(tmp_path):
    root = new_root(tmp_path)
    (root / "deploy" / "strategies.yml").unlink()
    with pytest.raises(Refused, match="no deploy/strategies.yml"):
        deployed_entries(root, tag(root))
    root = new_root(tmp_path / "c")
    entry = register(root, make_spec())
    registry = root / "deploy" / "strategies.yml"
    registry.write_text(registry.read_text().replace(entry.gate_report, "reports/other.md"))
    with pytest.raises(Refused, match="not its spec's"):
        deployed_entries(root, tag(root))
    registry.write_text("strategies:\n  - id: x\n    module: capstone.strategy.contract\n")
    with pytest.raises(Refused, match="registry at the tag is refused"):
        deployed_entries(root, tag(root))


def test_the_universe_at_the_tag_must_be_filled_and_non_empty(tmp_path):
    root = new_root(tmp_path)
    universe = deployed_universe(root, tag(root))
    assert universe.listings and not universe.is_template
    (root / "deploy" / "universe.csv").write_text(universe_text(version="TEMPLATE-1"))
    with pytest.raises(Refused, match="template"):
        deployed_universe(root, tag(root))
    (root / "deploy" / "universe.csv").write_text(universe_text(permnos=()))
    with pytest.raises(Refused, match="lists no stocks"):
        deployed_universe(root, tag(root))
    (root / "deploy" / "universe.csv").unlink()
    with pytest.raises(Refused, match="no universe file"):
        deployed_universe(root, tag(root))
