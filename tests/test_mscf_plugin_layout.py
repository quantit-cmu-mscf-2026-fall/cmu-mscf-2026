"""The mscf-pipeline plugin (mscf-plugin/) is well formed; its skills describe real code."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "mscf-plugin"
SKILLS = sorted(PLUGIN.glob("skills/*/SKILL.md"))
MARKDOWN = [*SKILLS, *PLUGIN.glob("skills/*/references/*.md"), PLUGIN / "README.md"]


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line)


def test_manifest_and_marketplace_entry():
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "mscf-pipeline" and plugin["version"]
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    (entry,) = [p for p in market["plugins"] if p["name"] == "mscf-pipeline"]
    assert (ROOT / entry["source"] / ".claude-plugin" / "plugin.json").is_file()


def test_skills_have_usable_frontmatter():
    assert {p.parent.name for p in SKILLS} == {
        "pipeline-boundaries",
        "shared-data",
        "run-ledger",
        "validate-candidates",
        "null-calibration",
        "research-pr",
        "paper-round",
    }
    for path in SKILLS:
        meta = frontmatter(path)
        assert meta["name"] == path.parent.name
        assert 40 < len(meta["description"]) <= 1024 and re.search(
            r"Use (when|before)", meta["description"]
        )
        assert len(path.read_text(encoding="utf-8").splitlines()) < 500


ALIASES = {"sd": "shared_data", "ev": "evaluate"}
MODULES = {
    "shared_data",
    "runlog",
    "evaluate",
    "synth",
    "backtest",
    "search_fdr",
    "cv",
    "calibration",
}


def test_skills_name_only_capstone_functions_that_exist():
    """A skill that names `sd.load` or `evaluate.holm` names something the code still has."""
    pattern = re.compile(
        r"\b(?:capstone\.)?(sd|ev|" + "|".join(sorted(MODULES)) + r")\.([a-z_]\w*)\b"
    )
    checked = 0
    for path in MARKDOWN:
        for alias, name in pattern.findall(path.read_text(encoding="utf-8")):
            module = importlib.import_module(f"capstone.{ALIASES.get(alias, alias)}")
            if name in {"py", "md", "json", "jsonl", "toml"}:
                continue
            assert hasattr(module, name), f"{path.parent.name}: {alias}.{name} not in {module}"
            checked += 1
    assert checked >= 30


def test_skills_point_at_repository_files_that_exist():
    pattern = re.compile(r"`((?:docs|deploy|\.github)/[\w./-]+\.(?:md|yml))`")
    seen = {m for path in MARKDOWN for m in pattern.findall(path.read_text(encoding="utf-8"))}
    assert seen, "expected the skills to name repository files"
    assert sorted(m for m in seen if not (ROOT / m).is_file()) == []


def test_the_plugin_carries_no_private_or_host_specific_text():
    banned = re.compile(r"caleb|ibkr|interactive brokers|OMP_NUM_THREADS|[A-Za-z]:\\[A-Za-z]", re.I)
    for path in [*MARKDOWN, *PLUGIN.rglob("*.json")]:
        assert not banned.search(path.read_text(encoding="utf-8")), path
