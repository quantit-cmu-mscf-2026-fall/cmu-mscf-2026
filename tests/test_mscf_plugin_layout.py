"""The mscf-pipeline plugin (mscf-plugin/) is well formed; its skills describe real code."""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import pkgutil
import re
from pathlib import Path

import capstone

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "mscf-plugin"
SKILLS = sorted(PLUGIN.glob("skills/*/SKILL.md"))
AGENTS = sorted(PLUGIN.glob("agents/*.md"))
COMMANDS = sorted(PLUGIN.glob("commands/*.md"))
MARKDOWN = [
    *SKILLS,
    *AGENTS,
    *COMMANDS,
    *PLUGIN.glob("skills/*/references/*.md"),
    PLUGIN / "README.md",
]


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line)


def test_manifest_and_marketplace_entry():
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "mscf-pipeline" and plugin["version"]
    hooks = json.loads((PLUGIN / plugin["hooks"]).read_text(encoding="utf-8"))
    commands = [
        h["command"] for groups in hooks["hooks"].values() for g in groups for h in g["hooks"]
    ]
    assert commands and all("${CLAUDE_PLUGIN_ROOT}/hooks/guard.py" in c for c in commands)
    assert (PLUGIN / "hooks" / "guard.py").is_file()
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
        "pipeline-view",
    }
    for path in SKILLS:
        meta = frontmatter(path)
        assert meta["name"] == path.parent.name
        assert 40 < len(meta["description"]) <= 1024 and re.search(
            r"Use (when|before)", meta["description"]
        )
        assert len(path.read_text(encoding="utf-8").splitlines()) < 500


def test_agents_are_read_only_and_defer_review_to_the_repo_harness():
    assert [p.stem for p in AGENTS] == ["ship-checker"]  # review is the repo's pr-reviewer
    assert (ROOT / ".claude" / "agents" / "pr-reviewer.md").is_file()
    for path in AGENTS:
        meta = frontmatter(path)
        body = path.read_text(encoding="utf-8")
        assert meta["name"] == path.stem and body.count("<example>") >= 2
        tools = {t.strip() for t in meta["tools"].split(",")}
        assert tools == {"Read", "Grep", "Glob", "Bash"}
        assert "/review-pr" in body and "REVIEW.md" in body


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


def _capstone_names() -> set[str]:
    """Every public name defined in any capstone module, plus the module names."""
    names: set[str] = set()
    for info in pkgutil.walk_packages(capstone.__path__, "capstone."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # an optional dependency is missing; skip that module
            continue
        names.update(n for n in dir(module) if not n.startswith("_"))
        names.add(info.name.rsplit(".", 1)[-1])
    return names


# Backticked words that are ledger columns or other libraries' names, and the Claude Code
# tool a command's `allowed-tools: Bash(...)` frontmatter names.
NOT_CAPSTONE = {"bh", "by", "q", "lfdr", "Path", "Series", "DataFrame", "Bash"}
CALL = re.compile(r"(?<![\w.-])([A-Za-z_]\w*)\((?!s\))")  # not "seed(s)" or CSS "cubic-bezier("
SPAN = re.compile(r"`([^`\n]+)`")


def _unqualified_names(text: str, catalog: bool) -> set[str]:
    """Names used without a module: calls like `log_run(...)` and, in the method catalog
    (references/methods.md), every identifier in a table's first column (`holm`)."""
    found = set(CALL.findall(text)) - set(vars(builtins))
    for line in text.splitlines():
        if catalog and line.startswith("|") and not set(line) <= set("|- "):
            for span in SPAN.findall(line.split("|")[1]):
                found.update(re.findall(r"^[A-Za-z_]\w*$", span.strip()))
    return {n for n in found - NOT_CAPSTONE if not n.startswith(("test_", "my_"))}


def _skill_dir(path: Path) -> Path | None:
    """The skills/<name>/ folder a markdown file belongs to, if any."""
    return next((p for p in path.parents if p.parent.name == "skills"), None)


def _skill_script_names(path: Path) -> set[str]:
    """Functions and classes a skill's own scripts define (pipeline-view's `track(...)`)."""
    skill = _skill_dir(path)
    if skill is None:
        return set()
    names: set[str] = set()
    for script in skill.rglob("*.py"):
        for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
    return names


def test_every_unqualified_name_the_skills_use_exists():
    """`holm` or `PurgedKFold(...)` named without a module is still something capstone has."""
    capstone_names = _capstone_names()
    checked = 0
    for path in MARKDOWN:
        known = capstone_names | _skill_script_names(path)
        for name in sorted(
            _unqualified_names(path.read_text(encoding="utf-8"), path.name == "methods.md")
        ):
            assert name in known, f"{path.parent.name}: `{name}` is not defined in capstone"
            checked += 1
    assert checked >= 25


FILE_REFERENCE = re.compile(
    r"`((?:docs|deploy|scripts|tests|\.github|\.claude)/[\w./-]+\.(?:md|yml|py))`"
    r"|\b([A-Z][A-Z_]{3,}\.md)\b"
)


def test_skills_point_at_repository_files_that_exist():
    """A path resolves from the repository root or, for a skill's own scripts, from its folder."""
    seen: dict[str, list[Path]] = {}
    for path in MARKDOWN:
        for quoted, bare in FILE_REFERENCE.findall(path.read_text(encoding="utf-8")):
            seen.setdefault(quoted or bare, []).append(path)
    assert {"CONTRIBUTING.md", "CLAUDE.md"} <= set(seen), (
        "expected the skills to name repository files"
    )
    roots = {m: [ROOT, *filter(None, (_skill_dir(p) for p in paths))] for m, paths in seen.items()}
    assert sorted(m for m in seen if not any((r / m).is_file() for r in roots[m])) == []


def test_the_plugin_carries_no_private_or_host_specific_text():
    banned = re.compile(
        r"ibkr|interactive brokers|OMP_NUM_THREADS|[A-Za-z]:\\[A-Za-z]|/Users/|/home/", re.I
    )
    for path in [*MARKDOWN, *PLUGIN.rglob("*.json"), PLUGIN / "hooks" / "guard.py"]:
        assert not banned.search(path.read_text(encoding="utf-8")), path
