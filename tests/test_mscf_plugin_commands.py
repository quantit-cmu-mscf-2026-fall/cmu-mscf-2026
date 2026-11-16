"""The plugin's slash commands wrap CLIs and tools that exist in this repository."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = sorted((ROOT / "mscf-plugin" / "commands").glob("*.md"))
RUN = re.compile(r"python -m (capstone(?:\.\w+)+)(?: (\w+))?")


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line)


def test_the_commands_are_the_shipped_ones_with_usable_frontmatter():
    assert [p.stem for p in COMMANDS] == ["qp-check"]
    for path in COMMANDS:
        meta = frontmatter(path)
        assert 20 < len(meta["description"]) <= 200
        assert meta["argument-hint"] and meta["allowed-tools"]
        assert len(path.read_text(encoding="utf-8").splitlines()) < 100


def test_every_module_and_subcommand_a_command_runs_exists():
    """A command that says `python -m capstone.x sub` names a module and a subcommand
    the code has; a command is added only together with the CLI it wraps."""
    seen = set()
    for path in COMMANDS:
        for module, sub in RUN.findall(path.read_text(encoding="utf-8")):
            spec = importlib.util.find_spec(module)
            assert spec is not None and spec.origin, f"{path.stem}: {module} does not exist"
            if sub:
                source = Path(spec.origin).read_text(encoding="utf-8")
                assert f'"{sub}"' in source, f"{path.stem}: {module} has no {sub!r} subcommand"
            seen.add((module, sub))
    assert ("capstone.runlog", "stats") in seen


def test_the_check_command_matches_the_repositorys_rules():
    text = (ROOT / "mscf-plugin" / "commands" / "qp-check.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert 'pytest -q -m "not network"' in text and 'pytest -q -m "not network"' in claude
    assert "ruff check" in text and "ruff format" in text
    assert "never type one in" in text.lower()
    assert "runs.jsonl" in text and "gitignored" in text


def test_the_commands_carry_no_private_or_host_specific_text():
    banned = re.compile(r"caleb|ibkr|interactive brokers|OMP_NUM_THREADS|[A-Za-z]:\\[A-Za-z]", re.I)
    for path in COMMANDS:
        assert not banned.search(path.read_text(encoding="utf-8")), path
