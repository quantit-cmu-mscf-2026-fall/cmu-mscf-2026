"""`registry_spec`: a registered strategy's spec file, read with its entry's module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from capstone.strategy import registry
from capstone.strategy.registry import (
    MODULES,
    PROGRAM_MODULE,
    REGISTRY_FILE,
    SAVED_PORTFOLIO_MODULE,
    SPEC_DIR,
    RegistryEntry,
    registry_spec,
    spec_from_toml,
)


class FakeSpec:
    def __init__(self, id):
        self.id = id

    @classmethod
    def from_toml(cls, text):
        return cls(text.strip())


def fake_module(monkeypatch, name, attr):
    module = types.ModuleType(name)
    setattr(module, attr, FakeSpec)
    monkeypatch.setitem(sys.modules, name, module)


def write(root: Path, sid: str, module: str, content: str) -> None:
    (root / SPEC_DIR).mkdir(parents=True, exist_ok=True)
    (root / REGISTRY_FILE).parent.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY_FILE).write_text(
        f'strategies:\n  - id: "{sid}"\n    module: "{module}"\n'
        '    gate_report: "r.json"\n    enabled: false\n',
        encoding="utf-8",
    )
    (root / SPEC_DIR / f"{sid}.toml").write_text(content, encoding="utf-8")


def test_modules_and_spec_dir():
    assert MODULES[1:] == (PROGRAM_MODULE, SAVED_PORTFOLIO_MODULE)
    assert SPEC_DIR == Path("deploy") / "strategies"
    for module in MODULES:
        RegistryEntry(id="a", module=module, gate_report="r.json")


def test_saved_portfolio_entry_is_read_with_its_module(tmp_path, monkeypatch):
    fake_module(monkeypatch, SAVED_PORTFOLIO_MODULE, "SavedProgramPortfolioSpec")
    write(tmp_path, "pf", SAVED_PORTFOLIO_MODULE, "pf")
    assert registry_spec(tmp_path, "pf").id == "pf"
    assert registry_spec(tmp_path, "pf", SAVED_PORTFOLIO_MODULE).id == "pf"


def test_program_entry_is_read_with_its_module(tmp_path, monkeypatch):
    fake_module(monkeypatch, PROGRAM_MODULE, "ProgramStrategySpec")
    write(tmp_path, "pg", PROGRAM_MODULE, "pg")
    assert registry_spec(tmp_path, "pg").id == "pg"


def test_a_spec_file_holding_another_id_is_refused(tmp_path, monkeypatch):
    fake_module(monkeypatch, PROGRAM_MODULE, "ProgramStrategySpec")
    write(tmp_path, "pg", PROGRAM_MODULE, "other")
    with pytest.raises(ValueError, match="holds spec id other"):
        registry_spec(tmp_path, "pg")


def test_an_unknown_module_is_refused():
    with pytest.raises(ValueError, match="module must be one of"):
        spec_from_toml("os.path", "")


def test_a_missing_spec_file_is_an_error(tmp_path, monkeypatch):
    fake_module(monkeypatch, PROGRAM_MODULE, "ProgramStrategySpec")
    write(tmp_path, "pg", PROGRAM_MODULE, "pg")
    (tmp_path / SPEC_DIR / "pg.toml").unlink()
    with pytest.raises(FileNotFoundError):
        registry_spec(tmp_path, "pg")


def test_registry_exports_the_names_lifecycle_actions_imports():
    for name in ("SAVED_PORTFOLIO_MODULE", "SPEC_DIR", "registry_spec", "INCUBATION_ONLY"):
        assert hasattr(registry, name)
