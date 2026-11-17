"""The deploy registry reader: the team's four-key schema, refusals, round trip."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from capstone.strategy.registry import (
    INCUBATION_ONLY,
    MODULE,
    PLUMBING_TEST,
    ROLES,
    STATES,
    RegistryEntry,
    _parse,
    _render,
    parse_registry,
    read_registry,
)

REPO = Path(__file__).resolve().parent.parent
ENTRY = (
    "strategies:\n  - id: momo-weekly-v2\n    module: capstone.strategy.contract\n"
    "    gate_report: reports/gate/momo.md\n    enabled: true\n"
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "deploy").mkdir()
    return tmp_path


def write(root: Path, text: str) -> None:
    (root / "deploy" / "strategies.yml").write_bytes(text.encode("utf-8"))


def test_an_entry_cannot_be_built_without_a_gate_report():
    with pytest.raises(ValueError, match="gate report"):
        RegistryEntry(id="abc", module=MODULE, gate_report="")
    entry = RegistryEntry(id="abc", module=MODULE, gate_report="r.md")
    assert entry.enabled is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.enabled = True


def test_an_entry_needs_an_id_a_known_module_and_a_real_bool():
    with pytest.raises(ValueError, match="id and a module"):
        RegistryEntry(id="", module=MODULE, gate_report="r.md")
    with pytest.raises(ValueError, match="module must be one of"):
        RegistryEntry(id="abc", module="os.path", gate_report="r.md")
    with pytest.raises(ValueError, match="true or false"):
        RegistryEntry(id="abc", module=MODULE, gate_report="r.md", enabled="yes")


def test_an_entry_in_the_file_without_a_gate_report_is_refused_on_read(root):
    write(
        root,
        "strategies:\n  - id: momo-weekly-v2\n    module: capstone.strategy.contract\n"
        "    enabled: true\n",
    )
    with pytest.raises(ValueError, match="gate_report"):
        read_registry(root)


def test_an_empty_gate_report_in_the_file_is_refused_on_read(root):
    write(root, ENTRY.replace("reports/gate/momo.md", '""'))
    with pytest.raises(ValueError, match="gate_report"):
        read_registry(root)


def test_a_hand_written_entry_with_plain_scalars_reads(root):
    write(root, "# header\n" + ENTRY)
    assert read_registry(root) == (
        RegistryEntry("momo-weekly-v2", MODULE, "reports/gate/momo.md", enabled=True),
    )


def test_quoted_scalars_read_as_their_content():
    text = (
        "strategies:\n  - id: 'it''s'\n    module: \"capstone.strategy.contract\"\n"
        '    gate_report: "reports/a b.md"\n    enabled: false\n'
    )
    assert parse_registry(text) == (RegistryEntry("it's", MODULE, "reports/a b.md", False),)


def test_an_unknown_key_is_refused():
    with pytest.raises(ValueError, match="expected one of"):
        parse_registry(ENTRY + "    owner: someone\n")


def test_a_repeated_key_is_refused():
    with pytest.raises(ValueError, match="once each"):
        parse_registry(ENTRY + "    enabled: false\n")


@pytest.mark.parametrize("value", ["[a, b]", "{a: 1}", "&anchor", "|", "", "# note"])
def test_a_value_that_is_not_a_plain_scalar_is_refused(value):
    with pytest.raises(ValueError, match="not a plain scalar|gate_report"):
        parse_registry(ENTRY.replace("reports/gate/momo.md", value))


def test_enabled_must_be_a_bool_not_a_string():
    with pytest.raises(ValueError, match="true or false"):
        parse_registry(ENTRY.replace("enabled: true", "enabled: yes"))
    with pytest.raises(ValueError, match="true or false"):
        parse_registry(ENTRY.replace("enabled: true", 'enabled: "true"'))


def test_a_line_outside_an_entry_is_refused():
    with pytest.raises(ValueError, match="not part of an entry"):
        parse_registry("strategies:\n  stray: 1\n")


def test_the_file_must_start_with_the_strategies_key():
    with pytest.raises(ValueError, match="no `strategies:` key"):
        parse_registry("# only a comment\n")
    with pytest.raises(ValueError, match="must start with"):
        parse_registry("other: 1\n")
    with pytest.raises(ValueError, match="content after"):
        parse_registry("strategies: []\n  - id: x\n")


def test_entries_keep_file_order_and_crlf_reads_like_lf():
    second = ENTRY.replace("momo-weekly-v2", "momo-weekly-v3").split("strategies:\n")[1]
    text = ENTRY + second
    assert [e.id for e in parse_registry(text)] == ["momo-weekly-v2", "momo-weekly-v3"]
    assert parse_registry(text.replace("\n", "\r\n")) == parse_registry(text)


def test_render_round_trips_byte_for_byte():
    header, entries = _parse("# header\n# more\n" + ENTRY)
    rendered = _render(header, entries)
    assert rendered == (
        "# header\n# more\nstrategies:\n"
        '  - id: "momo-weekly-v2"\n    module: "capstone.strategy.contract"\n'
        '    gate_report: "reports/gate/momo.md"\n    enabled: true\n'
    )
    assert _render(*_parse(rendered)) == rendered
    assert _render("# h\n", []) == "# h\nstrategies: []\n"


def test_the_committed_registry_parses_and_round_trips():
    path = REPO / "deploy" / "strategies.yml"
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")  # CRLF checkout
    header, entries = _parse(text)
    assert header.startswith("#")
    assert _render(header, entries) == text
    assert read_registry(REPO) == tuple(entries)
    assert all(entry.gate_report for entry in entries)


# --- optional keys: role, state, version -----------------------------------------------


def test_roles_and_states_are_closed_sets_with_safe_defaults():
    assert ROLES == ("deploy", INCUBATION_ONLY, PLUMBING_TEST)
    assert STATES == ("active", "suspended", "retired")
    entry = RegistryEntry(id="abc", module=MODULE, gate_report="r.md")
    assert (entry.role, entry.state, entry.version) == ("deploy", "active", "")
    assert entry.enabled is False
    with pytest.raises(ValueError, match="role"):
        RegistryEntry(id="abc", module=MODULE, gate_report="r.md", role="live")
    with pytest.raises(ValueError, match="state"):
        RegistryEntry(id="abc", module=MODULE, gate_report="r.md", state="paused")
    with pytest.raises(ValueError, match="version must be a string"):
        RegistryEntry(id="abc", module=MODULE, gate_report="r.md", version=3)


def test_an_entry_with_a_role_still_needs_a_gate_report():
    with pytest.raises(ValueError, match="gate report"):
        RegistryEntry(id="abc", module=MODULE, gate_report="", role=INCUBATION_ONLY)


def test_a_default_entry_renders_byte_for_byte_as_before():
    entry = RegistryEntry("abc", MODULE, "r.md")
    assert _render("# h\n", [entry]) == (
        "# h\nstrategies:\n"
        f'  - id: "abc"\n    module: "{MODULE}"\n'
        '    gate_report: "r.md"\n    enabled: false\n'
    )


def test_non_default_role_state_and_version_are_written_and_round_trip():
    entry = RegistryEntry(
        "abc", MODULE, "r.md", role=INCUBATION_ONLY, state="suspended", version="v2"
    )
    text = _render("", [entry])
    assert '    role: "incubation-only"\n' in text
    assert '    state: "suspended"\n' in text
    assert '    version: "v2"\n' in text
    assert parse_registry(text) == (entry,)
    assert _render(*_parse(text)) == text
    only_role = _render("", [RegistryEntry("abc", MODULE, "r.md", role=PLUMBING_TEST)])
    assert "state:" not in only_role and "version:" not in only_role


def test_a_hand_written_role_reads_and_an_unknown_one_is_refused(root):
    for role in ROLES:
        write(root, ENTRY + f"    role: {role}\n")
        assert read_registry(root)[0].role == role
    write(root, ENTRY + "    role: live\n")
    with pytest.raises(ValueError, match="role"):
        read_registry(root)
    write(root, ENTRY + "    state: paused\n")
    with pytest.raises(ValueError, match="state"):
        read_registry(root)
    with pytest.raises(ValueError, match="once each"):
        parse_registry(ENTRY + "    role: deploy\n    role: deploy\n")
