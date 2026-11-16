"""The deploy registry: `deploy/strategies.yml` entries.

The registry file's schema is in its header: one entry per strategy with
`id`, `module`, `gate_report` and `enabled`. Rules:

- No gate report, no entry: an entry without one is refused when built or
  read.
- `enabled` defaults to false. Turning a strategy on is a reviewed edit of
  the registry file, never a side effect of reading or rewriting it.
- A file that is read and rendered again is byte for byte what it was.

The project has no YAML dependency, so this module reads and writes the
registry's own narrow schema: a header of comments, then `strategies: []` or
a block list of entries with the keys in `KEYS` whose values are plain,
single-quoted or double-quoted scalars. Anything else is refused rather than
guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODULE = "capstone.strategy.contract"
MODULES = (MODULE,)
REGISTRY_FILE = Path("deploy") / "strategies.yml"
KEYS = ("id", "module", "gate_report", "enabled")


@dataclass(frozen=True)
class RegistryEntry:
    id: str
    module: str
    gate_report: str
    enabled: bool = False

    def __post_init__(self) -> None:
        if not self.gate_report:
            raise ValueError(f"strategy {self.id!r} has no gate report: no gate report, no deploy")
        if not self.id or not self.module:
            raise ValueError("an entry needs an id and a module")
        if self.module not in MODULES:
            raise ValueError(
                f"strategy {self.id!r}: module must be one of {MODULES}: {self.module!r}"
            )
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")


def basic_string(text: str) -> str:
    """A TOML basic string (also a YAML double-quoted scalar)."""
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _scalar(text: str) -> str | bool:
    text = text.strip()
    if text.startswith('"'):
        return json.loads(text)
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1].replace("''", "'")
    if text in ("true", "false"):
        return text == "true"
    if not text or text[0] in "[{&*!|>%@`#":
        raise ValueError(f"registry value {text!r} is not a plain scalar")
    return text


def _parse(text: str) -> tuple[str, list[RegistryEntry]]:
    """(header, entries) of a registry file."""
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    header = "".join(line + "\n" for line in lines[:i])
    if i == len(lines):
        raise ValueError("registry file has no `strategies:` key")
    top = lines[i].split("#", 1)[0].rstrip()
    if top == "strategies: []":
        rest = lines[i + 1 :]
        if any(line.strip() and not line.lstrip().startswith("#") for line in rest):
            raise ValueError("registry file has content after `strategies: []`")
        return header, []
    if top != "strategies:":
        raise ValueError(f"registry file must start with `strategies:`, got {lines[i]!r}")
    raw: list[dict] = []
    for line in lines[i + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        body = line.strip()
        if body.startswith("- "):
            raw.append({})
            body = body[2:].strip()
        elif not raw or not line.startswith("    "):
            raise ValueError(f"registry line {line!r} is not part of an entry")
        key, sep, value = body.partition(":")
        if not sep or key not in KEYS or key in raw[-1]:
            raise ValueError(f"registry line {line!r}: expected one of {KEYS}, once each")
        raw[-1][key] = _scalar(value)
    entries = []
    for item in raw:
        if not item.get("gate_report"):
            raise ValueError(f"registry entry {item.get('id')!r} has no gate_report")
        entries.append(RegistryEntry(**item))
    return header, entries


def _render(header: str, entries: list[RegistryEntry]) -> str:
    if not entries:
        return header + "strategies: []\n"
    body = "".join(
        f"  - id: {basic_string(e.id)}\n"
        f"    module: {basic_string(e.module)}\n"
        f"    gate_report: {basic_string(e.gate_report)}\n"
        f"    enabled: {'true' if e.enabled else 'false'}\n"
        for e in entries
    )
    return header + "strategies:\n" + body


def parse_registry(text: str) -> tuple[RegistryEntry, ...]:
    """The entries of a registry file's text, in file order (e.g. read at a git tag)."""
    return tuple(_parse(text)[1])


def read_registry(root: Path) -> tuple[RegistryEntry, ...]:
    """The entries of `<root>/deploy/strategies.yml`, in file order."""
    return parse_registry((Path(root) / REGISTRY_FILE).read_bytes().decode("utf-8"))
