"""Load and validate `autopilot.toml`: one section per stage, unknown keys refused."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

AUTO = 0  # a resources value of 0 means "size it from the machine"
REQUIRED = object()
KINDS = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "path": str,  # resolved against the config file's folder
    "list": list,
    "table": dict,
}


@dataclass(frozen=True)
class Key:
    """One config key: its kind, default (REQUIRED = no default) and meaning.

    `legacy` is the default the key had before it changed: a section frozen under
    that default still matches (`Config.section_identities`)."""

    kind: str
    default: object = REQUIRED
    help: str = ""
    legacy: object = REQUIRED

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown config kind {self.kind!r}")


CORE_SECTION = "autopilot"
CORE = {
    "run_root": Key("path", help="the registered run root; every stage writes under it"),
    "policy_id": Key("str", help="frozen at the first tick; reviewer is autopilot:<policy_id>"),
    "checkout": Key("path", None, "git checkout holding deploy/strategies.yml"),
    "input_archive": Key("path", None, "the qualified input archive (obtained out of band)"),
    "max_crashes": Key("int", 3, "consecutive crashes of one stage before a STOP"),
    "real_model_calls": Key(
        "bool", False, "without it a stage gets only an injected (fake) model client"
    ),
}
# Model-spend ceilings: owned by the core. The values are the caller's; the defaults allow
# no calls at all, and nothing in this module enforces a number (a spend guard reads them).
SPEND_SECTION = "spend"
SPEND = {
    "run_ceiling": Key("int", 0, "per-run call ceiling over every spend scope; 0 = no calls"),
    "global_ceiling": Key("int", 0, "global call ceiling over every run; 0 = no calls"),
    "runs_dir": Key(
        "path", None, "every folder in it is a run root counted toward the global ceiling"
    ),
    "accounting_roots": Key(
        "list", [], "other run roots outside `runs_dir`, also counted toward the global ceiling"
    ),
}
# Owned by the core, read by whatever sizes worker pools. Not frozen and not part of any
# stage's identity: no output depends on the worker count.
RESOURCES_SECTION = "resources"
RESOURCES = {
    "max_workers": Key(
        "int", AUTO, "cap on worker processes for auto-sized pools; 0 = the machine's cores"
    ),
    "memory_budget_mb": Key(
        "int", AUTO, "RAM (MB) a pool may plan for; 0 = 75% of the RAM available at the start"
    ),
}


class ConfigError(ValueError):
    """Every problem in the file, one per line."""


@dataclass(frozen=True)
class Config:
    path: Path
    sections: dict  # section name -> validated values
    section_of: dict  # stage name -> section name
    legacy: dict = field(default_factory=dict)  # section -> {key: (default, legacy default)}

    @property
    def core(self):
        return self.sections[CORE_SECTION]

    @property
    def run_root(self) -> Path:
        return Path(self.core["run_root"])

    def section_sha256(self, section) -> str:
        """Identity of one section; a stage's section is frozen once the stage starts."""
        text = json.dumps(self.sections[section], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def section_identities(self, section):
        """Every identity a frozen record of `section` may carry: the current one, and the
        one it had when a key now at its new default still had its `legacy` default."""
        values, changed = dict(self.sections[section]), False
        for key, (default, legacy) in self.legacy.get(section, {}).items():
            if values.get(key) == default:
                values[key], changed = legacy, True
        found = [self.section_sha256(section)]
        if changed:
            text = json.dumps(values, sort_keys=True, separators=(",", ":"))
            found.append(hashlib.sha256(text.encode()).hexdigest())
        return found


def _section(name, raw, schema, base, errors):
    values = {}
    if not isinstance(raw, dict):
        errors.append(f"[{name}] must be a table")
        return values
    for key in sorted(set(raw) - set(schema)):
        errors.append(f"[{name}] unknown key {key!r}")
    for key, spec in schema.items():
        if key not in raw:
            if spec.default is REQUIRED:
                errors.append(f"[{name}] missing required key {key!r}")
            else:
                values[key] = spec.default
            continue
        value = raw[key]
        wrong = not isinstance(value, KINDS[spec.kind]) or (
            isinstance(value, bool) and spec.kind != "bool"
        )
        if wrong:
            errors.append(f"[{name}] {key} must be {spec.kind}, got {type(value).__name__}")
            continue
        values[key] = str((base / value).resolve()) if spec.kind == "path" else value
    return values


def schemas(stages, extra=None):
    """Section name -> schema for the core, every stage and the ops sections (`extra`)."""
    out = {CORE_SECTION: CORE, SPEND_SECTION: SPEND, RESOURCES_SECTION: RESOURCES}
    owners = {CORE_SECTION: "core", SPEND_SECTION: "core", RESOURCES_SECTION: "core"}
    declared = [(s.section, s.schema, s.name) for s in stages]
    declared += [(name, schema, "ops") for name, schema in (extra or {}).items()]
    for name, schema, owner in declared:
        if name in out:
            raise ValueError(f"config section [{name}] is declared by {owners[name]} and {owner}")
        out[name], owners[name] = schema, owner
    return out


def validate(raw, stages, *, extra=None, path="autopilot.toml"):
    path = Path(path)
    known = schemas(stages, extra)
    errors = [f"unknown section [{name}]" for name in sorted(set(raw) - set(known))]
    sections = {
        name: _section(name, raw.get(name, {}), schema, path.resolve().parent, errors)
        for name, schema in known.items()
    }
    core, spend = sections[CORE_SECTION], sections[SPEND_SECTION]
    if core.get("real_model_calls") is True and spend.get("runs_dir") is None:
        # With real calls the global ceiling must count every other run; without a
        # discovered runs folder it would count only the listed roots.
        errors.append(
            "[spend] runs_dir is required when [autopilot] real_model_calls = true "
            "(the folder holding every run root, so the global ceiling counts them)"
        )
    for key, value in sections[RESOURCES_SECTION].items():
        if isinstance(value, int) and value < AUTO:
            errors.append(f"[{RESOURCES_SECTION}] {key} must be >= {AUTO} (0 = auto)")
    if errors:
        raise ConfigError("\n".join(errors))
    legacy = {
        name: {k: (v.default, v.legacy) for k, v in schema.items() if v.legacy is not REQUIRED}
        for name, schema in known.items()
    }
    legacy = {name: keys for name, keys in legacy.items() if keys}
    return Config(path, sections, {s.name: s.section for s in stages}, legacy)


def load(path, stages, *, extra=None):
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path}: {error}") from error
    return validate(raw, stages, extra=extra, path=path)
