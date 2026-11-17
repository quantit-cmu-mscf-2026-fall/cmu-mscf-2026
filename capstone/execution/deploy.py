"""Reading a deploy at its tag: the registry, the specs, the gate reports and
the universe file, each read from the tag's commit with `git show`, never from
the working tree. Anything that breaks a rule raises `Refused`.

Rules: only a tag `deploy-YYYY.MM.DD` is read (never a branch tip or HEAD); a
disabled entry is skipped, never run; an entry whose gate report says it is not
gated is refused; a deploy with nothing enabled, or with the template universe,
is refused.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path, PurePosixPath

from capstone.execution.universe import UNIVERSE_FILE, Universe, parse_universe
from capstone.strategy.contract import StrategySpec
from capstone.strategy.registry import REGISTRY_FILE, RegistryEntry, parse_registry

DEPLOY_TAG = re.compile(r"deploy-\d{4}\.\d{2}\.\d{2}")
SPEC_DIR = Path("deploy") / "strategies"  # one spec file per registry entry: <id>.toml

log = logging.getLogger(__name__)


class Refused(RuntimeError):
    """The runner will not run this deploy."""


def _git(repo: Path, *args: str) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    return out.stdout.decode("utf-8") if out.returncode == 0 else None


def resolve_tag(repo: Path, tag: str) -> str:
    """The commit of deploy tag `tag`; anything else (a branch, HEAD, a
    missing tag) is refused."""
    if not DEPLOY_TAG.fullmatch(tag):
        raise Refused(
            f"{tag!r} is not a deploy tag (deploy-YYYY.MM.DD); the runner only runs a tag, "
            "never a branch tip"
        )
    sha = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if not sha:
        raise Refused(f"there is no tag {tag}; the runner only runs a tag, never a branch tip")
    return sha.strip()


def _show(repo: Path, commit: str, path: Path) -> str | None:
    return _git(repo, "show", f"{commit}:{PurePosixPath(*path.parts)}")


def deployed_specs(repo: Path, commit: str) -> tuple[StrategySpec, ...]:
    """The enabled strategies' specs at `commit`, in registry order."""
    return tuple(spec for _, spec in deployed_entries(repo, commit))


def _gated(repo: Path, commit: str, entry: RegistryEntry) -> bool | None:
    """The `"gated"` value of the entry's gate report at the tag, if the report
    is a JSON object in the repository that states one; else None."""
    if "://" in entry.gate_report:
        return None
    text = _show(repo, commit, Path(entry.gate_report))
    try:
        report = json.loads(text) if text is not None else None
    except ValueError:
        return None
    gated = report.get("gated") if isinstance(report, dict) else None
    return gated if isinstance(gated, bool) else None


def check_gate_report(repo: Path, commit: str, entry: RegistryEntry) -> None:
    """Refuse an entry whose gate report at the tag says exactly that it is
    not gated."""
    if _gated(repo, commit, entry) is False:
        raise Refused(
            f"strategy {entry.id}: its gate report {entry.gate_report} says it is not gated "
            '("gated": false); a strategy that did not pass the gate is never deployed'
        )


def deployed_entries(repo: Path, commit: str) -> tuple[tuple[RegistryEntry, StrategySpec], ...]:
    """The enabled strategies' (registry entry, spec) at `commit`, in registry order."""
    text = _show(repo, commit, REGISTRY_FILE)
    if text is None:
        raise Refused(f"the deploy has no {REGISTRY_FILE.as_posix()}")
    try:
        entries = parse_registry(text)
    except ValueError as exc:
        raise Refused(f"the registry at the tag is refused: {exc}") from exc
    for entry in entries:
        if not entry.enabled:
            log.info("skipping disabled strategy %s", entry.id)
    specs = []
    for entry in (e for e in entries if e.enabled):
        spec_text = _show(repo, commit, SPEC_DIR / f"{entry.id}.toml")
        if spec_text is None:
            raise Refused(f"enabled strategy {entry.id} has no spec file at the tag")
        try:
            spec = StrategySpec.from_toml(spec_text)
        except ValueError as exc:
            raise Refused(f"spec file of {entry.id} is refused: {exc}") from exc
        if spec.id != entry.id:
            raise Refused(f"spec file {entry.id}.toml holds spec id {spec.id}")
        if spec.gate_report != entry.gate_report:
            raise Refused(f"strategy {entry.id}: the registry's gate report is not its spec's")
        check_gate_report(repo, commit, entry)
        specs.append((entry, spec))
    if not specs:
        raise Refused("no strategy is enabled at this tag; nothing runs, positions are left")
    return tuple(specs)


def deployed_universe(repo: Path, commit: str) -> Universe:
    text = _show(repo, commit, UNIVERSE_FILE)
    if text is None:
        raise Refused(f"the deploy has no universe file {UNIVERSE_FILE.as_posix()}")
    universe = parse_universe(text)
    if universe.is_template:
        raise Refused(
            f"{UNIVERSE_FILE.as_posix()} at the tag is the unfilled template (version "
            f"{universe.version}); fill it first"
        )
    if not universe.listings:
        raise Refused(f"{UNIVERSE_FILE.as_posix()} at the tag lists no stocks")
    return universe
