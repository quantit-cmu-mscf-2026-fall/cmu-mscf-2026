"""The command line: a dry run only, and no broker connection unless it is injected.

The default command opens nothing. These tests prove that with sockets disabled,
and that every refusal that needs only the arguments or the tag happens before
a factory is called.
"""

from __future__ import annotations

import ast
import socket
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_execution_runner import (  # noqa: F401  (ledger is an autouse fixture)
    AS_OF,
    CAPITAL,
    TAG,
    FrozenSnapshotSource,
    broker,
    deploy_repo,
    ledger,
    ledger_entries,
    live_panel,
    spec_,
)

from capstone.execution import cli
from capstone.execution.data import UnsupportedFieldError
from capstone.execution.runner import SEND_PATH, RiskLimits


@pytest.fixture
def repo(tmp_path):
    return deploy_repo(tmp_path / "repo", [spec_()])


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a network call was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def argv(repo, *extra, as_of=None):
    return [
        *("--tag", TAG, "--as-of", as_of or str(AS_OF.date()), "--repo", str(repo)),
        *("--capital", str(CAPITAL), "--max-weight", "0.5", "--max-order-notional", "1e9"),
        *("--out", str(repo.parent / "out")),
        *extra,
    ]


def spy():
    seen = []

    def factory(*args, **kwargs):
        seen.append((args, kwargs))
        raise AssertionError("a factory was called")

    return factory, seen


def test_the_default_command_refuses_without_building_a_connection(repo, no_network, monkeypatch):
    monkeypatch.setenv("CAPSTONE_KILL_SWITCH", str(repo.parent / "KILL"))
    with pytest.raises(SystemExit, match="no broker connection is configured"):
        cli.main(argv(repo, "--assumed-half-spread-bps", "5"))
    assert ledger_entries() == []
    assert not (repo.parent / "out").exists()


def test_one_factory_alone_is_not_enough(repo, no_network):
    factory, seen = spy()
    flags = ("--assumed-half-spread-bps", "5")
    with pytest.raises(SystemExit, match="no broker connection is configured"):
        cli.main(argv(repo, *flags), broker_factory=factory)
    with pytest.raises(SystemExit, match="no broker connection is configured"):
        cli.main(argv(repo, *flags), data_factory=factory)
    assert seen == []


def test_the_defaults_are_no_factories_and_there_is_no_connect_or_live_option():
    assert cli.main.__kwdefaults__ == {"broker_factory": None, "data_factory": None}
    assert not hasattr(cli, "connect_from_env")
    base = ["--tag", TAG, "--as-of", "2026-08-21"]
    for flag in ("--live", "--connect", "--host", "--port", "--account", "--client-id"):
        with pytest.raises(SystemExit):
            cli.parse_args([*base, flag])
    for word in ("connect", "live"):
        with pytest.raises(SystemExit):
            cli.parse_args([word, *base])


def test_the_cli_reads_no_connection_settings_and_imports_no_client_library():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    for word in (
        "socket",
        " connect(",
        ".connect(",
        "_HOST",
        "_PORT",
        "_CLIENT_ID",
        "_ACCOUNT",
        "getenv",
    ):
        assert word not in source
    allowed = {"capstone", "pandas", "numpy"} | set(sys.stdlib_module_names)
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    assert imported <= allowed, sorted(imported - allowed)


def test_paper_is_refused_naming_the_send_path_before_any_factory(repo):
    factory, seen = spy()
    with pytest.raises(SystemExit) as info:
        cli.main(
            argv(repo, "--paper", "--assumed-half-spread-bps", "5"),
            broker_factory=factory,
            data_factory=factory,
        )
    assert SEND_PATH in str(info.value.code)
    assert seen == [] and ledger_entries() == []


def test_the_spread_assumption_is_required_before_any_factory(repo):
    factory, seen = spy()
    with pytest.raises(SystemExit, match="assumed-half-spread-bps"):
        cli.main(argv(repo), broker_factory=factory, data_factory=factory)
    assert seen == []


def test_a_post_2020_date_is_refused_before_any_factory(repo):
    factory, seen = spy()
    with pytest.raises(SystemExit):
        cli.main(
            argv(repo, "--assumed-half-spread-bps", "5", as_of="2026-08-21"),
            broker_factory=factory,
            data_factory=factory,
        )
    assert seen == []


def test_a_capital_is_required(repo, monkeypatch):
    monkeypatch.delenv("CAPSTONE_PAPER_CAPITAL", raising=False)
    flags = ["--tag", TAG, "--as-of", str(AS_OF.date()), "--assumed-half-spread-bps", "5"]
    with pytest.raises(SystemExit, match="CAPSTONE_PAPER_CAPITAL"):
        cli.main([*flags, "--repo", str(repo)])


@pytest.mark.parametrize("expression", ["rank(cap)", "rank(returns - mkt_return)"])
def test_a_field_a_bar_source_lacks_is_refused_before_any_factory(tmp_path, expression):
    repo = deploy_repo(tmp_path / "repo", [spec_(expression=expression)])
    factory, seen = spy()
    with pytest.raises(UnsupportedFieldError):
        cli.main(
            argv(repo, "--assumed-half-spread-bps", "5"),
            broker_factory=factory,
            data_factory=factory,
        )
    assert seen == []


def test_the_cli_requires_a_tag_and_a_date():
    with pytest.raises(SystemExit):
        cli.parse_args(["--as-of", "2026-08-21"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--tag", TAG])
    args = cli.parse_args(["--tag", TAG, "--as-of", "2026-08-21"])
    assert args.as_of == pd.Timestamp("2026-08-21") and args.paper is False


def test_an_injected_pair_dry_runs_on_a_read_only_broker(repo, monkeypatch, capsys):
    seen = {}
    fake = broker()

    def broker_factory(env, *, readonly):
        seen["readonly"] = readonly
        return fake

    def data_factory(b, universe, args):
        seen["data"] = (args.assumed_half_spread_bps, universe.version)
        return FrozenSnapshotSource(live_panel())

    monkeypatch.setenv("CAPSTONE_KILL_SWITCH", str(repo.parent / "KILL"))
    code = cli.main(
        argv(repo, "--assumed-half-spread-bps", "5"),
        broker_factory=broker_factory,
        data_factory=data_factory,
    )
    assert code == 0
    assert seen == {"readonly": True, "data": (5.0, "2026-06-01")}
    assert "place" not in fake.calls and fake.placed == []
    assert "dry-run" in ledger_entries()[-1]["tags"]
    assert "dry-run:" in capsys.readouterr().out


def test_the_cli_takes_the_incubation_cap():
    base = ["--tag", TAG, "--as-of", "2026-08-21"]
    assert cli.parse_args(base).incubation_cap == RiskLimits.incubation_cap
    assert cli.parse_args([*base, "--incubation-cap", "0.05"]).incubation_cap == 0.05
