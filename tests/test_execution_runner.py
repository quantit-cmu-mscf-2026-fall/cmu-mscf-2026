"""The runner: run_once reads the registry at a deploy tag, builds the combined
book, checks the risk limits and writes the plan (dry run). It has no send path.

Fake broker, frozen synthetic data, a temp git repo with its own tag and a temp
ledger; never the real repo's tags, never a network or a broker connection.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_execution_universe import universe_text as venue_universe_text
from test_strategy_contract import _relabel, fake_rows, make_spec

from capstone import runlog
from capstone.execution import runner
from capstone.execution.broker import PAPER_FLAG, FakeBroker
from capstone.execution.data import FrozenSnapshotSource
from capstone.execution.runner import (
    LEDGER_NAME,
    SEND_PATH,
    Refused,
    RiskLimitError,
    RiskLimits,
    run_once,
)
from capstone.execution.universe import UniverseError
from capstone.factors.evaluation import HoldoutError, TradingSpec, build_panel
from capstone.strategy.contract import basic_string, save_spec

N_STOCKS, N_DAYS = 30, 60
LIVE_DATES = pd.bdate_range("2020-06-01", periods=N_DAYS)  # relabelled discovery dates
AS_OF = LIVE_DATES[-1]
ANCHOR = LIVE_DATES[10]
PAPER_ACCOUNT = "DU0000001"
TAG = "deploy-2026.08.24"
PERMNOS = tuple(range(10000, 10000 + N_STOCKS))
CAPITAL = 1_000_000.0
LOOSE = RiskLimits(max_gross=1.5, max_weight=0.5, max_order_notional=1e9)
INC = "incubation-only"


def symbol(permno: int) -> str:
    return f"T{permno - 10000:02d}"


def spec_(**overrides):
    return make_spec(
        **{"trading": TradingSpec(hold=5, neutral="none"), "anchor": str(ANCHOR.date())} | overrides
    )


def live_panel(seed=0):
    rows = fake_rows(n_stocks=N_STOCKS, n_days=N_DAYS, seed=seed)
    return _relabel(build_panel(rows, "synthetic-live-v1"), LIVE_DATES)


def prices():
    return {symbol(p): 20.0 + (p - 10000) for p in PERMNOS}


def git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def registry_text(entries) -> str:
    """entries: (id, gate_report, enabled) triples, or with a fourth role."""
    if not entries:
        return "# registry\nstrategies: []\n"
    body = "".join(
        f"  - id: {basic_string(i)}\n"
        f"    module: capstone.strategy.contract\n"
        f"    gate_report: {basic_string(g)}\n"
        f"    enabled: {'true' if e else 'false'}\n"
        + "".join(f"    role: {basic_string(r)}\n" for r in role)
        for i, g, e, *role in entries
    )
    return "# registry\nstrategies:\n" + body


def deploy_repo(root, specs, *, enabled=None, universe=None, registry=None, write_specs=True):
    """A git repo holding a registry of `specs`, their files and a universe file,
    committed on main and tagged TAG."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    deploy = root / "deploy"
    (deploy / "strategies").mkdir(parents=True)
    enabled = [True] * len(specs) if enabled is None else enabled
    if registry is None:
        registry = registry_text(
            [(s.id, s.gate_report, e) for s, e in zip(specs, enabled, strict=True)]
        )
    (deploy / "strategies.yml").write_bytes(registry.encode("utf-8"))
    if write_specs:
        for spec in specs:
            save_spec(spec, deploy / "strategies" / f"{spec.id}.toml")
    (deploy / "strategies" / ".keep").write_text("")
    if universe is not False:
        text = venue_universe_text(permnos=PERMNOS) if universe is None else universe
        (deploy / "universe.csv").write_bytes(text.encode("utf-8"))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "deploy")
    git(root, "tag", TAG)
    return root


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.delenv(PAPER_FLAG, raising=False)
    return tmp_path / "ledger"


@pytest.fixture
def repo(tmp_path):
    return deploy_repo(tmp_path / "repo", [spec_()])


def broker(holdings=None) -> FakeBroker:
    return FakeBroker(PAPER_ACCOUNT, 4002, holdings=dict(holdings or {}), quotes=prices())


def run(repo, b=None, *, mode="dry-run", limits=LOOSE, tag=TAG, data=None, **kw):
    return run_once(
        repo,
        tag,
        AS_OF,
        b if b is not None else broker(),
        data if data is not None else FrozenSnapshotSource(live_panel()),
        capital=CAPITAL,
        mode=mode,
        limits=limits,
        out_dir=repo.parent / "out",
        **kw,
    )


def ledger_entries():
    path = runlog.ledger_path()
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------
# Dry run only; nothing connects


def test_dry_run_plans_orders_and_never_calls_place(repo):
    spy = broker()
    result = run(repo, spy)
    assert spy.placed == []
    assert set(spy.calls) <= {"positions", "prices"}
    assert len(result.orders) > 10
    plan = json.loads(result.plan_path.read_text())
    assert plan["digest"] == result.digest
    assert plan["mode"] == "dry-run"
    assert len(plan["plan"]["orders"]) == len(result.orders)
    [entry] = ledger_entries()
    assert entry["name"] == LEDGER_NAME == "paper-run"
    assert "dry-run" in entry["tags"] and "paper" not in entry["tags"]
    assert entry["params"]["plan_digest"] == result.digest
    assert entry["params"]["tag"] == TAG
    assert entry["params"]["data_version"] == "synthetic-live-v1"


def test_paper_mode_is_refused_and_names_the_single_send_path(repo):
    spy = broker()
    with pytest.raises(Refused, match="send path") as info:
        run(repo, spy, mode="paper")
    assert SEND_PATH in str(info.value)
    assert spy.placed == [] and spy.calls == []
    assert ledger_entries() == []
    assert not (repo.parent / "out").exists()


def test_an_unknown_mode_is_refused(repo):
    with pytest.raises(ValueError, match="mode"):
        run(repo, mode="live")


def test_the_default_configuration_never_constructs_a_live_connection(repo):
    """The runner is handed a broker; it has no default one, imports no broker
    client library, and a run touches only the broker it was given."""
    assert inspect.signature(run_once).parameters["broker"].default is inspect.Parameter.empty
    assert runner.MODES == ("dry-run",)
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for word in ("socket", "connect(", "environ", "getenv"):
        assert word not in source
    # Every import is the package itself, the standard library or the data stack:
    # no third-party client library can come in through the runner.
    allowed = {"capstone", "pandas", "numpy"} | set(sys.stdlib_module_names)
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    assert imported <= allowed, sorted(imported - allowed)


def test_a_post_2020_book_is_refused_before_the_data_or_broker_is_asked(tmp_path):
    later = pd.bdate_range("2026-06-01", periods=N_DAYS)
    panel = _relabel(build_panel(fake_rows(n_stocks=N_STOCKS, n_days=N_DAYS), "x"), later)
    repo = deploy_repo(tmp_path / "repo", [spec_(anchor=str(later[10].date()))])
    spy = broker()

    class Untouched(FrozenSnapshotSource):
        def panel(self, fields, as_of):
            raise AssertionError("the data source was asked for a post-2020 panel")

    with pytest.raises(HoldoutError, match="authorized current-feature route"):
        run_once(
            repo,
            TAG,
            later[-1],
            spy,
            Untouched(panel),
            capital=CAPITAL,
            limits=LOOSE,
            out_dir=tmp_path / "out",
        )
    assert spy.calls == [] and spy.placed == []
    assert ledger_entries() == []


def test_account_ids_never_appear_unmasked_in_logs_plans_or_the_ledger(repo, caplog):
    caplog.set_level(logging.DEBUG)
    result = run(repo)
    out = repo.parent / "out"
    texts = [p.read_text() for p in out.iterdir()] + [runlog.ledger_path().read_text()]
    texts.append(caplog.text)
    assert len(texts) >= 4
    for text in texts:
        assert PAPER_ACCOUNT not in text
    plan = json.loads(result.plan_path.read_text())
    assert plan["account"] == "DU****01"
    assert "DU****01" in caplog.text


# ---------------------------------------------------------------------------
# Tags and the registry


@pytest.mark.parametrize("tag", ["main", "HEAD", "deploy-2026.09.01", "deploy-latest"])
def test_only_an_existing_deploy_tag_runs(repo, tag):
    with pytest.raises(Refused, match="tag"):
        run(repo, tag=tag)


def test_a_branch_named_like_a_deploy_tag_is_refused(repo):
    git(repo, "branch", "deploy-2026.09.01")
    with pytest.raises(Refused, match="tag"):
        run(repo, tag="deploy-2026.09.01")


def test_the_registry_is_read_at_the_tag_not_the_working_tree_or_branch_tip(repo):
    first = run(repo)
    (repo / "deploy" / "strategies.yml").write_text(registry_text([]))
    git(repo, "commit", "-qam", "empty the registry")
    (repo / "deploy" / "universe.csv").write_text("garbage")
    again = run(repo)
    assert again.digest == first.digest
    assert again.plan["commit"] == git(repo, "rev-parse", f"{TAG}^{{commit}}")


def test_disabled_entries_are_skipped(tmp_path):
    on, off = spec_(), spec_(trading=TradingSpec(hold=3, neutral="none"))
    repo = deploy_repo(tmp_path / "repo", [on, off], enabled=[True, False])
    result = run(repo)
    assert [s["id"] for s in result.plan["strategies"]] == [on.id]


def test_a_registry_with_nothing_enabled_is_refused(tmp_path):
    repo = deploy_repo(tmp_path / "repo", [spec_()], enabled=[False])
    with pytest.raises(Refused, match="enabled"):
        run(repo)


def test_an_enabled_entry_without_its_spec_file_is_refused(tmp_path):
    repo = deploy_repo(tmp_path / "repo", [spec_()], write_specs=False)
    with pytest.raises(Refused, match="spec"):
        run(repo)


def test_an_entry_without_a_gate_report_is_refused(tmp_path):
    spec = spec_()
    text = registry_text([(spec.id, "x", True)]).replace('gate_report: "x"', 'gate_report: ""')
    repo = deploy_repo(tmp_path / "repo", [spec], registry=text)
    with pytest.raises(Refused, match="gate"):
        run(repo)


def test_an_entry_whose_gate_report_differs_from_its_spec_is_refused(tmp_path):
    spec = spec_()
    text = registry_text([(spec.id, "reports/other.md", True)])
    repo = deploy_repo(tmp_path / "repo", [spec], registry=text)
    with pytest.raises(Refused, match="gate"):
        run(repo)


# ---------------------------------------------------------------------------
# Risk limits


@pytest.mark.parametrize(
    ("limits", "match"),
    [
        (RiskLimits(max_gross=0.5, max_weight=0.5, max_order_notional=1e9), "gross"),
        (RiskLimits(max_gross=1.5, max_weight=0.01, max_order_notional=1e9), "weight"),
        (RiskLimits(max_gross=1.5, max_weight=0.5, max_order_notional=1_000.0), "notional"),
    ],
)
def test_each_risk_limit_stops_the_run_before_anything_is_logged(repo, limits, match):
    spy = broker()
    with pytest.raises(RiskLimitError, match=match):
        run(repo, spy, limits=limits)
    assert spy.placed == []
    assert ledger_entries() == []


def test_the_kill_switch_file_stops_every_run(repo, tmp_path):
    kill = tmp_path / "KILL_SWITCH"
    kill.write_text("stop")
    spy = broker()
    with pytest.raises(RiskLimitError, match="kill switch"):
        run(repo, spy, limits=RiskLimits(1.5, 0.5, 1e9, kill_switch=kill))
    assert spy.placed == [] and spy.calls == []
    kill.unlink()
    run(repo, spy, limits=RiskLimits(1.5, 0.5, 1e9, kill_switch=kill))


# ---------------------------------------------------------------------------
# Determinism, the universe, the data


def test_the_same_inputs_give_the_same_orders_and_digest(repo):
    held = {symbol(10003): 500, symbol(10011): -200}
    first = run(repo, broker(holdings=held))
    bytes_first = first.plan_path.read_bytes()
    second = run(repo, broker(holdings=dict(reversed(list(held.items())))))
    assert second.digest == first.digest
    assert second.plan_path.read_bytes() == bytes_first
    assert second.orders == first.orders
    other = run(repo, broker(holdings={symbol(10003): 501}))
    assert other.digest != first.digest


def test_current_positions_are_netted_against_the_target(repo):
    flat = {o.symbol: o.shares for o in run(repo).orders}
    name = next(iter(flat))
    held = run(repo, broker(holdings={name: flat[name]})).orders
    assert name not in {o.symbol for o in held}


def test_the_broker_sees_the_venue_spelling_of_a_name(tmp_path):
    text = venue_universe_text(permnos=PERMNOS, venue_symbols={10003: "T03 B"})
    repo = deploy_repo(tmp_path / "repo", [spec_()], universe=text)
    spy = FakeBroker(PAPER_ACCOUNT, 4002, quotes=prices() | {"T03 B": 23.0})
    names = {o.symbol for o in run(repo, spy).orders}
    assert "T03 B" in names and "T03" not in names


def test_a_target_name_outside_the_universe_is_refused(tmp_path):
    text = venue_universe_text(permnos=range(10000, 10020))
    repo = deploy_repo(tmp_path / "repo", [spec_()], universe=text)
    with pytest.raises(UniverseError):
        run(repo)


def test_a_held_symbol_outside_the_universe_is_refused(repo):
    with pytest.raises(UniverseError, match="ZZZ"):
        run(repo, broker(holdings={"ZZZ": 10}))


def test_a_deploy_without_a_universe_file_is_refused(tmp_path):
    repo = deploy_repo(tmp_path / "repo", [spec_()], universe=False)
    with pytest.raises(Refused, match="universe"):
        run(repo)


class LookaheadSource:
    """A source that hands back data after as_of."""

    def panel(self, fields, as_of):
        return live_panel()


def test_data_after_as_of_is_refused(repo):
    with pytest.raises(Refused, match="after"):
        run_once(
            repo,
            TAG,
            pd.Timestamp(AS_OF) - pd.offsets.BDay(5),
            broker(),
            LookaheadSource(),
            capital=CAPITAL,
            limits=LOOSE,
            out_dir=repo.parent / "out",
        )


# ---------------------------------------------------------------------------
# Incubation-only entries run on capped capital (scaled, never refused)


def _repo(tmp_path, name, specs, entries):
    return deploy_repo(tmp_path / name, specs, registry=registry_text(entries))


def test_the_default_incubation_cap_is_a_tenth_and_must_be_a_fraction():
    assert RiskLimits().incubation_cap == 0.10
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="incubation_cap"):
            RiskLimits(incubation_cap=bad)


def test_an_incubation_only_strategy_is_scaled_to_the_cap_in_the_dry_run_plan(tmp_path):
    spec = spec_()
    full = run(_repo(tmp_path, "full", [spec], [(spec.id, spec.gate_report, True)]))
    spy = broker()
    capped = run(
        _repo(tmp_path, "capped", [spec], [(spec.id, spec.gate_report, True, INC)]),
        spy,
        limits=dataclasses.replace(LOOSE, incubation_cap=0.1),
    )
    assert spy.placed == []
    plan = json.loads(capped.plan_path.read_text())["plan"]
    [entry] = plan["strategies"]
    assert entry["role"] == INC and entry["capital_share"] == 0.1
    assert plan["limits"]["incubation_cap"] == 0.1
    assert entry["book"] == full.plan["strategies"][0]["book"]
    assert set(plan["target"]) == set(full.plan["target"])
    for permno, weight in plan["target"].items():
        assert weight == pytest.approx(0.1 * full.plan["target"][permno], abs=1e-15)
    assert sum(abs(w) for w in plan["target"].values()) <= 0.1 + 1e-12
    half_shares = sum(o.price / 2 for o in capped.orders)  # whole-share rounding
    assert sum(o.notional for o in capped.orders) <= 0.1 * CAPITAL + half_shares


def test_the_cap_binds_all_incubation_only_strategies_together(tmp_path):
    a, b, c = spec_(), spec_(hypothesis_id="H-002"), spec_(hypothesis_id="H-003")
    entries = [
        (a.id, a.gate_report, True),
        (b.id, b.gate_report, True, INC),
        (c.id, c.gate_report, True, INC),
    ]
    result = run(_repo(tmp_path, "repo", [a, b, c], entries))
    shares = {s["id"]: s["capital_share"] for s in result.plan["strategies"]}
    # deploy keeps its equal share (1/3); the two incubation-only split the 0.10 cap.
    assert shares == {a.id: pytest.approx(1 / 3), b.id: 0.05, c.id: 0.05}
    books = {s["id"]: pd.Series(s["book"]) for s in result.plan["strategies"]}
    expected = sum(books[i] * shares[i] for i in shares)
    expected = expected[expected != 0]
    got = pd.Series(result.plan["target"])
    pd.testing.assert_series_equal(
        got.sort_index(), expected.sort_index(), check_names=False, rtol=1e-12
    )


def test_a_cap_above_the_equal_share_leaves_the_equal_share(tmp_path):
    a, b = spec_(), spec_(hypothesis_id="H-002")
    entries = [(a.id, a.gate_report, True), (b.id, b.gate_report, True, INC)]
    result = run(
        _repo(tmp_path, "repo", [a, b], entries),
        limits=dataclasses.replace(LOOSE, incubation_cap=0.9),
    )
    shares = {s["id"]: s["capital_share"] for s in result.plan["strategies"]}
    assert shares == {a.id: 0.5, b.id: 0.5}


def test_a_deploy_without_enabled_incubation_only_entries_plans_as_before(tmp_path):
    spec, other = spec_(), spec_(hypothesis_id="H-002")
    entries = [(spec.id, spec.gate_report, True), (other.id, other.gate_report, False, INC)]
    plan = run(_repo(tmp_path, "repo", [spec, other], entries)).plan
    assert [s["id"] for s in plan["strategies"]] == [spec.id]  # disabled: skipped
    assert "incubation_cap" not in plan["limits"]
    assert set(plan["strategies"][0]) == {"id", "expression", "gate_report", "data_version", "book"}
    assert set(plan) == {
        "tag", "commit", "as_of", "capital", "universe_version", "data_versions",
        "limits", "strategies", "target", "positions", "prices", "orders",
    }  # fmt: skip


# ---------------------------------------------------------------------------
# The frozen-plan route's lazy lookup of the registered cap


def test_the_registered_incubation_cap_is_the_runners_default(monkeypatch):
    from capstone.execution.frozen import IncubationRouteRefused, registered_incubation_cap

    assert registered_incubation_cap() == RiskLimits.incubation_cap == 0.10
    monkeypatch.setitem(sys.modules, "capstone.execution.runner", None)  # unimportable
    with pytest.raises(IncubationRouteRefused, match="unavailable"):
        registered_incubation_cap()
