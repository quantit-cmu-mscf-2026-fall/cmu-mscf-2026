"""Tests for capstone.gate: registering a gate configuration before scoring.

- the config's hash is its content, the same in every process;
- `register` logs the config and hash to the ledger once, tagged "gate";
- the gate's own entries never count as trials;
- the input checks refuse returns past the search period, columns with no
  family, and too few bootstrap draws for stage 1.

Every test that touches the ledger points `CAPSTONE_LEDGER_DIR` at a
temporary directory.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import UTC, datetime

import pandas as pd
import pytest

from capstone import calibration, gate, runlog
from capstone.funnel import Thresholds
from capstone.gate import GateConfig, TrialScope, register

N_FAMILIES = 10
K = 5
TRIAL_NAME = "evaluate-factor"


@dataclasses.dataclass(frozen=True)
class Market:
    returns: pd.DataFrame
    family: pd.Series
    search_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp


@pytest.fixture(scope="module")
def market() -> Market:
    """3 years of search and 1 of holdout, on business days from 1990."""
    fs = calibration.make_family_set(N_FAMILIES, K, 0.3, 1.5, seed=1, n_search=756, n_holdout=252)
    dates = pd.bdate_range("1990-01-01", periods=len(fs.returns))
    return Market(
        returns=fs.returns.set_axis(dates),
        family=fs.family,
        search_end=dates[fs.n_search - 1],
        holdout_start=dates[fs.n_search],
        holdout_end=dates[-1],
    )


def config_for(m: Market, **overrides) -> GateConfig:
    kwargs = dict(
        thresholds=Thresholds(stage1="bootstrap"),
        n_boot=199,
        seed=0,
        search_end=m.search_end,
        holdout_start=m.holdout_start,
        holdout_end=m.holdout_end,
        trial_scope=TrialScope(name=TRIAL_NAME),
        data_version="synthetic",
        generator="test-gen",
    )
    kwargs.update(overrides)
    return GateConfig(**kwargs)


def search_of(m: Market) -> pd.DataFrame:
    return m.returns.loc[m.returns.index <= m.search_end]


def log_trials(n: int, name: str = TRIAL_NAME) -> None:
    """Append `n` trial entries to the (temporary) ledger in log_run's shape.

    Written directly rather than through `log_run`, which shells out to git
    once per entry: the gate reads the count, not how it was written.
    """
    path = runlog.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for i in range(n):
            entry = {
                "ts_utc": datetime.now(UTC).isoformat(),
                "name": name,
                "seed": i,
                "params": {"variant": i},
                "tags": [],
            }
            fh.write(json.dumps(entry) + "\n")


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


class TestGateConfig:
    def test_frozen(self, market):
        with pytest.raises(dataclasses.FrozenInstanceError):
            config_for(market).seed = 1

    def test_hash_is_content(self, market):
        a = config_for(market)
        b = config_for(
            market,
            search_end=str(market.search_end.date()),
            holdout_start=market.holdout_start.to_pydatetime(),
        )
        assert a.config_hash == b.config_hash
        assert len(a.config_hash) == 64
        for change in (
            {"seed": 1},
            {"n_boot": 399},
            {"thresholds": Thresholds(stage1="bootstrap", q4=0.2)},
            {"trial_scope": TrialScope(name="other")},
            {"holdout_end": market.holdout_end - pd.DateOffset(days=7)},
            {"generator": "other-gen"},
            {"data_version": "other"},
            {"factors_digest": "0" * 64},
        ):
            assert config_for(market, **change).config_hash != a.config_hash, change

    def test_hash_is_stable_across_processes(self, market):
        # A hash of the content, not of Python's per-process hash seed.
        config = config_for(market)
        assert GateConfig.from_dict(config.to_dict()) == config
        assert json.loads(json.dumps(config.to_dict())) == config.to_dict()
        code = (
            "import json, sys; from capstone.gate import GateConfig; "
            "print(GateConfig.from_dict(json.loads(sys.argv[1])).config_hash)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code, json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == config.config_hash

    @pytest.mark.parametrize(
        "change",
        [
            {"holdout_start": "search_end"},
            {"holdout_end": "search_end"},
            {"generator": " "},
            {"data_version": ""},
            {"n_boot": 0},
            {"n_boot": -1, "thresholds": Thresholds()},
            {"factors_digest": "not-a-digest"},
            # A missing date would be stored as "NaT", which sorts after
            # every ISO date, so the order check alone lets it through.
            {"holdout_end": None},
            {"holdout_end": float("nan")},
            {"holdout_end": "NaT"},
            {"search_end": None},
            {"holdout_start": None},
        ],
    )
    def test_bad_configs_raise_at_construction(self, market, change):
        change = {k: market.search_end if v == "search_end" else v for k, v in change.items()}
        with pytest.raises(ValueError):
            config_for(market, **change)

    def test_a_registered_config_with_no_holdout_end_is_refused(self, market):
        data = config_for(market).to_dict()
        data["holdout_end"] = None
        with pytest.raises(ValueError, match="holdout_end"):
            GateConfig.from_dict(data)

    def test_no_draws_are_needed_without_the_bootstrap(self, market):
        assert config_for(market, n_boot=0, thresholds=Thresholds(stage1="rho")).n_boot == 0


class TestRegistration:
    def test_register_logs_the_config_and_hash(self, ledger, market):
        config = config_for(market)
        entry = register(config)
        assert entry["name"] == gate.REGISTRATION_NAME
        assert entry["params"]["config_hash"] == config.config_hash
        assert entry["params"]["config"] == config.to_dict()
        assert entry["params"]["data_version"] == "synthetic"
        assert entry["tags"] == [gate.GATE_TAG]
        assert [e["name"] for e in runlog.read_entries()] == [gate.REGISTRATION_NAME]

    def test_registering_twice_logs_once(self, ledger, market):
        config = config_for(market)
        first = register(config)
        assert register(config) == first
        assert len(runlog.read_entries()) == 1

    def test_a_changed_config_has_no_registration(self, ledger, market):
        register(config_for(market))
        changed = config_for(market, thresholds=Thresholds(stage1="bootstrap", q1=0.2))
        assert gate._entries_for(gate.REGISTRATION_NAME, changed.config_hash) == []
        register(changed)
        assert len(runlog.read_entries()) == 2

    def test_gate_entries_are_not_trials(self, ledger, market):
        log_trials(12)
        log_trials(3, name="other-search")
        register(config_for(market))
        register(config_for(market, seed=1))
        assert gate._trial_count(TrialScope(name=TRIAL_NAME)) == 12
        assert gate._trial_count(TrialScope()) == 15
        assert gate._trial_count(TrialScope(exclude_tags=("synthetic",))) == 15


class TestInputChecks:
    def test_required_n_boot(self):
        assert gate.required_n_boot(10) == 199
        assert gate.required_n_boot(200) == 3999
        assert gate.required_n_boot(10, q=0.10) == 99
        with pytest.raises(ValueError):
            gate.required_n_boot(10, q=0.0)

    def test_too_few_bootstrap_draws_are_refused(self, market):
        floor = gate.required_n_boot(N_FAMILIES, Thresholds().q1)
        config = config_for(market, n_boot=floor - 1)
        with pytest.raises(ValueError, match=r"n_boot \+ 1 >= families / q1"):
            gate._checked_search(search_of(market), market.family, config)
        labels = gate._checked_search(search_of(market), market.family, config_for(market))
        pd.testing.assert_series_equal(labels, market.family.reindex(market.returns.columns))

    def test_the_floor_only_binds_the_bootstrap_stage_1(self, market):
        config = config_for(market, n_boot=0, thresholds=Thresholds(stage1="rho"))
        gate._checked_search(search_of(market), market.family, config)

    def test_returns_past_search_end_are_refused(self, market):
        with pytest.raises(ValueError, match="past search_end"):
            gate._checked_search(market.returns, market.family, config_for(market))

    def test_family_must_cover_every_column(self, market):
        family = market.family.drop(market.family.index[0])
        with pytest.raises(ValueError, match="family is missing"):
            gate._checked_search(search_of(market), family, config_for(market))

    def test_returns_must_be_dated_and_increasing(self, market):
        search = search_of(market)
        with pytest.raises(TypeError):
            gate._checked_search(search.reset_index(drop=True), market.family, config_for(market))
        with pytest.raises(ValueError, match="increasing"):
            gate._checked_search(search.iloc[::-1], market.family, config_for(market))
