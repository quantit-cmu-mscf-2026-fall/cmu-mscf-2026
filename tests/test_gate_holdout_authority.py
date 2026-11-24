"""`run_gate` opens the holdout through `holdout_authority` (#231, #363).

- real data cannot reach stage 4 without an authorized protocol for its config;
- a refused authority (unauthorized, another config, survivors outside the
  frozen ids, a period that misses the holdout) leaves the ledger and the
  authority store as they were: no gate look, no dataset look, nothing spent;
- an authorized protocol is spent with ISO dates before the gate logs its own
  look, and the provider reads with the open look's `access`;
- a new registration cannot renew a spent dataset look.

Synthetic markets from `gate_helpers` (they end before 2021), a temporary
ledger and a temporary authority store per test. No real data is read.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from gate_helpers import FUNNEL, Market, config_for, log_trials, looks, market, search_of
from holdout_fakes import approve, make_protocol

from capstone import holdout_authority as authority
from capstone import runlog
from capstone.artifacts import digest
from capstone.gate import GateConfig, register, run_gate

AUTHORIZER = "test-gate-owner"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
def store(tmp_path):
    return tmp_path / "authority"


@pytest.fixture(scope="module")
def planted() -> Market:
    """A planted market whose variant ids look like the store's (16 hex digits)."""
    m = market(seed=1, share_real=0.3)
    ids = {c: digest(c)[:16] for c in m.returns.columns}
    return dataclasses.replace(
        m, returns=m.returns.rename(columns=ids), family=m.family.rename(ids)
    )


class Provider:
    """Serves the market's returns; records each call and checks the access is live."""

    def __init__(self, m: Market):
        self.returns = m.returns
        self.calls: list[tuple] = []

    def __call__(self, ids, start, end, *, access=None):
        if access is not None:
            authority.validate_access(access, end=end.date().isoformat())
        self.calls.append((tuple(ids), start, end, access))
        idx = self.returns.index
        return self.returns.loc[(idx >= start) & (idx <= end), list(ids)]


def gate_looks() -> list[dict]:
    return [e for e in runlog.read_entries() if e["name"] == "gate-holdout-look"]


def dataset_looks() -> list[dict]:
    return [e for e in runlog.read_entries() if e["name"] == authority.LOOK_RUN_NAME]


def registered(m: Market, **overrides) -> GateConfig:
    log_trials(m.returns.shape[1])
    config = config_for(m, decision_rule=FUNNEL, **overrides)
    register(config)
    return config


def protocol_for(config: GateConfig, m: Market, **changes):
    kwargs = dict(
        period=(config.holdout_start, config.holdout_end),
        candidate_ids=sorted(m.returns.columns),
    )
    return make_protocol(config, **(kwargs | changes))


def gate(config, m, provider, protocol, store):
    return run_gate(
        config,
        search_of(m),
        m.family,
        holdout=provider,
        authorizer=AUTHORIZER,
        protocol=protocol,
        authority_root=store,
    )


def assert_nothing_spent(provider, protocol, store, state):
    assert provider.calls == []
    assert gate_looks() == []
    assert dataset_looks() == []
    assert authority.status(protocol, synthetic_root=store)["state"] == state


class TestRealDataNeedsTheAuthority:
    REAL = dict(
        search_end="2020-12-31",
        holdout_start="2021-01-01",
        holdout_end="2025-12-31",
        data_version="crsp-test",
    )

    def test_real_data_without_a_protocol_is_refused_before_anything_is_logged(
        self, ledger, planted, store
    ):
        config = registered(planted, **self.REAL)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed, match="real data"):
            gate(config, planted, provider, None, store)
        assert provider.calls == [] and gate_looks() == [] and dataset_looks() == []

    def test_real_data_refuses_a_synthetic_protocol(self, ledger, planted, store):
        config = registered(planted, **self.REAL)
        protocol = protocol_for(config, planted)
        approve(protocol, store)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed, match=authority.REAL_DATASET):
            gate(config, planted, provider, protocol, store)
        assert_nothing_spent(provider, protocol, store, "AUTHORIZED")


class TestARefusedAuthoritySpendsNothing:
    def test_an_unauthorized_protocol(self, ledger, planted, store):
        config = registered(planted)
        protocol = protocol_for(config, planted)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed, match="authorization"):
            gate(config, planted, provider, protocol, store)
        assert_nothing_spent(provider, protocol, store, "SEALED")
        assert not store.exists()

    def test_a_protocol_for_another_config(self, ledger, planted, store):
        config = registered(planted)
        other = config_for(planted, decision_rule=FUNNEL, seed=1)
        protocol = protocol_for(other, planted)
        approve(protocol, store)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed, match="different config"):
            gate(config, planted, provider, protocol, store)
        assert_nothing_spent(provider, protocol, store, "AUTHORIZED")

    def test_survivors_outside_the_frozen_ids(self, ledger, planted, store):
        config = registered(planted)
        protocol = protocol_for(config, planted, candidate_ids=["f" * 16])
        approve(protocol, store)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed, match="frozen final protocol"):
            gate(config, planted, provider, protocol, store)
        assert_nothing_spent(provider, protocol, store, "AUTHORIZED")

    def test_a_period_that_misses_the_holdout(self, ledger, planted, store):
        config = registered(planted)
        protocol = protocol_for(config, planted, period=(config.holdout_start, "1997-12-31"))
        approve(protocol, store)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed, match="does not cover"):
            gate(config, planted, provider, protocol, store)
        assert_nothing_spent(provider, protocol, store, "AUTHORIZED")


class TestAnAuthorizedProtocolOpensTheHoldoutOnce:
    def test_the_dataset_look_is_spent_with_iso_dates_before_the_gate_look(
        self, ledger, planted, store
    ):
        config = registered(planted)
        protocol = protocol_for(config, planted)
        approve(protocol, store)
        provider = Provider(planted)
        result = gate(config, planted, provider, protocol, store)

        (call,) = provider.calls
        ids, start, end, access = call
        assert access is not None  # validated inside the provider while the look was open
        assert (start, end) == (
            pd.Timestamp(config.holdout_start),
            pd.Timestamp(config.holdout_end),
        )
        survivors = tuple(result.scorecard["best"][result.scorecard["s3_pass"]])
        assert ids == survivors and ids

        entries = runlog.read_entries()
        names = [e["name"] for e in entries]
        assert names.index(authority.LOOK_RUN_NAME) < names.index("gate-holdout-look")
        (look,) = dataset_looks()
        assert look["params"]["dates"] == [config.holdout_start, config.holdout_end]
        assert look["params"]["candidates"] == list(survivors)
        assert len(looks(config.config_hash)) == 1
        assert authority.status(protocol, synthetic_root=store)["state"] == "SPENT"

    def test_a_new_registration_cannot_renew_the_dataset_look(self, ledger, planted, store):
        config = registered(planted)
        protocol = protocol_for(config, planted)
        approve(protocol, store)
        gate(config, planted, Provider(planted), protocol, store)

        renewed = registered(planted, seed=1)
        fresh = protocol_for(renewed, planted)
        provider = Provider(planted)
        with pytest.raises(authority.HoldoutSealed):
            approve(fresh, store)
        with pytest.raises(authority.HoldoutSealed):
            gate(renewed, planted, provider, fresh, store)
        assert provider.calls == []
        assert looks(renewed.config_hash) == []
        assert len(dataset_looks()) == 1
