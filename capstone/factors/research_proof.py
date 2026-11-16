"""Small independently calculated offline examples; no licensed observations or agent calls."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from capstone.factors.evaluation import Panel, TradingSpec
from capstone.factors.reference import january_diagnostic
from capstone.factors.research_contracts import (
    ConstructionEvidence,
    MethodSpec,
    ResearchPolicy,
    SourceClaim,
    qualify,
)
from capstone.factors.research_controller import ResearchController
from capstone.factors.research_relations import structural_relation
from capstone.factors.research_requests import RequestJournal, ResearchPaused
from capstone.factors.tree import identity_key, parse
from capstone.pipeline.budget import BudgetedClient
from capstone.strategy.contract import StrategySpec, target_weights


def development_checks():
    dates = pd.bdate_range("2000-01-03", periods=40)
    returns = pd.DataFrame({"A": 0.01, "B": -0.01}, index=dates)
    panel = Panel(
        {"returns": returns},
        returns.notna(),
        returns * 0,
        data_version="synthetic-research-controller-v1",
    )
    as_of = dates[10]
    spec = StrategySpec(
        "-ts_sum(returns, 5)", TradingSpec(), panel.data_version, dates[0].date().isoformat()
    )
    expected = pd.Series({"A": -0.5, "B": 0.5})
    weights = target_weights(spec, panel, as_of)
    pd.testing.assert_series_equal(weights, expected, check_names=False)
    wrong = target_weights(replace(spec, expression="ts_sum(returns, 5)"), panel, as_of)
    assert not np.allclose(wrong.reindex(expected.index), expected)
    changed = replace(panel, fields={"returns": returns.copy()})
    changed.fields["returns"].loc[dates > as_of] *= -100
    pd.testing.assert_series_equal(target_weights(spec, changed, as_of), weights)
    recent = returns.copy()
    recent.iloc[-5:] *= -3
    horizon_panel = replace(panel, fields={"returns": recent})
    short = target_weights(spec, horizon_panel, dates[-1])
    long = target_weights(
        replace(spec, expression="-ts_sum(returns, 21)"), horizon_panel, dates[-1]
    )
    assert not np.allclose(short, long)
    cohorts = pd.DataFrame(
        {
            "date": ["2000-01-03", "2000-01-03"],
            "formation": ["1997-12-31", "1999-12-31"],
            "long_return": [0.01, 0.02],
            "short_return": [0.0, 0.0],
        }
    )
    diagnostic = january_diagnostic(cohorts)
    assert diagnostic.age_months.tolist() == [25, 1]
    flow = SourceClaim(
        "https://example.org/synthetic",
        "fixture",
        "flow reversal",
        "no reversal",
        ("returns", "dated_flows"),
        True,
    )
    publication = replace(flow, required_inputs=("unrevised_vintages", "publication_dates"))
    assert qualify(flow, frozenset({"returns"}))[0] == "needs_data"
    assert qualify(publication, frozenset({"accounting"}))[0] == "needs_data"
    one, renamed = "ts_mean(returns, 5)", "ts_mean(returns, 21)"
    assert identity_key(parse(one)) != identity_key(parse(renamed))
    assert structural_relation(one, renamed)["same_structure"]
    return {
        name: True
        for name in (
            "renamed_copy",
            "wrong_direction",
            "wrong_horizon",
            "january_cohort",
            "unsupported_proxy",
            "missing_publication_time",
            "future_invariance",
            "shared_strategy_constructor",
        )
    }


class ScriptedProofClient:
    """A local deterministic fixture, visibly distinct from a real agent result."""

    def __init__(self):
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if kwargs.get("model") == "timeout-fixture":
            raise TimeoutError("synthetic interruption")
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="proof", input={"saved": True})],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def construction_handoff_checks(root):
    """A labelled synthetic adaptation, using the existing strategy constructor."""
    claim = SourceClaim(
        "https://example.org/synthetic-construction",
        "two-stock fixture",
        "buy five-day losers with demeaned signal weights",
        "wrong target shares",
        ("returns",),
        True,
    )
    entry = {
        "brief": {
            "packet_id": "known",
            "family": "liquidity_supply",
            "track": "replication_control",
            "method_key": "synthetic-reference",
        },
        "claim": {
            "source_url": claim.source_url,
            "method_location": claim.method_location,
            "claim": claim.claim,
            "falsifier": claim.falsifier,
            "required_inputs": claim.required_inputs,
            "full_methods": True,
        },
        "constructor": "reference",
    }
    controller = ResearchController(
        root / "construction.db",
        ResearchPolicy((("liquidity_supply", "replication_control", 1),)),
        [entry],
        ["returns"],
        generator_version="offline-fixture-v1",
    )
    try:
        if controller.state("known") is None:
            controller.reserve("known")
        if controller.state("known") == "assigned":
            controller.ground("known", claim)
        method = MethodSpec(
            controller.claim_id("known"),
            "adaptation",
            "buy losers",
            "all stocks",
            "demeaned signal",
            "daily",
            "one session",
            "next rebalance",
            "completed session",
            "long_short",
            ("synthetic engineering fixture",),
        )
        if controller.state("known") == "specification_ready":
            controller.freeze_spec("known", method)
        if controller.state("known") == "specification_frozen":

            def verifier(spec):
                dates = pd.bdate_range("2000-01-03", periods=12)
                returns = pd.DataFrame({"A": 0.01, "B": -0.01}, index=dates)
                panel = Panel(
                    {"returns": returns},
                    returns.notna(),
                    returns * 0,
                    data_version="synthetic-construction-fixture",
                )
                existing = StrategySpec(
                    "-ts_sum(returns, 5)",
                    TradingSpec(),
                    panel.data_version,
                    dates[0].date().isoformat(),
                )
                actual = target_weights(existing, panel, dates[-1])
                pd.testing.assert_series_equal(
                    actual, pd.Series({"A": -0.5, "B": 0.5}), check_names=False
                )
                return ConstructionEvidence(
                    spec.identity, "two-stock-independent-v1", True, True, True
                )

            controller.construct("known", verifier)
        if controller.state("known") == "construction_verified":
            controller.replay_handoff("known")
        assert controller.state("known") == "replay_ready"
        try:
            controller.validation_handoff("known", {"generator_policy": "old-generator"})
        except ValueError:
            pass
        else:
            raise AssertionError("stale calibration scope was accepted")
        handoff = controller.validation_handoff("known", {"generator_policy": controller.identity})
        assert handoff["existing_financial_funnel_required"] and not handoff["formal_admission"]
    finally:
        controller.close()
    return {
        "existing_constructor_handoff": True,
        "stale_calibration_scope_blocked": True,
        "scope_match_is_not_admission": True,
    }


def recovery_checks(root):
    client = ScriptedProofClient()
    manifest = {"call_limits": {"proof": 2}, "kind": "offline_fixture"}
    with RequestJournal(root / "cache.db", manifest) as journal:
        wrapped = BudgetedClient(client, 2, journal=journal, bucket="proof", task_id="cached")
        assert wrapped.create(model="fixture").content[0].input == {"saved": True}
    with RequestJournal(root / "cache.db", manifest) as journal:
        before = client.calls
        wrapped = BudgetedClient(client, 2, journal=journal, bucket="proof", task_id="cached")
        assert wrapped.create(model="fixture").content[0].input == {"saved": True}
        assert client.calls == before
    with RequestJournal(root / "uncertain.db", manifest) as journal:
        wrapped = BudgetedClient(client, 2, journal=journal, bucket="proof", task_id="uncertain")
        try:
            wrapped.create(model="timeout-fixture")
        except ResearchPaused:
            pass
        else:
            raise AssertionError("uncertain submission did not pause")
        before = client.calls
        try:
            wrapped.create(model="timeout-fixture")
        except ResearchPaused:
            pass
        else:
            raise AssertionError("uncertain request was retried")
        assert client.calls == before
        assert journal.report()["blocked"]
    return {
        "saved_response_replayed": True,
        "uncertain_request_not_retried": True,
        "fixture_model_calls_only": True,
    }
