from copy import deepcopy

import pandas as pd
import pytest

from capstone.factors.evaluation import HoldoutError
from capstone.factors.research_compiler import independent_example
from capstone.factors.research_contracts import canonical
from capstone.pipeline.reference_fixture import reference_panel
from capstone.strategy.compiled import (
    COMMON,
    RULES,
    CompiledStrategySpec,
    _window,
    validate_program,
)
from capstone.strategy.contract import load_spec, target_weights


def spec(method="weekly_reversal", side="long_only"):
    program = RULES[method] | COMMON | {"profile": side}
    return CompiledStrategySpec(
        method, canonical(program), "a" * 64, "b" * 64, "synthetic-reference-v2", "2000-12-29", side
    )


@pytest.mark.parametrize(
    "method,side",
    [
        (m, s)
        for m in RULES
        for s in ("long_only", "long_short")
        if not (m == "turn_of_month" and s == "long_short")
    ],
)
def test_compiled_program_independent_construction_and_legacy_loading(tmp_path, method, side):
    candidate = spec(method, side)
    assert all(independent_example(candidate).values())
    path = tmp_path / "spec.toml"
    path.write_text(candidate.to_toml())
    assert load_spec(path) == candidate


def test_calendar_next_close_timing_holiday_and_no_unscheduled_bets():
    calendar = pd.bdate_range("2020-11-02", "2021-02-01").difference(
        pd.to_datetime(["2020-11-26", "2020-12-25", "2021-01-01", "2021-01-18"])
    )
    # Nov25 decision -> Nov27 close fill -> Nov30 (last session) earns return.
    assert _window(calendar, "2020-11-25")
    # Dec2 decision -> Dec3 fill -> Dec4 is the fourth, excluded, session.
    assert not _window(calendar, "2020-12-02")
    assert not _window(calendar, "2020-11-20")
    with pytest.raises(ValueError, match="three future"):
        _window(calendar, "2021-02-01")


def test_wrong_horizon_proxy_profile_and_source_identity_cannot_compile():
    wrong = RULES["monthly_low_volatility"] | COMMON | {"profile": "long_only"}
    wrong["signal"] = "daily_volatility_1260"
    with pytest.raises(ValueError, match="mismatch"):
        validate_program("monthly_low_volatility", wrong, "long_only")
    with pytest.raises(ValueError, match="does not authorize"):
        spec("turn_of_month", "long_short")
    candidate = spec()
    with pytest.raises(ValueError, match="implementation changed"):
        CompiledStrategySpec(**(candidate.__dict__ | {"implementation": "c" * 64}))


def test_calendar_formation_names_and_whole_study_holdout_remain_protected():
    panel = reference_panel()
    candidate = spec("turn_of_month")
    baseline = target_weights(candidate, panel, "2006-01-30")
    panel.universe.loc["2006-01-30", 1] = False
    actual = target_weights(candidate, panel, "2006-01-30")
    assert 1 not in actual
    assert actual.sum() < baseline.sum()
    changed = deepcopy(panel)
    changed.universe.loc[pd.Timestamp("2021-01-04")] = True
    with pytest.raises(HoldoutError):
        target_weights(candidate, changed, "2006-01-30")


def proposal(packet, side="long_only"):
    from capstone.factors.research_compiler import ADAPTATIONS, PACKET_METHODS

    method = PACKET_METHODS[packet["id"]]
    return {
        "source_url": packet["source_url"],
        "method_location": packet["method_location"],
        "source_quote": " ".join(packet["methods"].split()[:8]),
        "rationale": "source-faithful fixture only",
        "required_inputs": packet["required_inputs"],
        "adaptations": sorted(ADAPTATIONS[method]),
        "program": RULES[method] | COMMON | {"profile": side},
    }


def test_source_and_dependency_checks_precede_verified_handoff(tmp_path):
    from test_research_controller import controller as make_controller
    from test_research_controller import packet as catalogue_packet

    from capstone.factors.research_compiler import compile_proposal
    from capstone.factors.research_contracts import SourceClaim

    row = catalogue_packet(name="turn_month", method="calendar")
    row["constructor"] = "research"
    row["claim"]["claim"] = (
        "Synthetic calendar rules use completed-session inputs and next-close execution."
    )
    packet = {"id": "turn_month", "methods": row["claim"]["claim"], **row["claim"]}
    packet["required_inputs"] = list(packet["required_inputs"])
    controller = make_controller(tmp_path / "controller.db", [row], packet["required_inputs"])
    try:
        controller.reserve(packet["id"])
        controller.ground(
            packet["id"],
            SourceClaim(
                packet["source_url"],
                packet["method_location"],
                packet["methods"],
                "fixture fails",
                tuple(packet["required_inputs"]),
                True,
            ),
        )
        good = proposal(packet)
        wrong = deepcopy(good)
        wrong["required_inputs"].append("observed_fund_flows")
        with pytest.raises(ValueError, match="unsupported data"):
            compile_proposal(
                controller, packet, wrong, "synthetic-reference-v2", "2000-12-29", "long_only"
            )
        assert controller.state(packet["id"]) == "specification_ready"
        candidate, _ = compile_proposal(
            controller, packet, good, "synthetic-reference-v2", "2000-12-29", "long_only"
        )
        assert controller.state(packet["id"]) == "construction_verified"
        resumed, _ = compile_proposal(
            controller, packet, good, "synthetic-reference-v2", "2000-12-29", "long_only"
        )
        assert resumed.id == candidate.id
        wrong = deepcopy(good)
        wrong["adaptations"].append("unreviewed_rule_change")
        with pytest.raises(ValueError, match="resumed program differs"):
            compile_proposal(
                controller, packet, wrong, "synthetic-reference-v2", "2000-12-29", "long_only"
            )
    finally:
        controller.close()
