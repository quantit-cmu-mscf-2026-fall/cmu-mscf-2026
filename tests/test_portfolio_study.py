from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.strategy.portfolio_study import (
    COMPARISON_SET,
    LEGACY_COMPARISON_SET,
    MODES,
    allocation_summary,
    comparisons,
    default_policy,
    saved_portfolio,
)
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.robust_allocation import AllocationPolicyV2

CLOCK = ProgramSchedule(kind="month_end", interval=1)
CALENDAR = pd.bdate_range("1995-06-01", "1996-12-31").drop(pd.Timestamp("1996-01-01"))


def spec(name, side, family):
    return SimpleNamespace(
        id=name, side=side, mechanism_id=family, anchor="1995-11-30", schedule=CLOCK
    )


def make_source(*, start="1996-01-02", end="1996-06-28", benchmarks=()):
    dates = CALENDAR[(CALENDAR >= start) & (CALENDAR <= end)]
    specs = [spec("a", "long_only", "fam1"), spec("b", "long_only", "fam2")]
    specs.append(spec("c", "long_short", "fam1"))
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(rng.normal(0, 0.01, (len(dates), 3)), index=dates, columns=list("abc"))
    return {
        "specs": specs,
        "inclusions": [
            {"family": "fam1", "profile": "long_only"},
            {"family": "fam2", "profile": "long_only"},
            {"family": "fam1", "profile": "long_short"},
        ],
        "inputs": SimpleNamespace(panel=SimpleNamespace(calendar=CALENDAR, data_version="v1")),
        "matrix": matrix,
        "report": {"benchmarks": list(benchmarks)},
        "manifest": {"costs": {"multiplier": 1}},
    }


def fixture_policy():
    return AllocationPolicyV2(lookback=30, minimum=10, window=5, step=5, annual_risk_limit=1)


def test_every_mode_family_omission_and_control_is_prespecified():
    entries = comparisons(make_source())
    assert {row["mode"] for row in entries} == set(MODES)
    # long-only: full, zero, 2 leave-one-out = 4; long-short: full, zero, 1 omit = 3;
    # times three modes, plus an exposure-matched control per long-only full pool x 2 proxies.
    assert len(entries) == 7 * 3 + 3 * 2
    assert len({row["id"] for row in entries}) == len(entries)
    zero = [row for row in entries if row["kind"] == "zero_cash"]
    assert zero and all(row["pool_ids"] == [] for row in zero)
    assert comparisons(make_source()) == entries


def test_legacy_set_has_no_controls_and_unknown_sets_are_refused():
    legacy = comparisons(make_source(), LEGACY_COMPARISON_SET)
    assert len(legacy) == 7 * 3
    assert comparisons(make_source(), COMPARISON_SET)[: len(legacy)] == legacy
    with pytest.raises(ValueError, match="unknown portfolio comparison set"):
        comparisons(make_source(), "pooled-portfolio-comparisons-v9")


def test_controls_link_registered_proxies_or_record_a_blocker():
    entries = comparisons(
        make_source(
            benchmarks=[
                {"kind": "cap_weight_universe_proxy", "profile": "long_only", "id": "cap-id"}
            ]
        )
    )
    controls = {
        row["control_kind"]: row
        for row in entries
        if row["kind"] == "exposure_matched_control" and row["mode"] == "equal_weight"
    }
    assert controls["cap_weight_universe_proxy"]["benchmark_control_id"] == "cap-id"
    assert controls["cap_weight_universe_proxy"]["construction_blockers"] == []
    assert controls["equal_weight_universe_proxy"]["construction_blockers"] == [
        "benchmark_control_missing"
    ]


def test_decision_sessions_are_month_ends_after_the_prior_session():
    entry = next(row for row in comparisons(make_source()) if row["kind"] == "full_pool")
    assert entry["construction_blockers"] == []
    assert entry["decision_sessions"][0] == "1995-11-30"
    assert entry["decision_sessions"][-1] == "1996-06-28"


def test_an_origin_before_the_calendar_blocks_construction():
    source = make_source()
    source["matrix"].index = source["matrix"].index - pd.Timedelta(days=2000)
    entry = next(row for row in comparisons(source) if row["kind"] == "full_pool")
    assert entry["construction_blockers"] == ["prior_completed_session_missing"]
    with pytest.raises(ValueError, match="prior_completed_session_missing"):
        saved_portfolio(entry, source, fixture_policy())


def test_mixed_schedules_are_not_supported():
    source = make_source()
    source["specs"][0].schedule = ProgramSchedule(kind="week_end", interval=1)
    entry = next(
        row
        for row in comparisons(source)
        if row["kind"] == "full_pool" and row["profile"] == "long_only"
    )
    assert entry["construction_blockers"] == ["only_synchronized_monthly_programs_are_supported"]


def test_default_policy_is_the_registered_v2_policy():
    assert default_policy() == AllocationPolicyV2()


def test_allocation_summary_counts_cash_decisions():
    saved = SimpleNamespace(
        id="def",
        policy=fixture_policy(),
        schedule=[
            {"weights": {}, "status": "insufficient_history_cash", "decision_session": "d1"},
            {"weights": {"a": 0.0}, "status": "cash", "decision_session": "d2"},
            {"weights": {"a": 0.5}, "status": "ok", "decision_session": "d3"},
        ],
    )
    summary = allocation_summary(saved)
    assert summary["decision_count"] == 3
    assert summary["first_positive_allocation"] == "d3"
    assert summary["insufficient_history_decisions"] == 1
    assert summary["all_cash_decisions"] == 2


def test_zero_cash_pool_saves_all_cash_decisions():
    source = make_source()
    source["specs"] = []
    entry = next(
        row
        for row in comparisons(make_source())
        if row["kind"] == "zero_cash" and row["profile"] == "long_only"
    )
    source["report"]["candidates"] = []
    source["manifest"]["costs"] = {}
    source["inputs"].panel.data_version = "v1"
    saved = saved_portfolio(entry, source, fixture_policy())
    assert len(saved.schedule) == len(entry["decision_sessions"])
    assert all(not any(value > 0 for value in row["weights"].values()) for row in saved.schedule)
    assert allocation_summary(saved)["all_cash_decisions"] == len(saved.schedule)
