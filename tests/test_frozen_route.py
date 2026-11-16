"""The incubation route and bundle thresholds of a frozen plan fail closed when their
unpublished collaborators (the runner's limits, the registered thresholds) are absent."""

import sys
import types

import pytest

from capstone.execution import frozen
from capstone.execution.evidence import INCUBATION_ROUTE, INCUBATION_UNQUALIFIED

ROUTE = {"route": INCUBATION_ROUTE, "incubation_cap": 0.05}


@pytest.fixture
def registered_cap(monkeypatch):
    module = types.ModuleType("capstone.execution.runner")
    module.RiskLimits = types.SimpleNamespace(incubation_cap=0.1)
    monkeypatch.setitem(sys.modules, "capstone.execution.runner", module)


def test_without_the_runner_limits_no_incubation_route_is_recorded(monkeypatch):
    monkeypatch.setitem(sys.modules, "capstone.execution.runner", None)
    with pytest.raises(frozen.IncubationRouteRefused) as error:
        frozen._route_record(ROUTE)
    assert str(error.value).startswith(INCUBATION_UNQUALIFIED) and "unavailable" in str(error.value)


def test_a_registered_route_is_paper_only(registered_cap):
    assert frozen._route_record(ROUTE) == {
        "route": INCUBATION_ROUTE,
        "incubation_cap": 0.05,
        "account": "paper",
    }


@pytest.mark.parametrize(
    "route",
    [
        None,
        {"route": INCUBATION_ROUTE},
        ROUTE | {"route": "live"},
        ROUTE | {"incubation_cap": 0.11},  # above the registered cap
        ROUTE | {"incubation_cap": 0},
        ROUTE | {"incubation_cap": True},
        ROUTE | {"incubation_cap": float("nan")},
        ROUTE | {"account": "other"},
    ],
)
def test_other_routes_and_caps_are_refused(registered_cap, route):
    with pytest.raises(frozen.IncubationRouteRefused):
        frozen._route_record(route)


def test_the_cap_bounds_capital_and_the_projected_book():
    content = {
        "reference_prices": {"1": 10.0},
        "account_positions": {"1": 100},
        "projected_shares": {"1": 50},
        "account_cash": 1000.0,
        "capital": 100.0,
        frozen.ROUTE_KEY: {"incubation_cap": 0.1},
    }
    assert frozen.incubation_exposure(content) == (2000.0, 100.0, 500.0)
    assert frozen.incubation_cap_exceeded(content) is True  # 500 > 0.1 x 2000
    assert frozen.incubation_cap_exceeded(content | {"projected_shares": {"1": 20}}) is False
    assert frozen.incubation_cap_exceeded(content | {"capital": 300.0}) is True


def test_a_bundle_without_threshold_records_needs_no_threshold_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "capstone.pipeline.registered_thresholds", None)
    assert frozen._bundle_thresholds(tmp_path) == {}


def test_a_recorded_threshold_file_cannot_be_verified_without_its_module(tmp_path, monkeypatch):
    (tmp_path / frozen.THRESHOLD_FILE).write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "capstone.pipeline.registered_thresholds", None)
    with pytest.raises(ImportError):
        frozen._bundle_thresholds(tmp_path)
