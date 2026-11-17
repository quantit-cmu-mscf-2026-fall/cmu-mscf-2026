"""The fault hook is inert unless explicitly enabled, and counts hits per point."""

import pytest

from capstone import faults
from capstone.faults import InjectedFault, maybe_fault


@pytest.fixture(autouse=True)
def fresh_plan(monkeypatch):
    for name in ("CAPSTONE_FAULT_INJECTION", "CAPSTONE_FAULT_AT"):
        monkeypatch.delenv(name, raising=False)
    faults._fault_plan.cache_clear()
    faults._fault_hits.clear()
    yield
    faults._fault_plan.cache_clear()
    faults._fault_hits.clear()


def enable(monkeypatch, plan):
    monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
    monkeypatch.setenv("CAPSTONE_FAULT_AT", plan)
    faults._fault_plan.cache_clear()


def test_plan_without_the_gate_is_ignored(monkeypatch):
    monkeypatch.setenv("CAPSTONE_FAULT_AT", "stage:step")
    faults._fault_plan.cache_clear()
    maybe_fault("stage:step")


def test_gate_without_a_plan_is_inert(monkeypatch):
    monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
    faults._fault_plan.cache_clear()
    maybe_fault("stage:step")


def test_first_hit_is_the_default_and_other_points_are_untouched(monkeypatch):
    enable(monkeypatch, "stage:step")
    maybe_fault("stage:other")
    with pytest.raises(InjectedFault, match="stage:step"):
        maybe_fault("stage:step")
    maybe_fault("stage:step")  # only the planned hit raises


def test_nth_hit_and_several_points(monkeypatch):
    enable(monkeypatch, "a:b:2, c:d:1,c:d:3")
    maybe_fault("a:b")
    with pytest.raises(InjectedFault, match=r"hit 2"):
        maybe_fault("a:b")
    maybe_fault("a:b")
    with pytest.raises(InjectedFault):
        maybe_fault("c:d")
    maybe_fault("c:d")
    with pytest.raises(InjectedFault, match=r"hit 3"):
        maybe_fault("c:d")


def test_injected_fault_is_a_runtime_error():
    assert issubclass(InjectedFault, RuntimeError)
