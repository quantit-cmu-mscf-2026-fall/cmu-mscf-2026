from types import SimpleNamespace

import pandas as pd
import pytest

from capstone.strategy import input_memo


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(input_memo, "ENABLED", True)


class Frame:
    """A weak-referenceable stand-in for a panel input."""


def counted():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    return calls, compute


def test_nothing_is_cached_outside_a_scope():
    calls, compute = counted()
    frame = Frame()
    assert input_memo.cached("k", (frame,), ("v",), compute) == 1
    assert input_memo.cached("k", (frame,), ("v",), compute) == 2
    assert not input_memo.active()


def test_same_objects_and_values_hit_inside_a_scope_and_the_scope_ends_cleanly():
    calls, compute = counted()
    frame = Frame()
    with input_memo.scope():
        assert input_memo.active()
        assert input_memo.cached("k", (frame,), ("v",), compute) == 1
        assert input_memo.cached("k", (frame,), ("v",), compute) == 1
        with input_memo.scope():  # nested scopes share one memo
            assert input_memo.cached("k", (frame,), ("v",), compute) == 1
        assert input_memo.active()
    assert not input_memo.active()
    assert len(calls) == 1


def test_a_different_object_or_plain_value_never_hits():
    calls, compute = counted()
    a, b = Frame(), Frame()
    with input_memo.scope():
        input_memo.cached("k", (a,), ("v",), compute)
        input_memo.cached("k", (b,), ("v",), compute)
        input_memo.cached("k", (a,), ("w",), compute)
        input_memo.cached("other", (a,), ("v",), compute)
    assert len(calls) == 4


def test_a_reused_id_after_the_owner_dies_does_not_hit():
    calls, compute = counted()
    with input_memo.scope():
        first = Frame()
        input_memo.cached("k", (first,), (), compute)
        entries = input_memo._state["entries"]
        (key,) = entries
        del first
        # Simulate id reuse: a new object under the same key must still miss.
        replacement = Frame()
        entries[("k", (), (id(replacement),))] = entries.pop(key)
        input_memo.cached("k", (replacement,), (), compute)
    assert len(calls) == 2


def test_unhashable_values_fall_through_to_the_computation():
    calls, compute = counted()
    with input_memo.scope():
        input_memo.cached("k", (Frame(),), ([1],), compute)
        input_memo.cached("k", (Frame(),), ([1],), compute)
    assert len(calls) == 2


def test_bypass_and_disabled_compute_without_the_memo(monkeypatch):
    calls, compute = counted()
    frame = Frame()
    with input_memo.scope():
        input_memo.cached("k", (frame,), (), compute)
        with input_memo.bypass():
            assert not input_memo.active()
            input_memo.cached("k", (frame,), (), compute)
        assert input_memo.active()
        input_memo.cached("k", (frame,), (), compute)
    assert len(calls) == 2
    monkeypatch.setattr(input_memo, "ENABLED", False)
    with input_memo.scope():
        assert not input_memo.active()


def test_scoped_decorator_opens_a_scope_for_the_call():
    seen = []

    @input_memo.scoped
    def run():
        seen.append(input_memo.active())

    run()
    assert seen == [True] and not input_memo.active()


def test_retain_keeps_one_owner_per_kind():
    calls, compute = counted()
    one, two, frame = Frame(), Frame(), Frame()
    with input_memo.scope():
        input_memo.retain("formation", one)
        input_memo.cached("formation", (frame,), (), compute)
        input_memo.retain("formation", one)
        input_memo.cached("formation", (frame,), (), compute)
        assert len(calls) == 1
        input_memo.retain("formation", two)
        input_memo.cached("formation", (frame,), (), compute)
    assert len(calls) == 2


def panel(**override):
    frames = {n: pd.DataFrame({"a": [1.0]}) for n in ("close", "returns")}
    base = dict(
        universe=pd.DataFrame(), half_spread=pd.DataFrame(), sic=None, calendar=None,
        company=None, fundamentals=None, sizing_prices=None, provenance=None,
        fields=frames, availability=None, data_version="v1",
    )  # fmt: skip
    return SimpleNamespace(**(base | override))


def test_panel_cache_is_keyed_on_every_panel_input():
    calls, compute = counted()
    spec = Frame()
    one = panel()
    with input_memo.scope():
        input_memo.panel_cached("form", spec, one, ("2020-01-02",), compute)
        input_memo.panel_cached("form", spec, one, ("2020-01-02",), compute)
        assert len(calls) == 1
        rebuilt = panel(fields={**one.fields, "close": one.fields["close"].copy()})
        input_memo.panel_cached("form", spec, rebuilt, ("2020-01-02",), compute)
        assert len(calls) == 2
        renamed = panel(data_version="v2")
        input_memo.panel_cached("form", spec, renamed, ("2020-01-02",), compute)
        assert len(calls) == 3


def test_a_non_panel_is_computed_so_the_computation_can_refuse_it():
    calls, compute = counted()
    with input_memo.scope():
        input_memo.panel_cached("form", Frame(), object(), (), compute)
        input_memo.panel_cached("form", Frame(), object(), (), compute)
    assert len(calls) == 2
