import pandas as pd

from capstone.strategy import input_memo


def test_nothing_is_cached_outside_a_scope():
    calls = []
    frame = pd.DataFrame({"a": [1.0]})
    for _ in range(2):
        input_memo.cached("k", (frame,), (1,), lambda: calls.append(1) or len(calls))
    assert len(calls) == 2


def test_a_scope_reuses_a_result_only_for_the_same_object_and_values():
    calls = []
    frame = pd.DataFrame({"a": [1.0]})
    copy = frame.copy()

    def run(obj, value):
        return input_memo.cached("k", (obj,), (value,), lambda: calls.append(1) or len(calls))

    with input_memo.scope():
        assert run(frame, 1) == run(frame, 1) == 1
        assert run(frame, 2) == 2  # another plain value
        assert run(copy, 1) == 3  # an equal but different object never hits
    assert run(frame, 1) == 4  # the scope's entries are gone


def test_bypass_computes_honestly_inside_a_scope():
    calls = []
    frame = pd.DataFrame({"a": [1.0]})
    with input_memo.scope():
        input_memo.cached("k", (frame,), (), lambda: calls.append(1))
        with input_memo.bypass():
            input_memo.cached("k", (frame,), (), lambda: calls.append(1))
        assert input_memo.active()
    assert len(calls) == 2


def test_a_scoped_function_opens_its_own_scope_once():
    seen = []

    @input_memo.scoped
    def run():
        seen.append(input_memo.active())
        with input_memo.scope():  # nested scopes share the outer one
            seen.append(input_memo.active())

    run()
    assert seen == [True, True]
    assert not input_memo.active()


def test_retain_frees_a_kind_when_its_owner_changes():
    first, second = object(), object()
    with input_memo.scope():
        input_memo.retain("formation", first)
        input_memo.cached("formation", (), (1,), lambda: "x")
        input_memo.retain("formation", first)  # same owner: kept
        assert input_memo.cached("formation", (), (1,), lambda: "y") == "x"
        input_memo.retain("formation", second)  # new owner: freed
        assert input_memo.cached("formation", (), (1,), lambda: "y") == "y"


def test_panel_cached_keys_on_every_panel_input():
    from capstone.factors.evaluation import Panel

    idx = pd.date_range("2000-01-03", periods=2)
    frame = pd.DataFrame({1: [1.0, 2.0]}, index=idx)
    panel = Panel({"close": frame}, frame.gt(0), frame * 0, data_version="v")
    other = Panel({"close": frame.copy()}, panel.universe, panel.half_spread, data_version="v")
    calls = []

    def run(p):
        return input_memo.panel_cached("t", "spec", p, (), lambda: calls.append(1) or len(calls))

    with input_memo.scope():
        assert run(panel) == run(panel) == 1
        assert run(other) == 2  # a rebuilt field frame never hits
    assert run(object()) == 3  # not a panel: computed, never cached
