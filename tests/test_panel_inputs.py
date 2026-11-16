"""The panel's optional inputs: absent by default, and part of what a memo keys on."""

import pandas as pd

from capstone.factors.evaluation import Panel
from capstone.strategy import input_memo


def small_panel(**extra):
    dates = pd.bdate_range("2020-01-02", periods=3)
    frame = pd.DataFrame(1.0, index=dates, columns=[1, 2])
    return Panel({"close": frame}, frame.astype(bool), frame * 0, data_version="v1", **extra)


def test_the_optional_inputs_default_to_absent():
    panel = small_panel()
    assert panel.company is None and panel.fundamentals is None
    assert panel.sizing_prices is None and panel.provenance is None


def test_a_real_panel_is_memoized_and_each_optional_input_changes_the_key():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    panel = small_panel()
    with input_memo.scope():
        first = input_memo.panel_cached("form", "spec", panel, ("2020-01-02",), compute)
        again = input_memo.panel_cached("form", "spec", panel, ("2020-01-02",), compute)
        assert first == again and len(calls) == 1
        panel.company = pd.DataFrame(1, index=panel.universe.index, columns=[1, 2])
        input_memo.panel_cached("form", "spec", panel, ("2020-01-02",), compute)
        assert len(calls) == 2
        panel.sizing_prices = panel.fields["close"].copy()
        input_memo.panel_cached("form", "spec", panel, ("2020-01-02",), compute)
        assert len(calls) == 3
