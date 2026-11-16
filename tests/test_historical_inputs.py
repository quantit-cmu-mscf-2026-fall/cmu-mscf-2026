import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone.pipeline import historical as h


def test_real_input_units_and_market_frames_are_explicit():
    daily = fake_rows(n_stocks=4, n_days=8, start="2000-01-03")
    daily["permco"] = daily.permno
    dates = pd.DatetimeIndex(sorted(daily.date.unique()))
    qualification = {"global_blockers": [], "method_blockers": {}}
    inputs = h.build_inputs(
        daily,
        dates,
        pd.DataFrame(columns=h.EVENT_COLUMNS),
        data_version="test",
        qualification=qualification,
    )
    first = daily.iloc[0]
    assert inputs.panel.fields["cap"].loc[first.date, first.permno] == first.dlycap * 1000
    assert isinstance(inputs.panel.fields["mkt_return"], pd.DataFrame)
    assert inputs.opens.loc[first.date, first.permno] == first.dlyopen
    # The qualified path cannot inherit the legacy cross-stock spread fallback or loss proxy.
    missing = daily.copy()
    missing.loc[0, ["dlybid", "dlyask", "dlyret"]] = float("nan")
    strict = h.build_inputs(
        missing,
        dates,
        pd.DataFrame(columns=h.EVENT_COLUMNS),
        data_version="test",
        qualification=qualification,
    )
    assert pd.isna(strict.half_spread_bps.loc[first.date, first.permno])
    assert pd.isna(strict.total_returns.loc[first.date, first.permno])
    broken = daily.copy()
    broken.loc[broken.index[0], "date"] = pd.Timestamp("2021-01-01")
    with pytest.raises(ValueError, match="boundary"):
        h.build_inputs(
            broken,
            dates,
            pd.DataFrame(columns=h.EVENT_COLUMNS),
            data_version="test",
            qualification=qualification,
        )


@pytest.mark.parametrize(
    "event",
    [
        {"kind": "dividend", "amount": 1, "pay_date": "1999-12-31"},
        {"kind": "split", "ratio": 0},
        {"kind": "conversion", "ratio": 1, "amount": 0},
    ],
)
def test_incomplete_event_terms_are_rejected(event):
    row = {"event_id": "event-1", "event_date": "2000-01-03", "permno": 1, **event}
    with pytest.raises(ValueError):
        h.validate_events(pd.DataFrame([row]))


def test_ex_date_and_payment_date_are_distinct_and_ordered():
    event = {
        "event_id": "d1",
        "event_date": "2000-01-03",
        "permno": 1,
        "kind": "dividend",
        "amount": 2.0,
        "pay_date": "2000-01-10",
    }
    assert len(h.validate_events(pd.DataFrame([event]))) == 1
    with pytest.raises(ValueError, match="unique"):
        h.validate_events(pd.DataFrame([event, event]))


def test_the_evaluation_memo_keys_on_every_input_of_a_real_panel():
    from capstone.factors.evaluation import Panel
    from capstone.strategy import input_memo

    idx = pd.date_range("2000-01-03", periods=2)
    frame = pd.DataFrame({1: [1.0, 2.0]}, index=idx)
    panel = Panel({"close": frame}, frame.gt(0), frame * 0, data_version="v")
    rebuilt = Panel({"close": frame.copy()}, panel.universe, panel.half_spread, data_version="v")
    calls = []

    def run(p):
        return input_memo.panel_cached("t", "spec", p, (), lambda: calls.append(1) or len(calls))

    with input_memo.scope():
        assert run(panel) == run(panel) == 1
        assert run(rebuilt) == 2  # a rebuilt field frame never hits
    assert run(panel) == 3  # outside a scope nothing is cached
