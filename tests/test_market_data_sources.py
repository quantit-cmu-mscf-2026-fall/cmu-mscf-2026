"""Decision bars from a frozen snapshot, and what a bar-only source may be asked for.

Synthetic CRSP-shaped rows only; nothing is fetched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone.execution.data import (
    BAR_FIELDS,
    FrozenSnapshotSource,
    UnsupportedFieldError,
    check_servable,
)
from capstone.factors.evaluation import Panel, build_panel


@pytest.fixture(scope="module")
def panel() -> Panel:
    return build_panel(fake_rows(n_stocks=6, n_days=40, seed=2), "synthetic-v1")


def test_a_snapshot_serves_every_row_through_as_of_and_none_after(panel):
    dates = panel.universe.index
    as_of = dates[25]
    served = FrozenSnapshotSource(panel).panel(("close", "volume"), as_of)
    assert served.universe.index[-1] == as_of and len(served.universe) == 26
    for name, frame in served.fields.items():
        assert frame.index.max() <= as_of, name
        assert frame.equals(panel.fields[name].loc[:as_of]), name
    pd.testing.assert_frame_equal(served.half_spread, panel.half_spread.loc[:as_of])
    assert served.sic is not None and served.sic.index.max() == as_of
    assert served.data_version == "synthetic-v1"
    assert len(panel.universe) == 40  # the snapshot itself is not cut


def test_the_market_return_series_is_cut_with_the_frames(panel):
    as_of = panel.universe.index[10]
    served = FrozenSnapshotSource(panel).panel(("close", "mkt_return"), as_of)
    assert served.fields["mkt_return"].index.max() == as_of


def test_rows_out_of_date_order_are_cut_by_date_not_by_position(panel):
    shuffled = Panel(
        {n: f.iloc[::-1] for n, f in panel.fields.items()},
        panel.universe.iloc[::-1],
        panel.half_spread.iloc[::-1],
        None if panel.sic is None else panel.sic.iloc[::-1],
        panel.data_version,
    )
    as_of = panel.universe.index[20]
    served = FrozenSnapshotSource(shuffled).panel(("close",), as_of)
    assert served.universe.index.is_monotonic_increasing and served.universe.index[-1] == as_of
    pd.testing.assert_frame_equal(served.fields["close"], panel.fields["close"].loc[:as_of])


def test_a_date_that_is_not_a_trading_date_of_the_snapshot_is_refused(panel):
    source = FrozenSnapshotSource(panel)
    with pytest.raises(ValueError, match="not a trading date"):
        source.panel(("close",), panel.universe.index[0] - pd.Timedelta(days=30))
    with pytest.raises(ValueError, match="not a trading date"):
        source.panel(("close",), panel.universe.index[-1] + pd.Timedelta(days=1))


def test_a_field_the_snapshot_lacks_is_refused_by_name(panel):
    with pytest.raises(ValueError, match=r"no field\(s\) \['book'\]"):
        FrozenSnapshotSource(panel).panel(("close", "book"), panel.universe.index[10])


def test_serving_does_not_change_what_was_frozen(panel):
    before = panel.fields["close"].copy()
    served = FrozenSnapshotSource(panel).panel(("close",), panel.universe.index[15])
    served.fields["close"].iloc[:] = np.nan
    pd.testing.assert_frame_equal(panel.fields["close"], before)


@pytest.mark.parametrize("field", ["shares", "cap", "mkt_return", "book", "earnings", "assets"])
def test_a_field_bars_cannot_serve_is_refused_with_the_field_named(field):
    with pytest.raises(UnsupportedFieldError, match=rf"'{field}'"):
        check_servable(("close", field))


def test_the_reason_a_field_is_missing_is_given():
    with pytest.raises(UnsupportedFieldError, match="shares outstanding"):
        check_servable(("cap",))


def test_a_sector_neutral_strategy_is_refused_bars_carry_no_sic_codes():
    with pytest.raises(UnsupportedFieldError, match="sector"):
        check_servable(("close",), "sector")
    check_servable(BAR_FIELDS, "none")
    check_servable((), "none")
