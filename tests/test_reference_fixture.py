import numpy as np
import pandas as pd
from reference_fakes import reference_panel

from capstone.factors.reference import field, month_ends, sessions


def test_fixture_is_deterministic_and_has_the_reference_fields():
    a, b = reference_panel(), reference_panel()
    assert set(a.fields) >= {"returns", "close", "cap", "volume", "mkt_excess", "smb", "hml", "rf"}
    for name in a.fields:
        pd.testing.assert_frame_equal(a.fields[name], b.fields[name])
    assert a.data_version == "synthetic-reference-v2"
    assert a.universe.shape == a.fields["returns"].shape == (len(a.universe), 120)


def test_fixture_calendar_extends_past_loaded_rows_and_feeds_reference_helpers():
    panel = reference_panel()
    assert panel.calendar.max() > panel.universe.index.max()
    assert set(panel.universe.index) <= set(panel.calendar)
    assert set(panel.availability) == set(panel.fields)
    assert len(sessions(panel.calendar, "2000-12-29")) > 0
    ends = month_ends(panel.calendar, "2001-01-31")
    assert ends[-1] == pd.Timestamp("2001-01-31")
    window = field(panel, "returns", ends[-1])
    assert window.index.max() <= ends[-1] and np.isfinite(window.iloc[-1]).all()
