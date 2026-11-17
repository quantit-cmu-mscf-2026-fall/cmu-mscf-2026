"""Observed-prefix route identity: version names and the observed-event closure."""

from types import SimpleNamespace

import pandas as pd
import pytest

from capstone.execution import prefix_protocol as protocol


def test_prefix_versions_are_distinct_and_program_and_benchmark_do_not_overlap():
    versions = (*protocol.PROGRAM_VERSIONS, *protocol.BENCHMARK_PREFIX_VERSIONS)
    assert len(set(versions)) == len(versions) == 6
    assert protocol.PROGRAM_VERSIONS[0] == protocol.EXECUTION_VERSION
    assert protocol.BENCHMARK_PREFIX_VERSIONS[0] == protocol.BENCHMARK_EXECUTION_VERSION
    assert all(v.startswith("program-observed-prefix-") for v in protocol.PROGRAM_VERSIONS)
    assert all(
        v.startswith("benchmark-observed-prefix-") for v in protocol.BENCHMARK_PREFIX_VERSIONS
    )


def test_observed_events_drop_all_null_future_columns_and_fix_dtypes():
    events = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "event_date": ["2000-01-03", "2000-01-04"],
            "permno": [101, 102],
            "kind": ["split", "split"],
            "factor": [2, 3],
            "later_schema": [None, None],
            "pay_date": ["2000-01-10", None],
        }
    )
    observed = protocol.observed_events(events)
    assert list(observed) == ["event_id", "event_date", "permno", "kind", "factor", "pay_date"]
    assert str(observed["event_date"].dtype) == "datetime64[ns]"
    assert observed["permno"].dtype == "int64"
    assert observed["factor"].dtype == "float64"
    assert observed["pay_date"].isna().tolist() == [False, True]
    assert "later_schema" in events  # the caller's frame is untouched


def test_required_columns_survive_even_when_empty():
    empty = pd.DataFrame(columns=["event_id", "event_date", "permno", "kind", "future"])
    assert list(protocol.observed_events(empty)) == ["event_id", "event_date", "permno", "kind"]


def test_the_prefix_input_closure_is_in_this_build_and_still_needs_events():
    # capstone.pipeline.prefix_inputs ships with the historical inputs, so the closure's
    # imports resolve and the first refusal is the protocol's own, not an import error.
    panel = SimpleNamespace(fields={}, universe=pd.DataFrame({101: [True]}))
    with pytest.raises(ValueError, match="requires supplied events"):
        protocol.canonical_inputs(panel, None, None, None, None, "2000-01-31")
