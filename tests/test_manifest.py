"""The digest must be blind to layout and sensitive to numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone import manifest


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((8, 4)),
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
        columns=["b", "d", "a", "c"],
    )


def _panels(seed: int = 0) -> dict[str, pd.DataFrame]:
    return {"close": _frame(seed), "returns": _frame(seed + 1)}


@pytest.fixture
def crsp_like() -> pd.DataFrame:
    """A tidy CRSP-shaped frame, including the negative-price convention."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    rows = []
    for permno in (10001, 10002):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "ticker": f"T{permno}",
                    "openprc": 10.0 + i,
                    "askhi": 11.0 + i,
                    "bidlo": 9.0 + i,
                    # A no-trade day is stored as a NEGATIVE midpoint by CRSP.
                    "prc": -(10.5 + i) if (permno == 10002 and i == 1) else 10.5 + i,
                    "vol": 1000.0 * (i + 1),
                    "ret": 0.01 * (i + 1),
                }
            )
    return pd.DataFrame(rows)


# --- normalisation: things that must NOT change the digest -------------------


def test_digest_ignores_row_and_column_order():
    """Two pulls differing only in ordering describe the same information set."""
    frame = _frame()
    shuffled = frame.iloc[::-1].reindex(["d", "c", "b", "a"], axis=1)
    assert manifest.frame_digest(frame) == manifest.frame_digest(shuffled)


def test_digest_ignores_nullable_extension_dtype():
    """Float64 arrives from parquet on sparse columns; it is not a data change."""
    frame = _frame()
    assert manifest.frame_digest(frame) == manifest.frame_digest(frame.astype("Float64"))


def test_digest_is_stable_across_repeated_calls():
    frame = _frame()
    assert manifest.frame_digest(frame) == manifest.frame_digest(frame)


# --- sensitivity: things that MUST change the digest -------------------------


def test_digest_detects_a_single_changed_value():
    frame = _frame()
    tweaked = frame.copy()
    tweaked.iloc[3, 2] += 1e-9
    assert manifest.frame_digest(frame) != manifest.frame_digest(tweaked)


def test_digest_detects_a_renamed_security():
    frame = _frame()
    assert manifest.frame_digest(frame) != manifest.frame_digest(frame.rename(columns={"a": "z"}))


def test_digest_detects_an_added_date():
    frame = _frame()
    extra = frame.iloc[[-1]].copy()
    extra.index = pd.DatetimeIndex([pd.Timestamp("2024-01-09")])
    assert manifest.frame_digest(frame) != manifest.frame_digest(pd.concat([frame, extra]))


def test_digest_separates_missing_from_zero():
    """NaN and 0.0 are different claims about what was observed."""
    with_nan, with_zero = _frame().copy(), _frame().copy()
    with_nan.iloc[0, 0] = np.nan
    with_zero.iloc[0, 0] = 0.0
    assert manifest.frame_digest(with_nan) != manifest.frame_digest(with_zero)


# --- manifest lifecycle ------------------------------------------------------


def test_manifest_round_trip_and_self_verification(tmp_path):
    panels = _panels()
    path = tmp_path / "data_manifest.json"
    written = manifest.write_manifest(panels, path, source="test")

    assert manifest.read_manifest(path) == written
    matches, mismatched = manifest.verify(panels, written)
    assert matches and mismatched == []


def test_coverage_is_read_off_the_data_not_declared():
    built = manifest.build_manifest(_panels(), source="test")
    assert built["coverage"]["n_dates"] == 8
    assert built["coverage"]["n_securities"] == 4
    assert built["coverage"]["start"].startswith("2024-01-01")
    assert built["coverage"]["panels"] == ["close", "returns"]


def test_empty_panel_set_is_rejected():
    with pytest.raises(ValueError, match="zero panels"):
        manifest.build_manifest({}, source="test")


# --- comparison --------------------------------------------------------------


def test_verify_names_the_panel_that_diverged():
    """A mismatch must say which panel, or it cannot be acted on."""
    panels = _panels()
    expected = manifest.build_manifest(panels, source="test")

    divergent = dict(panels)
    bumped = divergent["returns"].copy()
    bumped.iloc[0, 0] += 0.5
    divergent["returns"] = bumped

    matches, mismatched = manifest.verify(divergent, expected)
    assert not matches
    assert mismatched == ["returns"]


def test_a_dropped_panel_counts_as_a_mismatch():
    panels = _panels()
    expected = manifest.build_manifest(panels, source="test")
    matches, mismatched = manifest.verify({"close": panels["close"]}, expected)
    assert not matches
    assert mismatched == ["returns"]


def test_incomparable_digest_revisions_raise_rather_than_report_mismatch():
    """A code-version difference must not masquerade as a data difference."""
    expected = manifest.build_manifest(_panels(), source="test")
    stale = {**expected, "digest_revision": expected["digest_revision"] + 1}
    with pytest.raises(ValueError, match="digest_revision"):
        manifest.compare(stale, expected)


# --- CRSP adapter ------------------------------------------------------------


def test_crsp_panels_are_keyed_on_permno_and_cover_the_expected_fields(crsp_like):
    panels = manifest.panels_from_crsp_daily(crsp_like)
    assert set(panels) == set(manifest.CRSP_PANELS)
    assert list(panels["close"].columns) == [10001, 10002]
    assert len(panels["close"]) == 3


def test_crsp_panels_make_no_trade_prices_positive(crsp_like):
    """A negative CRSP prc is a stored bid/ask midpoint, not a negative price."""
    panels = manifest.panels_from_crsp_daily(crsp_like)
    assert (panels["close"].stack() > 0).all()
    assert panels["close"].loc[pd.Timestamp("2024-01-02"), 10002] == pytest.approx(11.5)


def test_crsp_panels_keyed_on_ticker_differ_from_permno(crsp_like):
    """Keying choice changes the object, so it must change the fingerprint."""
    by_permno = manifest.panels_from_crsp_daily(crsp_like)
    by_ticker = manifest.panels_from_crsp_daily(crsp_like, identifier="ticker")
    assert manifest.frame_digest(by_permno["close"]) != manifest.frame_digest(by_ticker["close"])


def test_crsp_adapter_rejects_an_absent_identifier(crsp_like):
    with pytest.raises(KeyError, match="cusip"):
        manifest.panels_from_crsp_daily(crsp_like, identifier="cusip")


def test_crsp_adapter_rejects_a_frame_with_no_known_fields():
    junk = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "permno": [1], "nonsense": [1.0]})
    with pytest.raises(KeyError, match="none of the expected columns"):
        manifest.panels_from_crsp_daily(junk)


# --- cache loading -----------------------------------------------------------


def test_load_panels_pivots_a_tidy_extract(crsp_like, tmp_path):
    path = tmp_path / "crsp_probe.parquet"
    crsp_like.to_parquet(path)
    panels = manifest.load_panels(path)
    assert set(panels) == set(manifest.CRSP_PANELS)


def test_load_panels_passes_through_an_already_pivoted_file(tmp_path):
    """Ken French and sample prices arrive as dates x securities already."""
    path = tmp_path / "french_probe.parquet"
    _frame().to_parquet(path)
    panels = manifest.load_panels(path)
    assert list(panels) == ["french_probe"]
    assert panels["french_probe"].shape == (8, 4)


def test_resolve_cache_path_accepts_an_explicit_parquet_path(tmp_path):
    path = tmp_path / "explicit.parquet"
    _frame().to_parquet(path)
    assert manifest.resolve_cache_path(str(path)) == path


def test_resolve_cache_path_lists_what_is_available_when_it_misses():
    """A bare 'not found' wastes time; cache names are chosen by whoever pulled."""
    with pytest.raises(FileNotFoundError, match="available:"):
        manifest.resolve_cache_path("definitely_not_a_cached_dataset")
