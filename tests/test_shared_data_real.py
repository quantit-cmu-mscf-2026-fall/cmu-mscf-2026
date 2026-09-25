"""The real shared files still hold every property the team relies on.

Runs only where the shared data is present (a teammate's machine, after
`shared_data.sync()`); skips cleanly elsewhere, including CI. Run with
`pytest -m data`. A failure here after a data refresh means the new pull broke
an assumption the loader or the documented rules depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd

pytestmark = pytest.mark.data

DAILY = "sp500_daily_1990_2025"


@pytest.fixture(scope="module")
def daily():
    if not (sd.cache_dir() / sd.MANIFEST).exists():
        pytest.skip(f"no shared data in {sd.cache_dir()}")
    return sd.load(
        DAILY,
        columns=[
            "date", "permno", "permco", "dlyret", "delret", "dlyretmissflg", "dlydelflg",
            "delactiontype", "dlybid", "dlyask", "dlycap", "in_sp500", "in_universe",
        ],
        auto_sync=False,
    )  # fmt: skip


def test_local_copy_matches_its_checksums():
    if not (sd.cache_dir() / sd.MANIFEST).exists():
        pytest.skip("no shared data")
    for name, digest in sd._read_manifest(sd.cache_dir() / sd.MANIFEST).items():
        assert sd._sha256(sd.cache_dir() / name) == digest, name


def test_shape_and_keys(daily):
    assert daily["date"].min() == pd.Timestamp("1990-01-02")
    assert not daily.duplicated(["date", "permno"]).any()


def test_membership_counts(daily):
    members = daily[daily["in_sp500"]].groupby("date")["permno"].nunique()
    universe = daily[daily["in_universe"]].groupby("date")["permno"].nunique()
    assert members.between(495, 510).all()
    assert universe.between(435, 495).all()
    assert (daily["in_universe"] <= daily["in_sp500"]).all()


def test_dlyret_already_includes_delisting_return(daily):
    rows = daily[daily["dlydelflg"] == "Y"]
    assert len(rows) > 600
    both = rows.dropna(subset=["dlyret", "delret"])
    np.testing.assert_allclose(both["dlyret"], both["delret"])


def test_treated_returns_complete_on_universe_rows(daily):
    need = ["permno", "date", "dlyret", "dlyretmissflg", "dlydelflg", "delactiontype"]
    ret = sd.daily_returns(daily[need])
    assert ret[daily["in_universe"]].notna().all()


def test_treated_spreads_complete_on_universe_rows(daily):
    spread = sd.quoted_spread(daily[["permno", "date", "dlybid", "dlyask", "in_universe"]])
    universe = spread[daily["in_universe"]]
    assert universe.notna().all()
    assert universe.between(0, sd.MAX_SPREAD).all()


def test_universe_tracks_the_market(daily):
    """Value-weighted universe return vs Ken French total market: catches bad data at scale."""
    from capstone.data import load_french

    try:
        french = load_french("factors_daily")
    except Exception as exc:  # network or cache unavailable
        pytest.skip(f"Ken French data unavailable: {exc}")
    frame = daily.sort_values(["permno", "date"]).copy()
    frame["weight"] = frame.groupby("permno")["dlycap"].shift(1)
    frame = frame[frame["in_universe"] & frame["weight"].notna() & frame["dlyret"].notna()]
    frame["wr"] = frame["weight"] * frame["dlyret"]
    by_day = frame.groupby("date")
    ours = by_day["wr"].sum() / by_day["weight"].sum()
    market = (french["Mkt-RF"] + french["RF"]).reindex(ours.index).dropna()
    assert ours.reindex(market.index).corr(market) > 0.98


def test_report_dates_present_for_quarters_feeding_the_universe(daily):
    """Quarters a universe stock can actually use: within the link's valid dates and
    the stock's time in the universe (plus one quarter before entry). Measured 0.48%
    missing on the 1990-2025 pull, 0.09% since 2000; the +90d fallback covers the rest.
    """
    links = sd.primary_links(sd.load("ccm_link", auto_sync=False))
    span = daily[daily["in_universe"]].groupby("permno")["date"].agg(["min", "max"])
    fundq = sd.load("comp_fundq", columns=["gvkey", "datadate", "rdq"], auto_sync=False)
    q = fundq.merge(links[["gvkey", "lpermno", "linkdt", "linkenddt"]], on="gvkey")
    q = q[q["datadate"].between(q["linkdt"], q["linkenddt"])]
    q = q.merge(span, left_on="lpermno", right_index=True)
    q = q[q["datadate"].between(q["min"] - pd.Timedelta(120, unit="D"), q["max"])]
    assert len(q) > 60_000
    assert q["rdq"].isna().mean() < 0.01
