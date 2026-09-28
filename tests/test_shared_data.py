"""The shared-data loader finds, verifies and serves the team's files safely.

Everything here runs against a fake Drive folder in tmp_path, so it needs no
real data and runs in CI. The contracts: a teammate's machine only ever reads
verified files; a half-synced or missing Drive never breaks a run; several
agents can sync at once; and the missing-value rules resolve exactly the cases
they claim to, without look-ahead.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd

REPO = Path(__file__).resolve().parents[1]


def _publish(drive: Path, frames: dict[str, pd.DataFrame]) -> None:
    drive.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(drive / f"{name}.parquet", index=False)
    entries = {p.name: sd._sha256(p) for p in drive.glob("*.parquet")}
    sd.write_manifest(drive / sd.MANIFEST, entries)


@pytest.fixture
def env(tmp_path, monkeypatch):
    drive = tmp_path / "drive" / sd.FOLDER_NAME
    cache = tmp_path / "cache"
    monkeypatch.setenv("CAPSTONE_DATA_DIR", str(drive))
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(cache))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2021-06-01"]),
            "permno": pd.array([1, 1, 2], dtype="Int64"),
            "ticker": ["AAA", "AAA", "BBB"],
            "dlyret": pd.array([0.01, None, 0.02], dtype="Float64"),
            "in_universe": [True, True, False],
        }
    )
    _publish(drive, {"daily": frame, "other": pd.DataFrame({"x": [1, 2]})})
    return drive, cache


# --- sync ------------------------------------------------------------------


def test_first_sync_copies_and_verifies_everything(env):
    drive, cache = env
    assert set(sd.sync().values()) == {"copied"}
    for name, digest in sd._read_manifest(drive / sd.MANIFEST).items():
        assert sd._sha256(cache / name) == digest


def test_second_sync_copies_nothing(env):
    sd.sync()
    assert set(sd.sync().values()) == {"up to date"}


def test_published_update_is_picked_up_and_changes_version(env):
    drive, _ = env
    sd.sync()
    before = sd.data_version()
    _publish(drive, {"other": pd.DataFrame({"x": [1, 2, 3]})})
    status = sd.sync()
    assert status == {"daily.parquet": "up to date", "other.parquet": "copied"}
    assert sd.data_version() != before


def test_half_synced_file_is_rejected_and_previous_copy_kept(env):
    drive, cache = env
    sd.sync()
    good = sd._sha256(cache / "other.parquet")
    entries = sd._read_manifest(drive / sd.MANIFEST)
    entries["other.parquet"] = "f" * 64  # manifest announces a file not yet arrived
    sd.write_manifest(drive / sd.MANIFEST, entries)
    with pytest.warns(UserWarning, match="not updated"):
        status = sd.sync()
    assert status["other.parquet"].startswith("FAILED")
    assert sd._sha256(cache / "other.parquet") == good
    assert not list(cache.glob("*.partial"))


def test_missing_drive_falls_back_to_local_copy(env, monkeypatch):
    sd.sync()
    monkeypatch.delenv("CAPSTONE_DATA_DIR")
    monkeypatch.setattr(sd, "_candidates", lambda: [])
    with pytest.warns(UserWarning, match="Drive folder not found"):
        assert set(sd.sync().values()) == {"local only"}
    assert len(sd.load("daily", auto_sync=False)) == 3


def test_no_drive_and_no_local_copy_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_DATA_DIR", raising=False)
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(sd, "_candidates", lambda: [])
    with pytest.raises(sd.SharedDataError, match="Add shortcut to Drive"):
        sd.sync()


def test_drive_folder_found_among_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_DATA_DIR", raising=False)
    empty, real = tmp_path / "G" / sd.FOLDER_NAME, tmp_path / "H" / sd.FOLDER_NAME
    _publish(real, {"x": pd.DataFrame({"a": [1]})})
    monkeypatch.setattr(sd, "_candidates", lambda: [empty, real])
    assert sd.find_drive_folder() == real


def test_manifest_is_lf_and_version_ignores_format(tmp_path):
    entries = {"b.parquet": "2" * 64, "a.parquet": "1" * 64}
    plain, starred = tmp_path / "plain", tmp_path / "starred"
    sd.write_manifest(plain, entries)
    assert b"\r" not in plain.read_bytes()  # sha256sum -c breaks on CRLF
    starred.write_bytes(b"2" * 64 + b" *b.parquet\r\n" + b"1" * 64 + b" *a.parquet\r\n")
    assert sd._read_manifest(starred) == entries
    assert sd.manifest_version(plain) == sd.manifest_version(starred)


def test_concurrent_syncs_all_succeed(env):
    """Several agents on one machine starting together must not collide."""
    code = (
        "import warnings; warnings.simplefilter('error');"
        "from capstone import shared_data as sd; print(sorted(set(sd.sync().values())))"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code], cwd=REPO, stdout=subprocess.PIPE, text=True)
        for _ in range(4)
    ]
    results = [(p.communicate()[0].strip(), p.returncode) for p in procs]
    assert all(rc == 0 for _, rc in results), results
    assert sorted(out for out, _ in results).count("['copied']") == 1
    _, cache = env
    assert not [p for p in cache.iterdir() if p.name.startswith(".")]


def test_sync_waits_for_the_lock_holder(env):
    """Deterministic: while another sync holds the lock, sync() must not run."""
    _, cache = env
    cache.mkdir(parents=True)
    finished = {}

    def run():
        status = sd.sync()
        finished.update(at=time.monotonic(), s=status)  # time taken AFTER sync returns

    with sd._sync_lock(cache):
        worker = threading.Thread(target=run)
        worker.start()
        time.sleep(1.0)
        released = time.monotonic()
        assert "at" not in finished, "sync ran while another sync held the lock"
    worker.join(timeout=30)
    assert finished["at"] >= released and set(finished["s"].values()) == {"copied"}


def test_copy_refreshes_the_lock(tmp_path):
    """Long downloads keep the lock alive, so waiting agents don't treat it as abandoned."""
    src = tmp_path / "big"
    src.write_bytes(b"x" * (3 * (1 << 22) + 1))  # four chunks
    beats = []
    sd._copy(src, tmp_path / "copy", lambda: beats.append(1))
    assert len(beats) == 4 and (tmp_path / "copy").read_bytes() == src.read_bytes()


def test_stale_lock_from_crashed_process_is_cleared(env, monkeypatch):
    _, cache = env
    monkeypatch.setattr(sd, "LOCK_STALE_AFTER", 5.0)
    monkeypatch.setattr(sd, "LOCK_TIMEOUT", 10.0)  # fail fast if it is never cleared
    cache.mkdir(parents=True)
    lock = cache / ".sync.lock"
    lock.write_text("12345")
    old = lock.stat().st_mtime - 60  # not refreshed for a minute
    os.utime(lock, (old, old))
    assert set(sd.sync().values()) == {"copied"}


def test_live_lock_is_respected_then_times_out(env, monkeypatch):
    """A recently refreshed lock belongs to a live sync: wait, never steal it."""
    _, cache = env
    monkeypatch.setattr(sd, "LOCK_TIMEOUT", 1.5)
    cache.mkdir(parents=True)
    (cache / ".sync.lock").write_text("12345")
    with pytest.raises(sd.SharedDataError, match="timed out"):
        sd.sync()
    assert (cache / ".sync.lock").exists()


# --- load ------------------------------------------------------------------


def test_load_filters_dates_and_returns_plain_numpy_dtypes(env):
    frame = sd.load("daily", start="2020-01-03")
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-03", "2021-06-01"]
    assert frame["permno"].dtype == "int64"
    assert frame["dlyret"].dtype == "float64" and np.isnan(frame["dlyret"].to_numpy()[0])
    assert frame["in_universe"].dtype == "bool"
    assert isinstance(frame["ticker"].dtype, pd.StringDtype)


def test_load_unknown_file(env):
    with pytest.raises(sd.SharedDataError, match="not in the shared data"):
        sd.load("nope")


# --- missing-value rules ---------------------------------------------------


def _rows(rows):
    cols = ["permno", "date", "dlyret", "dlyretmissflg", "dlydelflg", "delactiontype"]
    frame = pd.DataFrame(rows, columns=cols)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_daily_returns_resolves_each_documented_case():
    frame = _rows(
        [
            (1, "2020-01-02", np.nan, "NS", "N", None),  # first day: 0
            (1, "2020-01-03", np.nan, "MP", "N", None),  # missing price: 0
            (1, "2020-01-06", 0.05, "NA", "N", None),  # untouched
            (11786, "2023-03-13", np.nan, "DG", "Y", "GDR"),  # SVB: -100%
            (7, "2001-01-02", np.nan, "DG", "Y", "GDR"),  # other failure: -50%
            (8, "2005-01-03", np.nan, "DM", "Y", "MER"),  # merger: not guessed
            (9, "2003-03-18", -0.01, "NA", "N", None),
            (9, "2003-03-19", np.nan, "NT", "N", None),  # first untracked: -50%
            (9, "2003-03-20", np.nan, "NT", "N", None),  # later untracked: exit
        ]
    )
    ret = sd.daily_returns(frame)
    expected = [0.0, 0.0, 0.05, -1.0, -0.5, np.nan, -0.01, -0.5, np.nan]
    np.testing.assert_allclose(ret.to_numpy(), expected)


def test_daily_returns_keeps_input_order_and_does_not_mutate():
    frame = _rows(
        [
            (2, "2020-01-03", np.nan, "MP", "N", None),
            (1, "2020-01-02", 0.01, "NA", "N", None),
        ]
    )
    before = frame.copy()
    ret = sd.daily_returns(frame)
    assert list(ret.index) == list(frame.index) and list(ret) == [0.0, 0.01]
    pd.testing.assert_frame_equal(frame, before)


def test_quoted_spread_fills_without_look_ahead():
    dates = pd.bdate_range("2020-01-01", periods=14)
    rows = [(1, d, 99.9, 100.1, True) for d in dates[:6]]  # 20 bps history
    rows += [(1, dates[6], 100.0, 100.0, True)]  # locked quote: invalid
    rows += [(1, dates[7], 90.0, 110.0, True)]  # 2000 bps: invalid
    rows += [(1, d, 99.5, 100.5, True) for d in dates[8:]]  # 100 bps AFTER the gap
    rows += [(2, dates[7], 50.0, 50.05, True)]  # 10 bps, same day
    rows += [(3, dates[7], np.nan, np.nan, True)]  # no quote, no history
    frame = pd.DataFrame(rows, columns=["permno", "date", "dlybid", "dlyask", "in_universe"])
    spread = sd.quoted_spread(frame)
    assert spread.iloc[0] == pytest.approx(0.002)
    # Gap days use only earlier quotes (20 bps), never the later 100 bps ones.
    assert spread.iloc[6] == pytest.approx(0.002)
    assert spread.iloc[7] == pytest.approx(0.002)
    assert spread.iloc[8] == pytest.approx(0.01)
    assert spread.iloc[-1] == pytest.approx(0.05 / 50.025)  # day's cross-section median
    assert spread.notna().all()


def test_available_from_is_strictly_after_report_and_falls_back():
    days = pd.bdate_range("2020-01-01", "2020-12-31")
    fund = pd.DataFrame(
        {
            "datadate": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-12-31", None]),
            "rdq": pd.to_datetime(["2020-04-24", None, "2020-12-31", None]),
        }
    )
    got = sd.available_from(fund, days)
    assert got.iloc[0] == pd.Timestamp("2020-04-27")  # Friday report -> Monday
    assert got.iloc[1] == pd.Timestamp("2020-06-30")  # 2020-06-29 + 1 trading day
    assert pd.isna(got.iloc[2]) and pd.isna(got.iloc[3])  # past calendar / no dates


def test_primary_links_filters_and_fills_open_end():
    link = pd.DataFrame(
        {
            "gvkey": ["1", "2", "3"],
            "linktype": ["LU", "NR", "LC"],
            "linkprim": ["P", "P", "J"],
            "lpermno": [10.0, 20.0, 30.0],
            "linkdt": ["2000-01-01"] * 3,
            "linkenddt": [None] * 3,
        }
    )
    out = sd.primary_links(link)
    assert list(out["gvkey"]) == ["1"] and out["lpermno"].dtype == "int64"
    assert out["linkenddt"].iloc[0] > pd.Timestamp("2200-01-01")


def test_file_hashing_matches_hashlib(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"x" * 3_000_000)
    assert sd._sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
