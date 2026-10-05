"""The data-maintenance scripts build, document and publish the shared data correctly.

Covers scripts/pull_wrds.py (the universe flag), scripts/publish_shared_data.py
(preview vs apply, checksum file) and scripts/build_data_dictionary.py (code
meanings, fact checks). All on tiny hand-built frames and a fake Drive folder:
no WRDS login, no real data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pull = _script("pull_wrds")
publish = _script("publish_shared_data")
build = _script("build_data_dictionary")

COMMON = dict(
    sharetype="NS",
    securitytype="EQTY",
    securitysubtype="COM",
    usincflg="Y",
    issuertype="CORP",
    primaryexch="N",
    tradingstatusflg="A",
)
DELISTED = dict(
    sharetype="N/A",
    securitytype="N/A",
    securitysubtype="UNK",
    usincflg="N",
    issuertype="CORP",
    primaryexch="X",
    tradingstatusflg="D",
)


def _day(permno, date, in_sp500, **desc):
    return {"permno": permno, "date": pd.Timestamp(date), "in_sp500": in_sp500, **desc}


# --- pull_wrds: universe flag ---------------------------------------------


def test_universe_flag_cases():
    rows = [
        _day(1, "2020-01-02", True, **COMMON),  # member, common stock
        _day(1, "2020-01-03", True, **DELISTED),  # its delisting row: judged on prior day
        _day(2, "2020-01-02", True, **{**COMMON, "issuertype": "REIT"}),  # REIT
        _day(2, "2020-01-03", True, **DELISTED),  # REIT's delisting row stays out
        _day(3, "2020-01-02", True, **{**COMMON, "usincflg": "N"}),  # foreign-incorporated
        _day(4, "2020-01-02", False, **COMMON),  # not a member
    ]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=0)  # order must not matter
    flagged = pull.add_universe_flag(frame)
    keys = zip(flagged.permno, flagged.date.dt.day, strict=True)
    got = dict(zip(keys, flagged.in_universe, strict=True))
    assert got == {(1, 2): True, (1, 3): True, (2, 2): False, (2, 3): False,
                   (3, 2): False, (4, 2): False}  # fmt: skip
    assert list(flagged.index) == list(frame.index)


def test_universe_flag_leaves_other_rows_descriptors_alone():
    """Only delisting rows borrow the previous day's descriptors."""
    rows = [
        _day(1, "2020-01-02", True, **COMMON),
        _day(1, "2020-01-03", True, **{**COMMON, "primaryexch": "B"}),  # moved exchange
    ]
    flagged = pull.add_universe_flag(pd.DataFrame(rows))
    assert list(flagged.in_universe) == [True, False]


def test_delisting_fields_land_only_on_the_delisting_row():
    daily = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": pd.to_datetime(
                ["2025-06-02", "2025-06-03", "2025-06-04"] * 1 + ["2025-12-30", "2025-12-31"]
            ),
            "dlydelflg": ["N", "N", "Y", "N", "N"],
        }
    )
    delists = pd.DataFrame(
        {
            "permno": [1, 2],  # 2 delists after the sample ends
            "delistingdt": pd.to_datetime(["2025-06-03", "2026-02-15"]),
            "delret": [-0.2, 0.01],
        }
    )
    out = pull.attach_delistings(daily, delists)
    assert out["delret"].notna().tolist() == [False, False, True, False, False]


# --- publish_shared_data --------------------------------------------------


@pytest.fixture
def drive(tmp_path, monkeypatch):
    folder = tmp_path / sd.FOLDER_NAME
    folder.mkdir()
    (folder / "a.parquet").write_bytes(b"original")
    sd.write_manifest(folder / sd.MANIFEST, {"a.parquet": sd._sha256(folder / "a.parquet")})
    monkeypatch.setenv("CAPSTONE_DATA_DIR", str(folder))
    return folder


def _run_publish(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["publish_shared_data.py", *map(str, args)])
    publish.main()
    return capsys.readouterr().out


def test_publish_preview_writes_nothing(drive, tmp_path, monkeypatch, capsys):
    new = tmp_path / "a.parquet"
    new.write_bytes(b"updated")
    before = (drive / sd.MANIFEST).read_bytes()
    out = _run_publish(monkeypatch, capsys, new)
    assert "replace" in out and "preview only" in out
    assert (drive / "a.parquet").read_bytes() == b"original"
    assert (drive / sd.MANIFEST).read_bytes() == before


def test_publish_apply_replaces_adds_and_rewrites_manifest(drive, tmp_path, monkeypatch, capsys):
    changed, added = tmp_path / "a.parquet", tmp_path / "b.csv"
    changed.write_bytes(b"updated")
    added.write_bytes(b"new file")
    out = _run_publish(monkeypatch, capsys, changed, added, "--apply")
    assert "published 2 file(s)" in out
    manifest = drive / sd.MANIFEST
    assert b"\r" not in manifest.read_bytes()
    entries = sd._read_manifest(manifest)
    assert set(entries) == {"a.parquet", "b.csv"}
    for name, digest in entries.items():
        assert sd._sha256(drive / name) == digest


def test_publish_failure_partway_publishes_nothing(drive, tmp_path, monkeypatch, capsys):
    # Lauren's #27 review: a copy that fails verification after another file
    # was already replaced left Drive half-updated with a stale SHA256SUMS.
    changed, broken = tmp_path / "a.parquet", tmp_path / "b.csv"
    changed.write_bytes(b"updated")
    broken.write_bytes(b"will not verify")
    manifest_before = (drive / sd.MANIFEST).read_bytes()
    real_sha = sd._sha256

    def corrupt_b(path):
        return "0" * 64 if path.name.startswith(".b.csv") else real_sha(path)

    monkeypatch.setattr(sd, "_sha256", corrupt_b)
    monkeypatch.setattr(
        sys, "argv", ["publish_shared_data.py", str(changed), str(broken), "--apply"]
    )
    with pytest.raises(SystemExit, match="b.csv did not verify"):
        publish.main()
    assert (drive / "a.parquet").read_bytes() == b"original"
    assert (drive / sd.MANIFEST).read_bytes() == manifest_before
    assert sorted(p.name for p in drive.iterdir()) == ["SHA256SUMS", "a.parquet"]


def test_publish_unchanged_file_is_a_no_op(drive, monkeypatch, capsys):
    out = _run_publish(monkeypatch, capsys, drive / "a.parquet", "--apply")
    assert "unchanged" in out and "nothing to publish" in out


# --- build_data_dictionary ------------------------------------------------


@pytest.fixture
def meta():
    items = pd.DataFrame(
        {"itemname": ["DlyPrcFlg"], "itemflagtype": ["PC"], "itemdesc": ["Daily Price Flag"]}
    )
    items["key"] = items.itemname.str.lower()
    flags = pd.DataFrame(
        {"flagtype": ["PC", "PC"], "flagvalue": ["TR", "BA"],
         "flagdesc": ["Closing Trade Price", "Bid Ask Average"]}
    )  # fmt: skip
    return items.set_index("key"), flags


def test_code_values_use_crsp_meanings_then_documented_fallbacks(meta):
    items, flags = meta
    observed = pd.Series(["TR", "TR", "BA", "SU"])
    text = build.code_values("dlyprcflg", observed, items, flags)
    assert "TR = Closing Trade Price (2)" in text
    assert "BA = Bid Ask Average (1)" in text
    assert "SU = [not in CRSP flag table]" in text


def test_code_values_refuse_to_guess_an_unknown_code(meta):
    items, flags = meta
    with pytest.raises(ValueError, match="add it to OBSERVED"):
        build.code_values("dlyprcflg", pd.Series(["ZZ"]), items, flags)


def _claims_frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "dlyprc": [10.0, 5.0],
            "shrout": [100.0, 100.0],
            "dlycap": [1000.0, 500.0],
            "sharetype": ["NS", "N/A"],
            "dlydelflg": ["N", "Y"],
            "dlyret": [0.01, -0.5],
            "delret": [np.nan, -0.5],
            "dlyretmissflg": ["NA", "NA"],
            "delretmisstype": [None, "NA"],
            "dlyprcflg": ["TR", "DP"],
            "tradingstatusflg": ["A", "D"],
            "delistingdt": pd.to_datetime([None, "2020-01-02"]),
        }
    )


def test_fact_checks_pass_on_consistent_data():
    build.check_claims(_claims_frame())


def test_fact_checks_catch_a_broken_claim():
    frame = _claims_frame()
    frame.loc[1, "dlyret"] = -0.2  # dlyret no longer equals delret on the delisting row
    with pytest.raises(AssertionError, match="dlyret no longer equals delret"):
        build.check_claims(frame)


def test_markdown_escapes_and_lists_codes():
    dd = pd.DataFrame(
        [
            dict(file="f", column="a", source="s", description="x | y", definition="",
                 notes="", codes="TR = Trade (3)"),
        ]
    )  # fmt: skip
    text = build.to_markdown(dd)
    assert "## f" in text and "x \\| y" in text and "- `a`: TR = Trade (3)" in text
