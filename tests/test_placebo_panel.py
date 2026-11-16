"""The placebo panel (`capstone.pipeline.placebo`) and the refusal that keeps it off the main
ledger. Fixture rows only: no data, no model, no network.
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline import placebo

METHOD = "cross_section_permutation"


def _panel(n_stocks=100, n_dates=300, alpha_sd=0.003, noise_sd=0.01, seed=11):
    """Stocks with a persistent alpha each: a stock's trailing mean return predicts its
    next one (the planted signal). Extra fields ride along to check whole rows move."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2001-01-02", periods=n_dates)
    alpha = rng.normal(0.0, alpha_sd, n_stocks)
    frames = []
    for i in range(n_stocks):
        ret = alpha[i] + rng.normal(0.0, noise_sd, n_dates)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "permno": 10000 + i,
                    "in_universe": (rng.random(n_dates) > 0.05),
                    "dlyret": ret,
                    "dlyprc": 50.0 * np.cumprod(1.0 + ret),
                    "siccd": float(1000 + 10 * i),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    # an unbalanced panel: some stocks absent on some dates
    drop = rng.random(len(rows)) < 0.03
    return rows.loc[~drop].sort_values(["date", "permno"]).reset_index(drop=True)


def _ic_tstat(rows, window=60, id_column="permno"):
    """Mean cross-sectional rank IC of the trailing-mean signal for the next return, and
    its t-statistic across dates."""
    wide = rows.pivot(index="date", columns=id_column, values="dlyret").sort_index()
    signal = wide.rolling(window, min_periods=window // 2).mean()
    future = wide.shift(-1)
    ics = []
    for date in wide.index[window:-1]:
        s, f = signal.loc[date], future.loc[date]
        ok = s.notna() & f.notna()
        if ok.sum() >= 10:
            ics.append(s[ok].rank().corr(f[ok].rank()))
    ics = np.asarray(ics)
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))


def _row_multisets(rows):
    fields = [c for c in rows.columns if c not in ("date", "permno", "siccd")]
    return {
        date: sorted(map(tuple, group[fields].to_numpy().tolist()))
        for date, group in rows.groupby("date")
    }


# ---------------------------------------------------------------------------
# The permutation


def test_the_permutation_is_seeded_and_deterministic():
    rows = _panel()
    a = placebo.permute_cross_section(rows, 7)
    b = placebo.permute_cross_section(rows.sample(frac=1.0, random_state=3), 7)
    pd.testing.assert_frame_equal(a, b)  # same seed, any input order: same panel
    c = placebo.permute_cross_section(rows, 8)
    assert not a["dlyret"].equals(c["dlyret"])  # another seed: another panel
    assert not a["dlyret"].equals(rows["dlyret"])


def test_the_permutation_keeps_each_dates_rows_and_ids():
    rows = _panel()
    out = placebo.permute_cross_section(rows, 7)
    assert list(out.columns) == list(rows.columns)
    assert out[["date", "permno"]].equals(rows[["date", "permno"]])  # same ids per date
    assert _row_multisets(out) == _row_multisets(rows)  # same whole rows per date
    # rows move whole: each row's price still belongs to its own return path
    merged = out.merge(rows, on=["date", "dlyret"], suffixes=("", "_real"))
    assert len(merged) == len(rows)
    assert (merged["dlyprc"] == merged["dlyprc_real"]).all()
    # Sector codes belong to the target identity, not the donor's return history.
    assert out["siccd"].equals(rows["siccd"])
    # and the ids really are reshuffled
    assert (merged["permno"] != merged["permno_real"]).mean() > 0.9


def test_the_permutation_keeps_the_market_return():
    rows = _panel()
    out = placebo.permute_cross_section(rows, 7)
    pd.testing.assert_series_equal(
        out.groupby("date")["dlyret"].mean(), rows.groupby("date")["dlyret"].mean()
    )


def test_the_placebo_breaks_a_planted_signal():
    rows = _panel()
    ic, t = _ic_tstat(rows)
    assert ic > 0.1 and t > 10  # the planted signal is there
    for seed in (0, 1, 2):
        ic_p, t_p = _ic_tstat(placebo.placebo_rows(rows, METHOD, seed))
        assert abs(ic_p) < 0.02 and abs(t_p) < 3  # and gone in the placebo


def test_the_permutation_refuses_duplicate_rows():
    rows = _panel(n_stocks=5, n_dates=10)
    with pytest.raises(ValueError, match="one row per"):
        placebo.permute_cross_section(pd.concat([rows, rows.head(1)]), 0)


def test_an_unknown_method_or_bad_seed_is_refused():
    rows = _panel(n_stocks=5, n_dates=10)
    with pytest.raises(ValueError, match="method"):
        placebo.placebo_rows(rows, "bootstrap", 0)
    for seed in (None, -1, 1.5, True):
        with pytest.raises(ValueError, match="placebo_seed"):
            placebo.placebo_rows(rows, METHOD, seed)


def test_the_data_version_tag():
    real = "crsp-sp500-daily:abc123def456:0123456789ab"
    tag = placebo.placebo_version(METHOD, 42, real)
    assert tag == f"placebo:{METHOD}:42:{real}"
    assert tag.startswith("placebo:") and not real.startswith("placebo:")


# ---------------------------------------------------------------------------
# The ledger refusal


def test_check_ledger_refuses_no_ledger_and_the_main_one(tmp_path, monkeypatch):
    main = tmp_path / "experiments"
    monkeypatch.setattr(placebo, "main_ledger_dir", lambda: main)
    with pytest.raises(placebo.PlaceboRefused, match="CAPSTONE_LEDGER_DIR"):
        placebo.check_ledger({})
    for spelled in (str(main), str(main) + "/", str(main / ".." / "experiments")):
        with pytest.raises(placebo.PlaceboRefused, match="experiments"):
            placebo.check_ledger({"CAPSTONE_LEDGER_DIR": spelled})
    isolated = main / "placebo"
    assert placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(isolated)}) == isolated
    with pytest.raises(placebo.PlaceboRefused, match="placebo"):
        placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(tmp_path)})


def test_all_stock_identifiers_move_as_one_target_tuple():
    rows = _panel(n_stocks=20, n_dates=50)
    rows["permco"] = rows.permno // 2  # preserve two share classes per company
    for name in ("cusip", "ncusip", "ticker", "gvkey", "security_id"):
        rows[name] = (name + ":" + rows.permno.astype(str)).astype("string")
    rows["lpermno"], rows["lpermco"] = rows.permno, rows.permco
    rows["naics"] = rows.permno + 100000
    identities = [c for c in rows if c not in ("date", "dlyret", "dlyprc", "in_universe")]
    out = placebo.permute_cross_section(rows, 4, identity_columns=["security_id"])
    pd.testing.assert_frame_equal(out[["date", *identities]], rows[["date", *identities]])
    assert out.dtypes.equals(rows.dtypes)
    assert list(out.columns) == list(rows.columns)
    assert not out.dlyret.equals(rows.dlyret)


def test_alternate_company_identity_cannot_recover_the_planted_signal():
    rows = _panel()
    rows["permco"] = rows.permno + 50000
    for seed in (0, 1, 2):
        out = placebo.placebo_rows(rows, METHOD, seed)
        ic, t = _ic_tstat(out, id_column="permco")
        assert abs(ic) < 0.02 and abs(t) < 3


def test_placebo_null_calibration_and_daily_identity_independence():
    rows = _panel()
    statistics, repeats, opportunities = [], 0, 0
    for seed in range(40):
        out = placebo.placebo_rows(rows, METHOD, seed)
        statistics.append(_ic_tstat(out)[1])
        donors = out.merge(
            rows[["date", "permno", "dlyret"]], on=["date", "dlyret"], suffixes=("", "_donor")
        ).pivot(index="date", columns="permno", values="permno_donor")
        previous = donors.shift(1)
        valid = donors.notna() & previous.notna()
        repeats += ((donors == previous) & valid).to_numpy().sum()
        opportunities += valid.to_numpy().sum()
    statistics = np.asarray(statistics)
    assert abs(statistics.mean()) < 0.5
    # Forty fixed synthetic seeds: allow binomial variation around the 5% null rate.
    assert np.mean(np.abs(statistics) > 1.96) <= 0.15
    # About 1/100 with this panel; a five-day block permutation is about 0.8.
    assert 0.005 < repeats / opportunities < 0.025


@pytest.mark.parametrize("column", ["date", "permno", "permco"])
def test_missing_dates_or_identity_values_are_refused(column):
    rows = _panel(n_stocks=5, n_dates=10)
    rows["permco"] = rows.permno + 50000
    rows.loc[0, column] = pd.NaT if column == "date" else np.nan
    with pytest.raises(ValueError, match="missing"):
        placebo.permute_cross_section(rows, 0)


@pytest.fixture
def ledger_dirs(tmp_path, monkeypatch):
    current, primary = tmp_path / "worktree" / "experiments", tmp_path / "primary" / "experiments"
    current.mkdir(parents=True)
    primary.mkdir(parents=True)
    monkeypatch.setattr(placebo, "main_ledger_dir", lambda: current)
    # Mock only Git discovery; physical identity checks below use real temporary files.
    monkeypatch.setattr(
        placebo,
        "subprocess",
        SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=str(primary.parent / ".git"))),
        raising=False,
    )
    return current, primary


def test_primary_checkout_main_ledger_is_also_refused(ledger_dirs):
    _, primary = ledger_dirs
    with pytest.raises(placebo.PlaceboRefused, match="main trial count"):
        placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(primary)})
    isolated = primary / "placebo"
    assert placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(isolated)}) == isolated


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_isolated_ledger_file_cannot_alias_either_main_ledger(ledger_dirs, link_kind):
    current, primary = ledger_dirs
    isolated = current / "placebo"
    isolated.mkdir()
    target = primary / "runs.jsonl"
    target.write_text("synthetic fixture only\n", encoding="utf-8")
    alias = isolated / "runs.jsonl"
    try:
        if link_kind == "hardlink":
            os.link(target, alias)
        else:
            alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"{link_kind} unavailable: {exc}")
    with pytest.raises(placebo.PlaceboRefused, match="main trial count"):
        placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(isolated)})


def test_directory_alias_is_refused(ledger_dirs):
    current, primary = ledger_dirs
    isolated = current / "placebo"
    try:
        isolated.symlink_to(primary, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(placebo.PlaceboRefused, match="main trial count"):
        placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(isolated)})


def test_physical_samefile_check_handles_case_insensitive_aliases(tmp_path, monkeypatch):
    original, alias = tmp_path / "experiments", tmp_path / "EXPERIMENTS"
    monkeypatch.setattr(placebo.os.path, "normcase", lambda p: p)
    monkeypatch.setattr(placebo.os.path, "samefile", lambda a, b: True)
    assert placebo._same(original, alias)


def test_primary_checkout_is_discovered_from_git_common_dir(tmp_path, monkeypatch):
    from capstone import runlog

    primary, worktree = tmp_path / "primary", tmp_path / "worktree"
    subprocess.run(["git", "init", "-q", str(primary)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(primary),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(primary), "worktree", "add", "--detach", "-q", str(worktree)], check=True
    )
    (worktree / "capstone").mkdir()
    monkeypatch.setattr(runlog, "__file__", str(worktree / "capstone" / "runlog.py"))
    for main in (primary / "experiments", worktree / "experiments"):
        with pytest.raises(placebo.PlaceboRefused, match="main trial count"):
            placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(main)})
    isolated = primary / "experiments" / "placebo"
    assert placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(isolated)}) == isolated


def test_worktree_refuses_when_primary_ledger_cannot_be_discovered(tmp_path, monkeypatch):
    current = tmp_path / "worktree" / "experiments"
    current.parent.mkdir()
    (current.parent / ".git").write_text("gitdir: fixture\n", encoding="utf-8")
    monkeypatch.setattr(placebo, "main_ledger_dir", lambda: current)

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(placebo.subprocess, "run", unavailable)
    with pytest.raises(placebo.PlaceboRefused, match="primary"):
        placebo.check_ledger({"CAPSTONE_LEDGER_DIR": str(current / "placebo")})


def test_guard_and_apply_enforce_ledger_and_holdout_contract(ledger_dirs, monkeypatch):
    current, _ = ledger_dirs
    config = SimpleNamespace(placebo=METHOD, placebo_seed=3)
    rows = _panel(n_stocks=10, n_dates=15)
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    with pytest.raises(placebo.PlaceboRefused):
        placebo.guard(config)
    with pytest.raises(placebo.PlaceboRefused):
        placebo.apply(config, rows, "fixture-v1")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(current / "placebo"))
    placebo.guard(config)
    with pytest.raises(ValueError, match="no holdout"):
        placebo.guard(config, holdout=True)
    out, version = placebo.apply(config, rows, "fixture-v1")
    pd.testing.assert_frame_equal(out, placebo.placebo_rows(rows, METHOD, 3))
    assert not out.dlyret.equals(rows.dlyret)
    assert version == placebo.placebo_version(METHOD, 3, "fixture-v1")
    real = SimpleNamespace(placebo=None)
    placebo.guard(real, holdout=True)
    same, version = placebo.apply(real, rows, "fixture-v1")
    assert same is rows and version == "fixture-v1"
