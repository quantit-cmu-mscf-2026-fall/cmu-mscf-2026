"""Alpha-decay diagnostics on synthetic series."""

import numpy as np
import pandas as pd
import pytest

from capstone.analysis import decay
from capstone.analysis.outputs import output_dir


def _idx(start="1990-01-02", end="2020-12-31"):
    return pd.bdate_range(start, end)


def _series(idx, mean, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mean + rng.normal(0, sd, len(idx)), index=idx)


def test_planted_decay_is_significantly_negative():
    idx = _idx()
    years = idx.year.to_numpy() - 1990
    r = _series(idx, 0.0012 * (1 - years / 30))
    slope = decay.decay_slope(decay.annual_sharpes(r, 120))
    assert slope["slope_decade"] < -0.3
    assert slope["p"] < 0.05
    assert slope["n_years"] == 31


def test_constant_mean_has_no_slope():
    idx = _idx()
    ps = [
        decay.decay_slope(decay.annual_sharpes(_series(idx, 0.0005, seed=s), 120))["p"]
        for s in range(10)
    ]
    assert np.median(ps) > 0.1
    one = decay.decay_slope(decay.annual_sharpes(_series(idx, 0.0005, seed=1), 120))
    assert abs(one["slope_decade"]) < 0.6


def test_block_sharpe_matches_planted_step_and_partial_block_is_labelled():
    idx = _idx()
    r = _series(idx, np.where(idx.year < 2005, 0.002, 0.0), sd=0.005)
    b = decay.block_sharpes(r, 5, 120)
    assert list(b.index)[:2] == ["1990-94", "1995-99"]
    assert b.index[-1] == "2020"
    assert b["1990-94"] == pytest.approx(0.002 / 0.005 * np.sqrt(252), rel=0.25)
    assert abs(b["2010-14"]) < 1.2
    assert decay.block_edges(idx, 5)[-1][3] is True


def test_publication_split_detects_post_date_drop():
    idx = _idx()
    r = _series(idx, np.where(idx >= "2005-01-01", 0.0, 0.002), sd=0.01)
    out = decay.publication_split(r, pd.Timestamp("2005-01-01"))
    assert out["diff"] < -1
    assert out["p"] < 0.01


def test_dates_after_search_period_refused(tmp_path, monkeypatch):
    idx = pd.bdate_range("2019-01-01", "2021-02-01")
    df = pd.DataFrame({"a": 0.0}, index=idx)
    called = []
    monkeypatch.setattr(decay, "log_run", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="holdout"):
        decay.run_report(df, pd.Series({"a": "f"}), "r", out_root=tmp_path)
    assert not called


def _two_family(idx):
    df = pd.DataFrame(
        {
            "a1": _series(idx, 0.001, seed=1),
            "a2": _series(idx, 0.0002, seed=2),
            "b1": _series(idx, 0.0, seed=3),
        }
    )
    return df, pd.Series({"a1": "A", "a2": "A", "b1": "B"})


def test_ledger_written_before_results_and_statements_sourced(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(decay, "log_run", lambda *a, **k: events.append(("log", k)))
    real = decay.compute

    def spy(*a, **k):
        events.append(("compute", None))
        return real(*a, **k)

    monkeypatch.setattr(decay, "compute", spy)
    df, fam = _two_family(_idx())
    out = decay.run_report(df, fam, "r", out_root=tmp_path)
    assert [e[0] for e in events] == ["log", "compute"]
    assert events[0][1]["name"] == "decay-report"
    assert events[0][1]["tags"] == ["synthetic", "methodology"]
    assert out == output_dir("r", "decay", tmp_path)
    text = (out / "STATEMENTS.md").read_text(encoding="utf-8")
    assert "publication split was skipped" in text
    assert all("stats.json:" in ln for ln in text.splitlines() if ln.startswith("S"))
    assert (out / "data" / "slopes.parquet").exists()


def test_best_variant_selection_and_publication_table():
    df, fam = _two_family(_idx())
    assert decay.best_variants(df, fam).to_dict() == {"A": "a1", "B": "b1"}
    tables, stats, statements = decay.compute(df, fam, published={"A": "2005-01-01"})
    assert list(tables["publication"].index) == ["A"]
    assert {s.source for s in statements} <= set(stats)
    assert set(tables) >= {"blocks", "annual", "slopes", "publication"}
