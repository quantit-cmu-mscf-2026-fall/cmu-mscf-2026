import json
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone.strategy.contract import load_spec, target_weights
from capstone.strategy.reference import ReferenceStrategySpec, construction_check
from capstone.strategy.research import contracts


@pytest.fixture(scope="module")
def panel():
    return reference_panel()


EXECUTABLE = [c["id"] for c in contracts()["claims"] if c["status"] == "executable"]


@pytest.mark.parametrize("claim", EXECUTABLE)
@pytest.mark.parametrize("side", ["long_only", "long_short"])
def test_reference_books_obey_profile_and_share_the_production_contract(
    panel, claim, side, tmp_path
):
    spec = ReferenceStrategySpec(claim, panel.data_version, "2000-12-29", side)
    book = target_weights(spec, panel, "2005-12-30")
    assert len(book) > 0 and np.isfinite(book).all()
    assert book.abs().sum() <= 1 + 1e-12
    assert (book >= 0).all() if side == "long_only" else abs(book.sum()) < 1e-12
    assert construction_check(spec, book, panel, "2005-12-30", panel.calendar)[
        "construction_verified"
    ]
    path = tmp_path / "spec.toml"
    path.write_text(spec.to_toml(), encoding="utf-8")
    assert load_spec(path) == spec
    assert replace(spec, side="long_only" if side == "long_short" else "long_short").id != spec.id


def test_generated_candidate_cannot_pass_with_wrong_direction(panel):
    spec = ReferenceStrategySpec(EXECUTABLE[0], panel.data_version, "2000-12-29")
    book = target_weights(spec, panel, "2005-12-30")
    with pytest.raises(AssertionError):
        construction_check(spec, -book, panel, "2005-12-30", panel.calendar)


def test_future_rows_cannot_change_a_reference_book(panel):
    spec = ReferenceStrategySpec(EXECUTABLE[0], panel.data_version, "2000-12-29")
    before = target_weights(spec, panel, "2005-12-30")
    changed = replace(panel, fields={n: f.copy() for n, f in panel.fields.items()})
    for frame in changed.fields.values():
        frame.loc[frame.index > "2005-12-30"] *= 100
    pd.testing.assert_series_equal(before, target_weights(spec, changed, "2005-12-30"))


def test_missing_calendar_and_changed_method_are_rejected(panel):
    with pytest.raises(ValueError, match="method changed"):
        ReferenceStrategySpec(EXECUTABLE[0], panel.data_version, "2000-12-29", method_hash="bad")
    spec = ReferenceStrategySpec(EXECUTABLE[0], panel.data_version, "2000-12-29")
    with pytest.raises(ValueError, match="calendar"):
        target_weights(spec, replace(panel, calendar=None), "2005-12-30")


def test_independent_turnover_sorts_use_a_common_complete_history_sample(monkeypatch):
    from types import SimpleNamespace

    from capstone.strategy import reference

    scores = pd.Series(np.arange(100, dtype=float))
    turnover = -scores.copy()
    turnover.loc[0] = np.nan
    date = pd.Timestamp("2000-06-30")
    panel = SimpleNamespace(universe=pd.DataFrame(True, index=[date], columns=scores.index))
    monkeypatch.setattr(reference.features, "formation_return", lambda *a: scores)
    monkeypatch.setattr(reference.features, "mean_turnover", lambda *a: turnover)
    long, short = reference._legs(SimpleNamespace(method="volume_momentum_6m"), panel, date, None)
    assert set(long.index) == set(range(91, 100))
    assert set(short.index) == set(range(1, 10))


@pytest.mark.parametrize(
    "method,long_ids,short_ids",
    [
        ("momentum_6m", range(90, 100), range(10)),
        ("high_52week", range(70, 100), range(30)),
        ("ff3_idiosyncratic_volatility", range(20), range(80, 100)),
    ],
)
def test_published_tail_directions_and_value_weights_have_known_answers(
    monkeypatch, method, long_ids, short_ids
):
    from types import SimpleNamespace

    from capstone.strategy import reference

    date = pd.Timestamp("2000-06-30")
    score = pd.Series(np.arange(1, 101, dtype=float))
    caps = pd.DataFrame([score], index=[date])
    panel = SimpleNamespace(
        fields={"cap": caps},
        universe=caps.notna(),
        calendar=caps.index,
        availability={"cap": pd.Series([date], index=[date])},
    )
    monkeypatch.setattr(reference.features, "formation_return", lambda *a: score)
    monkeypatch.setattr(reference.features, "high_nearness", lambda *a: score)
    monkeypatch.setattr(reference.features, "ff3_residual_volatility", lambda *a: score)
    long, short = reference._legs(SimpleNamespace(method=method), panel, date, None)
    assert set(long.index) == set(long_ids) and set(short.index) == set(short_ids)
    if method.startswith("ff3"):
        assert np.allclose(long.sort_index(), np.arange(1, 21) / 210)
        assert np.allclose(short.sort_index(), np.arange(81, 101) / 1810)
    else:
        assert np.allclose(long, 1 / len(long_ids)) and np.allclose(short, 1 / len(short_ids))


def test_benchmarks_are_built_through_the_contract_and_cash_holds_nothing(panel):
    from capstone.strategy.reference import BenchmarkSpec

    cash = target_weights(
        BenchmarkSpec("cash", panel.data_version, "2000-12-29"), panel, "2005-12-30"
    )
    assert (cash == 0).all()
    equal = BenchmarkSpec("equal_weight_universe_proxy", panel.data_version, "2000-12-29")
    book = target_weights(equal, panel, "2005-12-30")
    held = panel.universe.loc["2005-12-30"].fillna(False).astype(bool)
    assert book.sum() == pytest.approx(1.0) and (book[held] > 0).all() and (book[~held] == 0).all()
    with pytest.raises(ValueError, match="unknown benchmark"):
        BenchmarkSpec("nonsense", panel.data_version, "2000-12-29")


def test_reference_file_is_loaded_by_schema_and_a_changed_method_is_archival_only(panel, tmp_path):
    from capstone.strategy.contract import save_spec
    from capstone.strategy.recorded import RecordedExecutionBlocked, RecordedStrategySpec
    from capstone.strategy.research import canonical, digest

    spec = ReferenceStrategySpec(EXECUTABLE[0], panel.data_version, "2000-12-29")
    path = tmp_path / "spec.toml"
    save_spec(spec, path)
    assert load_spec(path) == spec
    # An archived specification written under another method hash is readable, but it
    # is a record only: it cannot construct a book.
    payload = json.loads(tomllib.loads(spec.to_toml())["payload"]) | {"method_hash": "f" * 64}
    path.write_text(
        f"schema_version = 2\nid = {json.dumps(digest(payload)[:16])}\n"
        f"payload = {json.dumps(canonical(payload))}\n",
        encoding="utf-8",
    )
    record = load_spec(path)
    assert isinstance(record, RecordedStrategySpec)
    with pytest.raises(RecordedExecutionBlocked):
        target_weights(record, panel, "2005-12-30")
