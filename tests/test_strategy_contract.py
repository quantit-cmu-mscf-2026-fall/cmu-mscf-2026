"""The strategy contract: a frozen spec, its TOML file, and its target book.

Synthetic CRSP-shaped rows only; no shared data, no dates after 2020 except a
relabelled panel that checks the holdout guard.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone.backtest import backtest_frame, band_book
from capstone.factors import evaluation as ev
from capstone.factors.evaluation import (
    HoldoutError,
    Panel,
    TradingSpec,
    build_panel,
    factor_signal,
    trading_signal,
)
from capstone.factors.store import variant_id
from capstone.factors.tree import factor_id, parse
from capstone.strategy.contract import (
    InsufficientHistoryError,
    StrategySpec,
    load_spec,
    lookback,
    save_spec,
    target_weights,
)
from capstone.strategy.orders import orders

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


def fake_rows(n_stocks=30, n_days=160, *, spread=0.002, seed=0, start="2015-01-02"):
    """Daily rows shaped like the shared file (as in test_factor_evaluation)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    volume = np.exp(rng.normal(13, 0.5, (n_days, n_stocks)))
    ret = rng.normal(0, 0.02, (n_days, n_stocks))
    price = 50 * np.cumprod(1 + ret, axis=0)
    rows = []
    for j in range(n_stocks):
        for t, date in enumerate(dates):
            p = price[t, j]
            rows.append(
                {
                    "date": date,
                    "permno": 10000 + j,
                    "in_universe": True,
                    "dlyret": ret[t, j],
                    "dlyretmissflg": None,
                    "dlydelflg": "N",
                    "delactiontype": None,
                    "dlyopen": p,
                    "dlyhigh": p * 1.01,
                    "dlylow": p * 0.99,
                    "dlyprc": p,
                    "dlyvol": volume[t, j],
                    "shrout": 1000.0,
                    "dlycap": p * 1000.0,
                    "dlycumfacpr": 1.0,
                    "dlycumfacshr": 1.0,
                    "dlybid": p * (1 - spread / 2),
                    "dlyask": p * (1 + spread / 2),
                    "siccd": (2834, 3674, 6020)[j % 3],
                }
            )
    return pd.DataFrame(rows)


EXPRESSION = "rank(ts_mean(volume, 5)) - rank(delta(close, 3))"


def make_spec(**overrides) -> StrategySpec:
    values = {
        "expression": EXPRESSION,
        "trading": TradingSpec(hold=5, neutral="sector"),
        "data_version": "synthetic-v1",
        "anchor": "2015-01-02",
        "gross": 1.0,
        "hypothesis_id": "H-001",
        "gate_report": "reports/gate/H-001.md",
    }
    values.update(overrides)
    return StrategySpec(**values)


# ---------------------------------------------------------------------------
# The spec and its file


def test_the_spec_is_frozen():
    spec = make_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.gross = 2.0


def test_the_spec_derives_its_provenance_and_data_fields_from_its_content():
    spec = make_spec()
    tree = parse(EXPRESSION)
    assert spec.factor_id == factor_id(tree)
    assert spec.variant_id == variant_id(factor_id(tree), TradingSpec(5, "sector").as_dict())
    assert spec.fields == ("close", "volume")


def test_the_file_format_is_fixed_and_the_id_hashes_everything_else():
    spec = make_spec()
    fid = factor_id(parse(EXPRESSION))
    vid = variant_id(fid, TradingSpec(5, "sector").as_dict())
    body = (
        "# Strategy spec (capstone.strategy.contract). Written by code; "
        "the id is a hash of the rest.\n"
        'expression = "(rank(ts_mean(volume, 5)) - rank(delta(close, 3)))"\n'
        'anchor = "2015-01-02"\n'
        "gross = 1.0\n"
        "\n"
        "[trading]\n"
        "hold = 5\n"
        'neutral = "sector"\n'
        "\n"
        "[data]\n"
        'fields = ["close", "volume"]\n'
        'version = "synthetic-v1"\n'
        "\n"
        "[provenance]\n"
        f'factor_id = "{fid}"\n'
        f'variant_id = "{vid}"\n'
        'hypothesis_id = "H-001"\n'
        'gate_report = "reports/gate/H-001.md"\n'
    )
    expected_id = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    first, rest = body.split("\n", 1)
    assert spec.id == expected_id
    assert spec.to_toml() == f'{first}\nid = "{expected_id}"\n{rest}'


def test_the_spec_round_trips_through_its_file_byte_for_byte(tmp_path):
    spec = make_spec(gross=0.5, hypothesis_id='odd "quoted" \\ name\n')
    path = tmp_path / "spec.toml"
    save_spec(spec, path)
    loaded = load_spec(path)
    assert loaded == spec
    assert loaded.id == spec.id
    again = tmp_path / "again.toml"
    save_spec(loaded, again)
    assert again.read_bytes() == path.read_bytes()
    assert b"\r\n" not in path.read_bytes()


def test_equal_content_gives_an_equal_id_and_any_change_a_new_one():
    base = make_spec()
    assert make_spec().id == base.id
    # The same expression spelled differently is the same content.
    assert make_spec(expression="rank(ts_mean(volume,5))-rank(delta(close,3))").id == base.id
    changes = [
        {"expression": "rank(volume)"},
        {"trading": TradingSpec(hold=1, neutral="sector")},
        {"trading": TradingSpec(hold=5, neutral="none")},
        {"data_version": "synthetic-v2"},
        {"anchor": "2015-01-05"},
        {"gross": 2.0},
        {"hypothesis_id": "H-002"},
        {"gate_report": "reports/gate/other.md"},
    ]
    ids = {make_spec(**change).id for change in changes}
    assert base.id not in ids and len(ids) == len(changes)


def test_a_tampered_file_is_refused(tmp_path):
    path = tmp_path / "spec.toml"
    save_spec(make_spec(), path)
    path.write_bytes(path.read_bytes().replace(b"hold = 5", b"hold = 1"))
    with pytest.raises(ValueError, match="id"):
        load_spec(path)


@pytest.mark.parametrize(
    "bad",
    [
        {"expression": "close +"},
        {"gross": 0.0},
        {"gross": float("nan")},
        {"anchor": "not a date"},
        {"data_version": ""},
    ],
)
def test_an_invalid_spec_is_refused(bad):
    with pytest.raises(ValueError):
        make_spec(**bad)


def test_a_banded_spec_round_trips_its_file_and_band_zero_writes_no_band_line():
    plain = make_spec(trading=TradingSpec(5, "sector"))
    assert "band" not in plain.to_toml()
    banded = make_spec(trading=TradingSpec(5, "sector", 0.005))
    text = banded.to_toml()
    assert "\nband = 0.005\n" in text
    again = StrategySpec.from_toml(text)
    assert again == banded and again.id == banded.id != plain.id
    assert again.variant_id == variant_id(
        banded.factor_id, TradingSpec(5, "sector", 0.005).as_dict()
    )


def test_a_file_that_adds_a_band_without_changing_its_id_is_refused(tmp_path):
    path = tmp_path / "spec.toml"
    save_spec(make_spec(), path)
    text = path.read_bytes().decode("utf-8")
    text = text.replace('neutral = "sector"\n', 'neutral = "sector"\nband = 0.01\n')
    path.write_bytes(text.encode("utf-8"))
    with pytest.raises(ValueError, match="id"):
        load_spec(path)


def test_the_spec_id_is_the_same_in_a_fresh_process(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(TESTS)]),
        "PYTHONHASHSEED": "777",
        "CAPSTONE_LEDGER_DIR": str(tmp_path),
    }
    code = "import test_strategy_contract as t; print(t.make_spec().id)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == make_spec().id


def test_lookback_counts_the_rows_a_value_needs_before_its_own():
    assert lookback(parse("close")) == 0
    assert lookback(parse("ts_mean(close, 5)")) == 4
    assert lookback(parse("delta(close, 3)")) == 3
    assert lookback(parse("ts_mean(delta(close, 3), 5)")) == 7
    assert lookback(parse("rank(ts_std(returns, 20)) - ts_corr(close, volume, 10)")) == 19
    assert lookback(parse("ema(shift(volume, 2), 10)")) == 11


# ---------------------------------------------------------------------------
# Target weights


@pytest.fixture(scope="module")
def panel() -> Panel:
    return build_panel(fake_rows(), "synthetic-v1")


def evaluation_books(spec: StrategySpec, panel: Panel) -> pd.DataFrame:
    """Evaluation's weights, held the way `returns_from_weights` holds them."""
    weights = ev.signal_weights(
        ev.factor_signal(parse(spec.expression), panel), panel, spec.trading.neutral
    )
    return weights.fillna(0.0) * spec.gross


@pytest.mark.parametrize(
    ("hold", "neutral"), [(1, "none"), (5, "none"), (21, "none"), (5, "sector"), (21, "sector")]
)
def test_target_weights_are_evaluations_book_on_every_date(panel, hold, neutral):
    # Anchored at the panel's first date, the rebalance dates are evaluation's:
    # every `hold`-th row from the first.
    dates = panel.universe.index
    spec = make_spec(trading=TradingSpec(hold, neutral), anchor=str(dates[0].date()), gross=2.0)
    books = evaluation_books(spec, panel)
    need = lookback(parse(spec.expression))
    checked = 0
    for p in range(len(dates)):
        rebalance = (p // hold) * hold
        if rebalance < need:
            continue
        expected = books.iloc[rebalance].rename("weight")
        expected.index.name = "permno"
        got = target_weights(spec, panel, dates[p])
        # A weight that is zero up to rounding may or may not be listed: compare
        # on every stock, with an unlisted one as 0.
        pd.testing.assert_series_equal(
            got.reindex(expected.index, fill_value=0.0), expected, rtol=1e-12, atol=1e-14
        )
        assert set(expected.index[expected.abs() > 1e-14]) <= set(got.index)
        checked += 1
    assert checked > len(dates) // 2


def test_the_book_has_the_specs_gross_and_is_dollar_neutral(panel):
    spec = make_spec(gross=1.5)
    book = target_weights(spec, panel, panel.universe.index[100])
    assert book.abs().sum() == pytest.approx(1.5, rel=1e-12)
    assert book.sum() == pytest.approx(0.0, abs=1e-12)
    assert book.index.is_monotonic_increasing and book.index.name == "permno"


@pytest.mark.parametrize(
    "expression",
    [EXPRESSION, "ema(volume, 10)", "ts_corr(returns, mkt_return, 10)"],
)
def test_the_same_as_of_gives_the_same_book_whatever_history_is_loaded(expression):
    rows = fake_rows()
    dates = pd.DatetimeIndex(sorted(rows.date.unique()))
    full = build_panel(rows, "synthetic-v1")
    # A later window start that is not a multiple of `hold` rows from the
    # first: counting rebalances from the window would move them.
    late = build_panel(rows[rows.date >= dates[12]], "synthetic-v1")
    spec = make_spec(expression=expression, anchor=str(dates[30].date()))
    for as_of in dates[40:120]:
        a = target_weights(spec, full, as_of)
        b = target_weights(spec, late, as_of)
        pd.testing.assert_series_equal(a, b, check_exact=True)


def test_the_book_changes_only_on_anchored_rebalance_dates(panel):
    dates = panel.universe.index
    spec = make_spec(anchor=str(dates[30].date()))  # hold 5
    books = [target_weights(spec, panel, d) for d in dates[30:80]]
    for i in range(1, len(books)):
        same = books[i].equals(books[i - 1])
        assert same == (i % 5 != 0), i


def test_too_little_history_before_the_rebalance_date_raises(panel):
    dates = panel.universe.index
    spec = make_spec(expression="rank(ts_mean(volume, 20))", anchor=str(dates[0].date()))
    with pytest.raises(InsufficientHistoryError):
        target_weights(spec, panel, dates[10])
    # as_of has 20 rows of history, but its rebalance date (the anchor) has none.
    long_hold = make_spec(
        expression="rank(ts_mean(volume, 20))",
        trading=TradingSpec(21, "none"),
        anchor=str(dates[0].date()),
    )
    with pytest.raises(InsufficientHistoryError):
        target_weights(long_hold, panel, dates[20])
    assert len(target_weights(long_hold, panel, dates[21])) > 0


def test_mkt_return_needs_one_more_row_than_its_window(panel):
    # The panel's first row has no market return (it weights by the previous
    # day's cap), so a window starting there is not a full window.
    dates = panel.universe.index
    spec = make_spec(
        expression="ts_corr(returns, mkt_return, 10)",
        trading=TradingSpec(1, "none"),
        anchor=str(dates[0].date()),
    )
    with pytest.raises(InsufficientHistoryError, match="needs 10 earlier rows"):
        target_weights(spec, panel, dates[9])
    assert len(target_weights(spec, panel, dates[10])) > 0


def test_as_of_and_anchor_must_be_trading_dates_of_the_panel(panel):
    dates = panel.universe.index
    spec = make_spec(anchor=str(dates[30].date()))
    with pytest.raises(ValueError, match="as_of"):
        target_weights(spec, panel, pd.Timestamp("2015-01-03"))  # a Saturday
    with pytest.raises(ValueError, match="as_of"):
        target_weights(spec, panel, dates[20])  # before the anchor
    unanchored = make_spec(anchor="2014-06-02")
    with pytest.raises(ValueError, match="anchor"):
        target_weights(unanchored, panel, dates[60])


def _relabel(panel: Panel, index: pd.DatetimeIndex) -> Panel:
    def move(frame):
        return None if frame is None else frame.set_axis(index, axis=0)

    fields = {name: move(value) for name, value in panel.fields.items()}
    return Panel(
        fields, move(panel.universe), move(panel.half_spread), move(panel.sic), panel.data_version
    )


def test_a_panel_reaching_past_discovery_is_refused():
    rows = fake_rows(n_days=60)
    later = pd.bdate_range("2020-11-02", periods=60)  # runs into 2021
    panel = _relabel(build_panel(rows, "synthetic-v1"), later)
    spec = make_spec(anchor=str(later[0].date()))
    with pytest.raises(HoldoutError, match="authorized current-feature route"):
        target_weights(spec, panel, later[20])  # as_of in 2020: the panel still holds 2021
    with pytest.raises(HoldoutError, match="authorized current-feature route"):
        target_weights(spec, panel, later[50])
    assert ev.DISCOVERY_END < str(later[50].date())


def test_there_is_no_flag_that_admits_a_post_discovery_panel():
    import inspect

    from capstone.strategy import contract

    for function in (contract.target_weights, contract._target_weights):
        assert "live" not in inspect.signature(function).parameters
    panel = build_panel(fake_rows(), "synthetic-v1")
    dates = panel.universe.index
    spec = make_spec(anchor=str(dates[30].date()))
    with pytest.raises(TypeError):
        target_weights(spec, panel, dates[100], live=True)
    later = pd.bdate_range("2026-06-01", periods=len(dates))
    with pytest.raises(HoldoutError, match="authorized current-feature route"):
        target_weights(make_spec(anchor=str(later[30].date())), _relabel(panel, later), later[100])


# ---------------------------------------------------------------------------
# Determinism


def book_digest() -> str:
    """A hash of the bytes of one book and its orders; the subprocess test runs this."""
    panel = build_panel(fake_rows(), "synthetic-v1")
    dates = panel.universe.index
    spec = make_spec(anchor=str(dates[30].date()))
    target = target_weights(spec, panel, dates[100])
    current = target_weights(spec, panel, dates[94])
    trades = orders(target, current, min_trade=1e-4).trades
    h = hashlib.sha256()
    for series in (target, trades):
        h.update(series.index.to_numpy().astype("int64").tobytes())
        h.update(series.to_numpy().astype("float64").tobytes())
    return h.hexdigest()


def test_the_same_inputs_give_identical_books_and_orders_on_every_run():
    assert book_digest() == book_digest()


def test_a_fresh_process_gives_identical_books_and_orders(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(TESTS)]),
        "PYTHONHASHSEED": "12345",
        "CAPSTONE_LEDGER_DIR": str(tmp_path),
    }
    code = "import test_strategy_contract as t; print(t.book_digest())"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == book_digest()


# ---------------------------------------------------------------------------
# The no-trade band


@pytest.fixture(scope="module")
def panel_small() -> Panel:
    return build_panel(fake_rows(n_stocks=12, n_days=120, seed=3), "synthetic-v1")


def make_banded_spec(
    trading: TradingSpec, anchor, expression=EXPRESSION, gross=1.0
) -> StrategySpec:
    return StrategySpec(
        expression=expression,
        trading=trading,
        data_version="synthetic-v1",
        anchor=str(pd.Timestamp(anchor).date()),
        gross=gross,
    )


@pytest.mark.parametrize(
    ("hold", "neutral", "band"),
    [(1, "none", 0.01), (5, "none", 0.005), (5, "sector", 0.01), (1, "sector", 0.02)],
)
def test_target_weights_honour_the_band_as_evaluation_does(panel_small, hold, neutral, band):
    panel = panel_small
    dates = panel.universe.index
    spec = make_banded_spec(TradingSpec(hold, neutral, band), dates[0], gross=2.0)
    tree = parse(EXPRESSION)
    traded = trading_signal(factor_signal(tree, panel), panel, neutral)
    weights = ev.to_weights(traded, demean=neutral == "none")
    expected_books = band_book(weights, traded.isna(), band=band, hold=hold) * 2.0
    need = lookback(tree)
    books = {}
    for p in range(len(dates)):
        if (p // hold) * hold < need:
            continue
        got = target_weights(spec, panel, dates[p])
        expected = expected_books.iloc[p]
        expected = expected[expected != 0].rename("weight")
        expected.index.name = "permno"
        pd.testing.assert_series_equal(got, expected, rtol=1e-12, atol=1e-14)
        books[dates[p]] = got.reindex(panel.universe.columns, fill_value=0.0)
    assert len(books) > len(dates) // 2
    # The contract's books trade exactly the backtest's turnover.
    frame = backtest_frame(
        traded,
        panel.fields["returns"],
        cost_bps=panel.half_spread * 1e4,
        demean=neutral == "none",
        gross=1.0,
        hold=hold,
        band=band,
    )
    held = pd.DataFrame(books).T / 2.0
    turnover = held.diff().abs().sum(axis=1).shift(1).iloc[2:]
    np.testing.assert_allclose(
        turnover.to_numpy(), frame["turnover"].reindex(turnover.index).to_numpy(), atol=1e-12
    )
