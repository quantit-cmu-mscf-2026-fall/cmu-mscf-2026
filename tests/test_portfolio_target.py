"""Combining strategy books under an allocation, and the chronological selection wrapper."""

from dataclasses import dataclass, field

import pandas as pd
import pytest

from capstone.execution.deploy import Refused
from capstone.strategy import portfolio
from capstone.strategy.portfolio import PortfolioSpec, combined_book, portfolio_target
from capstone.strategy.selected_portfolio import SelectedPortfolioSpec

AS_OF = pd.Timestamp("2020-01-03")
DATES = pd.bdate_range("2020-01-01", "2020-01-10")


@dataclass(frozen=True)
class Spec:
    id: str
    hypothesis_id: str = "h"
    side: str = "long_only"
    fields: tuple = field(default=("close",))


@dataclass
class FakePanel:
    universe: pd.DataFrame
    data_version: str = "v1"


class FakeSource:
    def __init__(self, last=AS_OF):
        self.last = last

    def panel(self, fields, as_of):
        index = DATES[DATES <= self.last]
        return FakePanel(pd.DataFrame(True, index=index, columns=[1, 2, 3]))


BOOKS = {
    "a": pd.Series({1: 0.6, 2: 0.4}),
    "b": pd.Series({2: 0.5, 3: 0.5}),
}


@pytest.fixture(autouse=True)
def fake_books(monkeypatch):
    monkeypatch.setattr(portfolio, "target_weights", lambda spec, panel, as_of: BOOKS[spec.id])
    monkeypatch.setattr(portfolio, "FrozenSnapshotSource", lambda panel: FakeSource())


def test_combined_book_averages_or_weights_by_shares():
    specs = [Spec("a"), Spec("b")]
    mean, details = combined_book(specs, FakeSource(), AS_OF)
    assert mean.to_dict() == pytest.approx({1: 0.3, 2: 0.45, 3: 0.25})
    assert mean.index.name == "permno" and mean.name == "weight"
    assert details["data_versions"] == ["v1"]
    shared, _ = combined_book(specs, FakeSource(), AS_OF, shares=[0.5, 0.25])
    assert shared.to_dict() == pytest.approx({1: 0.3, 2: 0.325, 3: 0.125})


def test_combined_book_refuses_a_source_that_returns_later_dates():
    with pytest.raises(Refused, match="after 2020-01-03"):
        combined_book([Spec("a")], FakeSource(last=AS_OF + pd.Timedelta(days=3)), AS_OF)


def test_portfolio_target_attributes_each_strategy_and_checks_the_allocation():
    specs = [Spec("b"), Spec("a")]
    allocated = {"weights": {"a": 0.5, "b": 0.5}, "side": "long_only"}
    book, details = portfolio_target(specs, object(), AS_OF, allocated)
    assert book.to_dict() == pytest.approx({1: 0.3, 2: 0.45, 3: 0.25})
    attribution = details["attribution"]["strategies"]
    assert attribution["a"] == pytest.approx({1: 0.3, 2: 0.2})
    for bad in (
        {"weights": {"a": 1.0}, "side": "long_only"},
        {"weights": {"a": 0.5, "b": 0.5}, "side": "long_short"},
        {"weights": {"a": 0.7, "b": 0.7}, "side": "long_only"},
        {"weights": {"a": -0.5, "b": 1.5}, "side": "long_only"},
    ):
        with pytest.raises(ValueError):
            portfolio_target(specs, object(), AS_OF, bad)
    capped = {"weights": {"a": 0.25, "b": 0.25}, "side": "long_only", "schema_version": 3}
    assert portfolio_target(specs, object(), AS_OF, capped)[0].sum() == pytest.approx(0.5)


def test_portfolio_identity_depends_on_membership_mode_and_side():
    returns = pd.DataFrame(columns=["a", "b"])
    base = PortfolioSpec((Spec("a"), Spec("b")), returns)
    assert base.id == PortfolioSpec((Spec("b"), Spec("a")), returns).id
    assert base.id != PortfolioSpec((Spec("a"),), returns).id
    assert base.id != PortfolioSpec((Spec("a"), Spec("b")), returns, mode="correlation_groups").id
    with pytest.raises(ValueError, match="separate"):
        _ = PortfolioSpec((Spec("a"), Spec("b", side="long_short")), returns).side


def test_selection_is_chronological_and_precedes_its_period():
    returns = pd.DataFrame(index=DATES, columns=["a", "b"], dtype=float)
    schedule = (
        {"decision_session": "2020-01-01", "start": "2020-01-02", "selected": ["a"]},
        {"decision_session": "2020-01-03", "start": "2020-01-06", "selected": ["a", "b"]},
    )
    spec = SelectedPortfolioSpec((Spec("a"), Spec("b")), returns, schedule=schedule)
    assert spec.members(pd.Timestamp("2019-12-31")) == ()
    assert [s.id for s in spec.members("2020-01-02")] == ["a"]
    assert [s.id for s in spec.members("2020-01-03")] == ["a", "b"]
    assert spec.weights(None, "2019-12-31").empty
    assert spec.id != SelectedPortfolioSpec((Spec("a"), Spec("b")), returns).id
    late = SelectedPortfolioSpec(
        (Spec("a"),),
        returns,
        schedule=({"decision_session": "2020-01-03", "start": "2020-01-03", "selected": ["a"]},),
    )
    with pytest.raises(ValueError, match="precede"):
        late.members(AS_OF)
    unknown = SelectedPortfolioSpec(
        (Spec("a"),),
        returns,
        schedule=({"decision_session": "2020-01-01", "start": "2020-01-02", "selected": ["z"]},),
    )
    with pytest.raises(ValueError, match="unknown"):
        unknown.members(AS_OF)
