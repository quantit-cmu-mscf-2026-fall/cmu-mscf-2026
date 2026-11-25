"""The universe file: parsing, both-way lookups, and refusing the template."""

from __future__ import annotations

from pathlib import Path

import pytest

from capstone.execution.universe import UniverseError, parse_universe

REPO = Path(__file__).resolve().parents[1]


def universe_text(permnos=(10000, 10001), version="2026-06-01", venue_symbols=None) -> str:
    """A synthetic universe file; `venue_symbols` maps a permno to a spelling
    that differs from its symbol (default: the same)."""
    venue = dict(venue_symbols or {})
    rows = "".join(f"{p},T{p - 10000:02d},{venue.get(p, f'T{p - 10000:02d}')}\n" for p in permnos)
    return f"# synthetic universe\nversion = {version}\npermno,symbol,venue_symbol\n{rows}"


def test_the_universe_maps_permnos_to_symbols_both_ways():
    universe = parse_universe(universe_text(permnos=[10000, 10001]))
    assert universe.version == "2026-06-01"
    assert universe.symbol(10001) == "T01" and universe.venue_symbol(10001) == "T01"
    assert universe.permno("T00") == 10000
    with pytest.raises(UniverseError, match="10002"):
        universe.venue_symbol(10002)
    with pytest.raises(UniverseError, match="ZZZ"):
        universe.permno("ZZZ")


@pytest.mark.parametrize(
    "text",
    [
        "permno,symbol,venue_symbol\n1,A,A\n",
        "version = v1\npermno,symbol,venue_symbol\n1,A,A\n1,B,B\n",
        "version = v1\npermno,symbol,venue_symbol\n1,A,A\n2,A,B\n",
        "version = v1\npermno,symbol,venue_symbol\n1,a b,A\n",
        "version = v1\npermno,symbol,venue_symbol\nx,A,A\n",
        "version = v1\nsymbol,permno,venue_symbol\nA,1,A\n",
        "version = v1\npermno,symbol\n1,A\n",
        "version = v1\npermno,symbol,venue_symbol\n1,A\n",
        "version = v1\npermno,symbol,venue_symbol\n1,A,brk b\n",
        "version = v1\npermno,symbol,venue_symbol\n1,A,BRK  B\n",
        "version = v1\npermno,symbol,venue_symbol\n1,A,X\n2,B,X\n",
    ],
)
def test_a_malformed_universe_file_is_refused(text):
    with pytest.raises(UniverseError):
        parse_universe(text)


def test_the_universe_keeps_the_usual_ticker_and_the_venue_spelling():
    u = parse_universe(
        "version = v1\npermno,symbol,venue_symbol\n83443,BRK.B,BRK B\n14593,AAPL,AAPL\n"
    )
    assert u.symbol(83443) == "BRK.B" and u.venue_symbol(83443) == "BRK B"
    assert u.permno("BRK B") == 83443
    assert u.venue_symbols == ("AAPL", "BRK B")
    with pytest.raises(UniverseError, match=r"BRK\.B"):
        u.permno("BRK.B")
    with pytest.raises(UniverseError, match="99999"):
        u.venue_symbol(99999)


def test_the_committed_universe_is_a_template_and_claims_no_membership():
    text = (REPO / "deploy" / "universe.csv").read_text(encoding="utf-8")
    u = parse_universe(text)
    assert u.is_template and u.listings == ()
    assert "TEMPLATE" in text and "not" in text.lower()


def test_a_filled_universe_is_not_a_template():
    assert not parse_universe(universe_text()).is_template
    assert parse_universe(universe_text(version="template-x")).is_template
