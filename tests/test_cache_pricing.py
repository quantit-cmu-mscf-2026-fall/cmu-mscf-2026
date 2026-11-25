"""The cost estimate prices prompt-cache tokens.

With the system text marked cacheable, most input arrives as cache reads
and the first call of each prefix as a cache write. The estimate priced neither. A cache
write now costs 1.25x the model's input price and a cache read 0.1x. No model calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capstone.pipeline.usage import CACHE_READ, CACHE_WRITE, UsageRecorder, cost_estimate

PRICES = {"m": (4.0, 20.0)}


def _record(**tokens) -> dict:
    base = dict(
        model="m",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return base | tokens


def test_the_multipliers_are_the_list_ones():
    assert (CACHE_WRITE, CACHE_READ) == (1.25, 0.1)


def test_a_cache_write_costs_one_and_a_quarter_input():
    out = cost_estimate([_record(cache_creation_input_tokens=1_000_000)], prices=PRICES)
    assert out["usd"] == pytest.approx(5.0)


def test_a_cache_read_costs_a_tenth_of_input():
    out = cost_estimate([_record(cache_read_input_tokens=1_000_000)], prices=PRICES)
    assert out["usd"] == pytest.approx(0.4)


def test_all_four_token_kinds_add_up_and_are_kept_per_model():
    records = [
        _record(input_tokens=500_000, output_tokens=100_000, cache_creation_input_tokens=200_000),
        _record(input_tokens=500_000, cache_read_input_tokens=2_000_000),
    ]
    out = cost_estimate(records, prices=PRICES)
    # input 1M x 4 + output 0.1M x 20 + write 0.2M x 5 + read 2M x 0.4
    assert out["usd"] == pytest.approx(4.0 + 2.0 + 1.0 + 0.8)
    assert out["by_model"]["m"] == dict(
        calls=2,
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_creation_input_tokens=200_000,
        cache_read_input_tokens=2_000_000,
    )


def test_records_without_cache_fields_are_priced_as_before():
    records = [dict(model="m", input_tokens=1_000_000, output_tokens=1_000_000)]
    out = cost_estimate(records, prices=PRICES)
    assert out["usd"] == 24.0
    assert out["by_model"]["m"] == dict(calls=1, input_tokens=1_000_000, output_tokens=1_000_000)


def test_the_recorder_feeds_the_cache_tokens_to_the_estimate():
    usage = SimpleNamespace(
        input_tokens=100_000,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1_000_000,
    )
    response = SimpleNamespace(usage=usage)
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    recorder = UsageRecorder(fake)
    recorder.create(model="m", messages=[])
    out = cost_estimate(recorder.records, prices=PRICES)
    assert out["usd"] == pytest.approx(0.4 + 0.4)  # 0.1M input x 4, 1M read x 0.4


def test_a_model_without_a_price_is_named_and_not_priced():
    records = [
        _record(input_tokens=1_000_000, cache_read_input_tokens=1_000_000),
        _record(model="x", input_tokens=10),
    ]
    out = cost_estimate(records, prices=PRICES)
    assert out["unpriced"] == ["x"]
    assert out["usd"] == pytest.approx(4.0 + 0.4)
    assert out["by_model"]["x"]["calls"] == 1
