import copy

import pytest

from capstone.strategy import historical_benchmark as hb
from capstone.strategy.reference import BenchmarkSpec


def make(**kw):
    args = {
        "kind": "cap_weight_universe_proxy",
        "data_version": "synthetic-v1",
        "anchor": "2000-01-31",
    }
    return hb.HistoricalBenchmarkSpec(**(args | kw))


def test_record_round_trip_preserves_identity():
    spec = make()
    again = hb.HistoricalBenchmarkSpec.from_record(spec.to_record())
    assert again == spec and again.id == spec.id and len(spec.id) == 16


def test_legacy_benchmark_identity_and_no_generic_leaves():
    spec = make()
    assert spec.legacy == BenchmarkSpec(spec.kind, "synthetic-v1", "2000-01-31", "long_only")
    assert spec.legacy_benchmark_id == spec.legacy.id
    assert spec.specs == ()


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": "made_up"},
        {"side": "short_only"},
        {"kind": "equal_weight_universe_proxy", "side": "long_short"},
        {"version": 3},
        {"version": 2},  # eligibility rule belongs to v1
        {"phase_rule": "other"},
        {"data_version": ""},
        {"implementation": "stale"},
    ],
)
def test_invalid_specs_are_refused(kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_version_two_pairs_with_its_own_eligibility_rule():
    spec = make(version=2, eligibility_rule=hb.ELIGIBILITY_RULES[2])
    assert hb.HistoricalBenchmarkSpec.from_record(spec.to_record()) == spec


def test_momentum_control_may_be_long_short():
    assert make(kind="known_momentum_12_1", side="long_short").side == "long_short"


def test_tampered_records_are_rejected():
    record = make().to_record()
    changed = copy.deepcopy(record)
    changed["payload"]["data_version"] = "other"
    with pytest.raises(ValueError):
        hb.HistoricalBenchmarkSpec.from_record(changed)
    wrong_id = copy.deepcopy(record) | {"id": "0" * 16}
    with pytest.raises(ValueError):
        hb.HistoricalBenchmarkSpec.from_record(wrong_id)
    for bad in (None, {}, record | {"schema_version": 2}, record | {"extra": 1}):
        with pytest.raises(ValueError):
            hb.HistoricalBenchmarkSpec.from_record(bad)


def test_implementation_digest_is_deterministic_and_pinned_in_the_record():
    assert hb.implementation_digest() == hb.implementation_digest()
    assert make().to_record()["payload"]["implementation"] == hb.implementation_digest()
