"""A holdout look is spent on the data, not on the configuration.

The holdout is worth something only because the search had no chance to
flatter it, which is a property of the dataset. So the budget is one look per
dataset, and the question each test here answers is: *does changing this field
buy another look?*

Before `GateConfig.look_key`, the budget was keyed on `config_hash`, so every
hashed field bought one. Measured on signal-free data at `daccc5d`: four looks
under one dataset from nothing but `seed` 0, 1, 2, 3 — two of which accepted a
family. `test_a_seed_change_does_not_buy_a_look` is that case.

`NUISANCE` below is the enumeration: every hashed field that must not open the
holdout again, with a changed value for each. Add a hashed field to
`GateConfig` and `test_every_hashed_field_is_classified` fails until it is
listed here or in `OPENS_A_LOOK`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
from gate_helpers import (
    AUTHORIZER,
    TRIAL_NAME,
    Market,
    config_for,
    gate_once,
    log_trials,
    market,
    run,
)

from capstone import runlog
from capstone.funnel import Thresholds
from capstone.gate import (
    DecisionRule,
    GateConfig,
    HoldoutAlreadyOpened,
    holdout_looks,
    register,
)
from capstone.scorecard import TrialScope

# The funnel rule as the base: `run_gate` refuses the baseline rule unless
# stage1 is "bootstrap", so a baseline base config cannot vary stage1.
RULE = DecisionRule("funnel")

#: Hashed fields that must NOT open the holdout a second time, and a different
#: value for each. Grouped by why, which is the argument for excluding them.
NUISANCE: dict[str, dict] = {
    # Computational: cannot change which candidates are selected.
    "seed": {"seed": 7},
    "n_boot": {"n_boot": 399},
    # The pre-registered rules. Changing one after a disappointing holdout is
    # the move the funnel exists to prevent; it must not be rewarded.
    "thresholds.q1": {"thresholds": Thresholds(stage1="bootstrap", q1=0.20)},
    "thresholds.stage1": {"thresholds": Thresholds(stage1="rho")},
    "thresholds.alpha2": {"thresholds": Thresholds(stage1="bootstrap", alpha2=0.05)},
    "thresholds.final": {"thresholds": Thresholds(stage1="bootstrap", final="dsr")},
    "thresholds.q4": {"thresholds": Thresholds(stage1="bootstrap", q4=0.50)},
    "thresholds.d4": {"thresholds": Thresholds(stage1="bootstrap", d4=0.99)},
    "thresholds.final_trials": {
        "thresholds": Thresholds(stage1="bootstrap", final_trials="searched")
    },
    "decision_rule.rule": {"decision_rule": DecisionRule("baseline", q=0.05)},
    "decision_rule.q": {"decision_rule": DecisionRule("baseline", q=0.20)},
    # decision_rule.method takes only "bh" today, so it has no second value to vary.
    "min_holdout_coverage": {"min_holdout_coverage": 0.9},
    # Re-slicing one dataset is not new evidence; new rows change data_version.
    "holdout_end": None,  # these three are built per-market in `changed`
    "holdout_start": None,
    "search_end": None,
    # Free-text provenance; `_checked_stage3` refuses a non-None digest today,
    # but the look check runs first, so the refusal is still the look budget.
    "factors_digest": {"factors_digest": "a" * 64},
    # Narrowing the scope shrinks the trial count as well as minting a look.
    "trial_scope.name": {"trial_scope": TrialScope(name="something-else")},
    "trial_scope.exclude_tags": {"trial_scope": TrialScope(name=TRIAL_NAME, exclude_tags=("x",))},
    "trial_scope.include_tags": {"trial_scope": TrialScope(name=TRIAL_NAME, include_tags=("y",))},
    # Free-text provenance the gate cannot verify.
    "generator": {"generator": "a-different-generator"},
}

#: The only two fields that may open a look, and why.
OPENS_A_LOOK = {
    "data_version": "genuinely new data is genuinely new evidence",
    "holdout_round": "a declared second look at the same data",
}


def changed(config: GateConfig, field: str) -> GateConfig:
    """`config` with one field changed, for the `NUISANCE` case named `field`."""
    if field in ("holdout_end", "holdout_start", "search_end"):
        shift = {"holdout_end": 30, "holdout_start": 1, "search_end": -1}[field]
        moved = pd.Timestamp(getattr(config, field)) + pd.Timedelta(days=shift)
        return dataclasses.replace(config, **{field: moved.date().isoformat()})
    return dataclasses.replace(config, **NUISANCE[field])


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


@pytest.fixture
def spent(ledger):
    """A market whose holdout has been opened exactly once, and its config."""
    m = market(0)
    gate_once(m, decision_rule=RULE)
    assert len(holdout_looks()) == 1
    return m, config_for(m, decision_rule=RULE)


def open_holdout(m, config) -> None:
    """Register `config` and run the gate on `m`, as a second look would."""
    register(config)
    run(config, m)


# --- the enumeration -------------------------------------------------------


@pytest.mark.parametrize("field", sorted(NUISANCE))
def test_a_nuisance_change_does_not_buy_a_look(spent, field):
    """Every hashed field but the two in `OPENS_A_LOOK`, one test each."""
    m, config = spent
    with pytest.raises(HoldoutAlreadyOpened, match="spent on the data"):
        open_holdout(m, changed(config, field))
    assert len(holdout_looks()) == 1


def test_a_seed_change_does_not_buy_a_look(spent):
    """The measured failure: four looks from seeds 0-3, two accepting a family.

    Kept as its own test beside the enumeration because it is the case that
    was observed rather than reasoned about, and the one most likely to recur
    innocently -- re-running with another seed to check a result is stable is
    an ordinary thing to do.
    """
    m, config = spent
    for seed in (1, 2, 3):
        with pytest.raises(HoldoutAlreadyOpened):
            open_holdout(m, dataclasses.replace(config, seed=seed))
    assert len(holdout_looks()) == 1


def test_the_same_config_still_does_not_buy_a_look(spent):
    m, config = spent
    with pytest.raises(HoldoutAlreadyOpened):
        open_holdout(m, config)
    assert len(holdout_looks()) == 1


def test_a_fresh_candidate_set_does_not_buy_a_look(spent):
    """A screen that passed nothing must not be retried until something does."""
    m, config = spent
    other = market(11)
    with pytest.raises(HoldoutAlreadyOpened):
        open_holdout(
            Market(
                other.returns,
                other.family,
                other.truth,
                m.search_end,
                m.holdout_start,
                m.holdout_end,
            ),
            config,
        )
    assert len(holdout_looks()) == 1


# --- what may open a look --------------------------------------------------


def test_new_data_opens_a_look(spent):
    m, config = spent
    fresh = dataclasses.replace(config, data_version="synthetic-refreshed")
    open_holdout(m, fresh)
    keys = {look["look_key"] for look in holdout_looks()}
    assert keys == {config.look_key, fresh.look_key}


def test_a_declared_round_opens_a_look(spent):
    """The sanctioned second look: deliberate, and visible in the ledger."""
    m, config = spent
    second = dataclasses.replace(config, holdout_round=1)
    open_holdout(m, second)
    looks = holdout_looks()
    assert [look["holdout_round"] for look in looks] == [0, 1]
    assert looks[1]["look_key"].endswith("#1")


def test_a_declared_round_is_itself_spent_once(spent):
    m, config = spent
    second = dataclasses.replace(config, holdout_round=1)
    open_holdout(m, second)
    with pytest.raises(HoldoutAlreadyOpened, match="round 1"):
        open_holdout(m, dataclasses.replace(second, seed=99))
    assert len(holdout_looks()) == 2


# --- the key, the audit, and the field inventory ---------------------------


def test_the_look_key_is_the_data_and_the_round_and_nothing_else():
    m = market(0)
    config = config_for(m, decision_rule=RULE)
    assert config.look_key == f"{config.data_version}#0"
    for field in NUISANCE:
        assert changed(config, field).look_key == config.look_key, field


def test_every_hashed_field_is_classified():
    """A new hashed field must be put in one bucket or the other, on purpose.

    Without this, adding a field to `GateConfig` silently adds another way to
    buy a look -- which is exactly how the original hole appeared.
    """
    hashed = set(
        GateConfig(
            search_end="2020-12-31",
            holdout_start="2021-01-01",
            holdout_end="2025-12-31",
            data_version="synthetic",
            generator="g",
        ).to_dict()
    )
    hashed |= {"holdout_round"}  # omitted from content at its default
    classified = {name.split(".")[0] for name in NUISANCE} | set(OPENS_A_LOOK)
    assert hashed == classified, (
        f"unclassified hashed field(s): {sorted(hashed - classified)}; "
        f"unknown name(s): {sorted(classified - hashed)}"
    )


def test_the_refusal_says_what_to_do_instead(spent):
    m, config = spent
    with pytest.raises(HoldoutAlreadyOpened) as exc:
        open_holdout(m, dataclasses.replace(config, seed=5))
    message = str(exc.value)
    assert config.data_version in message
    assert AUTHORIZER in message
    assert "holdout_round 1" in message


def test_an_earlier_look_without_a_key_still_counts(ledger):
    """A look recorded before `look_key` existed is round 0 of its dataset.

    The budget has to be retroactive, or the change would hand everyone a
    fresh look on data already spent.
    """
    m = market(0)
    log_trials(m.returns.shape[1])
    config = config_for(m, decision_rule=RULE)
    runlog.log_run(
        "gate-holdout-look",
        params={"config_hash": "old" * 12, "data_version": config.data_version},
        tags=["gate", "synthetic"],
    )
    assert holdout_looks()[0]["look_key"] == f"{config.data_version}#0"
    with pytest.raises(HoldoutAlreadyOpened):
        open_holdout(m, config)


def test_holdout_looks_reports_each_look_for_audit(spent):
    m, config = spent
    open_holdout(m, dataclasses.replace(config, holdout_round=1))
    looks = holdout_looks()
    assert len(looks) == 2
    for look in looks:
        assert look["authorizer"] == AUTHORIZER
        assert look["data_version"] == config.data_version
        assert look["config_hash"] and look["ts_utc"]


def test_a_negative_round_is_refused():
    with pytest.raises(ValueError, match="holdout_round must be 0 or more"):
        GateConfig(
            search_end="2020-12-31",
            holdout_start="2021-01-01",
            holdout_end="2025-12-31",
            data_version="synthetic",
            generator="g",
            holdout_round=-1,
        )


def test_a_bool_round_is_refused():
    with pytest.raises(TypeError, match="holdout_round must be an integer"):
        GateConfig(
            search_end="2020-12-31",
            holdout_start="2021-01-01",
            holdout_end="2025-12-31",
            data_version="synthetic",
            generator="g",
            holdout_round=True,
        )


def test_the_registered_crsp_v1_hash_is_unchanged():
    """`holdout_round` is omitted from the content at its default.

    `docs/validation/framework.md` records crsp-v1's hash, and the ledger has
    a registration under it. A new field that moved the hash would retire a
    registered configuration.
    """
    data = json.loads(Path("scripts/gate_crsp_v1.json").read_text())["gate_config"]
    config = GateConfig.from_dict(data)
    assert config.holdout_round == 0
    assert "holdout_round" not in config.to_dict()
    assert config.config_hash == (
        "5f590f13f63e94f33fbdf960089c1fc01881a6fedf2437d8afbc3a0c396af7da"
    )
