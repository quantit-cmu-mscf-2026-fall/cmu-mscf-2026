"""The end-to-end driver's run config (`capstone.pipeline.config`): loading and refusals."""

from __future__ import annotations

from dataclasses import replace

import pytest

from capstone.factors.evaluation import TradingSpec
from capstone.factors.sources import FileSource
from capstone.pipeline.config import (
    DEFAULT_MAX_MODEL_CALLS,
    E2EConfig,
    IncubationSettings,
    Stage3Settings,
    load_config,
)
from capstone.pipeline.executor import AUTO
from capstone.pipeline.rows import REPO

SYNTHETIC = REPO / "config" / "e2e_synthetic.toml"
CRSP = REPO / "config" / "e2e_crsp.toml"


def _edited(tmp_path, old, new, source=SYNTHETIC):
    text = source.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "edited.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_the_synthetic_config_loads_with_its_own_papers_and_grid():
    config = load_config()
    assert config.run_tag == "e2e-run:synthetic-v1"
    assert config.papers == str(SYNTHETIC)
    assert [p["key"] for p in FileSource(config.papers).papers()] == [
        "synthetic:volume",
        "synthetic:reversal",
    ]
    assert config.scorer == TradingSpec(1, "none")
    assert len(config.specs) == 4  # holds [1, 5] x neutral ["none", "sector"]
    assert (config.paper_mode, config.concurrency) == ("sequential", AUTO)
    assert config.thresholds.stage1 == "rho"  # synthetic/CI configs only


def test_the_real_data_config_loads_the_pinned_model_and_the_bootstrap_stage_1():
    config = load_config(CRSP)
    assert config.factors.model == "claude-opus-5"  # the team's pin, never claude-opus-5-5
    assert config.thresholds.stage1 == "bootstrap"
    assert config.stage3.factors == "ff5_momentum"
    assert config.deepen_on_budget == "stop"
    assert config.search_end == "2020-12-31" and config.holdout_start == "2021-01-01"
    assert FileSource(config.papers).papers()  # the papers file resolves from the repository root


def test_the_real_data_config_states_the_registered_thresholds_and_generator():
    from capstone import gate
    from capstone.funnel import thresholds_dict

    registered = gate.load_registered(REPO / "scripts" / "gate_crsp_v1.json")
    config = load_config(CRSP)
    assert thresholds_dict(config.thresholds) == thresholds_dict(registered.thresholds)
    assert config.generator == registered.generator
    assert config.generator == "/".join(
        ["factor-trees", config.factors.model, config.factors.prompt_version]
        + [f"{config.deepen_selector}-deepen"]
    )
    assert config.n_boot == registered.n_boot
    assert (config.search_end, config.holdout_start, config.holdout_end) == (
        registered.search_end,
        registered.holdout_start,
        registered.holdout_end,
    )


def test_the_default_call_budget_is_300_and_a_run_can_set_its_own(tmp_path):
    assert DEFAULT_MAX_MODEL_CALLS == 300
    assert load_config(CRSP).max_model_calls == 300
    assert load_config().max_model_calls == 40
    text = SYNTHETIC.read_text(encoding="utf-8").replace("max_model_calls = 40\n", "")
    path = tmp_path / "no_budget.toml"
    path.write_text(text, encoding="utf-8")
    assert load_config(path).max_model_calls == 300


def test_an_unknown_section_or_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"unknown section \[extra\]"):
        load_config(_edited(tmp_path, "[trading]", "[extra]\nx = 1\n\n[trading]"))
    with pytest.raises(ValueError, match="unknown keys in .run."):
        load_config(_edited(tmp_path, "seed = 0\n# Data", "seed = 0\nmodle = 1\n# Data"))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('selector = "memory"', 'selector = "oracle"'),
        ('[stage3]\nfactors = "none"', '[stage3]\nfactors = "ff3"'),
        ("max_model_calls = 40", "max_model_calls = 0"),
        ('live_edge = "expected"', 'live_edge = "up"'),
        ('holdout_start = "2006-01-02"', 'holdout_start = "2005-06-01"'),
    ],
)
def test_a_bad_value_is_refused(tmp_path, old, new):
    with pytest.raises(ValueError, match="must be|need"):
        load_config(_edited(tmp_path, old, new))


def test_the_search_period_must_start_before_it_ends():
    config = load_config(CRSP)
    with pytest.raises(ValueError, match="search_start < search_end"):
        replace(config, search_start="2021-01-01")


def test_paper_mode_and_concurrency_are_read_from_the_run_section(tmp_path):
    path = _edited(tmp_path, "[run]\n", '[run]\npaper_mode = "isolated"\nconcurrency = 3\n')
    config = load_config(path)
    assert (config.paper_mode, config.concurrency) == ("isolated", 3)
    with pytest.raises(ValueError, match="paper_mode"):
        replace(config, paper_mode="other")
    with pytest.raises(ValueError, match="concurrency"):
        replace(config, concurrency=-1)


def test_dates_are_normalised_and_the_run_needs_a_spec_and_id():
    config = load_config()
    assert replace(config, search_end="2005-12-30 00:00").search_end == "2005-12-30"
    with pytest.raises(ValueError, match="at least one spec"):
        replace(config, specs=())
    with pytest.raises(ValueError, match="run_id"):
        replace(config, run_id="")
    assert isinstance(config, E2EConfig) and isinstance(config.specs, tuple)


def test_incubation_and_stage3_settings_check_themselves():
    assert IncubationSettings().confirm_horizon == 1260
    for bad in ({"confirm_alpha": 1.0}, {"confirm_horizon": 0}, {"block": 0}):
        with pytest.raises(ValueError):
            IncubationSettings(**bad)
    with pytest.raises(ValueError, match="confirm_accepted"):
        IncubationSettings(confirm_accepted=1)
    with pytest.raises(ValueError, match="stage3 factors"):
        Stage3Settings(factors="ff3")


def test_deepen_max_children_per_family_is_read_checked_and_kept_out_of_the_digest_unless_set(
    tmp_path,
):
    from capstone.pipeline.search_identity import config_digest

    config = load_config()
    assert config.deepen_max_children_per_family is None
    path = _edited(tmp_path, "rounds = 2\n", "rounds = 2\nmax_children_per_family = 3\n")
    capped = load_config(path)
    assert capped.deepen_max_children_per_family == 3
    assert config_digest(capped) != config_digest(config)
    unset = replace(capped, deepen_max_children_per_family=None, papers=config.papers)
    assert config_digest(unset) == config_digest(config)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_children_per_family"):
            replace(config, deepen_max_children_per_family=bad)


def test_the_data_tag_names_the_data_and_period_not_the_run():
    config = load_config()
    assert config.data_tag == "e2e-data:synthetic-crsp:start:2005-12-30"
    other = replace(config, run_id="another", search_start="2001-01-02")
    assert other.data_tag == "e2e-data:synthetic-crsp:2001-01-02:2005-12-30"
    assert config.run_tag != other.run_tag
