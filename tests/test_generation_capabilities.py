import json

from test_generation_freeze import setup_input

from capstone.factors import tree
from capstone.factors.generation import PROMPT_VERSION


def test_frozen_generation_explains_real_grammar_units_and_missing_input_boundary(tmp_path):
    _, frozen = setup_input(tmp_path, text="A price/volume research conjecture.")
    prompt = json.loads(frozen.prompt_json)
    assert PROMPT_VERSION == "source-grounded-generation-v5"
    # Fundamental fields (book, sales, ...) join tree.FIELDS only once fundamentals ship.
    assert set(tree.FIELDS) <= set(prompt["input_types"])
    assert prompt["input_types"]["cap"]["units"] == "USD^1"
    assert prompt["input_types"]["book"]["units"] == "USD^1"
    for name in (*tree.FUNCS, *tree.BINARY_WINDOW_FUNCS, *tree.BINARY_PARAMETER_FUNCS):
        assert name in prompt["expression_grammar"]
    assert "never omit" in prompt["input_requirements_notice"]
    assert "does not qualify" in prompt["input_requirements_notice"]
    assert any("ema:" in value for value in prompt["unsupported_constructions"])
    unsupported = " ".join(prompt["unsupported_constructions"])
    for capability in ("event portfolios", "overlapping holding cohorts", "cross-entity joins"):
        assert capability in unsupported
    assert "calendar" not in unsupported
    assert "90 days" not in prompt["expression_grammar"]
