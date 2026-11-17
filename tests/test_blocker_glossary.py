"""The blocker glossary: every blocker maps to a non-empty, plain-language way to clear it."""

from __future__ import annotations

import re

import pytest

from capstone.pipeline.blocker_glossary import CLEARED_BY, UNMAPPED, cleared_by

# Internal plan labels (unit numbers, decision numbers, broker product names) do not belong
# in the text a status report shows.
PRIVATE_LABEL = re.compile(r"\b(?:U|D|R)\d+[a-z]?\b|\bIBKR\b|docs/|\.md\b")


def test_every_row_is_a_non_empty_string_under_a_non_empty_code():
    assert CLEARED_BY
    for code, text in CLEARED_BY.items():
        assert isinstance(code, str) and code.strip() == code and code
        assert isinstance(text, str) and text.strip() == text and text


def test_no_row_names_an_internal_plan_label_or_a_document():
    offenders = {code: text for code, text in CLEARED_BY.items() if PRIVATE_LABEL.search(text)}
    assert not offenders
    assert not PRIVATE_LABEL.search(UNMAPPED)


def test_an_exact_code_maps_to_its_own_row():
    for code, text in CLEARED_BY.items():
        assert cleared_by(code) == text


def test_a_message_containing_a_code_maps_to_the_longest_row_it_contains():
    assert cleared_by("failed: price_missing for AAA") == CLEARED_BY["price_missing"]
    # "missing_input" and "missing input" are different rows; a message holds only one of them.
    assert cleared_by("bad missing_input here") == CLEARED_BY["missing_input"]
    assert cleared_by("x missing input y") == CLEARED_BY["missing input"]
    # Two rows in one message: the longer row wins.
    both = "HoldoutSealed after benchmark_control_missing"
    assert cleared_by(both) == CLEARED_BY["benchmark_control_missing"]


def test_an_unknown_blocker_is_unmapped_rather_than_guessed():
    assert cleared_by("nothing we know") == UNMAPPED
    assert cleared_by(None) == UNMAPPED
    assert UNMAPPED not in CLEARED_BY.values()


@pytest.mark.parametrize("code", ["paper_account_verified", "funds_verified", "data_verified"])
def test_broker_checks_share_one_row_and_never_name_a_product(code):
    assert CLEARED_BY[code] == "broker acceptance"
