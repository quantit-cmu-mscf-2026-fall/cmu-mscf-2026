import pytest

from capstone.strategy.research import claim_for, contracts, method_digest


def test_all_fifteen_claims_have_explicit_dispositions():
    catalog = contracts()
    assert len(catalog["claims"]) == 15
    assert len(catalog["papers"]) == 5
    assert sum(c["status"] == "executable" for c in catalog["claims"]) == 6
    assert sum(c["status"] == "diagnostic" for c in catalog["claims"]) == 5
    assert sum(c["status"] == "blocked" for c in catalog["claims"]) == 4
    for claim in catalog["claims"]:
        assert len(method_digest(claim["id"])) == 64
        if claim["status"] != "executable":
            with pytest.raises(ValueError, match="claim is"):
                claim_for(claim["id"], executable=True)


def test_january_is_a_cohort_diagnostic_and_not_a_continuous_factor():
    claim = claim_for("2c119c2ef01bbb47")
    assert claim["status"] == "diagnostic"
    assert claim["method"] == "reversal_36m"
    assert "January-only" in claim["check"]
