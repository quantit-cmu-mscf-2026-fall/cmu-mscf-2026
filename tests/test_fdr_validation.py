import numpy as np

from capstone.fdr_validation import fdr_stepup


def test_bh_and_by_select_expected_strategies():
    pvals = np.array([0.003, 0.018, 0.027, 0.041, 0.120])

    bh_reject, bh_adjusted, bh_cutoff = fdr_stepup(pvals, q=0.05, method="bh")
    by_reject, by_adjusted, by_cutoff = fdr_stepup(pvals, q=0.05, method="by")

    assert bh_reject.tolist() == [True, True, True, False, False]
    assert by_reject.tolist() == [True, False, False, False, False]
    assert bh_cutoff == 0.027
    assert by_cutoff == 0.003
    assert (bh_adjusted <= 0.05).tolist() == bh_reject.tolist()
    assert (by_adjusted <= 0.05).tolist() == by_reject.tolist()


def test_bh_uses_largest_passing_rank():
    # Rank 2 fails its threshold, but rank 3 passes.
    pvals = [0.009, 0.025, 0.028, 0.20, 0.30]

    reject, _, cutoff = fdr_stepup(pvals, q=0.05, method="bh")

    assert reject.tolist() == [True, True, True, False, False]
    assert cutoff == 0.028
