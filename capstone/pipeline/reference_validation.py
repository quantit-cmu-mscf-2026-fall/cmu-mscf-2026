"""Shared statistical methods for reference returns, with no implicit holdout access."""

from __future__ import annotations

import pandas as pd

from capstone import runlog
from capstone.evaluate import bh_adjusted, by_adjusted, holm_adjusted
from capstone.reference import generator_name
from capstone.search_fdr import bootstrap_family_pvalues

DISCOVERY_END = pd.Timestamp("2020-12-31")


def discovery_evidence(matrix, specs):
    """Family p-values on discovery-period returns; never reads or opens the holdout.

    The call is logged before the statistics are computed, so the run counts
    as a trial whatever it finds.
    """
    if matrix.index.max() > DISCOVERY_END:
        raise ValueError("discovery evidence cannot read the sealed holdout")
    family = pd.Series({s.id: s.claim_id for s in specs}).reindex(matrix.columns)
    if family.isna().any():
        raise ValueError("each return column must have a research family")
    runlog.log_run(
        "reference-discovery-screen",
        params={
            "generator": generator_name(),
            "variants": list(matrix),
            "data_version": specs[0].data_version,
            "bootstrap_draws": 1999,
            "resolution_q": 0.05,
        },
        seed=404,
        tags=["reference-v2", "gate"],
        notes="Discovery evidence only; underlying replays are logged trials.",
    )
    result = bootstrap_family_pvalues(matrix, family, n_boot=1999, seed=404, q=0.05)
    result["holm_adjusted"] = holm_adjusted(result.pvalue)
    result["by_adjusted"] = by_adjusted(result.pvalue)
    result["bh_adjusted"] = bh_adjusted(result.pvalue)
    return {
        "generator": generator_name(),
        "formally_admitted": False,
        "holdout_opened": False,
        "selection_policy_qualified": False,
        "families": [{"family": str(name), **row.to_dict()} for name, row in result.iterrows()],
    }
