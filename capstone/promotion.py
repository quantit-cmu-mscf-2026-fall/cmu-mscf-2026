"""Stage 4's outcomes: accepted, and (with promote="incubate") incubate.

Off by default. With the registered default `Thresholds.promote = "accept"` stage 4 has
one outcome: a stage-3 survivor whose BH-adjusted holdout p (`s4_stat`) is <= `q4` is
accepted, and nothing else leaves the gate. With `promote = "incubate"` there is a
second: a stage-3 survivor that fails `q4` but whose same statistic is <= `q4_incubate`
(registered with the config, q4 < q4_incubate < 1, before any look) is labelled
`incubate`. It is never accepted.

What happens to an incubating strategy (its horizon, its alpha, its capital cap, who
decides to promote or retire it) is a policy the team has not set; this module only
labels the outcome. `incubate` needs the "bh" stage 4: q4_incubate is a level on BH's
adjusted p, which the "dsr" gate does not have.
"""

from __future__ import annotations

import pandas as pd

from capstone.funnel import Thresholds

__all__ = ["decide", "outcomes"]


def outcomes(card: pd.DataFrame, th: Thresholds) -> tuple[pd.Series, pd.Series]:
    """(accepted, incubate) per family of a scorecard's stage columns.

    accepted: passed every gate (s1-s4). incubate: with promote="incubate", a
    stage-3 survivor not accepted whose `s4_stat` (NaN when it never reached
    stage 4, which never incubates) is <= q4_incubate; all False otherwise.
    """
    accepted = card["s1_pass"] & card["s2_pass"] & card["s3_pass"] & card["s4_pass"]
    if th.promote != "incubate":
        return accepted, pd.Series(False, index=card.index)
    incubate = card["s3_pass"] & ~accepted & (card["s4_stat"] <= th.q4_incubate)
    return accepted, incubate.astype(bool)


def decide(card: pd.DataFrame, th: Thresholds, expected_mean=None) -> pd.DataFrame:
    """Set `accepted` and `incubate` on a scorecard, and `expected_mean` for both.

    `expected_mean(card) -> Series` is the grading rule for the per-period mean a later
    live check should expect (the search and holdout means, haircut); it is passed in
    because grading is not part of this module. Without it `expected_mean` is left alone.
    """
    card["accepted"], card["incubate"] = outcomes(card, th)
    if expected_mean is not None:
        card["expected_mean"] = expected_mean(card).where(card["accepted"] | card["incubate"])
    return card
