"""Stage 4 label: `incubate`, next to the gate's `accepted`.

Off by default. With the registered default `Thresholds.promote = "accept"` stage 4 has
one outcome: a stage-3 survivor whose BH-adjusted holdout p (`s4_stat`) is <= `q4` is
accepted, and nothing else leaves the gate. Which families are accepted is the gate's
registered `decision_rule`; this module reads `accepted` and never recomputes it.
With `promote = "incubate"` there is a
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


def outcomes(card: pd.DataFrame, th: Thresholds) -> pd.Series:
    """`incubate` per family of a scorecard the gate has already decided.

    `card["accepted"]` is the gate's own decision (its registered `decision_rule`:
    the funnel's s1-s4, or the baseline's BH on s1) and is read, never recomputed.
    incubate: with promote="incubate", a stage-3 survivor the gate did NOT accept
    whose `s4_stat` (NaN when it never reached stage 4, which never incubates) is
    <= q4_incubate; all False otherwise.
    """
    if "accepted" not in card:
        raise ValueError("promotion reads the gate's `accepted` column; decide it first")
    accepted = card["accepted"].astype(bool)
    if th.promote != "incubate":
        return pd.Series(False, index=card.index)
    incubate = card["s3_pass"].astype(bool) & ~accepted & (card["s4_stat"] <= th.q4_incubate)
    return incubate.astype(bool)


def decide(card: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Set `incubate` on a scorecard; `accepted` and `expected_mean` stay the gate's.

    `expected_mean` is left alone: it is set only on accepted rows, so `notna()`
    keeps meaning accepted.
    """
    card["incubate"] = outcomes(card, th)
    return card
