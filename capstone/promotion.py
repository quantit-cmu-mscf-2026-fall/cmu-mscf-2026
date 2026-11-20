"""Stage 4 label: `incubate`, next to the gate's `accepted`.

Opt-in: it runs only when a caller passes `decide` a level. Stage 4 has one outcome: a
stage-3 survivor whose BH-adjusted holdout p (`s4_stat`) is <= `q4` is accepted, and
nothing else leaves the gate. Which families are accepted is the gate's registered
`decision_rule`; this module reads `accepted` and never recomputes it. `decide` adds a
second label: a stage-3 survivor the gate did not accept, whose same statistic is
<= `q4_incubate` (q4 < q4_incubate < 1, checked against the card's GateConfig), is
labelled `incubate`. It is never accepted.

What happens to an incubating strategy (its horizon, its alpha, its capital cap, who
decides to promote or retire it) is a policy the team has not set; this module only
labels the outcome. `decide` refuses a config that is not final="bh": q4_incubate is a level on BH's
adjusted p, which the "dsr" gate does not have.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

from capstone.gate import GateConfig

__all__ = ["decide", "outcomes"]


def outcomes(card: pd.DataFrame, config: GateConfig, q4_incubate: float) -> pd.Series:
    """`incubate` per family of a scorecard the gate has already decided.

    `config` is the GateConfig that produced the card; the level is checked against it,
    so it cannot be a threshold for some other gate. Refused unless `config` decides
    stage 4 with final == "bh" (s4_stat is then BH's adjusted p, lower is better) and
    q4 < q4_incubate < 1.

    `card["accepted"]` is the gate's own decision (its registered `decision_rule`:
    the funnel's s1-s4, or the baseline's BH on s1) and is read, never recomputed.
    incubate: a stage-3 survivor the gate did NOT accept whose `s4_stat` (NaN when it
    never reached stage 4, which never incubates) is <= q4_incubate.
    """
    th = config.thresholds
    if th.final != "bh":
        raise ValueError(
            f'incubate needs final == "bh" (got {th.final!r}): q4_incubate is a level on '
            "BH's adjusted p, and the other gates' s4_stat does not read that way"
        )
    if isinstance(q4_incubate, bool | np.bool_) or not isinstance(q4_incubate, numbers.Real):
        raise ValueError(f"q4_incubate must be a real number, got {q4_incubate!r}")
    if not th.q4 < q4_incubate < 1.0:
        raise ValueError(
            f"q4_incubate must satisfy q4 ({th.q4}) < q4_incubate < 1, got {q4_incubate}"
        )
    if "accepted" not in card:
        raise ValueError("promotion reads the gate's `accepted` column; decide it first")
    accepted = card["accepted"].astype(bool)
    incubate = card["s3_pass"].astype(bool) & ~accepted & (card["s4_stat"] <= q4_incubate)
    return incubate.astype(bool)


def decide(card: pd.DataFrame, config: GateConfig, q4_incubate: float) -> pd.DataFrame:
    """Set `incubate` on a scorecard; `accepted` and `expected_mean` stay the gate's.

    `expected_mean` is left alone: it is set only on accepted rows, so `notna()`
    keeps meaning accepted.
    """
    card["incubate"] = outcomes(card, config, q4_incubate)
    return card
