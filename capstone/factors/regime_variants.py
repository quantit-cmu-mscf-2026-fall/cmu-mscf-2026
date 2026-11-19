"""Where a regime variant is counted, and what it is spanned by.

A regime variant is a candidate whose tree applies a market-regime operator
(`tree.REGIME_FUNCS`: `volscale`, `bear_gate`). Its base formula is the tree with
those operators removed (`tree.regime_base`):

- every regime variant is one trial, counted in its base formula's family: when the
  base formula is itself a candidate, the variant takes that candidate's family,
  whatever hypothesis proposed it; otherwise it keeps its own;
- a regime variant should be spanned with its unscaled base factor as well, so a
  regime tweak of a known factor cannot pass as new: `regime_bases` names, for each
  variant, the candidate whose returns are that base factor. Nothing calls it for
  that yet: the stage-3 spanning is not wired in.

Pure functions on candidate -> expression and candidate -> family; nothing here
reads data, calls a model or writes to the ledger.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from capstone.factors.tree import identity_key, parse, regime_base, uses_regime

__all__ = ["regime_bases", "regime_families", "trials_per_family"]


def _trees(expressions: pd.Series) -> dict:
    return {c: parse(str(e), regime_operators=True) for c, e in expressions.items()}


def regime_bases(expressions: pd.Series) -> dict:
    """Regime variant -> the candidate that is its base formula (None if no candidate is)."""
    trees = _trees(expressions)
    by_key: dict[str, object] = {}
    for candidate in sorted(trees, key=str):  # the first candidate of a key, deterministically
        if not uses_regime(trees[candidate]):
            by_key.setdefault(identity_key(trees[candidate]), candidate)
    return {
        c: by_key.get(identity_key(regime_base(t)))
        for c, t in sorted(trees.items(), key=lambda kv: str(kv[0]))
        if uses_regime(t)
    }


def regime_families(expressions: pd.Series, family: pd.Series) -> pd.Series:
    """Each candidate's family, a regime variant's moved to its base formula's."""
    out = family.copy()
    for variant, base in regime_bases(expressions).items():
        if base is not None:
            out[variant] = family[base]
    return out


def trials_per_family(expressions: pd.Series, family: pd.Series) -> dict:
    """Trials charged per family: every candidate one, regime variants included."""
    return dict(sorted(Counter(regime_families(expressions, family).astype(str)).items()))
