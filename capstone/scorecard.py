"""The acceptance gate's scorecard and report: what was decided, and why.

`docs/validation/framework.md` ("The scorecard"): one row per hypothesis
family that reached stage 1, so a reader can see where each one stopped, and
a header for the batch. The gate fills it in; this module only fixes its
shape and writes and reads the gate report a deploy PR links to.

Row (index `family`), in `SCORECARD_COLUMNS` order:

- identity: `best` (the variant carried forward), `k` (variants searched),
  `generator`;
- per stage, the gate's statistic, threshold and pass flag:
  stage 1 `s1_stat` = BH-adjusted family p-value across families, against
  `q1`; stage 2 `s2_stat` = one-sided HAC p of `best`, against `alpha2`
  (1.0 = no gate); stage 3 (spanning) `s3_stat` against `s3_threshold`,
  with `s3_alpha` (annualised intercept) and `s3_t` (its HAC t) when
  `s3_measured` is True; until the spanning test exists `s3_measured` is
  False, those four numbers are NaN and `s3_pass` repeats stage 2; stage 4
  `s4_stat` = BH-adjusted holdout p across the stage-3 survivors against
  `q4` (or the deflated Sharpe against `d4`), NaN for families that never
  reached it;
- `accepted`: passed every gate;
- graded scores, reported and never used to decide: `p1` (the raw family
  p-value stage 1 adjusted), `by`, `holm` (adjusted across families),
  `lfdr`, `sharpe`, `psr`, `min_trl_years` (search period),
  `haircut_sharpe` (with the ledger trial count), `ho_sharpe`, `ho_p`,
  `ho_psr` (holdout, stage-4 families only);
- what a later live check needs: `is_mean` (search-period mean of `best`),
  `ho_mean`, `expected_mean` (the per-period mean to expect, accepted
  families) and `scale` (search-period standard deviation of `best`).

Stage 3's factor loadings for each measured family's `best` are
`GateResult.loadings` (family x factor; empty when stage 3 wasn't measured).

The report is a JSON file (header, accepted families, expected means,
loadings) with the scorecard as a CSV beside it, same stem. NaN is written as
null, so the JSON is standard.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone.funnel import Thresholds, thresholds_dict

SCORECARD_COLUMNS: tuple[str, ...] = (
    "best",
    "k",
    "generator",
    "s1_stat",
    "s1_threshold",
    "s1_pass",
    "s2_stat",
    "s2_threshold",
    "s2_pass",
    "s3_stat",
    "s3_threshold",
    "s3_pass",
    "s3_measured",
    "s3_alpha",
    "s3_t",
    "s4_stat",
    "s4_threshold",
    "s4_pass",
    "accepted",
    "p1",
    "by",
    "holm",
    "lfdr",
    "sharpe",
    "psr",
    "min_trl_years",
    "haircut_sharpe",
    "ho_sharpe",
    "ho_p",
    "ho_psr",
    "is_mean",
    "ho_mean",
    "expected_mean",
    "scale",
)
_BOOL_COLUMNS = ("s1_pass", "s2_pass", "s3_pass", "s3_measured", "s4_pass", "accepted")


@dataclass(frozen=True)
class TrialScope:
    """Which ledger trials count as this batch's search (`runlog.trial_count`).

    The gate's own entries are always left out. Tag filters are tuples; a
    bare string is refused, as `runlog.trial_count` refuses it.
    """

    name: str | None = None
    include_tags: tuple[str, ...] | None = None
    exclude_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("include_tags", "exclude_tags"):
            value = getattr(self, field)
            if isinstance(value, str):
                raise TypeError(f"{field} takes a list of tags, not a string: [{value!r}]")
            if value is not None:
                object.__setattr__(self, field, tuple(value))


@dataclass(frozen=True)
class GateHeader:
    """The batch header: what the gate was, and what the search looked like.

    `factors_digest` names stage 3's factor frame (None: not measured).
    `trial_count` is the ledger count for `trial_scope`; `search_fdr_estimate`
    is Storey's plug-in FDR of the stage-1 rejections on the family p-values
    (NaN when nothing passed stage 1); `pbo` is CSCV on the search period's
    variants (NaN when infeasible); `avg_correlation` is across every
    variant on the search period. `registered_utc` is when the config went
    on the ledger.
    """

    config_hash: str
    registered_utc: str
    generator: str
    data_version: str
    factors_digest: str | None
    thresholds: Thresholds
    n_boot: int
    seed: int
    trial_scope: TrialScope
    search_end: str
    holdout_start: str
    holdout_end: str
    trial_count: int
    n_families: int
    n_variants: int
    n_search_obs: int
    n_holdout_obs: int
    search_fdr_estimate: float
    pbo: float
    avg_correlation: float


HEADER_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(GateHeader))


@dataclass(frozen=True)
class GateResult:
    """What the gate decided: the scorecard, its header, the accepted
    families and stage 3's factor loadings (family x factor; empty when not
    measured)."""

    scorecard: pd.DataFrame
    header: GateHeader
    accepted: tuple
    loadings: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        # `read_report` takes the accepted families from the scorecard, in row
        # order; refuse a result that would not read back the same.
        on_card = tuple(self.scorecard.index[self.scorecard["accepted"]])
        if tuple(self.accepted) != on_card:
            raise ValueError(
                f"accepted {tuple(self.accepted)} disagrees with the scorecard's "
                f"accepted rows {on_card} (same families, in scorecard order)"
            )
        object.__setattr__(self, "accepted", on_card)

    @property
    def expected_means(self) -> dict:
        """Per accepted family, the per-period mean a live check should expect."""
        return dict(self.scorecard.loc[list(self.accepted), "expected_mean"])


def _plain(value):
    """JSON-safe: NaN to null, numpy scalars to Python ones."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    return value


def write_report(result: GateResult, path: str | Path) -> tuple[Path, Path]:
    """Write the gate report: `path` (JSON) and the scorecard beside it (CSV).

    Returns (json path, csv path). The JSON names the CSV, so a deploy PR
    needs to link only the JSON.
    """
    path = Path(path)
    csv_path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.scorecard.to_csv(csv_path)
    header = dataclasses.asdict(result.header)
    header["thresholds"] = thresholds_dict(result.header.thresholds)  # as registered
    report = {
        "header": header,
        "accepted": list(result.accepted),
        "expected_means": [[k, v] for k, v in result.expected_means.items()],
        "loadings": {
            "factors": [str(c) for c in result.loadings.columns],
            "rows": [
                [k, list(row)]
                for k, row in zip(result.loadings.index, result.loadings.to_numpy(), strict=True)
            ],
        },
        "scorecard_csv": csv_path.name,
    }
    path.write_text(json.dumps(_plain(report), indent=2, allow_nan=False), encoding="utf-8")
    return path, csv_path


# pandas 3 parses floats exactly by default and deprecates the argument;
# pandas 2 needs it to read back the floats it wrote.
_ROUND_TRIP = {} if int(pd.__version__.split(".")[0]) >= 3 else {"float_precision": "round_trip"}


def _number(value) -> float:
    return float("nan") if value is None else float(value)


def read_report(path: str | Path) -> GateResult:
    """Read a gate report written by `write_report` back into a `GateResult`."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    card = pd.read_csv(path.parent / report["scorecard_csv"], index_col=0, **_ROUND_TRIP)
    for column in _BOOL_COLUMNS:
        card[column] = card[column].astype(bool)
    h = report["header"]
    h["thresholds"] = Thresholds(**h["thresholds"])
    h["trial_scope"] = TrialScope(**h["trial_scope"])
    for name in ("search_fdr_estimate", "pbo", "avg_correlation"):
        h[name] = _number(h[name])
    accepted = tuple(card.index[card["accepted"]])
    if list(accepted) != report["accepted"]:
        raise ValueError(f"{path}: the accepted families disagree with the scorecard")
    rows = report["loadings"]["rows"]
    loadings = pd.DataFrame()
    if rows:
        loadings = pd.DataFrame(
            [[_number(v) for v in values] for _, values in rows],
            index=pd.Index([k for k, _ in rows], name="family"),
            columns=report["loadings"]["factors"],
        )
    return GateResult(card, GateHeader(**h), accepted, loadings)
