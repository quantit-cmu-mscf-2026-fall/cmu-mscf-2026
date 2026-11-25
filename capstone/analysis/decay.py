"""Alpha-decay diagnostics: does a family's best strategy degrade within 1990-2020?

Report-only. It reads an existing return matrix, decides nothing and adds no trial.

Selection (in-sample, on purpose): each family is represented by the variant with the
highest full-sample annualised Sharpe on the search period. That variant was picked
because it did well over the whole window, so its Sharpe is biased upward and, if
anything, early-heavy; every number here is a descriptive look at that selected
series, not an out-of-sample estimate.

- Block Sharpe: calendar blocks of `block_years` starting at the matrix's first year
  (1990-94, 1995-99, ..., 2015-19, 2020). A last block shorter than `block_years` is
  partial: it is kept, labelled with the years it has, and flagged in the statements.
- Annual Sharpe: one per calendar year with at least `min_days_year` returns.
- Decay slope: OLS of annual Sharpe on calendar year, Newey-West (HAC, Bartlett)
  standard errors with lag floor(4 (n/100)^(2/9)), t reference distribution.
  Reported per year and per decade.
- Pooled: median slope across families and the count with a negative slope below
  `alpha`. Uncorrected across families; families are not independent.
- Publication split (only with a published-date table): Welch t-test on daily
  returns before vs after the date, plus the annualised Sharpe on each side.
- Regimes (only with a market return series): `capstone.regime_report.regime_split`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from capstone.analysis.outputs import Stat, Statement, output_dir, write_outputs
from capstone.gate import HOLDOUT_OPENS
from capstone.runlog import log_run

TRADING_DAYS = 252
DEFAULTS = {
    "block_years": 5,
    "min_days_block": 120,
    "min_days_year": 120,
    "min_years_slope": 8,
    "alpha": 0.05,
}


def search_period_only(returns: pd.DataFrame) -> pd.DataFrame:
    """`returns` unchanged; raises ValueError if any date is on or after `HOLDOUT_OPENS`."""
    if len(returns) and returns.index.max().date() >= HOLDOUT_OPENS:
        raise ValueError(
            f"the matrix runs to {returns.index.max().date()}, past the search period: "
            f"the holdout opens {HOLDOUT_OPENS}"
        )
    return returns


def sharpe(r: pd.Series, min_days: int = 2) -> float:
    r = r.dropna()
    if len(r) < max(min_days, 2):
        return float("nan")
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def best_variants(returns: pd.DataFrame, family: pd.Series) -> pd.Series:
    """Family -> variant with the highest full-sample Sharpe (in-sample selection)."""
    out = {}
    for fam, members in family.groupby(family).groups.items():
        cols = [c for c in members if c in returns.columns]
        sh = pd.Series({c: sharpe(returns[c]) for c in cols}, dtype=float).dropna()
        if len(sh):
            out[str(fam)] = str(sh.idxmax())
    return pd.Series(out, name="variant", dtype=object)


def block_edges(index: pd.DatetimeIndex, block_years: int) -> list[tuple[str, int, int, bool]]:
    """(label, first_year, last_year, partial) calendar blocks from the first year."""
    first, last = int(index.min().year), int(index.max().year)
    out = []
    for y0 in range(first, last + 1, block_years):
        y1 = min(y0 + block_years - 1, last)
        label = f"{y0}-{str(y1)[2:]}" if y1 > y0 else str(y0)
        out.append((label, y0, y1, y1 - y0 + 1 < block_years))
    return out


def block_sharpes(r: pd.Series, block_years: int, min_days: int) -> pd.Series:
    return pd.Series(
        {
            label: sharpe(r[(r.index.year >= y0) & (r.index.year <= y1)], min_days)
            for label, y0, y1, _ in block_edges(r.index, block_years)
        }
    )


def annual_sharpes(r: pd.Series, min_days: int) -> pd.Series:
    return pd.Series({int(y): sharpe(g, min_days) for y, g in r.groupby(r.index.year)})


def decay_slope(annual: pd.Series, min_years: int = 8) -> dict:
    """OLS of annual Sharpe on year with Newey-West SEs; slope is per year and per decade."""
    a = annual.dropna()
    n = len(a)
    nan = float("nan")
    if n < max(min_years, 4) or a.nunique() < 2:
        return {"slope": nan, "slope_decade": nan, "se": nan, "t": nan, "p": nan, "n_years": n}
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    x = sm.add_constant(a.index.to_numpy(dtype=float))
    fit = sm.OLS(a.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=True)
    slope, se = float(fit.params[1]), float(fit.bse[1])
    return {
        "slope": slope,
        "slope_decade": slope * 10,
        "se": se,
        "t": float(fit.tvalues[1]),
        "p": float(fit.pvalues[1]),
        "n_years": n,
    }


def publication_split(r: pd.Series, date: pd.Timestamp) -> dict:
    """Sharpe before and on/after `date` and a Welch t-test on the daily means."""
    r = r.dropna()
    pre, post = r[r.index < date], r[r.index >= date]
    out = {"n_pre": len(pre), "n_post": len(post), "sharpe_pre": sharpe(pre, 20)}
    out["sharpe_post"] = sharpe(post, 20)
    out["diff"] = out["sharpe_post"] - out["sharpe_pre"]
    out["t"], out["p"] = float("nan"), float("nan")
    if len(pre) >= 20 and len(post) >= 20:
        t, p = sps.ttest_ind(post, pre, equal_var=False)
        out["t"], out["p"] = float(t), float(p)
    return out


def compute(
    returns: pd.DataFrame,
    family: pd.Series,
    *,
    published: Mapping[str, str] | None = None,
    mkt_return: pd.Series | None = None,
    **params,
) -> tuple[dict[str, pd.DataFrame], dict[str, Stat], list[Statement]]:
    """The tables, stats and statements for a run; `params` override `DEFAULTS`."""
    cfg = DEFAULTS | params
    returns = search_period_only(returns)
    best = best_variants(returns, family)
    blocks, annual, slopes, pub = {}, {}, {}, {}
    for fam, var in best.items():
        r = returns[var]
        blocks[fam] = block_sharpes(r, cfg["block_years"], cfg["min_days_block"])
        annual[fam] = annual_sharpes(r, cfg["min_days_year"])
        slopes[fam] = {"variant": var, **decay_slope(annual[fam], cfg["min_years_slope"])}
        if published and fam in published:
            pub[fam] = {"variant": var, "date": published[fam]} | publication_split(
                r, pd.Timestamp(published[fam])
            )
    slope_df = pd.DataFrame(slopes).T.rename_axis("family")
    for c in slope_df.columns.drop("variant"):
        slope_df[c] = slope_df[c].astype(float)
    tables = {
        "blocks": pd.DataFrame(blocks).T.rename_axis("family"),
        "annual": pd.DataFrame(annual).T.rename_axis("family"),
        "slopes": slope_df,
    }
    tables["annual"].columns = tables["annual"].columns.astype(str)
    if pub:
        p = pd.DataFrame(pub).T.rename_axis("family")
        for c in p.columns.drop(["variant", "date"]):
            p[c] = p[c].astype(float)
        tables["publication"] = p
    if mkt_return is not None:
        from capstone.regime_report import regime_split

        found = SimpleNamespace(survivors={}, scorecard={"best": best.to_dict()})
        split = regime_split(found, returns, mkt_return)
        tables["regimes"] = pd.DataFrame(
            {
                row["family"]: {
                    s: (
                        float("nan")
                        if row["regimes"][s]["sharpe"] is None
                        else row["regimes"][s]["sharpe"]
                    )
                    for s in split["states"]
                }
                for row in split["rows"]
            }
        ).T.rename_axis("family")

    valid = slope_df.dropna(subset=["slope"])
    alpha = cfg["alpha"]
    sig_neg = valid[(valid["slope"] < 0) & (valid["p"] < alpha)]
    partial = [e[0] for e in block_edges(returns.index, cfg["block_years"]) if e[3]]
    ins = ("matrix", "summary")
    stats = {
        "n_families": Stat(len(best), "families with a best variant", ins),
        "selection": Stat(
            "best full-sample Sharpe per family",
            "each family is represented by its highest full-sample search-period Sharpe "
            "variant (in-sample selection)",
            ins,
        ),
        "partial_blocks": Stat(
            partial, "calendar blocks shorter than block_years, kept and labelled", ("matrix",)
        ),
        "n_slopes": Stat(len(valid), "families with enough annual Sharpes for a slope", ins),
        "median_slope_decade": Stat(
            float(valid["slope_decade"].median()) if len(valid) else None,
            "median across families of the OLS slope of annual Sharpe on year, per decade",
            ins,
        ),
        "n_sig_negative": Stat(
            len(sig_neg),
            f"families with slope < 0 and HAC p < {alpha}, uncorrected across families",
            ins,
        ),
        "block_years": Stat(cfg["block_years"], "calendar block width in years", ()),
    }
    if len(valid):
        worst = valid["slope_decade"].idxmin()
        stats["steepest_decay"] = Stat(
            {"family": worst, "slope_decade": float(valid.loc[worst, "slope_decade"])},
            "the family with the most negative slope per decade",
            ins,
        )
    med = stats["median_slope_decade"].value
    statements = [
        Statement(
            "Each family is represented by its best variant by full-sample search-period "
            f"Sharpe ({len(best)} families).",
            "selection",
            "In-sample selection: the chosen variant's Sharpe is biased upward and the "
            "decay it shows is descriptive, not an out-of-sample estimate.",
        ),
        Statement(
            "The median decay slope across families is "
            + ("not available" if med is None else f"{med:+.3f} Sharpe per decade")
            + ".",
            "median_slope_decade",
            "Annual Sharpe on year, HAC standard errors; about 30 annual points per family.",
        ),
        Statement(
            f"{len(sig_neg)} of {len(valid)} families have a significantly negative decay "
            f"slope (p < {alpha}).",
            "n_sig_negative",
            "Uncorrected for multiple testing across families, which are not independent.",
        ),
    ]
    if partial:
        statements.append(
            Statement(
                f"The last block ({', '.join(partial)}) is shorter than {cfg['block_years']} "
                "years and is reported on the years it has.",
                "partial_blocks",
                "Its Sharpe rests on fewer days and is noisier than the full blocks.",
            )
        )
    if "publication" in tables:
        p = tables["publication"]
        stats["publication_median_diff"] = Stat(
            float(p["diff"].median()),
            "median across published families of Sharpe after minus before the date",
            ("matrix", "published"),
        )
        statements.append(
            Statement(
                f"Across {len(p)} families with a publication date, median Sharpe after minus "
                f"before is {stats['publication_median_diff'].value:+.2f}.",
                "publication_median_diff",
                "Welch t-test on daily returns per family; the post-publication window ends "
                "2020 and is short; uncorrected across families.",
            )
        )
    else:
        stats["publication_split"] = Stat(
            "skipped", "no published-date table was supplied", ("published",)
        )
        statements.append(
            Statement(
                "The pre/post publication split was skipped because the run's paper "
                "mapping is unavailable.",
                "publication_split",
            )
        )
    if "regimes" in tables:
        stats["regimes"] = Stat("computed", "regime_split on a supplied market return", ("mkt",))
        statements.append(
            Statement(
                "Regime-split Sharpe per family is in the regimes table.",
                "regimes",
                "Regimes use only the supplied market return through 2020.",
            )
        )
    else:
        stats["regimes"] = Stat("skipped", "no market return series supplied", ())
        statements.append(
            Statement(
                "The regime split was skipped: it needs a market return series, which "
                "comes from the panel and is not loaded by default.",
                "regimes",
            )
        )
    return tables, stats, statements


def run_report(
    returns: pd.DataFrame,
    family: pd.Series,
    run: str,
    *,
    published: Mapping[str, str] | None = None,
    mkt_return: pd.Series | None = None,
    out_root=None,
    inputs: dict | None = None,
    **params,
):
    """Refuse a post-search matrix, log the run, then compute and write the folder."""
    search_period_only(returns)
    log_run(
        name="decay-report",
        params={"run": run, **(DEFAULTS | params), **(inputs or {})},
        tags=["synthetic", "methodology"],
        notes="report-only decay diagnostics on an existing matrix; adds no trial",
    )
    tables, stats, statements = compute(
        returns, family, published=published, mkt_return=mkt_return, **params
    )
    return write_outputs(
        output_dir(run, "decay", out_root),
        title=f"Alpha decay: {run}",
        tables=tables,
        stats=stats,
        statements=statements,
    )


__all__ = ["compute", "run_report", "decay_slope", "block_sharpes", "publication_split"]
