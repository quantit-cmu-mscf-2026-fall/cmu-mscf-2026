"""What the agents produced, and how concentrated the search was: a report-only analysis.

Pure functions over a variant summary, a run summary and the attribution rows; the
CLI in `research/agent_bias.py` reads the files, logs the run and writes the folder.

Lineage. The factor store of a run is not kept, so a factor's origin is derived from
the order of first appearance in the variant summary: the run writes its seed
factors (from the papers' hypotheses) first and the deepen children after them. The
first `seed_factors` factors are `seed`, the rest `deepen_child`. The rule is checked
against what the run summary says (the papers' factor total, the children all in one
family, every deepen parent among the earlier factors); `lineage_check` reports it.

Price-level term. A term of the factor (see `capstone.factors.classify`) whose fields
are only close, open, high or low and which has no delta, ts_sum, ts_std, ts_cov,
ts_corr, ts_product and no division: the price itself, its log or rank, or its lag or
moving average. A price change (`delta(log(close), 21)`) or a ratio of prices
(`close / ts_max(close, 252)`) is not one. Prices were not split-adjusted in the old
scores, so a level is exposed to the split look-ahead where a change is far less so.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats as sps

from capstone.analysis.outputs import Stat, Statement
from capstone.factors.classify import classify
from capstone.factors.tree import ParseError, features_used, funcs_used, parse, unparse
from capstone.factors.zoo import default_zoo

PRICE_FIELDS = frozenset({"close", "open", "high", "low"})
NOT_A_LEVEL = frozenset({"delta", "ts_sum", "ts_std", "ts_cov", "ts_corr", "ts_product", "/"})
SEED, CHILD = "seed", "deepen_child"
SMALL = 5


def price_level_term(expression: str) -> bool:
    """True when the term is a bare price level (see the module docstring)."""
    try:
        node = parse(expression, regime_operators=True)
    except ParseError:
        return False
    fields = features_used(node)
    return bool(fields) and fields <= PRICE_FIELDS and not (funcs_used(node) & NOT_A_LEVEL)


def low_price_leg(term: Mapping) -> bool:
    """A price-level term that is long the low-priced stocks (negative sign)."""
    return term["sign"] < 0 and price_level_term(term["expr"])


def assign_lineage(summary: pd.DataFrame, run_summary: Mapping) -> pd.Series:
    """`seed` for the first `seed_factors` factors in order of first appearance, else child."""
    ids = list(dict.fromkeys(summary["factor_id"]))
    n_seed = int(run_summary["seed_factors"])
    return pd.Series([SEED] * n_seed + [CHILD] * (len(ids) - n_seed), index=pd.Index(ids))


def lineage_check(summary: pd.DataFrame, run_summary: Mapping, lineage: pd.Series) -> dict:
    """Counts and consistency checks of the lineage rule against the run summary."""
    family = summary.groupby("factor_id", sort=False)["hypothesis_id"].first()
    child_ids = lineage.index[lineage == CHILD]
    seed_ids = lineage.index[lineage == SEED]
    parents = {d["parent"] for d in run_summary["deepen"]["decisions"]}
    position = {f: i for i, f in enumerate(lineage.index)}
    papers_total = sum(p["factors"] for p in run_summary["papers"])
    return {
        "n_seed": len(seed_ids),
        "n_child": len(child_ids),
        "papers_factor_total": papers_total,
        "children_families": sorted(set(family[child_ids])),
        "children_in_one_family": family[child_ids].nunique() <= 1,
        "seed_count_matches_papers": len(seed_ids) == papers_total,
        "parents_known": all(p in position for p in parents),
        "n_parents_seed": sum(p in set(seed_ids) for p in parents),
        "n_parents_child": sum(p in set(child_ids) for p in parents),
        "seed_families": int(family[seed_ids].nunique()),
        "families_in_run": len(run_summary["families"]),
    }


def _best(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    return frame.groupby("factor_id", sort=False)[column].max().rename(name)


def factor_table(
    summary: pd.DataFrame,
    new: pd.DataFrame,
    run_summary: Mapping,
) -> pd.DataFrame:
    """One row per factor: lineage, family, classification, price-level flags, best scores."""
    lineage = assign_lineage(summary, run_summary)
    first = summary.groupby("factor_id", sort=False).agg(
        family=("hypothesis_id", "first"), expression=("expression", "first")
    )
    rows = {}
    for factor_id, expression in first["expression"].items():
        info = classify(parse(expression, regime_operators=True))
        levels = [price_level_term(t["expr"]) for t in info["terms"]]
        rows[factor_id] = {
            "archetype": info["archetype"],
            "archetypes": "|".join(info["archetypes"]),
            "n_terms": info["n_terms"],
            "interaction": info["interaction"],
            "horizon_bin": info["horizon_bin"],
            "primary_field_group": info["primary_field_group"],
            "zoo_share": info["zoo_share"],
            "nearest_zoo": info["nearest_zoo"],
            "term_archetypes": "|".join(t["archetype"] for t in info["terms"]),
            "has_price_level_term": any(levels),
            "has_low_price_leg": any(low_price_leg(t) for t in info["terms"]),
        }
    table = pd.concat([lineage.rename("lineage"), first[["family", "expression"]]], axis=1)
    table = table.join(pd.DataFrame.from_dict(rows, orient="index"))
    table["best_net_old"] = _best(summary, "sharpe_net", "best_net_old")
    table["best_net_new"] = _best(new, "sharpe_net", "best_net_new")
    table["best_gross_old"] = _best(summary, "sharpe_gross", "best_gross_old")
    table["best_gross_new"] = _best(new, "sharpe_gross", "best_gross_new")
    table["net_change"] = table["best_net_new"] - table["best_net_old"]
    table.index.name = "factor_id"
    return table


def zoo_table() -> pd.DataFrame:
    """Archetype of each zoo member (term archetypes joined with `|`)."""
    rows = {}
    for name, node in default_zoo():
        info = classify(node)
        rows[name] = {
            "expression": unparse(node),
            "archetype": info["archetype"],
            "term_archetypes": "|".join(t["archetype"] for t in info["terms"]),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("zoo_member")


def counts_by(table: pd.DataFrame, column: str, groups: str = "lineage") -> pd.DataFrame:
    return pd.crosstab(table[column], table[groups])


def term_counts(table: pd.DataFrame, groups: str = "lineage") -> pd.DataFrame:
    """Term archetype counts per group, one count per term."""
    exploded = table.assign(term=table["term_archetypes"].str.split("|")).explode("term")
    return pd.crosstab(exploded["term"], exploded[groups])


def chi_square(counts: pd.DataFrame, a: str, b: str) -> dict:
    """Pearson chi-square of two groups' counts over categories; flags small expected counts."""
    sub = counts[[a, b]]
    sub = sub[sub.sum(axis=1) > 0]
    if sub.shape[0] < 2 or (sub.sum() == 0).any():
        return {"chi2": None, "p": None, "dof": 0, "min_expected": None, "small": True}
    chi2, p, dof, expected = sps.chi2_contingency(sub.to_numpy())
    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "min_expected": float(expected.min()),
        "small": bool(expected.min() < SMALL),
    }


def hhi(counts) -> float:
    """Herfindahl index of shares: 1 / k when equal over k groups, 1 when in one."""
    arr = np.asarray(list(counts), dtype=float)
    total = arr.sum()
    return float(((arr / total) ** 2).sum()) if total > 0 else float("nan")


def concentration(table: pd.DataFrame, summary: pd.DataFrame, run_summary: Mapping) -> dict:
    variants = summary.groupby("hypothesis_id").size()
    children = table[table["lineage"] == CHILD].groupby("family").size()
    decisions = run_summary["deepen"]["decisions"]
    motifs = pd.Series([d["motif"] for d in decisions]).value_counts()
    outcome = dict(run_summary["deepen"]["children"])
    return {
        "children_per_family": children.to_dict(),
        "variants_per_family": variants.to_dict(),
        "n_families": int(variants.size),
        "hhi_variants": hhi(variants),
        "hhi_equal": 1 / variants.size,
        "largest_family": variants.idxmax(),
        "largest_family_share": float(variants.max() / variants.sum()),
        "hhi_factors": hhi(table.groupby("family").size()),
        "n_decisions": len(decisions),
        "motifs": motifs.to_dict(),
        "top_motif_share": float(motifs.iloc[0] / motifs.sum()),
        "hhi_motifs": hhi(motifs),
        "distinct_parents": len({d["parent"] for d in decisions}),
        "top_parent_share": float(
            pd.Series([d["parent"] for d in decisions]).value_counts().iloc[0] / len(decisions)
        ),
        "outcomes": outcome,
    }


def exposure(table: pd.DataFrame) -> dict:
    """Price-level share and old-to-new Sharpe change by lineage and by the flag."""
    share = table.groupby("lineage")["has_price_level_term"].agg(["sum", "size", "mean"])
    low = table.groupby("lineage")["has_low_price_leg"].agg(["sum", "size", "mean"])
    by_flag = table.groupby(["lineage", "has_price_level_term"])["net_change"].agg(
        ["size", "median"]
    )
    by_flag_only = table.groupby("has_price_level_term")["net_change"].agg(["size", "median"])
    return {
        "price_level_share": {k: float(v) for k, v in share["mean"].items()},
        "price_level_n": {k: int(v) for k, v in share["sum"].items()},
        "low_price_leg_share": {k: float(v) for k, v in low["mean"].items()},
        "net_change_median_by_flag": {
            str(k): {"n": int(r["size"]), "median": float(r["median"])}
            for k, r in by_flag_only.iterrows()
        },
        "net_change_median_by_lineage_flag": {
            f"{a}|{b}": {"n": int(r["size"]), "median": float(r["median"])}
            for (a, b), r in by_flag.iterrows()
        },
    }


def rebase_effect(rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per top-20 variant: gross with no fix vs with split rebasing only, and the drop."""
    off = rows[~rows["rebase"] & ~rows["csu"] & ~rows["strict"]].set_index("variant")["gross"]
    on = rows[rows["rebase"] & ~rows["csu"] & ~rows["strict"]].set_index("variant")["gross"]
    out = pd.DataFrame({"gross_no_fix": off, "gross_rebase": on}).dropna()
    out["drop"] = out["gross_no_fix"] - out["gross_rebase"]
    out["factor_id"] = [v.split("-")[0] for v in out.index]
    out = out.join(
        table[["lineage", "has_price_level_term"]], on="factor_id", how="left"
    ).rename_axis("variant")
    return out


def rebase_summary(effect: pd.DataFrame) -> dict:
    def block(frame):
        return {
            "n": int(len(frame)),
            "median_no_fix": float(frame["gross_no_fix"].median()) if len(frame) else None,
            "median_rebase": float(frame["gross_rebase"].median()) if len(frame) else None,
            "median_drop": float(frame["drop"].median()) if len(frame) else None,
        }

    return {
        "all": block(effect),
        "by_lineage": {k: block(g) for k, g in effect.groupby("lineage")},
        "by_price_level": {str(k): block(g) for k, g in effect.groupby("has_price_level_term")},
    }


def archetype_real_vs_placebo(
    new: pd.DataFrame, placebo: pd.DataFrame, table: pd.DataFrame
) -> pd.DataFrame:
    """Median gross Sharpe over variants, per factor archetype: real (new) and placebo."""
    archetype = table["archetype"]
    frame = pd.DataFrame(
        {
            "real": new["sharpe_gross"],
            "placebo": placebo["sharpe_gross"].reindex(new.index),
            "archetype": new["factor_id"].map(archetype),
        }
    )
    out = frame.groupby("archetype").agg(
        n_variants=("real", "size"),
        real_median_gross=("real", "median"),
        placebo_median_gross=("placebo", "median"),
    )
    out["real_minus_placebo"] = out["real_median_gross"] - out["placebo_median_gross"]
    return out


def _share(counts: pd.DataFrame, group: str) -> dict:
    total = counts[group].sum()
    return {k: float(v / total) for k, v in counts[group].items() if v}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def analyse(
    *,
    summary: pd.DataFrame,
    run_summary: Mapping,
    new: pd.DataFrame,
    placebo: pd.DataFrame,
    attribution: pd.DataFrame,
    inputs: tuple[str, ...] = (),
) -> tuple[dict[str, pd.DataFrame], dict[str, Stat], list[Statement]]:
    """All tables, stats and statements. Reads nothing but its arguments."""
    table = factor_table(summary, new, run_summary)
    zoo = zoo_table()
    check = lineage_check(summary, run_summary, table["lineage"])

    pooled = pd.concat([table.assign(group=table["lineage"]), zoo.assign(group="zoo")])
    arch = counts_by(pooled, "archetype", "group")
    terms = term_counts(pooled, "group")
    arch_chi = chi_square(arch, SEED, CHILD)
    term_chi = chi_square(terms, SEED, CHILD)
    conc = concentration(table, summary, run_summary)
    expo = exposure(table)
    effect = rebase_effect(attribution, table)
    rebase = rebase_summary(effect)
    real_vs_null = archetype_real_vs_placebo(new, placebo, table)

    tables = {
        "factors": table,
        "zoo": zoo,
        "archetype_counts": arch,
        "term_archetype_counts": terms,
        "rebase_effect_top20": effect,
        "archetype_real_vs_placebo": real_vs_null,
    }
    src = tuple(inputs)
    st = {
        "lineage_check": Stat(
            check,
            "lineage rule: first seed_factors factors in order of first appearance are seed, "
            "the rest deepen_child; counts and consistency checks against the run summary",
            src,
        ),
        "archetype_counts": Stat(
            {c: arch[c].to_dict() for c in arch.columns},
            "factor archetype counts by group (seed, deepen_child, zoo)",
            src,
        ),
        "archetype_seed_vs_child": Stat(
            arch_chi, "chi-square of factor archetype counts, seed vs deepen_child", src
        ),
        "term_archetype_counts": Stat(
            {c: terms[c].to_dict() for c in terms.columns},
            "term archetype counts by group (a factor has one or more terms)",
            src,
        ),
        "term_archetype_seed_vs_child": Stat(
            term_chi, "chi-square of term archetype counts, seed vs deepen_child", src
        ),
        "concentration": Stat(
            conc,
            "variants and children per family (hypothesis), HHI, motifs, deepen outcomes",
            src,
        ),
        "exposure": Stat(
            expo,
            "share of factors with a price-level term, and median old-to-new best net "
            "Sharpe change by lineage and flag",
            src,
        ),
        "rebase_top20": Stat(
            rebase,
            "top-20 old variants: gross Sharpe with no fix vs split rebasing only, by "
            "lineage and by price-level flag",
            src,
        ),
        "real_vs_placebo": Stat(
            real_vs_null.round(4).to_dict("index"),
            "median variant gross Sharpe by factor archetype, new scores vs placebo seed 0",
            src,
        ),
    }

    n_seed, n_child = check["n_seed"], check["n_child"]
    shared = (
        f"Lineage is inferred from row order (first {n_seed} factors are seed), not read "
        "from the factor store."
    )
    lin_caveat = shared + (
        ""
        if check["children_in_one_family"] and check["seed_count_matches_papers"]
        else " The checks against the run summary did not all pass; see the stat."
    )
    s = [
        Statement(
            f"{n_seed} factors are seed and {n_child} are deepen children; the children "
            f"sit in {len(check['children_families'])} family; "
            f"every deepen parent is a listed factor: {check['parents_known']}.",
            "lineage_check",
            lin_caveat,
        )
    ]
    small = " Counts are small, so the test is weak." if arch_chi["small"] else ""
    seed_top = max(_share(arch, SEED).items(), key=lambda kv: kv[1])
    child_top = max(_share(arch, CHILD).items(), key=lambda kv: kv[1])
    p_text = "n/a" if arch_chi["p"] is None else f"{arch_chi['p']:.3g}"
    s.append(
        Statement(
            f"The most common factor archetype is {seed_top[0]} among seeds "
            f"({_pct(seed_top[1])}) and {child_top[0]} among deepen children "
            f"({_pct(child_top[1])}); the chi-square p-value of seed vs child is {p_text}.",
            "archetype_seed_vs_child",
            "Archetypes come from draft rules." + small,
        )
    )
    tp_text = "n/a" if term_chi["p"] is None else f"{term_chi['p']:.3g}"
    s.append(
        Statement(
            f"Counting terms instead of whole factors, the seed vs child chi-square "
            f"p-value is {tp_text}; zoo shares are in the table for comparison.",
            "term_archetype_seed_vs_child",
            "Terms of one factor are not independent, so the p-value is indicative only."
            + (" Counts are small." if term_chi["small"] else ""),
        )
    )
    s.append(
        Statement(
            f"The {conc['n_families']} families hold {conc['largest_family_share']:.0%} of "
            f"variants in the largest one (HHI {conc['hhi_variants']:.3f}; "
            f"{conc['hhi_equal']:.3f} if equal).",
            "concentration",
            "Variants per factor are fixed at six, so this is the factor count in disguise.",
        )
    )
    s.append(
        Statement(
            f"The agent made {conc['n_decisions']} refinement choices on "
            f"{conc['distinct_parents']} parents; the top parent took "
            f"{conc['top_parent_share']:.0%} and the top motif "
            f"{conc['top_motif_share']:.0%}; children outcomes: {conc['outcomes']}.",
            "concentration",
            "Decisions are per refinement call; one call yields several children.",
        )
    )
    pl = expo["price_level_share"]
    s.append(
        Statement(
            "Share of factors with a price-level term: "
            + ", ".join(f"{k} {_pct(v)}" for k, v in sorted(pl.items()))
            + ".",
            "exposure",
            "A child inherits its parent's terms, so a high child share can be inheritance "
            "rather than separate choices.",
        )
    )
    flag = expo["net_change_median_by_flag"]
    s.append(
        Statement(
            "Median change in best net Sharpe from old to new scores: "
            + "; ".join(
                f"price-level={k}: {v['median']:+.2f} (n={v['n']})" for k, v in flag.items()
            )
            + ".",
            "exposure",
            "Other fixes act at the same time as split rebasing, and the flag is correlated "
            "with lineage.",
        )
    )
    all_ = rebase["all"]
    s.append(
        Statement(
            f"Among the top-{all_['n']} old variants, split rebasing alone moves median "
            f"gross Sharpe from {all_['median_no_fix']:.2f} to {all_['median_rebase']:.2f}.",
            "rebase_top20",
            "These are the winners of the old search, so they are selected on the old "
            "scores; by lineage and flag, see the stat. This is evidence on whether "
            "hill-climbing exploited the look-ahead, not proof: the top 20 carry the "
            "same price-level leg, so there is no unexposed comparison group.",
        )
    )
    gap = real_vs_null["real_minus_placebo"]
    best = gap.idxmax()
    s.append(
        Statement(
            f"Real minus placebo median gross Sharpe is largest for {best} "
            f"({gap.max():+.2f}); by archetype the table lists all.",
            "real_vs_placebo",
            "Variant-level medians of correlated variants; placebo is one seed.",
        )
    )
    return tables, st, s
