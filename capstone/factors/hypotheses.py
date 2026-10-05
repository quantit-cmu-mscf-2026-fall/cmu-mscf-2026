"""Paper -> hypotheses: state testable economic claims before any data is seen.

The model reads what is known about a paper (title, abstract, and the
screener's summary and methods when paperlog has them) and returns
hypotheses in AlphaAgent's five-part format. No market data and no
performance number are involved, so nothing here is a trial.
"""

from __future__ import annotations

import json
from collections import Counter

from capstone.factors.llm import FactorConfig, call_tool
from capstone.factors.store import KINDS, Hypothesis

_FIELDS = ("observation", "knowledge", "justification", "specification", "falsification_condition")

SYSTEM = (
    "You turn finance research papers into testable hypotheses for a cross-sectional "
    "equity factor search over daily US stock data. Each hypothesis is an "
    "economic claim made before seeing any data:\n"
    "- observation: the market behaviour the paper reports.\n"
    "- knowledge: the established finance concept behind it.\n"
    "- justification: why the effect should exist (risk, behaviour, frictions).\n"
    "- specification: what to measure, in terms of daily open, high, low, close, "
    "volume, total return, shares outstanding and market cap, so it can become a "
    "formula.\n"
    "- falsification_condition: the result that would show the hypothesis is wrong.\n"
    "- kind: 'market' if the claim is about stock returns and can become a formula; "
    "'method' if it is about how to search for, build or validate factors (for example "
    "'simpler factors decay less'). Method claims are kept for the record and never "
    "become factors, so do not dress one up as a market claim.\n"
    "State only what the paper supports. If the paper offers nothing testable with these "
    "daily fields, return an empty list rather than inventing a claim."
)


def _tool(k: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            **{name: {"type": "string"} for name in _FIELDS},
            "kind": {"type": "string", "enum": list(KINDS)},
        },
        "required": [*_FIELDS, "kind"],
        "additionalProperties": False,
    }
    return {
        "name": "record_hypotheses",
        "description": f"Record up to {k} testable hypotheses from the paper.",
        "input_schema": {
            "type": "object",
            "properties": {"hypotheses": {"type": "array", "items": item}},
            "required": ["hypotheses"],
            "additionalProperties": False,
        },
    }


def paper_prompt(paper: dict) -> str:
    """The parts of a paper record the model sees, as JSON."""
    keep = ("title", "abstract", "summary_en", "methods", "datasets", "markets", "published")
    return json.dumps({k: paper[k] for k in keep if paper.get(k)}, ensure_ascii=False, indent=2)


def extract(
    paper: dict, client, config: FactorConfig, usage: Counter | None = None
) -> list[Hypothesis]:
    """Ask the model for up to `hypotheses_per_paper` hypotheses about one paper.

    Entries with an empty field or an unknown kind are dropped. An empty list is a legitimate
    answer: the paper may hold nothing testable with daily stock data.
    """
    answer = call_tool(
        client,
        config,
        system=SYSTEM,
        user=paper_prompt(paper),
        tool=_tool(config.hypotheses_per_paper),
        usage=usage,
    )
    out: list[Hypothesis] = []
    for item in answer.get("hypotheses", [])[: config.hypotheses_per_paper]:
        try:
            fields = {n: str(item.get(n, "")) for n in _FIELDS}
            out.append(Hypothesis(paper_key=paper["key"], kind=str(item.get("kind", "")), **fields))
        except ValueError:
            continue
    return out
