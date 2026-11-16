"""Run papers through the pipeline: paper -> hypotheses -> factor trees.

One paper at a time. A paper counts as done, and is reported to its source
as done, only when every step for it finished; anything that raises leaves
it pending for the next run, and the error is reported, not swallowed.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field

from capstone.factors import store
from capstone.factors.hypotheses import extract
from capstone.factors.llm import FactorConfig
from capstone.factors.proposer import propose
from capstone.factors.zoo import default_zoo


@dataclass
class RunSummary:
    papers_done: list[str] = field(default_factory=list)
    papers_failed: dict[str, str] = field(default_factory=dict)
    hypotheses: Counter = field(default_factory=Counter)  # by kind: market, method
    outcomes: Counter = field(default_factory=Counter)
    usage: Counter = field(default_factory=Counter)  # calls, input_tokens, output_tokens

    def __str__(self) -> str:
        counts = ", ".join(f"{n} {status}" for status, n in sorted(self.outcomes.items()))
        kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(self.hypotheses.items()))
        lines = [
            f"papers: {len(self.papers_done)} done, {len(self.papers_failed)} failed",
            f"hypotheses: {sum(self.hypotheses.values())} ({kinds or 'none'})",
            f"proposals: {sum(self.outcomes.values())} ({counts or 'none'})",
            f"model calls: {self.usage['calls']}, tokens in {self.usage['input_tokens']}, "
            f"out {self.usage['output_tokens']}",
        ]
        lines += [f"  failed {key}: {error}" for key, error in self.papers_failed.items()]
        return "\n".join(lines)


def _citation(paper: dict) -> str:
    authors = paper.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    lead = f"{authors[0]} et al." if len(authors) > 1 else (authors[0] if authors else "")
    year = str(paper.get("published", ""))[:4]
    return paper.get("citation") or ", ".join(part for part in (lead, year) if part)


def run(con: sqlite3.Connection, source, client, config: FactorConfig) -> RunSummary:
    """Process every paper the source offers; mark each done only if it fully finished.

    Market hypotheses become factors; method hypotheses are stored and stop
    there. Each paper's model calls and tokens are recorded in the store,
    including for a paper that failed partway, since those calls were paid for.
    """
    summary = RunSummary()
    zoo = default_zoo()
    for paper in source.papers():
        key = paper["key"]
        usage: Counter = Counter()
        try:
            store.add_paper(
                con,
                key,
                paper["title"],
                url=paper.get("url") or "",
                citation=_citation(paper),
                source=type(source).__name__,
            )
            for hypothesis in extract(paper, client, config, usage=usage):
                hid = store.add_hypothesis(
                    con, hypothesis, model=config.model, prompt_version=config.prompt_version
                )
                summary.hypotheses[hypothesis.kind] += 1
                if hypothesis.kind != "market":
                    continue
                for outcome in propose(con, hid, client, config, zoo=zoo, usage=usage):
                    summary.outcomes[outcome.status] += 1
            # Inside the try: if the source cannot record the hand-off (paperlog's
            # database locked, say), this paper is reported as failed and stays
            # pending, and the rest of the run goes on.
            source.done([key])
            summary.papers_done.append(key)
        except Exception as exc:  # noqa: BLE001 - one paper's failure must not stop the run
            summary.papers_failed[key] = f"{type(exc).__name__}: {exc}"
        finally:
            summary.usage.update(usage)
            # The alignment judge runs on its own model, priced separately.
            for model, prefix in ((config.model, ""), (config.alignment_model, "alignment_")):
                if usage[f"{prefix}calls"]:
                    store.record_usage(
                        con,
                        key,
                        model,
                        calls=usage[f"{prefix}calls"],
                        input_tokens=usage[f"{prefix}input_tokens"],
                        output_tokens=usage[f"{prefix}output_tokens"],
                    )
    return summary
