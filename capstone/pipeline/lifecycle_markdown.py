"""LIFECYCLE.md: the human reading of the lifecycle report's LIFECYCLE.json (same content)."""

from __future__ import annotations

import json


def _cell(value):
    text = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return text.replace("|", r"\|").replace("\n", " ") or "-"


def _short(value):
    return "-" if not value else str(value)[:12]


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]


def _stages(document):
    rows = []
    for row in document["stages"]:
        pairs = zip(row["blockers"], row["cleared_by"], strict=True)
        cleared = "; ".join(f"`{b}` -> {c}" for b, c in pairs)
        rows.append([row["stage"], row["artifact"] or "-", _short(row["sha256"]), cleared])
    return _table(["Stage", "Artifact", "sha256", "Blocker -> cleared by"], rows)


def _conceptual(document):
    lines = []
    for run in document["conceptual"]:
        lines += [f"### {run['stage']} ({run.get('status', 'unverified')})", ""]
        lines += [f"- blocker: `{b}`" for b in run["blockers"]]
        for claim in run["claims"]:
            head = f"**{claim['packet_id']}** `{claim['status']}`"
            lines += ["", f"{head} claim {_short(claim['claim_id'])}"]
            lines += [f"- {k}: {_cell(v)}" for k, v in (claim["claim"] or {}).items()]
            lines.append(f"- equation: `{claim['equation'] or '-'}`")
            lines += [f"- declared default: {_cell(d)}" for d in claim["declared_defaults"]]
            lines += [f"- blocker: `{b}`" for b in claim["blockers"]]
        lines.append("")
    return lines


def _lineage(document):
    header = ["Source revisions", "Claim", "Method", "Program", "Portfolios", "Plans"]
    header += ["Journals", "Monitors"]
    rows = [
        [
            ", ".join(map(_short, c["source_revisions"])) or "-",
            _short(c["claim_id"]),
            _short(c["method_record_id"]),
            c["program_id"],
            ", ".join(map(_short, c["portfolios"])) or "-",
            ", ".join(c["plans"]) or "-",
            ", ".join(c["journals"]) or "-",
            ", ".join(c["monitors"]) or "-",
        ]
        for c in document["lineage"]["chains"]
    ]
    return _table(header, rows)


def _empirical(document):
    rows = []
    for row in document["empirical"]:
        if row["kind"] == "monitor":
            for sid, event in row["strategies"].items():
                numbers = f"e_value {event['e_value']}, n_obs {event['n_obs']}"
                numbers += f", {event['decision']}"
                rows.append(["monitor", row["stage"], sid, event["session"], numbers, "-"])
        elif "scenarios" in row:
            for name, value in row["scenarios"].items():
                numbers = {k: v for k, v in value.items() if k != "blockers"}
                found = row["blockers"] + value.get("blockers", [])
                cells = [row["kind"], row["stage"], row["id"], name]
                rows.append([*cells, json.dumps(numbers, sort_keys=True), found or "-"])
        else:
            skip = {"kind", "stage", "blockers"}
            numbers = {k: v for k, v in row.items() if k not in skip}
            cells = [row["kind"], row["stage"], "-", "-", json.dumps(numbers, sort_keys=True)]
            rows.append([*cells, row.get("blockers") or "-"])
    return _table(["Kind", "Stage", "Id", "Scenario", "Numbers", "Blockers"], rows)


def markdown(document):
    lines = [f"# Lifecycle: {document['run']}", "", f"`{document['schema']}`. Ids are"]
    lines += ["shortened to 12 characters; LIFECYCLE.json has them in full.", ""]
    lines += ["## Stages", "", *_stages(document), "", "## Recommendations", ""]
    for row in document["recommendations"]:
        lines.append(f"- {row['recommendation']}: {row['strategy_id']} -> `{row['action']}`")
    if not document["recommendations"]:
        lines.append("- none")
    lines += ["", "## Conceptual", "", *_conceptual(document)]
    lines += ["## Lineage", "", *_lineage(document), "", "## Empirical", ""]
    return "\n".join([*lines, *_empirical(document)]) + "\n"
