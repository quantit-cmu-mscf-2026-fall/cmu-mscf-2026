import json
from collections import defaultdict


def render(con, run_id: int, threshold: int, date: str) -> str:
    run = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    rows = con.execute(
        "SELECT * FROM papers WHERE screened_at>=? AND status='screened' AND relevance>=? "
        "ORDER BY relevance DESC, published DESC",
        (run["started"], threshold),
    ).fetchall()

    lines = [f"# Paper digest — {date}", ""]
    if True:
        lines.append(
            f"Window since {run['since']}. Fetched {run['fetched']} records, "
            f"{run['new']} new papers, {run['screened']} screened by the LLM, "
            f"{len(rows)} at relevance ≥ {threshold}."
        )
        lines.append("")
    if not rows:
        lines.append("Nothing above the threshold this run.")
        return "\n".join(lines) + "\n"

    groups = defaultdict(list)
    for r in rows:
        tags = json.loads(r["tags"] or "[]")
        groups[tags[0] if tags else "untagged"].append(r)

    for tag in sorted(groups, key=lambda t: -max(x["relevance"] for x in groups[t])):
        lines += [f"## {tag.replace('_', ' ').capitalize()}", ""]
        for r in groups[tag]:
            flags = []
            if (r["language"] or "en") not in ("en", ""):
                flags.append(f"original in {r['language']}")
            if r["lookahead_bias"] == "not_addressed":
                flags.append("⚠ look-ahead bias not addressed")
            if r["code_url"]:
                flags.append(f"[code]({r['code_url']})")
            title = r["title_en"] or r["title"]
            orig = f" ({r['title']})" if r["title_en"] and r["title_en"] != r["title"] else ""
            url = r["url"] or (f"https://doi.org/{r['doi']}" if r["doi"] else "")
            lines.append(f"**[{r['relevance']}] [{title}]({url})**{orig}  ")
            meta = f"{r['published'] or 'n.d.'} · {r['venue'] or 'unknown venue'}"
            lines.append(f"{meta}{' · ' + ' · '.join(flags) if flags else ''}  ")
            lines.append(f"{r['summary_en']}  ")
            lines.append(f"`{r['key']}`")
            lines.append("")
    lines.append("Record feedback with `paperlog review` so the log learns what you adopt.")
    return "\n".join(lines) + "\n"
