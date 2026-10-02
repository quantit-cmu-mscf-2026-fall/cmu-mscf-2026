"""Command line for the factor pipeline.

    python -m capstone.factors run --source file papers.toml
    python -m capstone.factors run --source paperlog --min-relevance 3 --limit 10
    python -m capstone.factors run --source paperlog --key arxiv:2502.16789
    python -m capstone.factors list
    python -m capstone.factors lineage <factor_id>
    python -m capstone.factors stats
    python -m capstone.factors methods [--paper KEY]

`methods` prints the method hypotheses (claims about how to search or
validate, which never become factors) as Markdown, by paper, to paste into
the paper's issue or a design note.

`run` calls the model and needs credentials and the agents extra. The
other commands only read the store (`--db`, else CAPSTONE_FACTOR_DB, else
experiments/factors.db).
"""

from __future__ import annotations

import argparse
import sys

from capstone.factors import store
from capstone.factors.llm import load_config, make_client
from capstone.factors.pipeline import run
from capstone.factors.sources import FileSource, PaperlogSource


def _run(args, con) -> int:
    config = load_config(args.config)
    if args.source == "file":
        if not args.path:
            print("--source file needs a path to a papers .toml file", file=sys.stderr)
            return 2
        source = FileSource(args.path, keys=args.key)
    else:
        source = PaperlogSource(args.paperlog_db, args.min_relevance, args.limit, keys=args.key)
    try:
        client = make_client()
    except (RuntimeError, ImportError) as exc:
        print(exc, file=sys.stderr)
        return 2
    summary = run(con, source, client, config)
    print(summary)
    return 1 if summary.papers_failed else 0


def _list(args, con) -> int:
    for row in store.factors(con):
        print(f"{row['id']}  {row['expression']}")
    return 0


def _lineage(args, con) -> int:
    rows = store.lineage(con, args.factor_id)
    if not rows:
        print(f"no factor {args.factor_id}", file=sys.stderr)
        return 1
    for row in rows:
        parent = f" from {row['parent_factor_id']}" if row["parent_factor_id"] else ""
        print(f"[{row['status']}, {row['edge_type']}{parent}] {row['expression']}")
        print(f"  paper: {row['paper_title']} ({row['paper_key']})")
        print(f"  hypothesis {row['hypothesis_id']}: {row['specification']}")
        print(f"  rationale: {row['rationale']}  [{row['model']}, {row['created_at']}]")
    return 0


def _methods(args, con) -> int:
    sql = (
        "SELECT h.*, p.title, p.url, p.citation FROM hypotheses h "
        "JOIN papers p ON p.key = h.paper_key WHERE h.kind = 'method'"
    )
    params: tuple = ()
    if args.paper:
        sql += " AND h.paper_key = ?"
        params = (args.paper,)
    rows = con.execute(sql + " ORDER BY p.key, h.created_at, h.id", params).fetchall()
    if not rows:
        print("no method hypotheses", file=sys.stderr)
        return 1
    paper = None
    for row in rows:
        if row["paper_key"] != paper:
            paper = row["paper_key"]
            cite = f" ({row['citation']})" if row["citation"] else ""
            print(f"\n## {row['title']}{cite}\n\n{row['url'] or paper}\n")
        print(f"- **Claim:** {row['observation']}")
        print(f"  - *Why:* {row['justification']}")
        print(f"  - *Test:* {row['specification']}")
        print(f"  - *Wrong if:* {row['falsification_condition']}")
    return 0


def _stats(args, con) -> int:
    counts = dict(con.execute("SELECT status, COUNT(*) FROM proposals GROUP BY status").fetchall())
    papers = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    hypotheses = con.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    print(f"papers {papers}, hypotheses {hypotheses}, factors {len(store.factors(con))}")
    print(f"proposals {sum(counts.values())}: {counts}")
    kinds = dict(con.execute("SELECT kind, COUNT(*) FROM hypotheses GROUP BY kind").fetchall())
    print(f"hypotheses by kind: {kinds}")
    for model, calls, tokens_in, tokens_out in con.execute(
        "SELECT model, SUM(calls), SUM(input_tokens), SUM(output_tokens) FROM usage GROUP BY model"
    ):
        print(f"{model}: {calls} calls, {tokens_in} tokens in, {tokens_out} tokens out")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capstone.factors")
    parser.add_argument("--db", help="factor store path (default: CAPSTONE_FACTOR_DB)")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="papers -> hypotheses -> factors")
    run_cmd.add_argument("--source", choices=("file", "paperlog"), required=True)
    run_cmd.add_argument("path", nargs="?", help="papers .toml file, for --source file")
    run_cmd.add_argument("--paperlog-db", help="paperlog database (default: PAPERLOG_DB)")
    run_cmd.add_argument("--min-relevance", type=int, default=0)
    run_cmd.add_argument("--limit", type=int, default=20)
    run_cmd.add_argument(
        "--key", action="append", help="run only this paper (e.g. arxiv:2502.16789); repeatable"
    )
    run_cmd.add_argument("--config", help="parameters file (default: config/factors.toml)")

    commands.add_parser("list", help="stored factors")
    lineage_cmd = commands.add_parser("lineage", help="paper and hypothesis behind a factor")
    lineage_cmd.add_argument("factor_id")
    commands.add_parser("stats", help="counts, including every proposal")
    methods_cmd = commands.add_parser("methods", help="method hypotheses as Markdown")
    methods_cmd.add_argument("--paper", help="only this paper's key")

    args = parser.parse_args(argv)
    con = store.connect(args.db)
    try:
        handler = {
            "run": _run,
            "list": _list,
            "lineage": _lineage,
            "stats": _stats,
            "methods": _methods,
        }
        return handler[args.command](args, con)
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
