import argparse
import csv
import json
import os
import sys
from datetime import date

import yaml

from . import collect, dedupe, digest, prefilter, resolve
from . import db as dbm
from .paths import config_path, db_path
from .sources import arxiv, openalex
from .sources import semantic_scholar as s2


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    learned = os.path.join(os.path.dirname(os.path.abspath(path)), "learned_queries.yaml")
    cfg["_learned_path"] = learned
    if os.path.exists(learned):
        with open(learned, encoding="utf-8") as f:
            cfg["_learned"] = yaml.safe_load(f) or {}
    else:
        cfg["_learned"] = {}
    return cfg


def learned(cfg, source):
    return list((cfg.get("_learned") or {}).get(source, []))


# ---------------------------------------------------------------- collection
def anchors_for_run(con, cfg, log=print):
    """Config anchors plus every adopted paper: the search grows toward whatever
    is actually producing hypotheses."""
    out, needs_refs = [], []
    wanted = [(t, "config") for t in cfg["semantic_scholar"].get("anchors", [])]
    if cfg["semantic_scholar"].get("adopted_as_anchors", True):
        cap = cfg["semantic_scholar"].get("max_adopted_anchors", 25)
        for r in con.execute(
            "SELECT title, s2_id FROM papers WHERE status='adopted' "
            "ORDER BY reviewed_at DESC LIMIT ?",
            (cap,),
        ):
            wanted.append((r["title"], "adopted"))

    for title, source in wanted:
        row = con.execute("SELECT * FROM anchors WHERE title=?", (title,)).fetchone()
        pid = row["s2_id"] if row else None
        if not pid:
            try:
                pid = s2.resolve(title)
            except Exception as e:
                log(f"  s2: could not resolve anchor '{title[:50]}': {e}")
                continue
            if not pid:
                log(f"  s2: no match for anchor '{title[:50]}'")
                continue
            con.execute(
                "INSERT OR REPLACE INTO anchors (title, s2_id, source) VALUES (?,?,?)",
                (title, pid, source),
            )
        out.append((title, pid))
        if source == "adopted" and not (row and row["refs_done"]):
            needs_refs.append((title, pid))
    con.commit()
    return out, needs_refs


def collect_papers(cfg, plan, anchors, ref_anchors, log=print):
    raw = []
    expl = plan.exploration
    if cfg["arxiv"].get("enabled", True):
        extra = learned(cfg, "arxiv") + (
            cfg.get("exploration", {}).get("arxiv", []) if expl else []
        )
        raw += arxiv.fetch(cfg["arxiv"], plan.since, plan.depth, extra, log)
    if cfg["openalex"].get("enabled", True):
        extra = learned(cfg, "openalex") + (
            cfg.get("exploration", {}).get("openalex", []) if expl else []
        )
        raw += openalex.fetch(cfg["openalex"], plan.since, plan.depth, extra, log)
    if cfg["semantic_scholar"].get("enabled", True) and anchors:
        cap = cfg["semantic_scholar"].get("max_citations_per_anchor", 2000) * plan.depth
        raw += s2.fetch_citations(anchors, plan.since, cap, log)
        if ref_anchors:
            raw += s2.fetch_references(
                ref_anchors, cfg["semantic_scholar"].get("max_references", 200), log
            )
    return raw


def store_batch(con, papers, run_id, log=print):
    batch = dedupe.dedupe_batch(papers)
    db_titles = dbm.all_titles(con)
    new_keys, seen = [], 0
    for p in batch:
        existing = dbm.find_existing(con, p) or dedupe.match_db(p, db_titles)
        if existing:
            dbm.add_sighting(con, existing, p)
            seen += 1
            continue
        key = p.key
        if con.execute("SELECT 1 FROM papers WHERE key=?", (key,)).fetchone():
            dbm.add_sighting(con, key, p)
            seen += 1
            continue
        dbm.insert(con, p, run_id, key)
        new_keys.append(key)
        db_titles.append((key, p.title_norm, p.arxiv_id, p.doi))
    con.commit()
    log(
        f"  {len(papers)} records -> {len(batch)} unique -> {len(new_keys)} new, "
        f"{seen} already logged"
    )
    return new_keys, seen


# ---------------------------------------------------------------- screening
def classify(con, keys, pf_cfg, loosen: bool):
    """Mark new papers as screening candidates or filtered (a revisitable label)."""
    cfg = dict(pf_cfg)
    if loosen:
        cfg["mode"] = "either"
    version = pf_cfg.get("version", 1)
    passed = []
    for k in keys:
        r = con.execute(
            "SELECT key, title, abstract, status FROM papers WHERE key=?", (k,)
        ).fetchone()
        if r["status"] == "duplicate":
            continue
        if prefilter.passes(r["title"], r["abstract"] or "", cfg):
            passed.append(k)
            con.execute("UPDATE papers SET prefilter_version=? WHERE key=?", (version, k))
        else:
            con.execute(
                "UPDATE papers SET status='filtered', relevance=0, prefilter_version=? WHERE key=?",
                (version, k),
            )
    con.commit()
    return passed


def revisit_filtered(con, pf_cfg, loosen: bool, log=print):
    """Papers the prefilter rejected under an older/stricter setting get another look."""
    cfg = dict(pf_cfg)
    if loosen:
        cfg["mode"] = "either"
    version = pf_cfg.get("version", 1)
    rows = con.execute(
        "SELECT key, title, abstract FROM papers WHERE status='filtered' AND "
        "(prefilter_version IS NULL OR prefilter_version < ? OR ?)",
        (version, 1 if loosen else 0),
    ).fetchall()
    revived = []
    for r in rows:
        if prefilter.passes(r["title"], r["abstract"] or "", cfg):
            con.execute(
                "UPDATE papers SET status='new', prefilter_version=? WHERE key=?",
                (version, r["key"]),
            )
            revived.append(r["key"])
    con.commit()
    if revived:
        log(f"  {len(revived)} previously filtered papers pass the current prefilter")
    return revived


def screen_keys(con, keys, sc, log=print):
    """Screen a list of keys. Returns how many succeeded."""
    if not keys:
        return 0
    import anthropic

    from .screen import Screener, save

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (or pass --no-screen)")
    s = Screener(sc)
    done, fails = 0, 0
    for k in keys:
        row = dbm.row_to_dict(con.execute("SELECT * FROM papers WHERE key=?", (k,)).fetchone())
        try:
            save(con, k, s.screen(row), sc["model"], sc.get("rubric_version", 1))
            done += 1
            fails = 0
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            log(f"  Anthropic rejected the API key ({e.status_code}). Stopping screening.")
            log("  Check ANTHROPIC_API_KEY; unscreened papers will be retried next run.")
            break
        except Exception as e:
            fails += 1
            log(f"  screening failed for {k}: {e}")
            if fails >= 5:
                log("  5 failures in a row; stopping. Papers will be retried next run.")
                break
        if done and done % 20 == 0:
            con.commit()
    con.commit()
    log(f"  screened {done} papers ({s.input_tokens:,} in / {s.output_tokens:,} out tokens)")
    return done


def candidates(con, cfg, fresh_keys, log=print):
    """Screening queue, in priority order: new papers, then backlog, then
    papers scored under an older rubric."""
    sc = cfg["screening"]
    budget = sc.get("budget_per_run", 300)
    queue = list(dict.fromkeys(fresh_keys))
    backlog = [
        r["key"]
        for r in con.execute("SELECT key FROM papers WHERE status='new' ORDER BY first_seen DESC")
    ]
    queue += [k for k in backlog if k not in set(queue)]
    if len(queue) < budget and sc.get("rescreen_when_rubric_changes", True):
        stale = [
            r["key"]
            for r in con.execute(
                "SELECT key FROM papers WHERE status='screened' AND "
                "(rubric_version IS NULL OR rubric_version < ?) "
                "ORDER BY COALESCE(relevance,0) DESC, published DESC LIMIT ?",
                (sc.get("rubric_version", 1), budget - len(queue)),
            )
        ]
        if stale:
            log(
                f"  {len(stale)} papers queued for rescreening "
                f"under rubric v{sc.get('rubric_version', 1)}"
            )
        queue += stale
    return queue[:budget]


# ---------------------------------------------------------------- run
def cmd_run(args, cfg):
    con = dbm.connect(args.db)
    coll = cfg.get("collection", {})
    target = args.target or coll.get("target_new_papers", 150)
    start = collect.base_since(con, coll, args.since, args.days)
    cur = con.execute("INSERT INTO runs (started, since) VALUES (?, ?)", (dbm.now(), start))
    run_id = cur.lastrowid
    con.commit()

    anchors, ref_anchors = ([], [])
    if cfg["semantic_scholar"].get("enabled", True):
        anchors, ref_anchors = anchors_for_run(con, cfg)
        print(
            f"Run {run_id}: {len(anchors)} citation anchors "
            f"({len(ref_anchors)} new adopted papers to backfill references from)"
        )

    plans = collect.ladder(coll, start)
    fresh, fetched, level, loosen = [], 0, 0, False
    for plan in plans:
        print(f"Level {plan.level}: {plan.describe()} (target {target}, have {len(fresh)})")
        raw = collect_papers(cfg, plan, anchors, ref_anchors if plan.level == 0 else [], print)
        fetched += len(raw)
        keys, _ = store_batch(con, raw, run_id)
        fresh += keys
        level, loosen = plan.level, plan.loosen_prefilter
        for label, _pid in ref_anchors:
            con.execute("UPDATE anchors SET refs_done=? WHERE title=?", (dbm.now(), label))
        con.commit()
        if len(fresh) >= target:
            break
        if plan.level > 0 and len(keys) < coll.get("min_yield_per_level", 5):
            print(f"  level added only {len(keys)} new papers; stopping escalation")
            break
    print(f"Collected {len(fresh)} new papers (levels 0-{level}, {fetched} records fetched)")

    emb = cfg.get("embeddings", {})
    if emb.get("enabled") and fresh:
        from . import embed

        embed.link_cross_language(con, fresh, emb)
        con.commit()

    passed = classify(con, fresh, cfg["prefilter"], loosen)
    print(f"{len(passed)} of the new papers pass the keyword prefilter")
    revisit_filtered(con, cfg["prefilter"], loosen)

    screened = 0
    if not args.no_screen:
        queue = candidates(con, cfg, passed)
        print(
            f"Screening queue: {len(queue)} papers "
            f"(budget {cfg['screening'].get('budget_per_run', 300)})"
        )
        screened = screen_keys(con, queue, cfg["screening"])

    con.execute(
        "UPDATE runs SET finished=?, fetched=?, new=?, screened=?, notes=? WHERE id=?",
        (dbm.now(), fetched, len(fresh), screened, f"level {level}", run_id),
    )
    con.commit()

    os.makedirs(args.digest_dir, exist_ok=True)
    today = date.today().isoformat()
    path = os.path.join(args.digest_dir, f"{today}.md")
    if screened == 0 and os.path.exists(path):
        print(f"Nothing newly screened; keeping existing digest {path}")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(digest.render(con, run_id, cfg["screening"].get("digest_threshold", 3), today))
        print(f"Digest written to {path}")
    if args.csv:
        export_csv(con, args.csv)
        print(f"CSV export written to {args.csv}")


# ---------------------------------------------------------------- add
def cmd_add(args, cfg):
    con = dbm.connect(args.db)
    paper, how = resolve.resolve(args.identifier)
    if not paper:
        sys.exit(f"Could not resolve '{args.identifier}': {how}")
    paper.sources = list(dict.fromkeys(paper.sources + ["manual"]))
    paper.found_via = list(dict.fromkeys(paper.found_via + ["manual"]))
    print(f"Resolved via {how}: {paper.title[:90]}")

    key = dbm.find_existing(con, paper) or dedupe.match_db(paper, dbm.all_titles(con))
    if key:
        dbm.add_sighting(con, key, paper)
        print(f"Already in the log as {key}")
    else:
        key = dbm.insert(con, paper, 0)
        con.execute(
            "UPDATE papers SET prefilter_version=? WHERE key=?",
            (cfg["prefilter"].get("version", 1), key),
        )
        print(f"Added as {key}")
    con.commit()

    if not args.no_screen:
        screen_keys(con, [key], cfg["screening"])
        row = con.execute("SELECT relevance, summary_en FROM papers WHERE key=?", (key,)).fetchone()
        if row and row["relevance"] is not None:
            print(f"  screener: {row['relevance']}/5 — {row['summary_en']}")
    if args.adopt:
        set_review(con, key, "adopted", args.rating, args.notes)
        print("  marked adopted")
    elif args.rating is not None or args.notes:
        set_review(con, key, None, args.rating, args.notes)
    if args.anchor or (args.adopt and cfg["semantic_scholar"].get("adopted_as_anchors", True)):
        title = con.execute("SELECT title FROM papers WHERE key=?", (key,)).fetchone()["title"]
        con.execute(
            "INSERT OR IGNORE INTO anchors (title, s2_id, source) VALUES (?,?,?)",
            (title, paper.s2_id, "manual" if args.anchor else "adopted"),
        )
        print("  will be used as a citation anchor in future runs")
    con.commit()
    print(key)
    return key


# ---------------------------------------------------------------- rescreen
def cmd_rescreen(args, cfg):
    con = dbm.connect(args.db)
    sc = cfg["screening"]
    v = sc.get("rubric_version", 1)
    if args.all:
        sql, params = (
            "SELECT key FROM papers WHERE status IN ('screened','read','adopted','rejected','new')"
            " ORDER BY COALESCE(relevance,0) DESC",
            (),
        )
    elif args.filtered:
        sql, params = (
            "SELECT key FROM papers WHERE status='filtered' ORDER BY first_seen DESC",
            (),
        )
    else:
        sql, params = (
            "SELECT key FROM papers WHERE status='screened' AND "
            "(rubric_version IS NULL OR rubric_version < ?) "
            "ORDER BY COALESCE(relevance,0) DESC",
            (v,),
        )
    keys = [r["key"] for r in con.execute(sql, params)][: args.limit]
    print(f"Rescreening {len(keys)} papers under rubric v{v}")
    before = {r["key"]: r["relevance"] for r in con.execute("SELECT key, relevance FROM papers")}
    screen_keys(con, keys, sc)
    moved = [
        (k, before.get(k), r["relevance"])
        for k in keys
        for r in [con.execute("SELECT relevance FROM papers WHERE key=?", (k,)).fetchone()]
        if before.get(k) is not None and r and r["relevance"] != before.get(k)
    ]
    if moved:
        print(f"\n{len(moved)} scores changed:")
        for k, old, new in sorted(moved, key=lambda x: -(abs((x[2] or 0) - (x[1] or 0))))[:20]:
            t = con.execute(
                "SELECT COALESCE(title_en, title) t FROM papers WHERE key=?", (k,)
            ).fetchone()["t"]
            print(f"  {old} -> {new}  {t[:80]}")


# ---------------------------------------------------------------- pending (ingestion handoff)
def cmd_pending(args, cfg):
    from . import api

    rows = api.pending(min_relevance=args.min_relevance, limit=args.limit, db_path=args.db)
    if args.claim:
        api.mark_ingested([r["key"] for r in rows], db_path=args.db)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            print(f"[{r['relevance']}] {r['key']}  {(r['title_en'] or r['title'])[:80]}")
        print(
            f"\n{len(rows)} papers pending ingestion"
            + (" (now claimed)" if args.claim else " (use --claim to mark them taken)")
        )


# ---------------------------------------------------------------- suggest queries
SUGGEST_SYSTEM = """You propose literature search queries. Given papers a quantitative research
team has adopted, suggest search queries that would surface similar future work.
Return JSON only: {"arxiv": [...], "openalex": [...]}. arXiv entries use arXiv query syntax
(e.g. 'all:"agentic workflow" AND (all:finance OR all:trading)'). OpenAlex entries are plain
keyword phrases; include Chinese-language phrases where the topic warrants it.
Propose at most 6 per source, and only queries meaningfully different from the existing ones."""


def cmd_suggest_queries(args, cfg):
    import anthropic

    con = dbm.connect(args.db)
    rows = [
        dbm.row_to_dict(r)
        for r in con.execute(
            "SELECT * FROM papers WHERE status='adopted' OR human_rating >= 4 "
            "ORDER BY reviewed_at DESC LIMIT 40"
        )
    ]
    if not rows:
        sys.exit("No adopted or highly rated papers yet; nothing to learn from.")
    payload = {
        "existing_arxiv": cfg["arxiv"].get("queries", []) + learned(cfg, "arxiv"),
        "existing_openalex": cfg["openalex"].get("queries", []) + learned(cfg, "openalex"),
        "adopted_papers": [
            {
                "title": r.get("title_en") or r["title"],
                "summary": r.get("summary_en", ""),
                "tags": r.get("tags", []),
            }
            for r in rows
        ],
    }
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=cfg["screening"]["model"],
        max_tokens=1500,
        system=SUGGEST_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    try:
        proposed = json.loads(text.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        sys.exit(f"Could not parse model output:\n{text[:500]}")

    merged = dict(cfg.get("_learned") or {})
    for src in ("arxiv", "openalex"):
        merged[src] = list(dict.fromkeys(list(merged.get(src, [])) + list(proposed.get(src, []))))
    print("Proposed queries:")
    for src in ("arxiv", "openalex"):
        for q in proposed.get(src, []):
            print(f"  [{src}] {q}")
    if args.write:
        with open(cfg["_learned_path"], "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
        print(f"\nWritten to {cfg['_learned_path']} (edit or prune it freely)")
    else:
        print(f"\nNot saved. Re-run with --write to add them to {cfg['_learned_path']}")


# ---------------------------------------------------------------- review / stats / export
def set_review(con, key, status, rating, notes):
    con.execute(
        "UPDATE papers SET status=COALESCE(?, status), human_rating=COALESCE(?, human_rating), "
        "human_notes=COALESCE(?, human_notes), reviewed_at=? WHERE key=?",
        (status, rating, notes, dbm.now(), key),
    )


def cmd_review(args, cfg):
    con = dbm.connect(args.db)
    if args.key:
        if not con.execute("SELECT 1 FROM papers WHERE key=?", (args.key,)).fetchone():
            sys.exit(f"No paper with key {args.key}")
        set_review(con, args.key, args.status, args.rating, args.notes)
        con.commit()
        print(f"Updated {args.key}")
        return
    threshold = cfg["screening"].get("digest_threshold", 3)
    rows = con.execute(
        "SELECT * FROM papers WHERE status='screened' AND relevance>=? "
        "ORDER BY relevance DESC, published DESC",
        (threshold,),
    ).fetchall()
    print(f"{len(rows)} screened papers awaiting review. [a]dopt [r]ead [x]reject [s]kip [q]uit\n")
    choice_map = {"a": "adopted", "r": "read", "x": "rejected"}
    for r in rows:
        print(f"[{r['relevance']}] {r['title_en'] or r['title']}")
        print(f"    {r['published']} · {r['venue']} · {r['url']}")
        print(f"    {r['summary_en']}")
        c = input("    > ").strip().lower()
        if c == "q":
            break
        if c not in choice_map:
            continue
        rating = input("    your rating 0-5 (enter to skip): ").strip()
        set_review(con, r["key"], choice_map[c], int(rating) if rating.isdigit() else None, None)
        con.commit()
        print()


def cmd_stats(args, cfg):
    con = dbm.connect(args.db)
    thr = cfg["screening"].get("digest_threshold", 3)
    rows = [
        dbm.row_to_dict(r) for r in con.execute("SELECT * FROM papers WHERE status!='duplicate'")
    ]
    rel = [r for r in rows if (r.get("relevance") or 0) >= thr]
    print(
        f"Papers: {len(rows)}  |  relevant (>= {thr}): {len(rel)}  |  "
        f"adopted: {sum(1 for r in rows if r.get('status') == 'adopted')}  |  "
        f"filtered: {sum(1 for r in rows if r.get('status') == 'filtered')}  |  "
        f"unscreened: {sum(1 for r in rows if r.get('status') == 'new')}"
    )

    counts = {}
    for r in rows:
        for s in r.get("sources") or []:
            counts[s] = counts.get(s, 0) + 1
    print(
        "\nBy source:  "
        + "  ".join(f"{s}={n}" for s, n in sorted(counts.items(), key=lambda x: -x[1]))
    )

    vers = con.execute(
        "SELECT rubric_version v, COUNT(*) n FROM papers WHERE relevance IS NOT NULL "
        "GROUP BY rubric_version ORDER BY v"
    ).fetchall()
    print("Rubric versions: " + "  ".join(f"v{r['v']}={r['n']}" for r in vers))
    ev = con.execute("SELECT COUNT(*) n FROM screenings").fetchone()["n"]
    print(f"Screening events recorded: {ev}")

    rated = [
        r for r in rows if r.get("human_rating") is not None and r.get("relevance") is not None
    ]
    if rated:
        mad = sum(abs(r["human_rating"] - r["relevance"]) for r in rated) / len(rated)
        print(f"\nCalibration on {len(rated)} rated papers: mean |screener - you| = {mad:.2f}")
        misses = sorted(
            [r for r in rated if r["human_rating"] - r["relevance"] >= 2],
            key=lambda r: r["relevance"] - r["human_rating"],
        )
        if misses:
            print(
                f"  {len(misses)} papers you rated much higher than the screener "
                "(the misses worth fixing the rubric for):"
            )
            for r in misses[:10]:
                print(
                    f"   screener {r['relevance']} vs you {r['human_rating']}: "
                    f"{(r.get('title_en') or r['title'])[:70]}"
                )
    else:
        print("\nNo human ratings yet; rate papers with `review` to check calibration.")


EXPORT_COLS = [
    "key",
    "status",
    "relevance",
    "human_rating",
    "rubric_version",
    "title_en",
    "title",
    "published",
    "venue",
    "language",
    "countries",
    "tags",
    "summary_en",
    "lookahead_bias",
    "markets",
    "methods",
    "datasets",
    "code_url",
    "url",
    "sources",
    "found_via",
]


def export_csv(con, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(EXPORT_COLS)
        for r in con.execute(
            "SELECT * FROM papers WHERE status NOT IN ('filtered','duplicate') "
            "ORDER BY COALESCE(relevance,-1) DESC, published DESC"
        ):
            d = dbm.row_to_dict(r)
            w.writerow(
                [
                    "; ".join(map(str, d.get(c))) if isinstance(d.get(c), list) else d.get(c)
                    for c in EXPORT_COLS
                ]
            )


def cmd_export(args, cfg):
    export_csv(dbm.connect(args.db), args.out)
    print(f"Wrote {args.out}")


# ---------------------------------------------------------------- entry point
def main(argv=None):
    ap = argparse.ArgumentParser(prog="paperlog")
    ap.add_argument(
        "--config", default=None, help="default: $PAPERLOG_CONFIG or the bundled config.yaml"
    )
    ap.add_argument("--db", default=None, help="default: $PAPERLOG_DB or ./papers.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="collect, screen, write digest")
    r.add_argument("--since", help="YYYY-MM-DD start of the first level's window")
    r.add_argument("--days", type=int, help="look back N days instead")
    r.add_argument("--target", type=int, help="new papers to aim for before escalating stops")
    r.add_argument("--no-screen", action="store_true")
    r.add_argument("--digest-dir", default="digests")
    r.add_argument("--csv", default="papers.csv", help="'' to skip")

    a = sub.add_parser("add", help="add a paper you found (arXiv id/URL, DOI, S2 URL, or title)")
    a.add_argument("identifier")
    a.add_argument("--adopt", action="store_true", help="mark adopted (and use as a search anchor)")
    a.add_argument("--anchor", action="store_true", help="use as a citation anchor")
    a.add_argument("--rating", type=int, choices=range(0, 6))
    a.add_argument("--notes")
    a.add_argument("--no-screen", action="store_true")

    rs = sub.add_parser("rescreen", help="rescore papers under the current rubric")
    rs.add_argument("--all", action="store_true", help="every paper, not just stale ones")
    rs.add_argument("--filtered", action="store_true", help="papers the prefilter rejected")
    rs.add_argument("--limit", type=int, default=500)

    p = sub.add_parser("pending", help="papers not yet ingested downstream")
    p.add_argument("--min-relevance", type=int, default=0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    p.add_argument("--claim", action="store_true", help="mark them ingested")

    sq = sub.add_parser("suggest-queries", help="propose new search queries from adopted papers")
    sq.add_argument("--write", action="store_true")

    v = sub.add_parser("review", help="record your feedback")
    v.add_argument("key", nargs="?")
    v.add_argument("--status", choices=["read", "adopted", "rejected"])
    v.add_argument("--rating", type=int, choices=range(0, 6))
    v.add_argument("--notes")

    sub.add_parser("stats", help="coverage, rubric versions, calibration")
    e = sub.add_parser("export", help="CSV for Google Sheets / Excel")
    e.add_argument("--out", default="papers.csv")

    args = ap.parse_args(argv)
    args.config = args.config or config_path()
    args.db = args.db or db_path()
    cfg = load_config(args.config)
    {
        "run": cmd_run,
        "add": cmd_add,
        "rescreen": cmd_rescreen,
        "pending": cmd_pending,
        "suggest-queries": cmd_suggest_queries,
        "review": cmd_review,
        "stats": cmd_stats,
        "export": cmd_export,
    }[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
