"""A stronger "teacher" model labels papers once, so a cheaper screen can be scored against it.

The teacher reads the same title and abstract Jev reads, the same rubric text and the same
draft tag definitions (research/jev_paper_tags.toml), and answers with structured output:
a relevance level 0-5, a list of taxonomy tags and a one-sentence reason. Differences from
Jev are then about the model, not the instructions.

Answers are appended to a JSONL cache keyed on paper and design hash (model, system prompt,
schema, effort, max_tokens), so a rerun resumes without paying twice. Errors, refusals and
truncated replies are written as error rows and are never treated as labels. The caller logs
the run to the ledger before `run_teacher` is called; the call cap is checked before any call.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jev_paper_screen as jps

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_TOKENS = 4000
WORKERS = 8
MAX_RETRIES = 6
PRICE_INPUT_PER_M = 2.0
PRICE_OUTPUT_PER_M = 10.0
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


def build_system(cfg: dict, definitions: dict[str, str]) -> str:
    """The fixed, cacheable instructions: reader, rubric verbatim, tag definitions."""
    team = " ".join(cfg["team_description"].split())
    tag_lines = "\n".join(f"- {tag}: {text}" for tag, text in definitions.items())
    return (
        "You label academic papers for a research team's paper-screening pipeline. You see "
        "only the title and abstract.\n\n"
        f"The reader is {team}\n\n"
        "Relevance: give one integer 0-5 using this rubric. Two kinds of value share one "
        "scale, METHODS and FINDINGS; score whichever is higher. Publication year does not "
        "matter.\n\n"
        f"{cfg['rubric'].strip()}\n\n"
        "Tags: list every taxonomy tag whose question below is answered yes for the paper "
        "(none, one or several). Use only these tag names.\n\n"
        f"{tag_lines}\n\n"
        "Also give a one-sentence reason for the relevance level."
    )


def build_schema(tags: list[str], top: int = 5) -> dict:
    return {
        "type": "object",
        "properties": {
            "relevance": {"type": "integer", "enum": list(range(top + 1))},
            "tags": {"type": "array", "items": {"type": "string", "enum": list(tags)}},
            "reason": {"type": "string"},
        },
        "required": ["relevance", "tags", "reason"],
        "additionalProperties": False,
    }


def design_hash(system: str, schema: dict, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    blob = json.dumps(
        {"m": model, "s": system, "j": schema, "e": EFFORT, "t": max_tokens}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def user_message(paper: dict) -> str:
    state = jps.build_state(paper)
    return f"Title: {state['title']}\n\nAbstract: {state['abstract']}"


class TeacherParseError(ValueError):
    """The reply is not a valid label."""


def parse_label(text: str, tags: list[str], top: int = 5) -> dict:
    try:
        data = json.loads(text)
        relevance, got, reason = data["relevance"], data["tags"], data["reason"]
    except (ValueError, KeyError, TypeError):
        raise TeacherParseError("reply is not the expected JSON object") from None
    if isinstance(relevance, bool) or not isinstance(relevance, int) or not 0 <= relevance <= top:
        raise TeacherParseError(f"relevance {relevance!r} is not an integer in 0..{top}")
    if not isinstance(got, list) or any(t not in tags for t in got):
        raise TeacherParseError(f"tags {got!r} are not a subset of the taxonomy")
    if not isinstance(reason, str):
        raise TeacherParseError("reason is not a string")
    return {"relevance": relevance, "tags": sorted(set(got)), "reason": reason.strip()}


def _usage(response) -> dict:
    u = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def label_one(
    client, paper: dict, system: str, schema: dict, tags: list[str], design: str, max_tokens: int
) -> dict:
    """One teacher call; any failure becomes an error row rather than a label."""
    row = {"key": paper["key"], "model": MODEL, "design": design, "error": ""}
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_message(paper)}],
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["seconds"] = time.perf_counter() - start
        return row
    row["seconds"] = time.perf_counter() - start
    row["usage"] = _usage(response)
    row["stop_reason"] = getattr(response, "stop_reason", None)
    if row["stop_reason"] == "refusal":
        row["error"] = "refusal"
    elif row["stop_reason"] != "end_turn":
        row["error"] = f"stop_reason {row['stop_reason']}"
    else:
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        try:
            row |= parse_label(text, tags)
        except TeacherParseError as exc:
            row["error"] = str(exc)
    return row


def load_cache(path: Path, design: str) -> dict:
    cache: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("design") == design and not row["error"]:
                    cache[row["key"]] = row
    return cache


def pending(papers: list[dict], out: Path, design: str) -> list[dict]:
    cache = load_cache(out / "teacher_cache.jsonl", design)
    return [p for p in papers if p["key"] not in cache]


def check_cap(papers: list[dict], out: Path, design: str, max_calls: int) -> int:
    """Number of calls needed; refuses (SystemExit) when over the cap."""
    needed = len(pending(papers, out, design))
    if needed > max_calls:
        raise SystemExit(
            f"refusing to start: {needed} teacher calls needed, --max-teacher-calls {max_calls}"
        )
    return needed


def run_teacher(
    papers: list[dict],
    system: str,
    tags: list[str],
    out: Path,
    client,
    max_calls: int,
    workers: int = WORKERS,
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[dict], int]:
    """Label every uncached paper; returns (label rows in sample order, error count this run)."""
    out.mkdir(parents=True, exist_ok=True)
    schema = build_schema(tags)
    design = design_hash(system, schema, MODEL, max_tokens)
    todo = pending(papers, out, design)
    if len(todo) > max_calls:
        raise SystemExit(
            f"refusing to start: {len(todo)} teacher calls needed, --max-teacher-calls {max_calls}"
        )
    cache = load_cache(out / "teacher_cache.jsonl", design)
    errors = 0
    with open(out / "teacher_cache.jsonl", "a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [
                pool.submit(label_one, client, p, system, schema, tags, design, max_tokens)
                for p in todo
            ]
            for future in as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if row["error"]:
                    errors += 1
                else:
                    cache[row["key"]] = row
    return [cache[p["key"]] for p in papers if p["key"] in cache], errors


def make_client():  # pragma: no cover - needs the SDK and a key
    import anthropic

    return anthropic.Anthropic(max_retries=MAX_RETRIES)


def cost_usd(rows: list[dict]) -> dict:
    """Token totals and list-price cost: cache reads 0.1x, 5-minute cache writes 1.25x input."""
    use = [r.get("usage", {}) for r in rows]
    tok = {
        k: sum(u.get(k, 0) for u in use)
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    }
    usd = (
        tok["input_tokens"] * PRICE_INPUT_PER_M
        + tok["cache_read_input_tokens"] * PRICE_INPUT_PER_M * CACHE_READ_FACTOR
        + tok["cache_creation_input_tokens"] * PRICE_INPUT_PER_M * CACHE_WRITE_FACTOR
        + tok["output_tokens"] * PRICE_OUTPUT_PER_M
    ) / 1e6
    return {**tok, "usd": usd}
