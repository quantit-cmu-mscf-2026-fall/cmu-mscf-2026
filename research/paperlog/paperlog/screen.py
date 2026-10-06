"""LLM screening: relevance score, tags, extraction, English translation.

Uses forced tool use so the output always matches the schema.
"""

import json
import os

from .db import now

DEFAULT_TEAM = "a quantitative research team"

SYSTEM = """You screen academic papers for {team}
Judge only from the text given. If the abstract is missing, judge from the title, keep \
the score conservative, and say so in notes. Papers may be in Chinese; read them directly \
and give English translations.

Relevance rubric:
{rubric}

Look-ahead bias: when a paper evaluates an LLM on historical market data, check whether it \
addresses leakage from the model's training data overlapping the test period (e.g. test \
periods after the training cutoff, anonymised tickers, or explicit leakage tests). \
Use "not_applicable" if there is no market backtest or prediction evaluation."""


def tool_schema(taxonomy: list[str]) -> dict:
    return {
        "name": "record_screening",
        "description": "Record the screening result for one paper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "relevance": {"type": "integer", "minimum": 0, "maximum": 5},
                "tags": {"type": "array", "items": {"type": "string", "enum": taxonomy}},
                "title_en": {
                    "type": "string",
                    "description": "English title (translate if not English).",
                },
                "summary_en": {
                    "type": "string",
                    "description": (
                        "2-3 sentences: what they did and what it means for our pipeline."
                    ),
                },
                "methods": {"type": "array", "items": {"type": "string"}},
                "datasets": {"type": "array", "items": {"type": "string"}},
                "markets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "e.g. US equities, China A-shares, crypto; empty if none.",
                },
                "eval_method": {"type": "string"},
                "code_url": {"type": "string", "description": "Empty if not mentioned."},
                "lookahead_bias": {
                    "type": "string",
                    "enum": ["controlled", "not_addressed", "not_applicable", "unclear"],
                },
                "notes": {"type": "string"},
            },
            "required": ["relevance", "tags", "title_en", "summary_en", "lookahead_bias"],
        },
    }


def build_prompt(p: dict) -> str:
    return json.dumps(
        {
            "title": p["title"],
            "abstract": p.get("abstract") or "(no abstract available)",
            "venue": p.get("venue") or "",
            "language": p.get("language") or "",
            "institutions": (p.get("institutions") or [])[:8],
            "published": p.get("published") or "",
        },
        ensure_ascii=False,
        indent=1,
    )


class Screener:
    def __init__(self, cfg: dict, client=None):
        self.model = cfg["model"]
        self.rubric_version = cfg.get("rubric_version", 1)
        team = (cfg.get("team_description") or DEFAULT_TEAM).strip().rstrip(".") + "."
        self.system = SYSTEM.format(team=team, rubric=cfg["rubric"].strip())
        self.tool = tool_schema(cfg["taxonomy"])
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.input_tokens = 0
        self.output_tokens = 0

    def screen(self, paper: dict) -> dict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=self.system,
            tools=[self.tool],
            tool_choice={"type": "tool", "name": "record_screening"},
            messages=[{"role": "user", "content": build_prompt(paper)}],
        )
        usage = getattr(msg, "usage", None)
        if usage:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("model returned no tool call")


def save(con, key: str, r: dict, model: str, rubric_version: int = 1) -> None:
    con.execute(
        "INSERT INTO screenings (key, screened_at, model, rubric_version, relevance, tags, "
        "summary_en, payload) VALUES (?,?,?,?,?,?,?,?)",
        (
            key,
            now(),
            model,
            rubric_version,
            int(r.get("relevance", 0)),
            json.dumps(r.get("tags", [])),
            r.get("summary_en", ""),
            json.dumps(r, ensure_ascii=False),
        ),
    )
    con.execute(
        """UPDATE papers SET status='screened', screened_at=?, screen_model=?, relevance=?, tags=?,
           title_en=?, summary_en=?, methods=?, datasets=?, markets=?, eval_method=?, code_url=?,
           lookahead_bias=?, screen_notes=?, rubric_version=?,
           screen_count=COALESCE(screen_count,0)+1 WHERE key=?""",
        (
            now(),
            model,
            int(r.get("relevance", 0)),
            json.dumps(r.get("tags", [])),
            r.get("title_en", ""),
            r.get("summary_en", ""),
            json.dumps(r.get("methods", []), ensure_ascii=False),
            json.dumps(r.get("datasets", []), ensure_ascii=False),
            json.dumps(r.get("markets", []), ensure_ascii=False),
            r.get("eval_method", ""),
            r.get("code_url", ""),
            r.get("lookahead_bias", "unclear"),
            r.get("notes", ""),
            rubric_version,
            key,
        ),
    )
