"""Cheap keyword gate before any LLM call. Tuned for recall, not precision:
the goal is only to avoid paying to screen obviously unrelated papers."""


def passes(title: str, abstract: str, cfg: dict) -> bool:
    text = f"{title} {abstract}".lower()
    agent = any(t.lower() in text for t in cfg.get("agent_terms", []))
    finance = any(t.lower() in text for t in cfg.get("finance_terms", []))
    if cfg.get("mode", "both") == "either":
        return agent or finance
    return agent and finance
