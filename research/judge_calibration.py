"""Calibrate the factor-alignment judge across backends on items with known labels.

The judge (`capstone.factors.alignment`) answers three yes/no checks: does the
formula use the specified variables, point the specified way, and measure over
the specified horizon. To pick a model and a threshold on evidence, this builds
items whose answers are known by construction from stored market proposals:

- original: the proposal as stored; expected to pass all three checks.
- direction: the formula negated; expected direction=False.
- horizon: every window rescaled x4 (or /4 near the top); expected horizon=False.
- variables: the primary field swapped for one from another group; expected
  variables=False.
- mismatch: the formula paired with a hypothesis from another paper; expected
  to fail overall (a single check is not defined, so no per-check label).

Label noise is real: an original may not implement its hypothesis, a hypothesis
may state no horizon (then a rescaled window is still "aligned"), and a swapped
field may be an acceptable equivalent. `--labels` overrides the expected labels
of originals from a hand-labelled CSV.

Backends: `claude:<model>` runs `alignment.judge` with that model; `jev:jev-latest`
posts the same three questions to the Jev system-one endpoint, which returns a
probability of yes per check. The judge never sees returns or performance, so no
run here is a ledger trial of a factor; the run is logged once, before any call.

    python research/judge_calibration.py --store DB --n 60 --seed 0 \
        --backends claude:claude-haiku-5-5,claude:claude-sonnet-5-5,jev:jev-latest

Each answer is appended to cache.jsonl in the output folder, so a rerun resumes
without paying twice. Credentials come from the environment, never a file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from capstone import runlog
from capstone.analysis.outputs import Stat, Statement, output_dir, write_outputs
from capstone.factors import alignment, tree
from capstone.factors.hypotheses import _FIELDS
from capstone.factors.llm import FactorConfig, grammar_help, load_config, make_client

REPO = Path(__file__).resolve().parents[1]
CHECKS = alignment.CHECKS
KINDS = ("original", "direction", "horizon", "variables", "mismatch")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
THRESHOLDS = tuple(round(0.1 * i, 1) for i in range(1, 10))
DEFAULT_THRESHOLD = 0.5

_spec = importlib.util.spec_from_file_location(
    "alignment_calibration", REPO / "scripts" / "alignment_calibration.py"
)
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)


# ---------------------------------------------------------------------------
# Items


@dataclass
class Item:
    item_id: str
    kind: str
    proposal_id: int
    hypothesis: dict
    expression: str
    rationale: str
    expected: dict  # check -> True / False / None (not defined)
    aligned: bool  # expected overall: every check passes


def _map(node: tree.Node, fn: Callable[[tree.Node], tree.Node]) -> tree.Node:
    """Rebuild the tree bottom-up, applying `fn` to every rebuilt node."""
    if isinstance(node, tree.Unary):
        node = tree.Unary(node.op, _map(node.operand, fn))
    elif isinstance(node, tree.BinOp):
        node = tree.BinOp(node.op, _map(node.left, fn), _map(node.right, fn))
    elif isinstance(node, tree.Call):
        arg2 = None if node.arg2 is None else _map(node.arg2, fn)
        node = tree.Call(node.func, _map(node.arg, fn), node.window, arg2)
    return fn(node)


def negate(node: tree.Node) -> tree.Node:
    """The formula with its sign flipped (a root negation is removed rather than doubled)."""
    if isinstance(node, tree.Unary) and node.op == "neg":
        return node.operand
    return tree.Unary("neg", node)


def rescale_windows(node: tree.Node) -> tree.Node | str:
    """Every window x4, or /4 when x4 is too large; a string says why it is not applicable."""
    windows = [n.window for n in tree.walk(node) if isinstance(n, tree.Call) and n.window]
    if not windows:
        return "no windows to rescale"
    if max(windows) * 4 <= tree.WINDOW_MAX:
        factor = lambda w: w * 4  # noqa: E731
    elif min(windows) >= 4:
        factor = lambda w: w // 4  # noqa: E731
    else:
        return "windows span too wide a range to rescale within the allowed window range"

    def fn(n: tree.Node) -> tree.Node:
        if isinstance(n, tree.Call) and n.window:
            return tree.Call(n.func, n.arg, factor(n.window), n.arg2)
        return n

    return _map(node, fn)


# primary field -> candidates in order; the first one the formula does not already read wins
SWAPS = {
    "returns": ("volume", "cap", "shares"),
    "close": ("cap", "volume"),
    "open": ("volume", "cap"),
    "high": ("volume", "cap"),
    "low": ("volume", "cap"),
    "volume": ("returns", "close"),
    "shares": ("returns", "close"),
    "cap": ("returns", "volume"),
    "book": ("returns", "volume"),
    "earnings": ("returns", "volume"),
    "sales": ("returns", "volume"),
    "assets": ("returns", "volume"),
}


def swap_primary_field(node: tree.Node) -> tuple[tree.Node, str, str] | str:
    """Replace the most-read per-stock field with one from another group, or say why not."""
    used = tree.features_used(node)
    counts = Counter(
        n.name for n in tree.walk(node) if isinstance(n, tree.Field_) and n.name in SWAPS
    )
    if not counts:
        return "no swappable per-stock field"
    primary = max(counts, key=lambda name: (counts[name], name))
    target = next((c for c in SWAPS[primary] if c not in used), None)
    if target is None:
        return f"every swap candidate for {primary} is already read"

    def fn(n: tree.Node) -> tree.Node:
        return tree.Field_(target) if isinstance(n, tree.Field_) and n.name == primary else n

    return _map(node, fn), primary, target


def _valid(node: tree.Node) -> str | None:
    """The unparsed expression if it reads back unchanged, else None."""
    text = tree.unparse(node)
    try:
        return text if tree.unparse(tree.parse(text)) == text else None
    except tree.ParseError:
        return None


def load_labels(path: Path) -> dict[int, dict]:
    """Hand labels for originals: proposal_id, `label` (1 aligned, 0 not), per-check 1/0."""
    out: dict[int, dict] = {}
    # utf-8-sig: a sheet saved from Excel starts with a byte-order mark.
    with open(path, newline="", encoding="utf-8-sig") as handle:
        import csv

        for row in csv.DictReader(handle):
            pid = int(row["proposal_id"])
            entry: dict = {}
            for key in ("label", *CHECKS):
                value = (row.get(key) or "").strip()
                if value in {"0", "1"}:
                    entry[key] = value == "1"
            out[pid] = entry
    return out


def build_items(
    proposals: list[dict],
    n: int,
    seed: int,
    labels: dict[int, dict] | None = None,
) -> tuple[list[Item], list[dict]]:
    """A seeded sample of `n` originals and their perturbations; skips carry a reason."""
    usable = []
    for p in sorted(proposals, key=lambda p: p["proposal_id"]):
        try:
            tree.parse(p["expression"])
        except tree.ParseError:
            continue
        usable.append(p)
    originals = sorted(
        random.Random(seed).sample(usable, min(n, len(usable))), key=lambda p: p["proposal_id"]
    )
    labels = labels or {}
    items: list[Item] = []
    skipped: list[dict] = []

    def skip(pid: int, kind: str, why: str) -> None:
        skipped.append({"proposal_id": pid, "kind": kind, "reason": why})

    for p in originals:
        pid = p["proposal_id"]
        hyp = {name: p[name] for name in _FIELDS} | {"paper_key": p["paper_key"]}
        rationale = p["rationale"] or ""
        node = tree.parse(p["expression"])
        label = labels.get(pid, {})
        if label.get("label") is False:  # misaligned: only a stated check has a known answer
            expected = {c: label.get(c) for c in CHECKS}
            aligned = False
        else:
            expected = {c: label.get(c, True) for c in CHECKS}
            aligned = all(expected.values())
        items.append(
            Item(
                f"{pid}:original",
                "original",
                pid,
                hyp,
                p["expression"],
                rationale,
                expected,
                aligned,
            )
        )

        def add(kind, expression, hypothesis, expected, pid=pid, rationale=rationale) -> None:
            if expression is None:
                skip(pid, kind, "perturbed formula does not parse back")
                return
            items.append(
                Item(f"{pid}:{kind}", kind, pid, hypothesis, expression, rationale, expected, False)
            )

        if label.get("label") is False or not aligned:
            for kind in ("direction", "horizon", "variables"):
                skip(pid, kind, "original hand-labelled misaligned")
        else:
            add(
                "direction",
                _valid(negate(node)),
                hyp,
                {"variables": True, "direction": False, "horizon": True},
            )
            scaled = rescale_windows(node)
            if isinstance(scaled, str):
                skip(pid, "horizon", scaled)
            else:
                add(
                    "horizon",
                    _valid(scaled),
                    hyp,
                    {"variables": True, "direction": True, "horizon": False},
                )
            swapped = swap_primary_field(node)
            if isinstance(swapped, str):
                skip(pid, "variables", swapped)
            else:
                add(
                    "variables",
                    _valid(swapped[0]),
                    hyp,
                    {"variables": False, "direction": True, "horizon": True},
                )
        others = [q for q in usable if q["paper_key"] != p["paper_key"]]
        if not others:
            skip(pid, "mismatch", "no hypothesis from another paper")
            continue
        q = random.Random(f"{seed}:{pid}:mismatch").choice(others)
        other = {name: q[name] for name in _FIELDS} | {"paper_key": q["paper_key"]}
        items.append(
            Item(
                f"{pid}:mismatch", "mismatch", pid, other, p["expression"], rationale,
                {c: None for c in CHECKS}, False,
            )
        )  # fmt: skip
    return items, skipped


def subset(items: list[Item], k: int | None) -> list[Item]:
    """The first `k` items taking one kind at a time, so a small run covers every kind."""
    if k is None or k >= len(items):
        return items
    queues = {kind: [i for i in items if i.kind == kind] for kind in KINDS}
    out: list[Item] = []
    while len(out) < k and any(queues.values()):
        for kind in KINDS:
            if queues[kind] and len(out) < k:
                out.append(queues[kind].pop(0))
    return out


# ---------------------------------------------------------------------------
# Backends


@dataclass
class Answer:
    checks: dict = field(default_factory=dict)  # check -> bool (Claude)
    probs: dict = field(default_factory=dict)  # check -> P(yes) (Jev)
    reason: str = ""
    error: str = ""
    usage: dict = field(default_factory=dict)


def _state(item: Item, rationale: str) -> str:
    lines = [f"{name}: {item.hypothesis[name]}" for name in _FIELDS]
    return (
        "Hypothesis\n" + "\n".join(lines) + f"\n\nFormula: {item.expression}\n"
        f"Author's rationale: {rationale or '(none)'}"
    )


class ClaudeBackend:
    """`alignment.judge` with the given model; a malformed answer is an error row."""

    paid = True

    def __init__(self, model: str, client, config: FactorConfig | None = None):
        self.name = f"claude:{model}"
        self.client = client
        self.config = replace(config or load_config(), alignment_model=model)

    def answer(self, item: Item, rationale: str) -> Answer:
        usage: Counter = Counter()
        try:
            verdict = alignment.judge(
                self.client, self.config, item.hypothesis, item.expression, rationale, usage
            )
        except ValueError as exc:  # JudgeAnswerError, or the model never called the tool
            return Answer(error=str(exc), usage=dict(usage))
        return Answer(checks=dict(verdict.checks), reason=verdict.reason, usage=dict(usage))


class JevError(ValueError):
    """The Jev response lacks a probability for a check."""


def jev_questions() -> tuple[dict, str]:
    """The three questions and the grammar notes, taken from the Claude judge's own prompt."""
    intro_and_checks, grammar = alignment.SYSTEM.split("\n\n", 2)[0::2][0], grammar_help()
    intro, bullets = intro_and_checks.split("Answer three questions:\n")
    questions = {}
    for check in CHECKS:
        match = re.search(rf"^- {check}: (.*?)(?=^- |\Z)", bullets, re.M | re.S)
        if match is None:
            raise RuntimeError(f"alignment.SYSTEM has no {check} question")
        questions[check] = {
            "type": "noul",
            "instructions": f"{intro.strip()}\nQuestion: {match.group(1).strip()}\n"
            "Answer yes or no.",
        }
    return questions, grammar


def parse_jev(response: dict) -> dict:
    """P(yes) per check from a systemone response; anything else raises `JevError`."""
    try:
        answers = response["answers"]
        probs = {check: answers[check]["noul"] for check in CHECKS}
    except (KeyError, TypeError) as exc:
        raise JevError(f"response lacks answers for every check: {exc!r}") from None
    for check, p in probs.items():
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise JevError(f"{check} probability {p!r} is not a number in [0, 1]")
    return {check: float(p) for check, p in probs.items()}


def http_post(url: str, headers: dict, body: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode())


class JevBackend:
    paid = False

    def __init__(self, model: str, key: str, post=http_post, sleep=time.sleep, retries: int = 3):
        self.name = f"jev:{model}"
        self.model = model
        self.post, self.sleep, self.retries = post, sleep, retries
        self.headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "judge-calibration/1",
        }
        self.questions, self.grammar = jev_questions()

    def answer(self, item: Item, rationale: str) -> Answer:
        body = {
            "model": self.model,
            "state": _state(item, rationale) + "\n\nGrammar notes\n" + self.grammar,
            "questions": self.questions,
        }
        for attempt in range(self.retries + 1):
            try:
                response = self.post(JEV_URL, self.headers, body)
                break
            except urllib.error.HTTPError as exc:
                if (
                    exc.code not in (429, 500, 502, 503, 504, 529) or attempt == self.retries
                ):  # 529: overloaded
                    raise
                self.sleep(2.0**attempt)
        try:
            probs = parse_jev(response)
        except JevError as exc:
            return Answer(error=str(exc), usage={"calls": 1})
        return Answer(probs=probs, usage={"calls": 1})


def jev_key(environ=None) -> str:
    """TYPESAFE_API_KEY from the environment, else the Windows user environment."""
    env = os.environ if environ is None else environ
    if env.get("TYPESAFE_API_KEY"):
        return env["TYPESAFE_API_KEY"]
    try:  # pragma: no cover - Windows only
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
            return winreg.QueryValueEx(handle, "TYPESAFE_API_KEY")[0]
    except (ImportError, OSError):
        raise SystemExit("TYPESAFE_API_KEY is not set") from None


# ---------------------------------------------------------------------------
# Running with a resumable cache


def _load_cache(path: Path) -> dict:
    cache: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[(row["backend"], row["item_id"], row["no_rationale"])] = row
    return cache


def run(
    items: list[Item], backends: list, out: Path, no_rationale: bool, max_calls: int | None
) -> list[dict]:
    """Answer every (backend, item) not yet cached; returns one row per pair."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cache.jsonl"
    cache = _load_cache(path)
    pending = {
        b.name: sum((b.name, i.item_id, no_rationale) not in cache for i in items) for b in backends
    }
    paid = sum(pending[b.name] for b in backends if b.paid)
    if max_calls is not None and paid > max_calls:
        raise SystemExit(
            f"refusing to start: {paid} Claude calls needed, --max-calls is {max_calls}"
        )
    runlog.log_run(
        name="judge-calibration",
        tags=["synthetic", "methodology"],
        params={
            "items": len(items),
            "backends": [b.name for b in backends],
            "no_rationale": no_rationale,
            "pending_calls": pending,
            "max_calls": max_calls,
        },
    )
    rows = []
    with open(path, "a", encoding="utf-8") as handle:
        for backend in backends:
            for item in items:
                key = (backend.name, item.item_id, no_rationale)
                if key not in cache:
                    answer = backend.answer(item, "" if no_rationale else item.rationale)
                    row = {
                        "backend": backend.name,
                        "item_id": item.item_id,
                        "no_rationale": no_rationale,
                        "checks": answer.checks,
                        "probs": answer.probs,
                        "reason": answer.reason,
                        "error": answer.error,
                        "usage": answer.usage,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    cache[key] = row
                rows.append(cache[key])
    return rows


# ---------------------------------------------------------------------------
# Metrics


def _decision(row: dict, check: str, threshold: float) -> bool | None:
    """Pass / fail for one check; None for an error row."""
    if row["error"]:
        return None
    if row["checks"]:
        return bool(row["checks"][check])
    return row["probs"][check] >= threshold


def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def metrics_at(rows: list[dict], items: dict[str, Item], threshold: float) -> dict:
    """Rates for one backend at one threshold. Error rows are not passes and leave denominators."""
    out: dict = {"n": len(rows), "errors": sum(bool(r["error"]) for r in rows)}
    out["error_rate"] = _rate(out["errors"], len(rows))
    for check in CHECKS:
        fp_n = fp = fr_n = fr = acc_n = acc = 0
        for r in rows:
            item, d = items[r["item_id"]], _decision(r, check, threshold)
            want = item.expected[check]
            if d is None or want is None:
                continue
            acc_n += 1
            acc += d == want
            if want is False:
                fp_n += 1
                fp += d
            elif item.kind == "original":
                fr_n += 1
                fr += not d
        out[check] = {
            "false_pass": fp, "false_pass_n": fp_n, "false_pass_rate": _rate(fp, fp_n),
            "false_reject": fr, "false_reject_n": fr_n, "false_reject_rate": _rate(fr, fr_n),
            "accuracy": _rate(acc, acc_n), "accuracy_n": acc_n,
        }  # fmt: skip
    overall = {"n": 0, "correct": 0, "caught": 0, "misaligned": 0, "rejected": 0, "originals": 0}
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        if r["error"]:
            continue
        item = items[r["item_id"]]
        passes = all(_decision(r, c, threshold) for c in CHECKS)
        overall["n"] += 1
        overall["correct"] += passes == item.aligned
        if not item.aligned:
            overall["misaligned"] += 1
            overall["caught"] += not passes
            by_kind.setdefault(item.kind, [0, 0])[0] += not passes
            by_kind[item.kind][1] += 1
        elif item.kind == "original":
            overall["originals"] += 1
            overall["rejected"] += not passes
    out["overall"] = {
        "accuracy": _rate(overall["correct"], overall["n"]),
        "detection": _rate(overall["caught"], overall["misaligned"]),
        "detection_n": overall["misaligned"],
        "caught": overall["caught"],
        "false_reject_rate": _rate(overall["rejected"], overall["originals"]),
        "false_reject": overall["rejected"],
        "false_reject_n": overall["originals"],
        "by_kind": {k: {"caught": v[0], "n": v[1]} for k, v in by_kind.items()},
    }
    return out


def brier(rows: list[dict], items: dict[str, Item]) -> dict:
    out = {}
    for check in CHECKS:
        errs = [
            (r["probs"][check] - float(items[r["item_id"]].expected[check])) ** 2
            for r in rows
            if not r["error"] and r["probs"] and items[r["item_id"]].expected[check] is not None
        ]
        out[check] = sum(errs) / len(errs) if errs else float("nan")
    return out


def reliability(rows: list[dict], items: dict[str, Item]) -> pd.DataFrame:
    """Ten equal-width probability bins per check: how often "yes" was right in each."""
    recs = []
    for check in CHECKS:
        bins: dict[int, list[tuple[float, bool]]] = {}
        for r in rows:
            want = items[r["item_id"]].expected[check]
            if r["error"] or not r["probs"] or want is None:
                continue
            p = r["probs"][check]
            bins.setdefault(min(int(p * 10), 9), []).append((p, want))
        for b in range(10):
            got = bins.get(b, [])
            recs.append(
                {
                    "check": check,
                    "bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}",
                    "n": len(got),
                    "mean_p": sum(p for p, _ in got) / len(got) if got else float("nan"),
                    "frac_expected_true": sum(w for _, w in got) / len(got)
                    if got
                    else float("nan"),
                }
            )
    return pd.DataFrame(recs)


def usage_totals(rows: list[dict]) -> dict:
    total: Counter = Counter()
    for r in rows:
        for k, v in r["usage"].items():
            total[k.removeprefix("alignment_")] += v
    return dict(total)


# ---------------------------------------------------------------------------
# Outputs


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _pct(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "0/0"


def report(
    rows: list[dict],
    items: list[Item],
    skipped: list[dict],
    backends: list[str],
    out: Path,
    no_rationale: bool,
    args: dict,
) -> Path:
    by_id = {i.item_id: i for i in items}
    stats: dict[str, Stat] = {}
    statements: list[Statement] = []
    files = ("answers.csv", "items.csv")

    def stat(key: str, value, definition: str, inputs: tuple[str, ...] = files) -> None:
        stats[key] = Stat(value, definition, inputs)

    counts = Counter(i.kind for i in items)
    for kind in KINDS:
        stat(f"n_items.{kind}", counts[kind], f"{kind} items in the set", ("items.csv",))
    stat("n_skipped", len(skipped), "perturbations not applicable", ("skipped.csv",))
    stat("no_rationale", no_rationale, "the judge was given an empty rationale", ())
    items_table = pd.DataFrame(
        [
            {
                "item_id": i.item_id,
                "kind": i.kind,
                "proposal_id": i.proposal_id,
                "expression": i.expression,
                "aligned": i.aligned,
                **{f"expect_{c}": i.expected[c] for c in CHECKS},
            }
            for i in items
        ]
    )
    skipped_table = pd.DataFrame(skipped, columns=["proposal_id", "kind", "reason"])
    answers, summary, sweep, rel = [], [], [], []
    caveat = (
        "Expected labels are by construction and the originals' labels are unchecked: "
        "label noise (an original that does not implement its hypothesis, a hypothesis with no "
        "stated horizon, an acceptable field swap) shifts every rate. Samples are small."
    )
    for name in backends:
        mine = [r for r in rows if r["backend"] == name]
        s = _slug(name)
        is_jev = any(r["probs"] for r in mine)
        for r in mine:
            shown = r["probs"] or r["checks"] or {}
            answers.append(
                {
                    "backend": name,
                    "item_id": r["item_id"],
                    "kind": by_id[r["item_id"]].kind,
                    "error": r["error"],
                    "reason": r["reason"],
                    **{f"{c}_{'p' if is_jev else 'pass'}": shown.get(c) for c in CHECKS},
                }
            )
        m = metrics_at(mine, by_id, DEFAULT_THRESHOLD)
        usage = usage_totals(mine)
        stat(f"{s}.error_rate", m["error_rate"], f"{name}: share of items answered with an error")
        stat(
            f"{s}.calls",
            usage.get("calls", 0),
            f"{name}: billed calls in the cache",
            ("cache.jsonl",),
        )
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        stat(f"{s}.tokens", tokens, f"{name}: input plus output tokens", ("cache.jsonl",))
        for t in THRESHOLDS if is_jev else (None,):
            mt = metrics_at(mine, by_id, DEFAULT_THRESHOLD if t is None else t)
            for check in CHECKS:
                sweep.append({"backend": name, "threshold": t, "check": check} | mt[check])
            overall = {k: v for k, v in mt["overall"].items() if k != "by_kind"}
            sweep.append({"backend": name, "threshold": t, "check": "overall"} | overall)
        at = f" at threshold {DEFAULT_THRESHOLD}" if is_jev else ""
        for check in CHECKS:
            c = m[check]
            stat(
                f"{s}.false_pass.{check}",
                c["false_pass_rate"],
                f"{name}: share of items whose {check} check is expected False that the judge "
                f"passed{at}",
            )
            stat(
                f"{s}.false_reject.{check}",
                c["false_reject_rate"],
                f"{name}: share of originals whose {check} check the judge failed{at}",
            )
            stat(
                f"{s}.accuracy.{check}",
                c["accuracy"],
                f"{name}: {check} decisions matching the expected label{at}",
            )
            summary.append({"backend": name, "check": check, **c})
            fp = _pct(c["false_pass"], c["false_pass_n"])
            fr = _pct(c["false_reject"], c["false_reject_n"])
            statements.append(
                Statement(
                    f"{name} passed the {check} check on {fp} items built to fail it and failed "
                    f"it on {fr} originals{at}.",
                    f"{s}.false_pass.{check}",
                    caveat,
                )
            )
        o = m["overall"]
        stat(
            f"{s}.detection",
            o["detection"],
            f"{name}: share of misaligned items with at least one check failed{at}",
        )
        stat(
            f"{s}.overall_false_reject",
            o["false_reject_rate"],
            f"{name}: share of originals with at least one check failed{at}",
        )
        caught = _pct(o["caught"], o["detection_n"])
        wrong = _pct(o["false_reject"], o["false_reject_n"])
        statements.append(
            Statement(
                f"{name} flagged {caught} misaligned items and wrongly flagged {wrong} originals"
                f"{at}.",
                f"{s}.detection",
                caveat,
            )
        )
        if is_jev:
            for check, v in brier(mine, by_id).items():
                stat(
                    f"{s}.brier.{check}",
                    v,
                    f"{name}: Brier score of P(yes) for {check} against the expected label",
                    ("answers.csv",),
                )
            rel.append(reliability(mine, by_id).assign(backend=name))
    if not statements:
        raise SystemExit("no backend produced any rows")
    stat("run_params", args, "arguments of this run", ())
    tables = {
        "items": items_table,
        "skipped": skipped_table,
        "answers": pd.DataFrame(answers),
        "metrics": pd.DataFrame(summary),
        "sweep": pd.DataFrame(sweep),
    }
    if rel:
        tables["reliability"] = pd.concat(rel, ignore_index=True)
    return write_outputs(
        out, title="Alignment judge calibration", tables=tables, stats=stats, statements=statements
    )


# ---------------------------------------------------------------------------
# CLI


def make_backends(spec: str) -> list:
    backends = []
    client = None
    for part in spec.split(","):
        kind, _, model = part.strip().partition(":")
        if kind == "claude" and model:
            client = client or make_client()
            backends.append(ClaudeBackend(model, client))
        elif kind == "jev" and model:
            backends.append(JevBackend(model, jev_key()))
        else:
            raise SystemExit(f"unknown backend {part!r}; use claude:<model> or jev:<model>")
    return backends


def main(argv: list[str] | None = None, backends: list | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backends", required=True)
    ap.add_argument("--no-rationale", action="store_true")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-items", type=int, help="smoke runs: one item of each kind in turn")
    ap.add_argument(
        "--statuses", default="stored", help="proposal statuses to sample originals from"
    )
    ap.add_argument("--label", help="run label for the default output folder")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    statuses = set(a.statuses.split(","))
    proposals = [p for p in _cal._proposals(a.store) if p["status"] in statuses]
    labels = load_labels(a.labels) if a.labels else None
    items, skipped = build_items(proposals, a.n, a.seed, labels)
    items = subset(items, a.max_items)
    out = a.out or output_dir("judge-calibration", a.label or f"n{a.n}-seed{a.seed}")
    built = backends if backends is not None else make_backends(a.backends)
    rows = run(items, built, out, a.no_rationale, a.max_calls)
    params = {
        "store": a.store.name, "n": a.n, "seed": a.seed, "backends": a.backends,
        "statuses": a.statuses, "max_items": a.max_items, "n_items": len(items),
    }  # fmt: skip
    path = report(rows, items, skipped, [b.name for b in built], out, a.no_rationale, params)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
