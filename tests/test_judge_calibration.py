"""Judge calibration: labelled items by construction, backends, metrics, resumable cache.

Fake backends and a fake HTTP post only; no API calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from capstone.factors import store, tree
from capstone.factors.hypotheses import _FIELDS
from capstone.factors.llm import FactorConfig

SCRIPT = Path(__file__).resolve().parents[1] / "research" / "judge_calibration.py"
_spec = importlib.util.spec_from_file_location("judge_calibration", SCRIPT)
jc = importlib.util.module_from_spec(_spec)
sys.modules["judge_calibration"] = jc
_spec.loader.exec_module(jc)

EXPRESSIONS = [
    "-ts_mean(volume / shares, 126)",
    "ts_sum(returns, 21)",
    "rank(close)",  # no window: horizon skipped
    "ts_mean(volume, 1000)",  # x4 too big: /4
    "ts_corr(returns, volume, 5)",  # primary is volume; returns is read, so close
]


def _proposals(n=5):
    out = []
    for i in range(n):
        out.append(
            {
                "proposal_id": i + 1,
                "paper_key": f"paper{i % 3}",
                **{name: f"{name} {i}" for name in _FIELDS},
                "expression": EXPRESSIONS[i % len(EXPRESSIONS)],
                "rationale": f"because {i}",
                "status": "stored",
            }
        )
    return out


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger" / "runs.jsonl"


def _items(n=5, labels=None):
    return jc.build_items(_proposals(), n, 0, labels)


# -- perturbations ---------------------------------------------------------


def test_perturbations_carry_the_expected_labels_and_parse_back():
    items, _ = _items()
    assert {i.kind for i in items} == set(jc.KINDS)
    for item in items:
        text = tree.unparse(tree.parse(item.expression))
        assert tree.unparse(tree.parse(text)) == text
        want = {
            "original": (True, True, True),
            "direction": (True, False, True),
            "horizon": (True, True, False),
            "variables": (False, True, True),
            "mismatch": (None, None, None),
        }[item.kind]
        assert tuple(item.expected[c] for c in jc.CHECKS) == want
        assert item.aligned is (item.kind == "original")
    by_id = {i.item_id: i for i in items}
    assert by_id["2:direction"].expression == "-(ts_sum(returns, 21))"
    assert by_id["1:direction"].expression == "ts_mean((volume / shares), 126)"  # neg removed
    assert by_id["2:horizon"].expression == "ts_sum(returns, 84)"
    assert by_id["4:horizon"].expression == "ts_mean(volume, 250)"  # 1000 x4 > 1260: /4
    assert by_id["2:variables"].expression == "ts_sum(volume, 21)"
    assert by_id["5:variables"].expression == "ts_corr(returns, close, 5)"


def test_mismatch_pairs_the_formula_with_another_papers_hypothesis():
    items, _ = _items()
    originals = {i.proposal_id: i for i in items if i.kind == "original"}
    mismatches = [i for i in items if i.kind == "mismatch"]
    assert mismatches
    for item in mismatches:
        original = originals[item.proposal_id]
        assert item.hypothesis["paper_key"] != original.hypothesis["paper_key"]
        assert item.expression == original.expression


def test_inapplicable_perturbations_are_skipped_with_a_reason():
    items, skipped = _items()
    assert "3:horizon" not in {i.item_id for i in items}
    reasons = {(s["proposal_id"], s["kind"]): s["reason"] for s in skipped}
    assert reasons[(3, "horizon")] == "no windows to rescale"
    assert (3, "variables") not in reasons  # close -> cap is applicable
    node = tree.parse("ts_mean(returns, 2) + ts_mean(returns, 1200)")
    assert "range" in jc.rescale_windows(node)  # neither x4 nor /4 fits
    assert "already read" in jc.swap_primary_field(
        tree.parse("ts_corr(volume, returns, 5) + close")
    )


def test_sampling_is_seeded_and_labels_override_originals():
    a, _ = jc.build_items(_proposals(5), 3, 7)
    b, _ = jc.build_items(_proposals(5), 3, 7)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    assert sum(i.kind == "original" for i in a) == 3
    items, skipped = _items(labels={1: {"label": False}, 2: {"horizon": False}})
    by_id = {i.item_id: i for i in items}
    assert by_id["1:original"].aligned is False
    assert by_id["1:original"].expected == {c: None for c in jc.CHECKS}
    assert "1:direction" not in by_id and "1:mismatch" in by_id
    assert any(s["proposal_id"] == 1 and "hand-labelled" in s["reason"] for s in skipped)
    assert by_id["2:original"].expected["horizon"] is False and not by_id["2:original"].aligned


def test_subset_covers_every_kind_first():
    items, _ = _items()
    assert [i.kind for i in jc.subset(items, 5)] == list(jc.KINDS)


# -- Claude backend ---------------------------------------------------------


class FakeClient:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.messages = self

    def create(self, **kwargs):
        block = SimpleNamespace(
            type="tool_use", name=kwargs["tools"][0]["name"], input=self.answers.pop(0)
        )
        return SimpleNamespace(
            content=[block], usage=SimpleNamespace(input_tokens=100, output_tokens=10)
        )


def _ok(**fails):
    return {**{c: c not in fails for c in jc.CHECKS}, "reason": "r"}


def test_claude_backend_judges_and_a_strict_parse_error_is_an_error_row_not_a_pass():
    bad = {"variables": "true", "direction": True, "horizon": True, "reason": "r"}
    backend = jc.ClaudeBackend("m", FakeClient(_ok(), bad), FactorConfig())
    item = _items()[0][0]
    good = backend.answer(item, "r")
    assert good.checks == {c: True for c in jc.CHECKS} and good.usage["alignment_calls"] == 1
    err = backend.answer(item, "r")
    assert err.error and not err.checks
    row = {
        "backend": "claude:m",
        "item_id": item.item_id,
        "no_rationale": False,
        "checks": {},
        "probs": {},
        "reason": "",
        "error": err.error,
        "usage": {},
    }
    m = jc.metrics_at([row], {item.item_id: item}, 0.5)
    assert m["error_rate"] == 1.0 and m["errors"] == 1
    assert m["overall"]["detection_n"] == 0 and m["variables"]["false_reject_n"] == 0


def test_no_rationale_sends_an_empty_rationale(tmp_path):
    seen = []

    class Spy(FakeClient):
        def create(self, **kwargs):
            seen.append(kwargs["messages"][0]["content"])
            return super().create(**kwargs)

    backend = jc.ClaudeBackend("m", Spy(_ok(), _ok()), FactorConfig())
    items, _ = _items()
    jc.run(items[:1], [backend], tmp_path / "a", no_rationale=False, max_calls=None)
    jc.run(items[:1], [backend], tmp_path / "b", no_rationale=True, max_calls=None)
    assert "Author's rationale: because" in seen[0]
    assert "Author's rationale: (none)" in seen[1]


# -- Jev backend ------------------------------------------------------------


def _jev_response(yes=0.9, **override):
    p = dict.fromkeys(jc.CHECKS, yes) | override
    return {"answers": {c: {"noul": p[c]} for c in jc.CHECKS}}


def test_jev_questions_repeat_the_claude_judges_checks():
    questions, grammar = jc.jev_questions()
    assert set(questions) == set(jc.CHECKS)
    assert "Answer yes if the specification states none" in questions["horizon"]["instructions"]
    assert "higher value mean" in questions["direction"]["instructions"]
    assert all(q["type"] == "noul" for q in questions.values())
    assert "ts_mean" in grammar


def test_jev_response_parsing_and_request_shape():
    sent = []

    def post(url, headers, body):
        sent.append((url, headers, body))
        return _jev_response(0.8, direction=0.1)

    backend = jc.JevBackend("jev-latest", "KEY", post=post)
    item = _items()[0][0]
    answer = backend.answer(item, "why")
    assert answer.probs == {"variables": 0.8, "direction": 0.1, "horizon": 0.8}
    url, headers, body = sent[0]
    assert url == jc.JEV_URL and headers["Authorization"] == "Bearer KEY"
    assert body["model"] == "jev-latest" and set(body["questions"]) == set(jc.CHECKS)
    assert item.expression in body["state"] and "Author's rationale: why" in body["state"]
    with pytest.raises(jc.JevError):
        jc.parse_jev({"answers": {"variables": {"noul": 0.5}}})
    with pytest.raises(jc.JevError):
        jc.parse_jev(_jev_response(1.5))
    bad = jc.JevBackend("jev-latest", "KEY", post=lambda *a: {"answers": {}})
    assert bad.answer(item, "").error


def test_jev_retries_on_429_and_5xx_then_gives_up():
    attempts, naps = [], []

    def flaky(url, headers, body):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 429 if len(attempts) == 1 else 503, "x", {}, None)
        return _jev_response()

    item = _items()[0][0]
    backend = jc.JevBackend("jev-latest", "K", post=flaky, sleep=naps.append)
    assert backend.answer(item, "").probs
    assert len(attempts) == 3 and len(naps) == 2

    def forbidden(url, headers, body):
        raise urllib.error.HTTPError(url, 401, "no", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        jc.JevBackend("jev-latest", "K", post=forbidden, sleep=naps.append).answer(item, "")


# -- metrics ----------------------------------------------------------------


class Fake:
    """A backend with a fixed answer per item kind."""

    paid = True

    def __init__(self, name, fails, calls=None):
        self.name, self.fails, self.calls = name, fails, calls if calls is not None else []

    def answer(self, item, rationale):
        self.calls.append(item.item_id)
        fails = self.fails.get(item.kind, ())
        return jc.Answer(
            checks={c: c not in fails for c in jc.CHECKS}, usage={"calls": 1, "input_tokens": 10}
        )


def test_metrics_on_a_backend_with_known_answers(tmp_path):
    items, _ = _items()
    by_id = {i.item_id: i for i in items}
    # catches direction and horizon, never variables; wrongly fails horizon on originals
    backend = Fake(
        "fake",
        {
            "direction": ("direction",),
            "horizon": ("horizon",),
            "original": ("horizon",),
            "mismatch": ("variables",),
        },
    )
    rows = jc.run(items, [backend], tmp_path, False, None)
    m = jc.metrics_at(rows, by_id, 0.5)
    n_kind = {k: sum(i.kind == k for i in items) for k in jc.KINDS}
    assert m["direction"]["false_pass_rate"] == 0.0
    assert m["horizon"]["false_pass_rate"] == 0.0
    assert m["variables"]["false_pass_rate"] == 1.0
    assert m["variables"]["false_pass_n"] == n_kind["variables"]
    assert m["horizon"]["false_reject_rate"] == 1.0
    assert m["variables"]["false_reject_rate"] == 0.0
    o = m["overall"]
    assert o["detection_n"] == sum(not i.aligned for i in items)
    assert o["caught"] == n_kind["direction"] + n_kind["horizon"] + n_kind["mismatch"]
    assert o["false_reject_rate"] == 1.0  # every original fails horizon
    assert o["by_kind"]["variables"] == {"caught": 0, "n": n_kind["variables"]}


def test_jev_threshold_decision_brier_and_reliability():
    items, _ = _items()
    by_id = {i.item_id: i for i in items}

    def row(item, p):
        return {
            "backend": "j",
            "item_id": item.item_id,
            "no_rationale": False,
            "checks": {},
            "probs": dict.fromkeys(jc.CHECKS, p),
            "reason": "",
            "error": "",
            "usage": {},
        }

    rows = [row(by_id["2:original"], 0.95), row(by_id["2:direction"], 0.4)]
    assert jc.metrics_at(rows, by_id, 0.5)["direction"]["false_pass_rate"] == 0.0
    assert jc.metrics_at(rows, by_id, 0.3)["direction"]["false_pass_rate"] == 1.0
    b = jc.brier(rows, by_id)
    assert b["direction"] == pytest.approx(((0.95 - 1) ** 2 + (0.4 - 0) ** 2) / 2)
    rel = jc.reliability(rows, by_id)
    d = rel[(rel.check == "direction") & (rel.n > 0)]
    assert len(rel) == 30 and sorted(d.bin) == ["0.4-0.5", "0.9-1.0"]
    assert sorted(d.frac_expected_true) == [0.0, 1.0]


# -- running: refusal, cache, ledger ---------------------------------------


def test_max_calls_refuses_before_any_call_or_ledger_entry(tmp_path, ledger):
    items, _ = _items()
    calls: list = []
    backends = [Fake("a", {}, calls), Fake("b", {}, calls)]
    with pytest.raises(SystemExit, match="refusing"):
        jc.run(items, backends, tmp_path, False, max_calls=len(items) * 2 - 1)
    assert not calls and not ledger.exists()
    jc.run(items, backends, tmp_path, False, max_calls=len(items) * 2)
    assert len(calls) == len(items) * 2


def test_unpaid_backends_do_not_count_against_max_calls(tmp_path):
    items, _ = _items()
    jev = jc.JevBackend("jev-latest", "K", post=lambda *a: _jev_response())
    assert len(jc.run(items, [jev], tmp_path, False, max_calls=0)) == len(items)


def test_rerun_resumes_from_the_cache(tmp_path):
    items, _ = _items()
    calls: list = []
    backend = Fake("a", {}, calls)
    jc.run(items[:4], [backend], tmp_path, False, None)
    assert len(calls) == 4
    rows = jc.run(items, [backend], tmp_path, False, None)
    assert len(calls) == len(items) and len(rows) == len(items)  # only the rest was paid for
    jc.run(items, [backend], tmp_path, False, None)
    assert len(calls) == len(items)
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == len(items)
    jc.run(items[:1], [backend], tmp_path, True, None)  # other rationale mode: not cached
    assert len(calls) == len(items) + 1


def test_the_ledger_entry_is_written_before_the_first_call(tmp_path, ledger):
    seen = []

    class Watch(Fake):
        def answer(self, item, rationale):
            seen.append(ledger.exists() and ledger.read_text())
            return super().answer(item, rationale)

    items, _ = _items()
    jc.run(items[:2], [Watch("a", {})], tmp_path, False, None)
    entry = json.loads(seen[0].splitlines()[0])
    assert entry["name"] == "judge-calibration" and entry["tags"] == ["synthetic", "methodology"]
    assert len(ledger.read_text().splitlines()) == 1


# -- end to end -------------------------------------------------------------


def test_main_writes_sourced_statements(tmp_path):
    db = tmp_path / "f.db"
    con = store.connect(db)
    for key in ("pa", "pbb"):
        store.add_paper(con, key, key)
        hid = store.add_hypothesis(
            con,
            store.Hypothesis(
                paper_key=key,
                observation=f"o {key}",
                knowledge="k",
                justification="j",
                specification="s",
                falsification_condition="f",
                kind="market",
            ),
        )
        for expression in (f"-ts_sum(returns, {len(key)})", f"ts_mean(volume, 20) * {len(key)}"):
            store.add_factor(con, tree.parse(expression), hid, rationale="r")
    con.close()
    before = db.read_bytes()
    out = tmp_path / "out"
    fake = Fake("claude:fake", {"direction": ("direction",)})
    path = jc.main(
        ["--store", str(db), "--n", "4", "--backends", "claude:fake", "--out", str(out)],
        backends=[fake],
    )
    assert db.read_bytes() == before
    stats = json.loads((path / "stats.json").read_text())
    assert stats["claude_fake.false_pass.direction"]["value"] == 0.0
    assert stats["claude_fake.false_pass.variables"]["value"] == 1.0
    assert stats["n_items.original"]["value"] == 4
    text = (path / "STATEMENTS.md").read_text()
    assert "stats.json: claude_fake.false_pass.direction" in text and "label noise" in text
    assert (path / "data" / "sweep.csv").exists()


def test_labels_saved_from_excel_with_a_byte_order_mark_are_read(tmp_path):
    path = tmp_path / "labels.csv"
    text = "proposal_id,label,variables,direction,horizon\n7,1,1,,0\n"
    path.write_text(text, encoding="utf-8-sig")
    labels = jc.load_labels(path)
    assert labels == {7: {"label": True, "variables": True, "horizon": False}}
