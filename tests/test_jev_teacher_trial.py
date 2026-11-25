"""Sonnet teacher labelling and the Jev tuning trial: prompt, cache, cap, ledger, tuning, metrics.

Fake client and fake HTTP post only; no API calls.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import jev_paper_screen as jps  # noqa: E402
import jev_teacher_trial as jtt  # noqa: E402
import teacher_label as tl  # noqa: E402

CFG = jps.load_config()
TAGS = list(CFG["taxonomy"])
DEFS = jps.load_definitions(TAGS)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("TYPESAFE_API_KEY", "K")
    return tmp_path / "ledger" / "runs.jsonl"


def _reply(text, stop="end_turn", usage=None):
    usage = usage or {"input_tokens": 30, "output_tokens": 50, "cache_read_input_tokens": 1500}
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(**usage),
    )


def _label(level, tags=(), reason="because"):
    return json.dumps({"relevance": level, "tags": list(tags), "reason": reason})


class FakeClient:
    def __init__(self, respond, on_call=None):
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)
        self._respond, self._on_call = respond, on_call

    def _create(self, **kw):
        self.calls.append(kw)
        if self._on_call:
            self._on_call()
        return self._respond(kw)


def _paper(key, title="T", abstract="A"):
    return {"key": key, "title": title, "abstract": abstract}


# -- teacher prompt and request ---------------------------------------------


def test_system_prompt_has_the_rubric_verbatim_and_every_tag_definition():
    system = tl.build_system(CFG, DEFS)
    assert CFG["rubric"].strip() in system
    for tag, definition in DEFS.items():
        assert f"- {tag}: {definition}" in system
    assert tl.build_schema(TAGS)["properties"]["tags"]["items"]["enum"] == TAGS
    assert tl.build_schema(TAGS)["properties"]["relevance"]["enum"] == [0, 1, 2, 3, 4, 5]


def test_request_uses_structured_output_cached_system_and_title_abstract_only(tmp_path):
    client = FakeClient(lambda kw: _reply(_label(3, [TAGS[0]])))
    system = tl.build_system(CFG, DEFS)
    rows, errors = tl.run_teacher(
        [_paper("a", "My Title", "My abstract")], system, TAGS, tmp_path, client, 5, workers=1
    )
    kw = client.calls[0]
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"]["effort"] == "medium"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in kw and "tools" not in kw and "thinking" not in kw
    assert kw["messages"][0]["content"] == "Title: My Title\n\nAbstract: My abstract"
    assert errors == 0 and rows[0]["relevance"] == 3 and rows[0]["tags"] == [TAGS[0]]


@pytest.mark.parametrize(
    "response",
    [
        _reply("", stop="refusal"),
        _reply(_label(3), stop="max_tokens"),
        _reply("not json"),
        _reply(_label(7)),
        _reply(_label(2, ["no_such_tag"])),
    ],
)
def test_refusal_truncation_and_invalid_replies_are_error_rows_not_labels(tmp_path, response):
    client = FakeClient(lambda kw: response)
    rows, errors = tl.run_teacher(
        [_paper("a")], tl.build_system(CFG, DEFS), TAGS, tmp_path, client, 5, workers=1
    )
    assert rows == [] and errors == 1
    logged = json.loads((tmp_path / "teacher_cache.jsonl").read_text().splitlines()[0])
    assert logged["error"] and "relevance" not in logged
    assert tl.pending([_paper("a")], tmp_path, logged["design"]) == [_paper("a")]  # retried


def test_api_exception_is_an_error_row(tmp_path):
    def boom(kw):
        raise RuntimeError("overloaded")

    rows, errors = tl.run_teacher(
        [_paper("a")], "sys", TAGS, tmp_path, FakeClient(boom), 5, workers=1
    )
    assert rows == [] and errors == 1


def test_teacher_cache_resumes_and_cap_refuses_before_any_call(tmp_path):
    papers = [_paper(f"p{i}") for i in range(6)]
    client = FakeClient(lambda kw: _reply(_label(1)))
    system = tl.build_system(CFG, DEFS)
    with pytest.raises(SystemExit, match="refusing"):
        tl.run_teacher(papers, system, TAGS, tmp_path, client, 5)
    assert client.calls == []
    rows, _ = tl.run_teacher(papers, system, TAGS, tmp_path, client, 6, workers=3)
    assert len(client.calls) == 6 and [r["key"] for r in rows] == [p["key"] for p in papers]
    tl.run_teacher(papers, system, TAGS, tmp_path, client, 0, workers=3)
    assert len(client.calls) == 6
    other = tl.build_system(CFG | {"rubric": CFG["rubric"] + "\n"}, DEFS) + " changed"
    with pytest.raises(SystemExit, match="refusing"):
        tl.run_teacher(papers, other, TAGS, tmp_path, client, 0)  # new prompt: cache not reused


def test_cost_prices_cache_reads_at_a_tenth():
    row = {"usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}
    row2 = {"usage": {"cache_read_input_tokens": 1_000_000}}
    assert tl.cost_usd([row, row2])["usd"] == pytest.approx(2 + 10 + 0.2)


# -- metrics and tuning ------------------------------------------------------


def test_weighted_spearman_matches_plain_with_equal_weights_and_moves_with_weights():
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 3, 2, 4, 6, 5]
    assert jtt.wspearman(x, y, np.ones(6)) == pytest.approx(jps.spearman(x, y))
    heavy_left = jtt.wspearman(x, y, np.array([10, 10, 10, 1, 1, 1.0]))
    assert heavy_left != pytest.approx(jps.spearman(x, y))
    assert np.isnan(jtt.wspearman([1, 1, 1], [1, 2, 3], np.ones(3)))


def test_f1_vector_and_macro_skip_tags_with_no_truth():
    truth = np.array([[1, 0], [1, 0], [0, 0]], bool)
    pred = np.array([[1, 1], [0, 0], [0, 0]], bool)
    f1 = jtt.f1_vector(pred, truth)
    assert f1[0] == pytest.approx(2 / 3) and np.isnan(f1[1])
    assert jtt.macro_f1(pred, truth) == pytest.approx(2 / 3)


def test_thresholds_are_chosen_on_the_data_and_default_to_half_without_positives():
    p = pd.DataFrame({"a": [0.9, 0.35, 0.3, 0.1, 0.05, 0.2], "b": [0.5] * 6})
    truth = pd.DataFrame({"a": [1, 1, 1, 0, 0, 0], "b": [0] * 6}).astype(bool)
    thr = jtt.fit_thresholds(p, truth)
    assert thr["a"] == 0.3 and thr["b"] == 0.5  # 0.3 catches all three positives, no negatives


def test_platt_fixes_overconfident_probabilities_and_skips_tiny_classes():
    rng = np.random.default_rng(0)
    p = np.clip(rng.uniform(0, 1, 2000), 0.01, 0.99)
    truth = rng.uniform(size=2000) < 0.5 * p + 0.25  # true rate is a flattened version of p
    params = jtt.fit_platt(p, truth)
    after = jtt.apply_platt(p, params)
    assert jtt.brier(after, truth) < jtt.brier(p, truth)
    assert jtt.ece(after, truth) < jtt.ece(p, truth)
    assert jtt.fit_platt(p[:50], np.arange(50) < 2) == (1.0, 0.0)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    p = np.array([0.25] * 4 + [0.75] * 4)
    truth = np.array([1, 0, 0, 0, 1, 1, 1, 0], bool)
    assert jtt.ece(p, truth) == pytest.approx(0.0)


def test_relevance_map_learns_which_jev_score_matters_and_stays_in_range():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "jev_methods": rng.uniform(0, 5, n),
            "jev_findings": rng.uniform(0, 5, n),
            "jev_p_high": rng.uniform(0, 1, n),
        }
    )
    df["sonnet_relevance"] = np.round(df.jev_findings).astype(int)
    model = jtt.fit_relevance_map(df)
    coef = dict(zip(jtt.FEATURES, model["coef"], strict=True))
    assert coef["jev_findings"] > 3 * abs(coef["jev_methods"])
    out = jtt.apply_relevance_map(model, df)
    assert out.min() >= 0 and out.max() <= 5
    assert jps.spearman(out, df.sonnet_relevance) > jps.spearman(
        df.jev_methods, df.sonnet_relevance
    )


def test_split_is_seeded_disjoint_and_stratified():
    df = pd.DataFrame(
        {"key": [f"k{i}" for i in range(60)], "haiku_relevance": [i % 3 for i in range(60)]}
    )
    a = jtt.split_halves(df, 0)
    assert a.equals(jtt.split_halves(df, 0)) and not a.equals(jtt.split_halves(df, 1))
    for level in range(3):
        assert (a[df.haiku_relevance == level] == "train").sum() == 10


def test_population_weights_restore_the_real_level_counts_and_zero_the_adopted():
    df = pd.DataFrame(
        {
            "haiku_relevance": [0, 0, 4, 4, 4, 5],
            "status": ["screened"] * 5 + ["adopted"],
        }
    )
    w = jtt.population_weights(df, {0: 100, 4: 6, 5: 9})
    assert w.tolist() == [50, 50, 2, 2, 2, 0]
    pop = jtt.population_counts(
        [
            {"status": "screened", "relevance": 0},
            {"status": "filtered", "relevance": 0},
            {"status": "adopted", "relevance": 5},
            {"status": "new", "relevance": None},
        ]
    )
    assert pop == {0: 2}


def test_bootstrap_gives_zero_difference_for_identical_systems_and_is_seeded():
    rng = np.random.default_rng(2)
    n = 60
    truth_tags = rng.uniform(size=(n, 4)) < 0.4
    system = {"score": rng.uniform(0, 5, n), "high": rng.uniform(size=n) < 0.5, "tags": truth_tags}
    systems = {name: system for name in jtt.SYSTEMS}
    args = (systems, np.round(system["score"]), truth_tags, np.ones(n), np.arange(n) % 3, 40, 0)
    boot = jtt.bootstrap(*args)
    assert np.all(boot["macro_f1.tuned_minus_raw"] == 0)
    assert np.all(boot["spearman.tuned_minus_haiku"] == 0)
    again = jtt.bootstrap(*args)
    assert np.array_equal(boot["spearman.raw"], again["spearman.raw"], equal_nan=True)
    lo, hi = jtt.ci(boot["spearman.raw"])
    assert lo < hi


# -- end to end --------------------------------------------------------------


def _export(tmp_path, per_level=12):
    rows = [
        {
            "key": "adopt1", "title": "L5_adopted", "abstract": "abs", "published": "2025",
            "status": "adopted", "relevance": 5, "tags": ["survey"],
        }
    ]  # fmt: skip
    for level in range(6):
        for i in range(per_level):
            rows.append(
                {
                    "key": f"s{level}_{i}",
                    "title": f"L{level}_{i}",
                    "abstract": f"abs {i}",
                    "published": "2025",
                    "status": "screened",
                    "relevance": level,
                    "tags": [TAGS[0]] if i % 2 else [],
                }  # fmt: skip
            )
    for i in range(20):
        rows.append(
            {
                "key": f"f{i}",
                "title": "F",
                "abstract": "x",
                "published": "2025",
                "status": "filtered",
                "relevance": 0,
                "tags": [],
            }  # fmt: skip
        )
    rows.append(
        {
            "key": "n1",
            "title": "N",
            "abstract": "x",
            "published": "2025",
            "status": "new",
            "relevance": None,
            "tags": [],
        }  # fmt: skip
    )
    path = tmp_path / "papers.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def _level_of(title):
    return int(title.split("_")[0][1:])


def _jev_response(level, odd):
    def score(x):
        return {
            "type": "score", "score": float(x), "confidence": 0.9,
            "probabilities": {"0": 0.1, "4": 0.2, "5": 0.2},
        }  # fmt: skip

    answers = {"methods_relevance": score(level), "findings_relevance": score(level)}
    for t in TAGS:
        answers[f"tag_{t}"] = {"type": "noul", "noul": 0.8 if odd and t == TAGS[0] else 0.1}
    return {"answers": answers, "usage": {"input_tokens": 500, "output_tokens": 0}}


def test_main_runs_end_to_end_with_one_ledger_entry_before_any_call(tmp_path, env):
    export = _export(tmp_path)
    out = tmp_path / "out"
    seen_ledger: list[int] = []

    def ledger_lines():
        seen_ledger.append(len(env.read_text().splitlines()) if env.exists() else 0)

    def respond(kw):
        title = kw["messages"][0]["content"].split("\n")[0].removeprefix("Title: ")
        odd = int(title.split("_")[1]) % 2 if title[1:].split("_")[1].isdigit() else 0
        return _reply(_label(min(5, _level_of(title) + odd), [TAGS[0]] if odd else []))

    client = FakeClient(respond, on_call=ledger_lines)
    posts: list[dict] = []

    def post(url, headers, body):
        ledger_lines()
        posts.append(body)
        title = body["state"]["title"]
        odd = int(title.split("_")[1]) % 2 if title[1:].split("_")[1].isdigit() else 0
        return _jev_response(_level_of(title), odd)

    seed_cache = tmp_path / "seed.jsonl"
    jps.run(  # a pre-existing Jev cache for the same design, to be copied in and reused
        [{"key": "s0_0", "title": "L0_0", "abstract": "abs 0"}],
        jps.build_questions(CFG, DEFS),
        TAGS,
        tmp_path / "seedrun",
        "K",
        5,
        post=lambda u, h, b: _jev_response(0, 0),
        sleep=lambda s: None,
        log=False,
    )
    seed_cache.write_text((tmp_path / "seedrun" / "cache.jsonl").read_text())
    n_args = [
        "--papers", str(export), "--out", str(out), "--n-per-level", "10", "--seed", "0",
        "--jev-cache-from", str(seed_cache), "--bootstrap", "30", "--workers", "2",
    ]  # fmt: skip
    path = jtt.main(n_args, post=post, sleep=lambda s: None, client=client)

    n_sample = 6 * 10 + 1
    assert len(client.calls) == n_sample
    assert len(posts) <= n_sample and len(posts) >= n_sample - 1  # the seeded cache was reused
    assert set(seen_ledger) == {1}  # the single ledger entry predates every Sonnet and Jev call
    entry = json.loads(env.read_text().splitlines()[0])
    assert {"methodology", "jev", "teacher"} <= set(entry["tags"])
    assert len(env.read_text().splitlines()) == 1
    for body in posts:  # Jev reads only title and abstract: nothing from the teacher
        assert set(body["state"]) == {"title", "abstract"}
        assert "reason" not in json.dumps(body)

    stats = json.loads((path / "stats.json").read_text())
    assert stats["n.split"]["value"] == {"train": 31, "test": 30}
    assert stats["population.by_haiku_level"]["value"]["0"] == 12 + 20
    assert "differences" in stats and "haiku.vs_sonnet" in stats and "calibration" in stats
    statements = (path / "STATEMENTS.md").read_text()
    assert "95% CI" in statements and "not ground truth" in statements
    sheet = pd.read_csv(path / "test_spotcheck.csv", dtype=str, keep_default_na=False)
    assert (
        len(sheet) == 30 and (sheet.human_relevance == "").all() and (sheet.human_tags == "").all()
    )
    assert {
        "sonnet_relevance",
        "haiku_relevance",
        "jev_raw_relevance",
        "jev_tuned_relevance",
    } <= set(sheet.columns)
    assert stats["tuning.thresholds"]["value"].keys() == dict.fromkeys(TAGS).keys()

    calls_before = (len(client.calls), len(posts))
    jtt.main(n_args, post=post, sleep=lambda s: None, client=client)  # rerun pays nothing
    assert (len(client.calls), len(posts)) == calls_before


def test_caps_refuse_before_logging_or_calling(tmp_path, env):
    export = _export(tmp_path)
    client = FakeClient(lambda kw: _reply(_label(1)))
    base = ["--papers", str(export), "--out", str(tmp_path / "o"), "--n-per-level", "5"]
    with pytest.raises(SystemExit, match="refusing"):
        jtt.main([*base, "--max-teacher-calls", "3"], client=client)
    with pytest.raises(SystemExit, match="refusing"):
        jtt.main([*base, "--max-jev-calls", "3"], client=client)
    assert client.calls == [] and not env.exists()
