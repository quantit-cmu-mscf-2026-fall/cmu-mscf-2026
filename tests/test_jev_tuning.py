"""Frozen Jev tuning, the full tagging pass, the labelling queue and the human-label refit.

Fakes only: cached Jev answers are constructed, the HTTP post is a fake, and no API is called.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import jev_paper_screen as jps  # noqa: E402
import jev_queue as jq  # noqa: E402
import jev_refit as jr  # noqa: E402
import jev_tag as jtag  # noqa: E402
import jev_tuning as jt  # noqa: E402

TAGS, DESIGN = jt.current_design()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("TYPESAFE_API_KEY", "K")
    return tmp_path / "ledger" / "runs.jsonl"


def ledger(env) -> list[dict]:
    return [json.loads(ln) for ln in env.read_text().splitlines()] if env.exists() else []


# -- synthetic data -----------------------------------------------------------


def cache_row(key, rel, survey):
    high = 0.9 if rel >= 4 else 0.1
    return {
        "key": key,
        "model": jps.MODEL,
        "design": DESIGN,
        "error": "",
        "methods": {"expected": float(rel), "p_high": high},
        "findings": {"expected": float(max(rel - 1, 0)), "p_high": high},
        "tags": {t: (0.9 if survey else 0.1) if t == "survey" else 0.1 for t in TAGS},
        "usage": {"input_tokens": 1000},
        "seconds": 0.1,
    }


def key(i):
    return f"doi:10.1/x{i:03d}"


def world(n=80):
    """Papers, cache rows and Sonnet labels: relevance i%6, 'survey' on every third paper."""
    rows = {key(i): cache_row(key(i), i % 6, i % 3 == 0) for i in range(n)}
    sonnet = {
        key(i): {"relevance": i % 6, "tags": ["survey"] if i % 3 == 0 else []} for i in range(n)
    }
    papers = [
        {
            "key": key(i),
            "title": f"Paper {i}",
            "abstract": f"Abstract {i} " * 80,
            "published": "2024-01-01",
            "status": ["screened", "new", "filtered"][i % 3],
            "relevance": i % 6,
            "haiku_tags": ["survey"] if i % 3 == 0 else [],
        }
        for i in range(n)
    ]
    return papers, rows, sonnet


def write_cache(path: Path, rows) -> Path:
    text = "".join(json.dumps(r) + "\n" for r in rows.values())
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def write_teacher(path: Path, sonnet) -> Path:
    path.write_text(
        "".join(
            json.dumps({"key": k, "design": "d1", "error": "", **v, "reason": "r"}) + "\n"
            for k, v in sonnet.items()
        ),
        encoding="utf-8",
    )
    return path


def fitted(n=80) -> jt.Tuning:
    _, rows, sonnet = world(n)
    table = jt.training_table(jt.jev_features(rows, TAGS), sonnet, "sonnet")
    return jt.Tuning.fit(table, TAGS, DESIGN)


# -- tuning file ----------------------------------------------------------------


def test_save_load_round_trip_keeps_every_field(tmp_path):
    tuning = fitted()
    path = tmp_path / "tuning.json"
    tuning.save(path)
    again = jt.load_tuning(path, DESIGN)
    assert again.to_dict() == json.loads(path.read_text())
    assert again.thresholds == tuning.thresholds
    assert again.relevance["high_cut"] == tuning.relevance["high_cut"]
    assert again.fitted_on == {"sonnet": 80, "human": 0}
    data = json.loads(path.read_text())
    assert data["model"] == jps.MODEL and data["design_hash"] == DESIGN
    assert set(data["tags"]["survey"]) == {"threshold", "platt_a", "platt_b"}
    assert {"coef", "intercept", "mean", "std", "high_cut"} <= set(data["relevance"])
    assert data["created"]


def test_load_refuses_a_different_question_design(tmp_path):
    path = tmp_path / "tuning.json"
    fitted().save(path)
    with pytest.raises(jt.TuningError, match="refit before applying"):
        jt.load_tuning(path, "000000000000")


def test_tag_cli_refuses_a_stale_tuning_before_any_call(tmp_path):
    papers, _, _ = world(3)
    pj = tmp_path / "p.jsonl.gz"
    with gzip.open(pj, "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(p | {"tags": p["haiku_tags"]}) + "\n" for p in papers))
    tuning = fitted()
    tuning.design = "000000000000"
    tuning.save(tmp_path / "tuning.json")

    def post(*_):
        raise AssertionError("no call may be made")

    with pytest.raises(jt.TuningError):
        jtag.main(
            [
                "--papers",
                str(pj),
                "--tuning",
                str(tmp_path / "tuning.json"),
                "--out",
                str(tmp_path / "o"),
            ],
            post=post,
        )


def test_apply_gives_the_expected_thresholds_and_calibrated_probabilities():
    tuning = jt.Tuning(
        model=jps.MODEL,
        design=DESIGN,
        tags=["survey", "finance_ml"],
        thresholds={"survey": 0.4, "finance_ml": 0.7},
        platt={"survey": (2.0, 0.0), "finance_ml": (1.0, 0.0)},
        relevance={
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "coef": [1.0, 0.0, 0.0],
            "intercept": 0.5,
            "high_cut": 3.0,
        },
        fitted_on={"sonnet": 1, "human": 0},
        created="t",
    )
    df = pd.DataFrame(
        {
            "jev_methods": [3.0, 1.0],
            "jev_findings": [0.0, 2.0],
            "jev_relevance": [3.0, 2.0],
            "jev_p_high": [0.5, 0.0],
            "p_survey": [0.6, 0.3],
            "p_finance_ml": [0.6, 0.8],
        }
    )
    out = tuning.apply(df)
    assert list(out.jev_relevance_raw) == [3.0, 2.0]
    assert list(out.jev_relevance) == [3.5, 1.5]  # methods + 0.5
    assert list(out.jev_high) == [True, False]
    assert list(out.jev_tags) == ["survey", "finance_ml"]  # 0.6>=0.4; 0.3<0.4, 0.8>=0.7
    logit = math.log(0.6 / 0.4)
    assert out.p_survey[0] == pytest.approx(1 / (1 + math.exp(-2 * logit)))
    assert out.p_finance_ml[1] == pytest.approx(0.8)  # identity Platt


def test_diverged_platt_falls_back_to_identity():
    p = np.array([0.50] * 20 + [0.52] * 20)  # separable, but only by a huge slope
    truth = p > 0.51
    assert jt.stable_platt(p, truth) == (1.0, 0.0)


# -- queue ---------------------------------------------------------------------------


def queue_frame(n=300):
    papers, rows, sonnet = world(n)
    tuning = fitted(n)
    df = jtag.tag_frame(papers, list(rows.values()), tuning, dict(list(sonnet.items())[:120]))
    df.loc[df.index[:5], "status"] = "adopted"
    return df, tuning


def test_queue_has_the_right_counts_reasons_and_no_duplicates():
    df, tuning = queue_frame()
    q = jq.build_queue(df, tuning, seed=1)
    assert q.pick_reason.value_counts().to_dict() == {"random": 40, "uncertain": 40, "disagree": 20}
    assert q.key.is_unique and q.queue_id.is_unique
    assert list(q.columns) == jq.COLUMNS
    assert (q[["human_relevance", "human_tags", "human_note"]] == "").all().all()
    status = df.set_index("key").status
    assert set(status[q.key]) <= set(jq.POOL)  # adopted papers are never queued
    sonnet_keys = set(df.key[df.sonnet_relevance.notna()])
    dis = q[q.pick_reason == "disagree"]
    assert set(dis.key) <= sonnet_keys
    assert (dis.sonnet_relevance != "").all()
    assert q.abstract_snippet.str.len().max() <= jq.SNIPPET + 3


def test_queue_is_seeded_and_uncertain_picks_sit_near_the_cut():
    df, tuning = queue_frame()
    a, b = jq.build_queue(df, tuning, seed=1), jq.build_queue(df, tuning, seed=1)
    c = jq.build_queue(df, tuning, seed=2)
    assert a.equals(b)
    assert set(a[a.pick_reason == "random"].key) != set(c[c.pick_reason == "random"].key)
    pool = df[df.status.isin(jq.POOL)]
    cut = tuning.relevance["high_cut"]
    nearest = (pool.jev_relevance - cut).abs().min()
    unc = a[a.pick_reason == "uncertain"].merge(df[["key", "jev_relevance"]], on="key")
    assert ((unc.jev_relevance_y - cut).abs().min()) == pytest.approx(nearest)


def test_links_follow_the_key_scheme():
    assert jq.link_for("doi:10.1093/rfs/hhv059") == "https://doi.org/10.1093/rfs/hhv059"
    assert jq.link_for("arxiv:2008.13651") == "https://arxiv.org/abs/2008.13651"
    assert jq.link_for("title:f694210a890622fa") == ""


# -- full pass ---------------------------------------------------------------------


def test_full_pass_asks_only_for_uncached_papers_and_writes_the_files(tmp_path, env):
    papers, rows, sonnet = world(12)
    pj = tmp_path / "p.jsonl.gz"
    with gzip.open(pj, "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(p | {"tags": p["haiku_tags"]}) + "\n" for p in papers))
    fitted(12).save(tmp_path / "tuning.json")
    cached = {k: v for k, v in list(rows.items())[:9]}
    write_cache(tmp_path / "old.jsonl", cached)
    calls = []

    def post(url, headers, body):
        calls.append(body["state"]["title"])
        i = int(body["state"]["title"].split()[1])
        r = cache_row(key(i), i % 6, i % 3 == 0)
        answers = {
            "methods_relevance": {
                "score": r["methods"]["expected"],
                "probabilities": {"4": r["methods"]["p_high"]},
            },
            "findings_relevance": {
                "score": r["findings"]["expected"],
                "probabilities": {"4": r["findings"]["p_high"]},
            },
        } | {f"tag_{t}": {"noul": p} for t, p in r["tags"].items()}
        return {"answers": answers, "usage": {"input_tokens": 10}}

    out = tmp_path / "out"
    jtag.main(
        ["--papers", str(pj), "--tuning", str(tmp_path / "tuning.json"), "--out", str(out),
         "--cache-from", str(tmp_path / "old.jsonl"), "--max-calls", "5"],
        post=post,
    )  # fmt: skip
    assert sorted(calls) == ["Paper 10", "Paper 11", "Paper 9"]
    assert len(ledger(env)) == 1
    tagged = pd.read_csv(out / "tagged_all.csv", encoding="utf-8-sig")
    assert len(tagged) == 12
    assert list(tagged.columns) == jtag.tagged_columns(TAGS)
    assert tagged.link.iloc[0] == "https://doi.org/10.1/x000"
    assert (out / "label_queue.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert "human_relevance" in (out / "LABELING.md").read_text()


# -- reading human labels -----------------------------------------------------------


def write_labels(path: Path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, jq.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: "" for c in jq.COLUMNS} | r)
    return path


def test_empty_and_partial_rows_are_ignored_and_bad_rows_reported(tmp_path):
    path = write_labels(
        tmp_path / "q.csv",
        [
            {
                "key": "a",
                "pick_reason": "random",
                "human_relevance": "4",
                "human_tags": "survey; finance_ml",
            },
            {"key": "b"},  # nothing filled in
            {"key": "c", "human_note": "just a note"},  # note only: ignored
            {"key": "d", "human_relevance": "2"},  # no tags is a valid label
            {"key": "e", "human_relevance": "7"},
            {"key": "f", "human_relevance": "3", "human_tags": "surveys"},
            {"key": "g", "human_tags": "survey"},  # tags without relevance
            {"key": "h", "human_relevance": "2.5"},
        ],
    )
    labels, problems, ignored = jr.read_human_labels([path], TAGS)
    assert labels == {
        "a": {"relevance": 4, "tags": ["finance_ml", "survey"], "pick_reason": "random"},
        "d": {"relevance": 2, "tags": [], "pick_reason": ""},
    }
    assert ignored == 2
    assert len(problems) == 4
    assert any("surveys" in p for p in problems) and any("'7'" in p for p in problems)
    assert any("no human_relevance" in p for p in problems)


def test_human_labels_override_sonnet_and_are_added_where_sonnet_has_none():
    sonnet = {"a": {"relevance": 1, "tags": []}, "b": {"relevance": 2, "tags": []}}
    human = {"a": {"relevance": 5, "tags": ["survey"]}, "z": {"relevance": 3, "tags": []}}
    kept, counts = jr.combine(sonnet, human)
    assert kept == {"b": sonnet["b"]}
    assert counts == {"overridden": 1, "added": 1}


# -- refit end to end ----------------------------------------------------------------


def refit_setup(tmp_path, n_random=12, n_other=6):
    papers, rows, sonnet = world(80)
    cache = write_cache(tmp_path / "cache.jsonl.gz", rows)  # .gz is accepted
    teacher = write_teacher(tmp_path / "teacher_cache.jsonl", sonnet)
    prev = tmp_path / "prev.json"
    fitted().save(prev)
    tuning = jt.load_tuning(prev, DESIGN)
    tagged = jtag.finish(jtag.tag_frame(papers, list(rows.values()), tuning, sonnet), TAGS)
    tagged.to_csv(tmp_path / "tagged_all.csv", index=False, encoding="utf-8-sig")
    # human labels: every one agrees with the truth except the 'other' picks, which are wrong
    lab = []
    for i in range(n_random):
        lab.append(
            {"key": key(i), "pick_reason": "random", "human_relevance": str(i % 6),
             "human_tags": "survey" if i % 3 == 0 else ""}
        )  # fmt: skip
    for i in range(n_random, n_random + n_other):
        lab.append(
            {
                "key": key(i),
                "pick_reason": "uncertain",
                "human_relevance": "0",
                "human_tags": "survey",
            }
        )
    labels = write_labels(tmp_path / "q.csv", lab)
    return cache, teacher, prev, labels


def run_refit(tmp_path, cache, teacher, prev, labels, extra=()):
    return jr.main(
        ["--labels", str(labels), "--jev-cache", str(cache), "--teacher", str(teacher),
         "--tagged", str(tmp_path / "tagged_all.csv"), "--prev-tuning", str(prev),
         "--out", str(tmp_path / "refit"), *extra]
    )  # fmt: skip


def test_refit_overrides_sonnet_and_scores_on_random_human_rows_only(tmp_path, env):
    cache, teacher, prev, labels = refit_setup(tmp_path)
    run_refit(tmp_path, cache, teacher, prev, labels)
    out = tmp_path / "refit"
    new = json.loads((out / "tuning.json").read_text())
    assert new["fitted_on"] == {"sonnet": 80 - 18, "human": 18}  # 18 human rows replace Sonnet's
    assert new["human_weight"] == 3
    stats = json.loads((out / "stats.json").read_text())
    assert stats["labels"]["value"]["overridden"] == 18
    metrics = stats["metrics"]["value"]
    assert metrics["raw"]["n"] == 12  # the 6 uncertain-pick rows are not scored
    assert metrics["tuned_with_human"]["n"] == 12
    assert metrics["sonnet"]["n"] == 12
    assert set(metrics) == set(jr.SYSTEMS)
    assert metrics["raw"]["spearman"] == pytest.approx(1.0)  # raw Jev here equals the labels
    assert (out / "STATEMENTS.md").read_text().count("stats.json") >= 2
    retagged = pd.read_csv(out / "tagged_all.csv", encoding="utf-8-sig")
    assert len(retagged) == 80 and "jev_high" in retagged
    entries = [e for e in ledger(env) if e["name"] == "jev-refit"]
    assert len(entries) == 1 and entries[0]["params"]["human_labels"] == 18


def test_refit_reports_no_unbiased_estimate_with_fewer_than_ten_random_rows(tmp_path):
    cache, teacher, prev, labels = refit_setup(tmp_path, n_random=9)
    run_refit(tmp_path, cache, teacher, prev, labels)
    stats = json.loads((tmp_path / "refit" / "stats.json").read_text())
    assert "metrics" not in stats
    assert stats["eval.n_random_labelled"]["value"] == 9
    assert "no unbiased accuracy estimate" in (tmp_path / "refit" / "STATEMENTS.md").read_text()


def test_refit_refuses_papers_missing_from_the_jev_cache(tmp_path, env):
    cache, teacher, prev, labels = refit_setup(tmp_path)
    _, rows, _ = world(80)
    del rows[key(79)]
    short = write_cache(tmp_path / "short.jsonl", rows)
    with pytest.raises(SystemExit, match="never calls the API"):
        run_refit(tmp_path, short, teacher, prev, labels)
    assert not [e for e in ledger(env) if e["name"] == "jev-refit"]
    assert not (tmp_path / "refit" / "tuning.json").exists()


def test_refit_with_only_bad_rows_stops(tmp_path):
    cache, teacher, prev, _ = refit_setup(tmp_path)
    bad = write_labels(tmp_path / "bad.csv", [{"key": key(0), "human_relevance": "9"}])
    with pytest.raises(SystemExit, match="no usable human labels"):
        run_refit(tmp_path, cache, teacher, prev, bad)
