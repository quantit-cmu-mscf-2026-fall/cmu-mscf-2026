"""Offline verification and an opt-in, resumable 120-request research pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capstone.factors import store
from capstone.factors.llm import load_config, make_client
from capstone.factors.research_artifacts import (
    atomic_json,
    checkpoint,
    manifest,
    read_checkpoint,
    register,
    write_pilot_report,
)
from capstone.factors.research_benchmark import (
    DEFAULT_BENCHMARK,
    catalogue,
    load_benchmark,
    pilot_configs,
    pilot_policy,
    run_controlled,
    run_legacy,
)
from capstone.factors.research_contracts import ResearchPolicy, digest
from capstone.factors.research_controller import ResearchController
from capstone.factors.research_proof import (
    construction_handoff_checks,
    development_checks,
    recovery_checks,
)
from capstone.factors.research_relations import (
    attach_constructions,
    coverage_scorecard,
    research_graph,
    write_relationship_report,
)
from capstone.factors.research_requests import RequestAbandoned, RequestJournal, ResearchPaused
from capstone.factors.research_review_v2 import review_packet, summarize
from capstone.pipeline.budget import BudgetedClient, CallBudgetExceeded


def verify(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    checks = development_checks() | recovery_checks(root) | construction_handoff_checks(root)
    if not all(checks.values()):
        raise AssertionError("offline conformance failed")
    result = {
        "mode": "offline_fixtures",
        "checks": checks,
        "paid_model_calls": 0,
        "historical_data_reads": 0,
        "financial_admissions": 0,
    }
    atomic_json(root / "conformance.json", result)
    return result


def _saved(root, frozen_id):
    saved = {arm: {} for arm in ("A", "B", "C")}
    envelopes = {arm: [] for arm in saved}
    for arm in saved:
        for path in sorted((root / "cases" / arm).glob("*.json")):
            row = read_checkpoint(path, frozen_id)
            saved[arm][row["packet"]] = row["artifact"]
            envelopes[arm].append(row)
    reviews = {}
    for path in sorted((root / "reviews").glob("*.json")):
        row = read_checkpoint(path, frozen_id)
        reviews.setdefault(row["packet"], {})[row["pass"]] = row["judgments"]
    reviews = {key: [rows[i] for i in sorted(rows)] for key, rows in reviews.items()}
    return saved, envelopes, reviews


def _case_store(root, arm, packet, envelopes):
    folder = root / "stores" / arm
    folder.mkdir(parents=True, exist_ok=True)
    attempt = len(list(folder.glob(packet["id"] + "-attempt-*.db")))
    path = folder / f"{packet['id']}-attempt-{attempt}.db"
    con = store.connect(path)
    if envelopes:
        previous = max(envelopes, key=lambda row: row["sequence"])
        source = store.connect(root / previous["store"])
        try:
            source.backup(con)
        finally:
            source.close()
    return con, path


def _run_case(root, packet, arm, config, client, journal, controller, envelopes):
    con, path = _case_store(root, arm, packet, envelopes)
    budget = BudgetedClient(client, 30, journal=journal, bucket=arm, task_id=packet["id"])
    try:
        try:
            result = (
                run_controlled(packet, con, budget, config, controller)
                if arm == "B"
                else run_legacy(packet, con, budget, config)
            )
        except RequestAbandoned:
            result = {"status": "request_abandoned", "construction_verified": False}
        except (ValueError, TypeError, KeyError) as error:
            if arm == "B" and controller.state(packet["id"]) in (
                "assigned",
                "specification_ready",
                "specification_frozen",
            ):
                controller.record_failure(
                    packet["id"], "interpretation_failed", {"error_class": type(error).__name__}
                )
            result = {
                "status": "interpretation_failed",
                "error_class": type(error).__name__,
                "construction_verified": False,
                "financial_admission": False,
            }
        return {
            "packet": packet["id"],
            "artifact": result,
            "store": path.relative_to(root).as_posix(),
            "sequence": len(envelopes),
        }
    finally:
        con.close()


def _reviews(root, benchmark, results, reviews, client, config, journal, frozen_id):
    for packet in sorted(benchmark["packets"], key=lambda p: p["source_order"]):
        key = packet["id"]
        if not all(key in results[arm] for arm in ("A", "B", "C")):
            continue
        if any(
            results[arm][key]["status"] in ("request_abandoned", "budget_exhausted")
            for arm in results
        ):
            continue
        while len(reviews.get(key, [])) < 2:
            index = len(reviews.get(key, []))
            failure_path = root / "review-failures" / f"{key}-{index}.json"
            if failure_path.exists():
                break
            budget = BudgetedClient(
                client, 30, journal=journal, bucket="review", task_id=f"review:{key}:{index}"
            )
            try:
                judgments = review_packet(
                    packet,
                    {arm: results[arm][key] for arm in results},
                    budget,
                    config,
                    index,
                    version=benchmark.get("review_version", 1),
                    qualified_inputs=benchmark["qualified_inputs"],
                )
            except CallBudgetExceeded:
                return
            except (ValueError, TypeError, KeyError) as error:
                checkpoint(
                    failure_path,
                    {
                        "error_class": type(error).__name__,
                        "status": "review_failed",
                        **(
                            {"reason": str(error)[:160]}
                            if benchmark.get("review_version") == 2
                            else {}
                        ),
                    },
                    frozen_id,
                )
                break
            checkpoint(
                root / "reviews" / f"{key}-{index}.json",
                {"packet": key, "pass": index, "judgments": judgments},
                frozen_id,
            )
            reviews.setdefault(key, []).append(judgments)


def _report(root, frozen, controller, journal, stop_reason=None):
    results, _, reviews = _saved(root, digest(frozen))
    accounting = journal.report()
    summary = summarize(
        frozen["benchmark"],
        results,
        reviews,
        accounting,
        integrity_passed=all(frozen["integrity_checks"].values()),
    )
    summary["mode"] = frozen["mode"]
    summary["stop_reason"] = stop_reason
    states = {
        p["id"]: controller.state(p["id"])
        for p in frozen["benchmark"]["packets"]
        if controller.state(p["id"]) is not None
    }
    entries = catalogue(frozen["benchmark"])
    graph = attach_constructions(research_graph(entries, states), results, frozen["benchmark"])
    atomic_json(root / "relationships.json", graph)
    atomic_json(root / "transitions.json", controller.events())
    atomic_json(root / "recovery.json", accounting["events"])
    write_relationship_report(
        root / "relationships.html", graph, coverage_scorecard(entries, states)
    )
    write_pilot_report(root, frozen, summary, accounting, stop_reason)
    return summary


def execute(
    root,
    benchmark_path=DEFAULT_BENCHMARK,
    *,
    resume=False,
    client=None,
    offline_fixture=False,
    config=None,
):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark(benchmark_path)
    config = load_config() if config is None else config
    if not config.alignment_model:
        raise ValueError("the configured review model is required; no substitution")
    checks = verify(root / "offline-proof")["checks"]
    configs = pilot_configs(config)
    policy = pilot_policy()
    frozen = manifest(benchmark, configs, checks, policy)
    frozen["mode"] = "offline_fixture" if offline_fixture else "real_agent_exploratory"
    if not offline_fixture and (
        frozen["release"]["dirty"] or not frozen["release"].get("merged_main")
    ):
        raise ValueError("live pilot requires clean merged code")
    if offline_fixture and client is None:
        raise ValueError("offline fixture requires an explicitly supplied fake client")
    supplied_client = client is not None
    controller = None
    try:
        with RequestJournal(root / "requests.db", frozen) as journal:
            manifest_path = root / "manifest.json"
            if manifest_path.exists():
                if not resume or digest(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                ) != digest(frozen):
                    raise ValueError("use resume with the identical frozen study")
            elif resume:
                raise ValueError("no frozen study to resume")
            else:
                atomic_json(manifest_path, frozen)
            register(root, frozen)
            controller = ResearchController(
                root / "controller.db",
                policy,
                catalogue(benchmark),
                benchmark["qualified_inputs"],
                generator_version=digest(frozen),
            )
            results, envelopes, reviews = _saved(root, digest(frozen))
            if all(len(rows) == 8 for rows in results.values()) and all(
                len(reviews.get(p["id"], [])) == 2 for p in benchmark["packets"]
            ):
                return _report(root, frozen, controller, journal)
            if client is None:
                try:
                    client = make_client().with_options(max_retries=0, timeout=120)
                except Exception as error:
                    return _report(
                        root,
                        frozen,
                        controller,
                        journal,
                        "model_client_unavailable: " + type(error).__name__,
                    )
            halted, stop_reason = set(), None
            for arm in results:
                if any(v["status"] == "budget_exhausted" for v in results[arm].values()):
                    halted.add(arm)
            try:
                packets = {p["id"]: p for p in benchmark["packets"]}
                for iteration in range(8):
                    arms = ["A", "B", "C"]
                    arms = arms[iteration % 3 :] + arms[: iteration % 3]
                    for arm in arms:
                        if arm in halted:
                            continue
                        pending = [p for p in benchmark["packets"] if p["id"] not in results[arm]]
                        if not pending:
                            continue
                        if arm == "B":
                            unfinished = [
                                p for p in pending if controller.state(p["id"]) is not None
                            ]
                            brief = None if unfinished else controller.next_brief()
                            if not unfinished and brief is None:
                                continue
                            packet = unfinished[0] if unfinished else packets[brief.packet_id]
                        else:
                            packet = min(pending, key=lambda p: p["source_order"])
                        try:
                            envelope = _run_case(
                                root,
                                packet,
                                arm,
                                configs[arm],
                                client,
                                journal,
                                controller,
                                envelopes[arm],
                            )
                        except CallBudgetExceeded:
                            halted.add(arm)
                            envelope = {
                                "packet": packet["id"],
                                "artifact": {"status": "budget_exhausted"},
                                "store": "",
                                "sequence": len(envelopes[arm]),
                            }
                        checkpoint(
                            root / "cases" / arm / f"{packet['id']}.json", envelope, digest(frozen)
                        )
                        results[arm][packet["id"]] = envelope["artifact"]
                        envelopes[arm].append(envelope)
                        print(
                            f"saved {arm}/{packet['id']}: {envelope['artifact']['status']}",
                            flush=True,
                        )
                        _reviews(
                            root,
                            benchmark,
                            results,
                            reviews,
                            client,
                            config,
                            journal,
                            digest(frozen),
                        )
                        _report(root, frozen, controller, journal)
                _reviews(root, benchmark, results, reviews, client, config, journal, digest(frozen))
            except (ResearchPaused, CallBudgetExceeded) as error:
                stop_reason = str(error)
            except (ValueError, TypeError, KeyError) as error:
                stop_reason = "saved response needs review: " + type(error).__name__
            return _report(root, frozen, controller, journal, stop_reason)
    finally:
        if controller is not None:
            controller.close()
        if not supplied_client and client is not None:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "start", "resume", "report", "resolve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--request-id")
    parser.add_argument("--decision", choices=("retry", "abandon"))
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.mode == "verify":
        result = verify(args.output)
    elif args.mode in ("start", "resume"):
        result = execute(args.output, args.benchmark, resume=args.mode == "resume")
    else:
        frozen = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        with RequestJournal(args.output / "requests.db", frozen) as journal:
            if args.mode == "resolve":
                if not all((args.request_id, args.decision, args.reason)):
                    parser.error("resolve requires request-id, decision and a nonsecret reason")
                journal.resolve(args.request_id, args.decision, args.reason)
                result = journal.report()
            else:
                policy = ResearchPolicy(**frozen["research_policy"])
                controller = ResearchController(
                    args.output / "controller.db",
                    policy,
                    catalogue(frozen["benchmark"]),
                    [],
                    generator_version=digest(frozen),
                )
                try:
                    old = args.output / "summary.json"
                    reason = (
                        json.loads(old.read_text()).get("stop_reason") if old.exists() else None
                    )
                    result = _report(args.output, frozen, controller, journal, reason)
                finally:
                    controller.close()
    print(json.dumps(result, sort_keys=True))
    if args.mode in ("start", "resume") and not result["complete_matched_study"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
