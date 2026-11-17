"""Deterministic opt-in scheduler and verified research handoffs, without broker access."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from capstone.factors.idea_routes import IdeaCatalogueEntry, catalogue_rows
from capstone.factors.research_contracts import (
    AssignedBrief,
    ConstructionEvidence,
    MethodSpec,
    ResearchPolicy,
    SourceClaim,
    canonical,
    digest,
    qualify,
)
from capstone.factors.research_records import put_artifact


class ResearchController:
    """Catalogue, capabilities and constructor bindings are curator-owned frozen inputs."""

    def __init__(self, path, policy, catalogue, qualified_inputs, *, generator_version):
        if not isinstance(policy, ResearchPolicy) or not generator_version:
            raise ValueError("versioned policy and generator required")
        self.policy = policy
        self.inputs = frozenset(qualified_inputs)
        catalogue = catalogue_rows(catalogue)
        self.catalogue = {}
        methods = {}
        for entry in catalogue:
            brief = AssignedBrief(**entry["brief"])
            idea = IdeaCatalogueEntry.from_dict(entry) if entry.get("version") == 2 else None
            claim = idea if idea is not None else SourceClaim(**entry["claim"])
            if brief.packet_id in self.catalogue:
                raise ValueError("duplicate packet")
            if brief.method_key in methods and methods[brief.method_key] != brief.family:
                raise ValueError("canonical method cannot change family")
            if idea is None and entry["constructor"] not in (
                "formula",
                "reference",
                "research",
                "unavailable",
            ):
                raise ValueError("unknown constructor binding")
            methods[brief.method_key] = brief.family
            self.catalogue[brief.packet_id] = (brief, claim, entry["constructor"])
        manifest = {
            "policy": asdict(policy),
            "catalogue": sorted(catalogue, key=canonical),
            "qualified_inputs": sorted(self.inputs),
            "generator": generator_version,
        }
        self.identity = digest(manifest)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS research_identity (payload TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS research_briefs
              (id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL, repairs INTEGER);
            CREATE TABLE IF NOT EXISTS research_events
              (seq INTEGER PRIMARY KEY, brief TEXT NOT NULL, state TEXT NOT NULL,
               reason TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_artifacts
              (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_research_events_update
              BEFORE UPDATE ON research_events BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_research_events_delete
              BEFORE DELETE ON research_events BEGIN SELECT RAISE(ABORT,'append only'); END;
        """)
        try:
            with self.transaction():
                old = self.db.execute("SELECT payload FROM research_identity").fetchone()
                if old and old[0] != canonical(manifest):
                    raise ValueError("research manifest changed; create a new study")
                if old is None:
                    self.db.execute(
                        "INSERT INTO research_identity VALUES (?)",
                        (canonical(manifest),),
                    )
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _artifact(self, kind, payload):
        return put_artifact(self.db, kind, payload)

    def _event(self, packet, state, reason, payload):
        self.db.execute("UPDATE research_briefs SET state=? WHERE id=?", (state, packet))
        self.db.execute(
            "INSERT INTO research_events(brief,state,reason,payload) VALUES (?,?,?,?)",
            (packet, state, reason, canonical(payload)),
        )

    def state(self, packet):
        row = self.db.execute("SELECT state FROM research_briefs WHERE id=?", (packet,)).fetchone()
        return None if row is None else row[0]

    def _require(self, packet, *states):
        if self.state(packet) not in states:
            raise ValueError(f"unsupported transition from {self.state(packet)}")

    def reserve(self, packet):
        brief, _, _ = self.catalogue[packet]
        with self.transaction():
            if self.state(packet) is not None:
                return False
            used = [
                AssignedBrief(**json.loads(r[0]))
                for r in self.db.execute("SELECT payload FROM research_briefs")
            ]
            if any(b.method_key == brief.method_key and b.track == brief.track for b in used):
                return False
            limit = {(f, t): n for f, t, n in self.policy.reservations}.get(
                (brief.family, brief.track), 0
            )
            if sum(b.family == brief.family and b.track == brief.track for b in used) >= limit:
                return False
            self.db.execute(
                "INSERT INTO research_briefs VALUES (?,?,?,0)",
                (packet, canonical(asdict(brief)), "assigned"),
            )
            self._event(
                packet,
                "assigned",
                "reserved original family/track capacity",
                asdict(brief),
            )
            return True

    def next_brief(self):
        """Fewest previously reserved attempts; deterministic ties, no performance feedback."""
        used = [
            AssignedBrief(**json.loads(r[0]))
            for r in self.db.execute("SELECT payload FROM research_briefs")
        ]
        counts = {f: sum(b.family == f for b in used) for f, _, _ in self.policy.reservations}
        ordered = sorted(
            self.catalogue,
            key=lambda p: (
                counts.get(self.catalogue[p][0].family, 0),
                self.catalogue[p][0].family,
                self.catalogue[p][0].track,
                p,
            ),
        )
        for packet in ordered:
            if self.reserve(packet):
                return self.catalogue[packet][0]
        return None

    def _latest(self, packet, field):
        for row in self.db.execute(
            "SELECT payload FROM research_events WHERE brief=? ORDER BY seq DESC",
            (packet,),
        ):
            payload = json.loads(row[0])
            if field in payload:
                return payload[field]
        raise ValueError(f"missing {field} evidence")

    def events(self):
        return [
            dict(row) | {"payload": json.loads(row["payload"])}
            for row in self.db.execute("SELECT * FROM research_events ORDER BY seq")
        ]

    def pending_papers(self, source):
        """Read an existing FileSource/PaperlogSource without changing global ingestion."""
        papers = list(source.papers())
        if len({p["key"] for p in papers}) != len(papers):
            raise ValueError("duplicate source packet")
        if any(p["key"] not in self.catalogue for p in papers):
            raise ValueError("source has an unregistered packet; freeze its catalogue first")
        return [p for p in sorted(papers, key=lambda p: p["key"]) if self.state(p["key"]) is None]

    def ground(self, packet, proposal: SourceClaim):
        _, inventory, _ = self.catalogue[packet]
        if isinstance(inventory, IdeaCatalogueEntry):
            raise ValueError("idea catalogue requires the assess_idea route")
        with self.transaction():
            self._require(packet, "assigned")
            if proposal.source_url != inventory.source_url:
                self._event(
                    packet,
                    "unsupported_claim",
                    "source identity changed",
                    asdict(proposal),
                )
                return "unsupported_claim"
            # Agent omissions cannot remove dependencies, improve source access, or expand scope.
            effective = replace(
                proposal,
                full_methods=inventory.full_methods,
                universe=inventory.universe if inventory.universe != "sp500" else proposal.universe,
                required_inputs=tuple(
                    set(proposal.required_inputs) | set(inventory.required_inputs)
                ),
            )
            state, missing = qualify(effective, self.inputs)
            claim_id = self._artifact("source_claim", asdict(effective))
            self._event(
                packet,
                state,
                "source and capability inventory",
                {
                    "claim_id": claim_id,
                    "missing": missing,
                    "agent_declared": asdict(proposal),
                    "effective": asdict(effective),
                },
            )
            return state

    def claim_id(self, packet):
        """Identity to bind into a proposed or frozen MethodSpec."""
        return self._latest(packet, "claim_id")

    def frozen_spec_id(self, packet):
        """Read the grounded specification binding when resuming a construction."""
        return self._latest(packet, "spec_id")

    def freeze_spec(self, packet, spec: MethodSpec):
        with self.transaction():
            self._require(packet, "specification_ready", "repair_requested")
            if spec.claim_id != self._latest(packet, "claim_id"):
                raise ValueError("specification must bind the grounded claim")
            route = self.catalogue[packet][2]
            key = self._artifact("method_spec", asdict(spec))
            state = "specification_frozen" if route != "unavailable" else "constructor_unavailable"
            self._event(
                packet,
                state,
                "existing constructor binding",
                {"spec_id": key, "route": route},
            )
            return key

    def construct(self, packet, verifier):
        """A trusted executable verifier, never a tool-call field named `passed`."""
        with self.transaction():
            self._require(packet, "specification_frozen")
            spec_id = self._latest(packet, "spec_id")
            payload = self.db.execute(
                "SELECT payload FROM research_artifacts WHERE id=?", (spec_id,)
            ).fetchone()[0]
            evidence = verifier(MethodSpec(**json.loads(payload)))
            if not isinstance(evidence, ConstructionEvidence) or evidence.spec_id != spec_id:
                raise ValueError("verifier returned unrelated evidence")
            key = self._artifact("construction_evidence", asdict(evidence))
            state = "construction_verified" if evidence.passed else "construction_failed"
            row = self.db.execute(
                "SELECT repairs FROM research_briefs WHERE id=?", (packet,)
            ).fetchone()
            if not evidence.passed and row[0] < self.policy.max_repairs:
                state = "repair_requested"
                self.db.execute(
                    "UPDATE research_briefs SET repairs=repairs+1 WHERE id=?", (packet,)
                )
            self._event(
                packet,
                state,
                "deterministic construction verifier",
                {"evidence_id": key},
            )
            return state

    def replay_handoff(self, packet):
        with self.transaction():
            self._require(packet, "construction_verified")
            evidence = {
                "spec_id": self._latest(packet, "spec_id"),
                "evidence_id": self._latest(packet, "evidence_id"),
                "generator_policy": self.identity,
                "formal_admission": False,
            }
            self._event(packet, "replay_ready", "register before common replay", evidence)
            return evidence

    def record_failure(self, packet, kind, evidence):
        """Operational/interpretation failures cannot overwrite a data-blocked economic claim."""
        if kind not in ("interpretation_failed", "infrastructure_failed", "budget_exhausted"):
            raise ValueError("unknown non-economic failure reason")
        with self.transaction():
            self._require(packet, "assigned", "specification_ready", "specification_frozen")
            self._event(packet, kind, "research process failure, not an economic result", evidence)

    def validation_handoff(self, packet, calibration):
        """Scope compatibility only; the existing financial funnel still owns admission."""
        self._require(packet, "replay_ready")
        if (
            not isinstance(calibration, dict)
            or calibration.get("generator_policy") != self.identity
        ):
            raise ValueError("financial calibration does not cover this generator/policy")
        return {
            "spec_id": self._latest(packet, "spec_id"),
            "generator_policy": self.identity,
            "existing_financial_funnel_required": True,
            "formal_admission": False,
        }


class RecordsController:
    """A controller's procedure records, without re-binding its research manifest (the
    gate report's read of a reconciliation, `validation.program_gate`)."""

    transaction = ResearchController.transaction

    def __init__(self, path):
        uri = f"file:{Path(path).resolve().as_posix()}?mode=rw"
        self.db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
