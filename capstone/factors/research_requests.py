"""Durable model requests: each send is counted before it goes out; a saved response replays."""

from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace

from capstone.artifacts import canonical, digest
from capstone.faults import maybe_fault
from capstone.journal_db import locked_journal


class ResearchPaused(RuntimeError):
    """The saved study needs credits, credentials, or explicit request resolution."""


class RequestAbandoned(ResearchPaused):
    """An operator abandoned an uncertain request; retain it as an incomplete case."""


def _plain(response):
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(response).items()}
    if isinstance(response, dict):
        return {k: _plain(v) for k, v in response.items()}
    if isinstance(response, list):
        return [_plain(v) for v in response]
    return response


def _response(payload):
    data = json.loads(payload)
    data["content"] = [SimpleNamespace(**block) for block in data.get("content", [])]
    data["usage"] = SimpleNamespace(**data.get("usage", {}))
    return SimpleNamespace(**data)


class RequestJournal:
    """One owner per study, using the shared journal lock and SQLite settings.

    `manifest["call_limits"]` maps each bucket to its frozen allowance. The ceiling is
    caller-supplied; nothing here chooses a model or a spend cap.
    """

    def __init__(self, path, manifest):
        self.manifest = json.loads(canonical(manifest))
        self._limits = MappingProxyType(dict(self.manifest["call_limits"]))
        if not self.limits or any(type(n) is not int or n < 0 for n in self.limits.values()):
            raise ValueError("nonnegative frozen bucket limits required")
        self._manager = locked_journal(path, message="another dispatcher owns this request journal")
        self.db = self._manager.__enter__()
        try:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS request_manifest (payload TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS model_requests
              (id TEXT PRIMARY KEY, logical TEXT NOT NULL, bucket TEXT NOT NULL,
               attempt INTEGER NOT NULL, request TEXT NOT NULL, response TEXT,
               state TEXT NOT NULL, error TEXT);
            CREATE TABLE IF NOT EXISTS request_events
              (seq INTEGER PRIMARY KEY, request_id TEXT NOT NULL, state TEXT NOT NULL,
               detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS request_resolutions
              (request_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS response_receipts
              (request_id TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_request_events_update
              BEFORE UPDATE ON request_events BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_request_events_delete
              BEFORE DELETE ON request_events BEGIN SELECT RAISE(ABORT,'append only'); END;
            """)
            with self.db:
                old = self.db.execute("SELECT payload FROM request_manifest").fetchone()
                if old and old[0] != canonical(self.manifest):
                    raise ValueError("request manifest changed")
                if old is None:
                    self.db.execute(
                        "INSERT INTO request_manifest VALUES (?)",
                        (canonical(self.manifest),),
                    )
                interrupted = self.db.execute(
                    "SELECT id FROM model_requests WHERE state='inflight'"
                ).fetchall()
                for (rid,) in interrupted:
                    self.db.execute(
                        "UPDATE model_requests SET state='uncertain',"
                        "error='interrupted_before_checkpoint' WHERE id=?",
                        (rid,),
                    )
                    self._event(
                        rid, "uncertain", "previous owner exited before a response checkpoint"
                    )
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._manager is not None:
            manager, self._manager = self._manager, None
            manager.__exit__(None, None, None)

    @property
    def limits(self):
        return self._limits

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def count(self, bucket):
        return self.db.execute(
            "SELECT COUNT(*) FROM model_requests WHERE bucket=?", (bucket,)
        ).fetchone()[0]

    def _event(self, request, state, detail=""):
        self.db.execute(
            "INSERT INTO request_events(request_id,state,detail) VALUES (?,?,?)",
            (request, state, detail),
        )

    def _blocked(self):
        return self.db.execute("""SELECT id,state FROM model_requests r
            WHERE state IN ('inflight','uncertain')
              AND NOT EXISTS(SELECT 1 FROM request_resolutions s WHERE s.request_id=r.id)
            LIMIT 1""").fetchone()

    def _response_checked(self, request_id, payload):
        receipt = self.db.execute(
            "SELECT digest FROM response_receipts WHERE request_id=?", (request_id,)
        ).fetchone()
        if receipt is None or receipt[0] != digest(json.loads(payload)):
            raise ValueError("response checkpoint contents changed")
        return _response(payload)

    def resolve(self, request_id, decision, reason):
        """Explicit operator decision; immutable, and a retry consumes capacity again."""
        if decision not in ("retry", "abandon") or not reason.strip():
            raise ValueError("explicit retry/abandon decision and reason required")
        with self.db:
            row = self.db.execute(
                "SELECT state FROM model_requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None or row[0] not in ("inflight", "uncertain"):
                raise ValueError("request is not awaiting resolution")
            self.db.execute(
                "INSERT INTO request_resolutions VALUES (?,?,?)",
                (request_id, decision, reason),
            )
            self._event(request_id, "resolved", decision + ": " + reason)

    def call(self, client, bucket, task, ordinal, kwargs):
        """Send once, or replay the saved response for this (bucket, task, ordinal)."""
        if getattr(client, "max_retries", 0) != 0:
            raise ValueError("disable hidden SDK retries before using the journal")
        if bucket not in self.limits:
            raise ValueError("unknown budget bucket")
        logical = digest({"bucket": bucket, "task": task, "ordinal": ordinal})
        request = canonical(kwargs)
        rows = self.db.execute(
            "SELECT id,request,response,state FROM model_requests WHERE logical=? ORDER BY attempt",
            (logical,),
        ).fetchall()
        if rows and any(row[1] != request for row in rows):
            raise ValueError("checkpoint request inputs changed")
        if rows and rows[-1][3] == "completed":
            return self._response_checked(rows[-1][0], rows[-1][2])
        if rows:
            resolution = self.db.execute(
                "SELECT decision FROM request_resolutions WHERE request_id=?",
                (rows[-1][0],),
            ).fetchone()
            if resolution is not None and resolution[0] == "abandon":
                raise RequestAbandoned("request explicitly abandoned: " + rows[-1][0])
            if resolution is None:
                raise ResearchPaused("request needs explicit resolution: " + rows[-1][0])
        blocked = self._blocked()
        if blocked:
            raise ResearchPaused(f"study paused: {blocked[0]} ({blocked[1]})")
        if self.count(bucket) >= self.limits[bucket]:
            from capstone.pipeline.budget import CallBudgetExceeded

            raise CallBudgetExceeded(f"frozen {bucket} allowance exhausted")
        attempt = len(rows)
        rid = digest({"logical": logical, "attempt": attempt})
        with self.db:
            self.db.execute(
                "INSERT INTO model_requests VALUES (?,?,?,?,?,NULL,'inflight',NULL)",
                (rid, logical, bucket, attempt, request),
            )
            self._event(rid, "reserved", "outbound request counted before sending")
        maybe_fault("runner:after_reserve")  # reserved, response not saved
        try:
            response = client.messages.create(**kwargs)
            payload = canonical(_plain(response))
            with self.db:
                self.db.execute(
                    "INSERT INTO response_receipts VALUES (?,?)",
                    (rid, digest(json.loads(payload))),
                )
                self.db.execute(
                    "UPDATE model_requests SET response=?,state='completed' WHERE id=?",
                    (payload, rid),
                )
                self._event(rid, "completed", "raw response saved before parsing")
        except BaseException as exc:
            detail = f"uncertain_send:{type(exc).__name__}"
            with self.db:
                self.db.execute(
                    "UPDATE model_requests SET state='uncertain',error=? WHERE id=?",
                    (detail, rid),
                )
                self._event(rid, "uncertain", detail)
            raise ResearchPaused(f"saved request {rid}: uncertain_send") from None
        return _response(payload)

    def report(self):
        rows = self.db.execute(
            "SELECT id,bucket,attempt,response,state,error FROM model_requests ORDER BY rowid"
        ).fetchall()
        result = []
        for rid, bucket, attempt, response, state, error in rows:
            if response is not None and state == "completed":
                self._response_checked(rid, response)
            result.append(
                {
                    "id": rid,
                    "bucket": bucket,
                    "attempt": attempt,
                    "state": state,
                    "error": error,
                    "usage": {} if response is None else json.loads(response).get("usage", {}),
                }
            )
        return {
            "limits": dict(self.limits),
            "requests": result,
            "outbound_reservations": len(result),
            "blocked": self._blocked() is not None,
            "events": [
                {"sequence": seq, "request_id": rid, "state": state, "detail": detail}
                for seq, rid, state, detail in self.db.execute(
                    "SELECT * FROM request_events ORDER BY seq"
                )
            ],
        }
