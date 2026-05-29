from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from rlm_harness.kernel.events import AnyRunEvent, parse_event


class TraceStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  thread_id TEXT NOT NULL,
                  task TEXT NOT NULL,
                  workspace TEXT NOT NULL,
                  status TEXT NOT NULL,
                  started_at INTEGER NOT NULL,
                  finished_at INTEGER
                );

                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  ts INTEGER NOT NULL,
                  kind TEXT NOT NULL,
                  node TEXT,
                  payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS events_run_id ON events(run_id, id);
                """
            )

    def start_run(self, task: str, workspace: str, thread_id: Optional[str] = None) -> str:
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (run_id, thread_id, task, workspace, status, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, thread_id or run_id, task, workspace, "running", int(time.time())),
            )
        return run_id

    def finish_run(self, run_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, int(time.time()), run_id),
            )

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, thread_id, task, workspace, status, started_at, finished_at
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def latest_run_for_thread(self, thread_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, thread_id, task, workspace, status, started_at, finished_at
                FROM runs
                WHERE thread_id = ?
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def list_runs(
        self,
        limit: int = 20,
        thread_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if thread_id:
            query = """
                SELECT run_id, thread_id, task, workspace, status, started_at, finished_at
                FROM runs
                WHERE thread_id = ?
                ORDER BY started_at DESC, rowid DESC
                LIMIT ?
            """
            params = (thread_id, limit)
        else:
            query = """
                SELECT run_id, thread_id, task, workspace, status, started_at, finished_at
                FROM runs
                ORDER BY started_at DESC, rowid DESC
                LIMIT ?
            """
            params = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def event(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        node: Optional[str] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events (run_id, ts, kind, node, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, int(time.time()), kind, node, json.dumps(payload, sort_keys=True)),
            )

    def next_sequence(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["count"]) + 1

    def record_typed_event(self, event: AnyRunEvent) -> None:
        self.event(
            event.run_id,
            event.kind,
            event.model_dump(mode="json"),
            node=event.node,
        )

    def iter_events(self, run_id: str) -> Iterable[sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, ts, kind, node, payload_json FROM events WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return rows

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": int(row["id"]),
                "ts": int(row["ts"]),
                "kind": str(row["kind"]),
                "node": row["node"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in self.iter_events(run_id)
        ]

    def typed_events(self, run_id: str) -> list[AnyRunEvent]:
        typed = []
        for event in self.events(run_id):
            payload = event["payload"]
            if not isinstance(payload, dict):
                continue
            if payload.get("kind") != event["kind"] or "event_id" not in payload:
                continue
            try:
                typed.append(parse_event(payload))
            except ValueError:
                continue
        return typed

    def run_summary(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        events = self.events(run_id)
        final_answer = None
        for event in reversed(events):
            if event["kind"] == "completion":
                final_answer = event["payload"].get("final_answer")
                break
            if event["kind"] == "final":
                final_answer = event["payload"].get("final_answer")
                break
        return {**run, "event_count": len(events), "final_answer": final_answer}

    def render_report(self, run_id: str) -> str:
        if self.get_run(run_id) is None:
            raise KeyError(f"unknown run_id: {run_id}")
        lines = [f"Trace report: {run_id}"]
        for row in self.iter_events(run_id):
            payload = json.loads(row["payload_json"])
            node = row["node"] or "-"
            lines.append(
                "[{}] {} {}".format(
                    row["kind"],
                    node,
                    json.dumps(payload, sort_keys=True),
                )
            )
        return "\n".join(lines)
