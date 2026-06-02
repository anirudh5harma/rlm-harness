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
                  payload_json TEXT NOT NULL,
                  parent_id INTEGER
                );

                CREATE INDEX IF NOT EXISTS events_run_id ON events(run_id, id);
                CREATE INDEX IF NOT EXISTS events_parent ON events(parent_id);
                """
            )
            # Phase E: add the `parent_id` column to an existing
            # `events` table that was created before Phase E. The
            # `CREATE TABLE IF NOT EXISTS` above is a no-op when
            # the table already exists, so the new column has to
            # be added via ALTER. SQLite's `ALTER TABLE ADD
            # COLUMN` is idempotent only via the duplicate-column
            # check we do here.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            if "parent_id" not in columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN parent_id INTEGER"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS events_parent ON events(parent_id)"
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
        parent_id: Optional[int] = None,
    ) -> int:
        """Append an event. Returns the new event id.

        `parent_id` links this event into the run's tree. The
        supervisor uses it to build a turn's chain
        (turn_started → iteration_started → observation →
        turn_finished). A `None` parent is the root of a chain
        (typically the run's `run_started` event).
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (run_id, ts, kind, node, payload_json, parent_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(time.time()),
                    kind,
                    node,
                    json.dumps(payload, sort_keys=True),
                    parent_id,
                ),
            )
            return int(cursor.lastrowid)

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
                "SELECT id, ts, kind, node, payload_json, parent_id "
                "FROM events WHERE run_id = ? ORDER BY id",
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
                "parent_id": (
                    int(row["parent_id"])
                    if row["parent_id"] is not None
                    else None
                ),
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

    def write_jsonl_tree(
        self, run_id: str, path: Path
    ) -> int:
        """Write the run's events as a JSONL tree to `path`.

        One event per line, in offset order. The `parent_id`
        field on each row links the events into a tree. The
        output is a portable, line-delimited snapshot of the
        run; replay tools read it back the same way.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, ts, kind, node, payload_json, parent_id
                FROM events
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = json.loads(row["payload_json"])
                record = {
                    "id": int(row["id"]),
                    "ts": int(row["ts"]),
                    "kind": str(row["kind"]),
                    "node": row["node"],
                    "parent_id": (
                        int(row["parent_id"])
                        if row["parent_id"] is not None
                        else None
                    ),
                    "payload": payload,
                }
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
        return len(rows)

    def read_jsonl_tree(self, path: Path) -> list[dict[str, Any]]:
        """Read a JSONL tree back into a list of event dicts.

        The reverse of `write_jsonl_tree`. Used by replay.
        """
        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return events

    def timeline_summary(self, run_id: str) -> dict[str, Any]:
        """Compact timeline for `harness trace show <run-id>`.

        Returns the run summary plus the ordered sequence of
        events, in offset order, with the parent_id linkage
        preserved. This is what a developer sees when they ask
        the harness "what happened in this run?".
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        events = self.events(run_id)
        # Surface the final answer; mirrors `run_summary` but
        # exposed as a top-level key for the CLI.
        final_answer = None
        for event in reversed(events):
            if event["kind"] == "completion":
                final_answer = event["payload"].get("final_answer")
                break
            if event["kind"] == "final":
                final_answer = event["payload"].get("final_answer")
                break
        return {
            "run_id": run_id,
            "thread_id": run["thread_id"],
            "task": run["task"],
            "workspace": run["workspace"],
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "event_count": len(events),
            "final_answer": final_answer,
            "events": events,
        }

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
