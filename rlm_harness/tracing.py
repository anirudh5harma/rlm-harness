from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional


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

    def iter_events(self, run_id: str) -> Iterable[sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, ts, kind, node, payload_json FROM events WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return rows

    def render_report(self, run_id: str) -> str:
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
