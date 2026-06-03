"""Regression test: the Phase E migration is correct on a
pre-Phase-E trace db.

Real-world bug report (post-Phase-H release): a user on a
fresh install whose trace db was created by an earlier
release hit `sqlite3.OperationalError: no such column:
parent_id` on the first run after upgrading.

Root cause: `_init_schema` ran
`CREATE INDEX events_parent ON events(parent_id)` in the
initial `executescript` *before* the `ALTER TABLE` that adds
the column. The `CREATE TABLE IF NOT EXISTS` was a no-op on
the existing table (no `parent_id` column), so the subsequent
`CREATE INDEX` failed.

This test pre-creates a pre-Phase-E trace db, opens a new
`TraceStore` against it, and asserts the migration runs
without error and the new column is present.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from rlm_harness.tracing import TraceStore

PRE_PHASE_E_SCHEMA = """
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


class TraceMigrationTests(unittest.TestCase):
    def test_init_schema_migrates_pre_phase_e_db(self):
        """A pre-Phase-E trace db (no `parent_id` column) opens
        cleanly. The migration adds the column and the index.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = Path(temp_dir) / "trace.db"
            # Pre-create the legacy schema.
            with sqlite3.connect(str(trace_db)) as conn:
                conn.executescript(PRE_PHASE_E_SCHEMA)
                conn.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        "r-legacy",
                        "t",
                        "old task",
                        "/tmp",
                        "done",
                        0,
                        0,
                    ),
                )
                conn.execute(
                    "INSERT INTO events (run_id, ts, kind, node, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        "r-legacy",
                        0,
                        "run_started",
                        "supervisor",
                        "{}",
                    ),
                )
                conn.commit()

            # Now open with the new TraceStore; the migration
            # should run without error.
            store = TraceStore(trace_db)
            # The old event is still readable.
            events = store.events("r-legacy")
            self.assertEqual(len(events), 1)

            # The new column is present and accessible.
            with sqlite3.connect(str(trace_db)) as conn:
                row = conn.execute(
                    "SELECT parent_id FROM events WHERE run_id = ?",
                    ("r-legacy",),
                ).fetchone()
            self.assertIsNone(row[0])

            # The new index is present.
            with sqlite3.connect(str(trace_db)) as conn:
                indexes = {
                    row[1]  # PRAGMA index_list returns (seq, name, unique, origin, partial)
                    for row in conn.execute(
                        "PRAGMA index_list(events)"
                    ).fetchall()
                }
            self.assertIn("events_parent", indexes)

    def test_init_schema_idempotent_on_already_migrated_db(self):
        """A trace db that already has `parent_id` and the index
        opens cleanly. Re-running the migration is a no-op.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = Path(temp_dir) / "trace.db"
            # First open: creates the new schema.
            TraceStore(trace_db)
            # Second open: must not fail.
            store = TraceStore(trace_db)
            self.assertTrue(store.get_run("anything") is None)


if __name__ == "__main__":
    unittest.main()
