"""Tests for the JSONL session tree + replay (Phase E).

The pivot plan's Phase E gate:

    "Given a run id, a developer can see what context the model
     saw, what action it chose, what happened, and why the final
     answer was produced."

The trace store is the spine of debug and improvement. This
test exercises:

* The events table carries a `parent_id` so a turn's events
  form a tree (turn_started → iteration → observation → ...
  → turn_finished).
* A JSONL tree file is written to disk, one event per line, in
  offset order. Re-reading the file reproduces the same
  sequence.
* `harness trace show <run-id>` produces a compact timeline
  that surfaces context, action, outcome, and final answer.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness.tracing import TraceStore


class TraceStoreTreeTests(unittest.TestCase):
    def test_events_table_has_parent_id_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(Path(temp_dir) / "trace.db")
            run_id = store.start_run("x", temp_dir, thread_id="t")
            event_id = store.event(
                run_id,
                "turn_started",
                {"turn_index": 0},
                node="supervisor",
            )
            # The event row has a `parent_id` column. The first
            # event in a run has no parent (`parent_id IS NULL`).
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT parent_id FROM events WHERE id = ?",
                    (event_id,),
                ).fetchone()
            self.assertIsNone(row["parent_id"])

    def test_event_with_explicit_parent_links_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(Path(temp_dir) / "trace.db")
            run_id = store.start_run("x", temp_dir, thread_id="t")
            parent_id = store.event(
                run_id, "turn_started", {"turn_index": 0}, node="supervisor"
            )
            child_id = store.event(
                run_id,
                "iteration_started",
                {"iteration": 1},
                node="rlm",
                parent_id=parent_id,
            )
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT parent_id FROM events WHERE id = ?", (child_id,)
                ).fetchone()
            self.assertEqual(row["parent_id"], parent_id)

    def test_jsonl_tree_round_trips(self):
        """The JSONL tree file is a portable, line-delimited
        representation of the run. Re-reading it reproduces the
        same sequence of events.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(Path(temp_dir) / "trace.db")
            run_id = store.start_run("x", temp_dir, thread_id="t")
            store.event(run_id, "run_started", {"task": "x"}, node="supervisor")
            store.event(run_id, "turn_started", {"turn_index": 0}, node="supervisor")
            store.event(
                run_id,
                "iteration_started",
                {"iteration": 1},
                node="rlm",
                parent_id=2,
            )
            store.event(run_id, "turn_finished", {"turn_index": 0}, node="supervisor")

            # Write the JSONL tree.
            tree_path = Path(temp_dir) / "tree.jsonl"
            store.write_jsonl_tree(run_id, tree_path)
            self.assertTrue(tree_path.exists())

            # Read it back and confirm one event per line, in
            # offset order, with the parent_id linkage intact.
            lines = tree_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            kinds = [json.loads(line)["kind"] for line in lines]
            self.assertEqual(
                kinds,
                [
                    "run_started",
                    "turn_started",
                    "iteration_started",
                    "turn_finished",
                ],
            )
            # The third event has parent_id == 2 (the turn_started).
            iteration = json.loads(lines[2])
            self.assertEqual(iteration["parent_id"], 2)

    def test_timeline_summary_surfaces_key_phases(self):
        """`harness trace show` returns a compact dict with the
        run summary, ordered key events, and the final answer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(Path(temp_dir) / "trace.db")
            run_id = store.start_run("summarize", temp_dir, thread_id="t")
            store.event(
                run_id,
                "run_started",
                {"task": "summarize", "max_turns": 5},
                node="supervisor",
            )
            store.event(
                run_id,
                "turn_started",
                {"turn_index": 0, "query": "summarize"},
                node="supervisor",
            )
            store.event(
                run_id,
                "model_completion",
                {"content": "ok", "model": "test"},
                node="rlm",
            )
            store.event(
                run_id,
                "turn_finished",
                {
                    "turn_index": 0,
                    "status": "done",
                    "iterations": 1,
                    "subcalls": 0,
                },
                node="supervisor",
            )
            store.record_typed_event(
                _build_completion_event(run_id, "Project summary.", store)
            )
            store.finish_run(run_id, "done")

            timeline = store.timeline_summary(run_id)
            self.assertEqual(timeline["run_id"], run_id)
            self.assertEqual(timeline["status"], "done")
            self.assertEqual(timeline["final_answer"], "Project summary.")
            kinds = [event["kind"] for event in timeline["events"]]
            self.assertIn("run_started", kinds)
            self.assertIn("turn_started", kinds)
            self.assertIn("turn_finished", kinds)


def _build_completion_event(run_id: str, summary: str, store: TraceStore):
    from rlm_harness.actions import CompleteTaskAction, CompletionStatus
    from rlm_harness.kernel.events import CompletionEvent

    return CompletionEvent.from_action(
        run_id=run_id,
        sequence=store.next_sequence(run_id),
        node="supervisor",
        action=CompleteTaskAction(
            summary=summary, status=CompletionStatus.SUCCESS
        ),
    )


if __name__ == "__main__":
    unittest.main()
