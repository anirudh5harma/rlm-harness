"""Tests for the `harness trace show` and `harness trace replay`
commands (Phase E.2 + E.3).

The pivot plan's Phase E gate:

    "Given a run id, a developer can see what context the model
     saw, what action it chose, what happened, and why the final
     answer was produced."

These tests assert the developer-facing commands work
end-to-end against a recorded run.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlm_harness.actions import CompleteTaskAction, CompletionStatus
from rlm_harness.kernel.events import CompletionEvent
from rlm_harness.tracing import TraceStore


def _write_sample_run(trace_db: Path) -> str:
    """Write a run with the events a developer would want to
    inspect: run_started, turn_started, model_completion,
    turn_finished, completion. Returns the run id.
    """
    store = TraceStore(trace_db)
    run_id = store.start_run("summarize the project", str(trace_db.parent), thread_id="t")
    store.event(
        run_id,
        "run_started",
        {"task": "summarize the project", "max_turns": 5},
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
        {"content": "stub response", "model": "test"},
        node="rlm",
    )
    store.event(
        run_id,
        "turn_finished",
        {"turn_index": 0, "status": "done", "iterations": 1},
        node="supervisor",
    )
    store.record_typed_event(
        CompletionEvent.from_action(
            run_id=run_id,
            sequence=store.next_sequence(run_id),
            node="supervisor",
            action=CompleteTaskAction(
                summary="Project is a CLI for tests.",
                status=CompletionStatus.SUCCESS,
            ),
        )
    )
    store.finish_run(run_id, "done")
    return run_id


class TraceCliTests(unittest.TestCase):
    def test_trace_show_human_readable_includes_final_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = Path(temp_dir) / "trace.db"
            run_id = _write_sample_run(trace_db)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlm_harness.cli",
                    "trace",
                    "--trace-db",
                    str(trace_db),
                    "show",
                    run_id,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Final answer:", result.stdout)
            self.assertIn("Project is a CLI for tests.", result.stdout)
            self.assertIn("Status:    done", result.stdout)

    def test_trace_show_json_returns_timeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = Path(temp_dir) / "trace.db"
            run_id = _write_sample_run(trace_db)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlm_harness.cli",
                    "trace",
                    "--trace-db",
                    str(trace_db),
                    "show",
                    run_id,
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["run_id"], run_id)
            self.assertEqual(payload["status"], "done")
            self.assertEqual(payload["final_answer"], "Project is a CLI for tests.")
            self.assertGreaterEqual(payload["event_count"], 5)

    def test_trace_replay_writes_jsonl_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = Path(temp_dir) / "trace.db"
            run_id = _write_sample_run(trace_db)
            out_path = Path(temp_dir) / "tree.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlm_harness.cli",
                    "trace",
                    "--trace-db",
                    str(trace_db),
                    "replay",
                    run_id,
                    "--out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(out_path.exists())
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 5)
            kinds = [json.loads(line)["kind"] for line in lines]
            self.assertIn("run_started", kinds)
            self.assertIn("turn_started", kinds)
            self.assertIn("turn_finished", kinds)
            self.assertIn("completion", kinds)

    def test_trace_replay_round_trip(self):
        """`replay` writes a JSONL tree that re-reads losslessly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = Path(temp_dir) / "trace.db"
            run_id = _write_sample_run(trace_db)
            out_path = Path(temp_dir) / "tree.jsonl"
            store = TraceStore(trace_db)
            store.write_jsonl_tree(run_id, out_path)
            readback = store.read_jsonl_tree(out_path)
            self.assertEqual(len(readback), 5)
            # The first event is `run_started`; the last is the
            # completion.
            self.assertEqual(readback[0]["kind"], "run_started")
            self.assertEqual(readback[-1]["kind"], "completion")


if __name__ == "__main__":
    unittest.main()
