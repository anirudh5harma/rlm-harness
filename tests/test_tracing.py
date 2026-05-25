import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from rlm_harness import cli
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


class TraceStoreTests(unittest.TestCase):
    def test_records_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("task", temp_dir)
            traces.event(run_id, "kind", {"value": 1}, node="node")
            report = traces.render_report(run_id)
            self.assertIn("Trace report", report)
            self.assertIn('"value": 1', report)

    def test_lists_runs_and_summarizes_final_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("task", temp_dir, thread_id="thread-a")
            traces.event(run_id, "final", {"final_answer": "done"}, node="done")
            traces.finish_run(run_id, "done")

            runs = traces.list_runs(thread_id="thread-a")
            summary = traces.run_summary(run_id)

        self.assertEqual(runs[0]["run_id"], run_id)
        self.assertEqual(summary["final_answer"], "done")
        self.assertEqual(summary["event_count"], 1)

    def test_cli_trace_commands_and_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            memory_db = str(Path(temp_dir) / "memory.db")

            with contextlib.redirect_stdout(io.StringIO()):
                first = cli.main(
                    [
                        "run",
                        "remember resume context",
                        "--no-sandbox",
                        "--trace-db",
                        trace_db,
                        "--memory-db",
                        memory_db,
                        "--thread-id",
                        "thread-resume-cli",
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--quiet",
                    ]
                )
                resumed = cli.main(
                    [
                        "resume",
                        "thread-resume-cli",
                        "continue resume context",
                        "--no-sandbox",
                        "--trace-db",
                        trace_db,
                        "--memory-db",
                        memory_db,
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--quiet",
                    ]
                )
            traces = TraceStore(Path(trace_db))
            runs = traces.list_runs(thread_id="thread-resume-cli")

        self.assertEqual(first, 0)
        self.assertEqual(resumed, 0)
        self.assertEqual(len(runs), 2)

    def test_cli_run_json_and_trace_report_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "run",
                        "json output task",
                        "--no-sandbox",
                        "--no-memory",
                        "--trace-db",
                        trace_db,
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            output = stdout.getvalue()

            report_stdout = io.StringIO()
            with contextlib.redirect_stdout(report_stdout):
                report_exit = cli.main(
                    [
                        "trace",
                        "--trace-db",
                        trace_db,
                        "report",
                        payload["run_id"],
                        "--json",
                    ]
                )
            report = json.loads(report_stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report_exit, 0)
        self.assertTrue(output.startswith("{\n"))
        self.assertEqual(payload["status"], "done")
        self.assertIn("Stub response", payload["response"])
        self.assertEqual(report["run_id"], payload["run_id"])

    def test_cli_run_text_output_is_only_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "run",
                        "text output task",
                        "--no-sandbox",
                        "--no-memory",
                        "--trace-db",
                        trace_db,
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                    ]
                )
            output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Stub response", output)
        self.assertNotIn("Trace report", output)
        self.assertNotIn("run_started", output)

    def test_json_payload_uses_final_state_answer_for_error_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            traces = TraceStore(Path(trace_db))
            run_id = traces.start_run("bad tool call", temp_dir)
            traces.finish_run(run_id, "error")
            state = HarnessState(
                task="bad tool call",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                status="error",
                final_answer="ToolError: path must be a non-empty string",
            )

            payload = cli.run_output_payload(
                Namespace(trace_db=trace_db),
                traces,
                run_id,
                state,
            )

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["response"], "ToolError: path must be a non-empty string")

    def test_cli_accepts_task_without_run_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "default task alias",
                        "--no-sandbox",
                        "--no-memory",
                        "--trace-db",
                        trace_db,
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["task"], "default task alias")

    def test_cli_help_hides_internal_commands_and_options(self):
        help_stdout = io.StringIO()
        with contextlib.redirect_stdout(help_stdout):
            exit_code = cli.main([])
        help_text = help_stdout.getvalue()

        run_help = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(run_help):
            cli.parser().parse_args(["run", "--help"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("run", help_text)
        self.assertIn("resume", help_text)
        self.assertNotIn("benchmark-model", help_text)
        self.assertNotIn("sandbox", help_text)
        self.assertIn("--model", run_help.getvalue())
        self.assertNotIn("--base-url", run_help.getvalue())
        self.assertNotIn("--checkpoint-db", run_help.getvalue())

    @unittest.skipIf(not module_available("langgraph"), "langgraph is not installed")
    @unittest.skipIf(
        not module_available("langgraph.checkpoint.sqlite"),
        "langgraph SQLite checkpointer is not installed",
    )
    def test_cli_langgraph_stream_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            checkpoint_db = Path(temp_dir) / "checkpoints.db"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "run",
                        "stream checkpoint task",
                        "--no-sandbox",
                        "--no-memory",
                        "--graph-backend",
                        "langgraph",
                        "--trace-db",
                        trace_db,
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--checkpoint-db",
                        str(checkpoint_db),
                        "--stream",
                        "--quiet",
                    ]
                )
            output = stdout.getvalue()
            checkpoint_exists = checkpoint_db.exists()

        self.assertEqual(exit_code, 0)
        self.assertIn("graph_update", output)
        self.assertTrue(checkpoint_exists)


if __name__ == "__main__":
    unittest.main()
