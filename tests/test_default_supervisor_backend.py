"""Tests for the default-supervisor gate.

The pivot plan's Phase A.5 contract is that the supervisor is the
default control plane: a default ``harness`` invocation must route
through the supervisor, not the legacy simple/langgraph graph. The
legacy paths are preserved for explicit ``--graph-backend
simple|langgraph`` and for older configs that pin ``auto`` (which is
treated as an alias for ``supervisor``).

This test file pins three guarantees:

* ``--graph-backend`` defaults to ``supervisor``.
* A default invocation that fails the legacy path's stub ``ask``
  smoke test (a stub repl block that calls ``project_summary()``)
  exits cleanly with ``status=done`` because the local REPL now
  has the workspace tool surface.
* A default invocation records exactly one row in the ``runs``
  table per run (no double ``start_run`` between ``run_task`` and
  ``run_supervisor_graph``).
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness import cli
from rlm_harness.tracing import TraceStore


class DefaultSupervisorTests(unittest.TestCase):
    def test_default_graph_backend_is_supervisor(self):
        """The CLI parser must default ``--graph-backend`` to the
        supervisor on every task subcommand (``run``, ``ask``,
        ``work``, ``plan``). The legacy ``auto`` value is
        treated as an alias for the same path so older pinned
        scripts keep working.
        """
        from rlm_harness.cli import parser as build_parser

        cli_parser = build_parser()
        # The task subcommands are nested under the root
        # subparsers. Walk the action tree to find each
        # subparser action and inspect its choices.
        for action in cli_parser._actions:
            if not hasattr(action, "choices") or not isinstance(
                action.choices, dict
            ):
                continue
            for name, sub in action.choices.items():
                if name not in {"run", "ask", "work", "plan"}:
                    continue
                for sub_action in sub._actions:
                    if "--graph-backend" in (sub_action.option_strings or []):
                        self.assertEqual(
                            sub_action.default,
                            "supervisor",
                            f"--graph-backend default on `{name}` must be supervisor",
                        )
                        self.assertIn("supervisor", sub_action.choices)
                        self.assertIn("auto", sub_action.choices)
                        break
                else:
                    self.fail(f"--graph-backend not found on `{name}` subcommand")

    def test_default_run_dispatches_to_supervisor_and_exits_done(self):
        """A default ``harness <prompt>`` invocation must exit 0
        with status=done, and the answer must come from the
        supervisor's local REPL — which now exposes the
        workspace tool surface (project_summary, read_file,
        etc.) so a stub repl block that calls
        ``project_summary()`` resolves cleanly.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Demo\n\nA tiny demo project for the default-supervisor gate.\n",
                encoding="utf-8",
            )
            (workspace / "pyproject.toml").write_text(
                '[project]\nname = "demo"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "ask",
                        "what is this project",
                        "--workspace",
                        str(workspace),
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--no-memory",
                        "--no-sandbox",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "done")
        # The stub repl block for a project-summary prompt calls
        # ``project_summary()``; the supervisor's local REPL
        # surfaces a "Project Summary" answer that includes a
        # "Verification" line.
        self.assertIn("Project Summary", payload["final_answer"])
        self.assertIn("Verification", payload["final_answer"])

    def test_default_run_records_exactly_one_runs_row(self):
        """A regression test for the previous bug where
        ``run_task`` and ``run_supervisor_graph`` each called
        ``traces.start_run``, producing two ``runs`` rows for
        one invocation. The supervisor now accepts the
        ``run_id`` the CLI already created, so the trace has
        exactly one row per invocation.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_db = str(Path(temp_dir) / "traces.db")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "run",
                        "single runs row",
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
            runs = TraceStore(Path(trace_db)).list_runs(limit=10)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], payload["run_id"])

    def test_auto_graph_backend_routes_to_supervisor(self):
        """Pinning ``--graph-backend auto`` (the legacy default)
        must route to the supervisor. Older pinned scripts that
        pass ``--graph-backend auto`` explicitly must get the
        new control plane, not the legacy langgraph path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Demo\n\nA tiny demo project for the auto-alias gate.\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "ask",
                        "what is this project",
                        "--workspace",
                        str(workspace),
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--no-memory",
                        "--no-sandbox",
                        "--graph-backend",
                        "auto",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "done")

    def test_default_run_surfaces_provider_503_as_clean_error(self):
        """A regression test for the user-reported bug where a
        default ``harness ask <prompt>`` invocation against a
        provider returning HTTP 503 surfaced the
        ``__rlm_stream_error__:...`` prefix in the final
        answer. With the streaming retry path in place, the
        user must see a clean error (the prefix must NOT leak
        into the final answer).
        """
        from unittest.mock import patch

        from rlm_harness.model_client import LMClient
        from rlm_harness.types import TokenEvent

        class AlwaysFailingClient(LMClient):
            def __init__(self):
                super().__init__(
                    provider="stub",
                    model="stub",
                    max_stream_retries=2,
                    stream_retry_base_delay_s=0.0,
                )

            def stream(self, messages, max_tokens=512, temperature=0.2):
                yield TokenEvent(
                    type="start", model=self.model, provider=self.provider
                )
                yield TokenEvent(
                    type="error",
                    error="HTTP 503 Service Unavailable (after 2 attempts)",
                    provider=self.provider,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Demo\n\nTiny project.\n", encoding="utf-8"
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), patch(
                "rlm_harness.task_runtime.build_client", lambda args: AlwaysFailingClient()
            ):
                exit_code = cli.main(
                    [
                        "ask",
                        "hi",
                        "--workspace",
                        str(workspace),
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--no-memory",
                        "--no-sandbox",
                        "--json",
                    ]
                )
            output = stdout.getvalue()
            stderr_output = stderr.getvalue()

        self.assertEqual(exit_code, 1)
        # The legacy bug was that the final answer carried the
        # ``__rlm_stream_error__:...`` prefix. With the fix in
        # place, the error is surfaced through the CLI's normal
        # ``Error:`` channel instead.
        self.assertNotIn("__rlm_stream_error__", output)
        self.assertNotIn("__rlm_stream_error__", stderr_output)
        # The error message must reach the user via the JSON
        # ``error`` field on stdout.
        payload = json.loads(output)
        self.assertEqual(payload["status"], "error")
        self.assertIn("HTTP 503", payload["error"])


if __name__ == "__main__":
    unittest.main()
