import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import (
    GraphRuntimeConfig,
    Nodes,
    parse_numbered_plan,
    parse_python_action,
)
from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import DockerREPL, SandboxConfig
from rlm_harness.tracing import TraceStore
from rlm_harness.types import Completion, HarnessState

IMAGE = "rlm-harness-sandbox:test"


def docker_available():
    completed = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


class BadThenGoodActionClient(LMClient):
    def __init__(self):
        super().__init__(provider="stub")
        self.action_calls = 0

    def complete(self, messages, max_tokens=512, temperature=0.2):
        user_text = ""
        for message in reversed(list(messages)):
            if message.role == "user":
                user_text = message.content
                break

        if "Return a concise numbered plan" in user_text:
            content = "1. Inspect workspace\n2. Print file names"
        elif "Return only valid JSON" in user_text:
            self.action_calls += 1
            if self.action_calls == 1:
                content = "not json"
            else:
                content = '{"type":"python","code":"print(123)"}'
        elif "Decide whether the task is complete" in user_text:
            content = "done"
        else:
            content = "done"

        return Completion(
            content=content,
            model="bad-then-good",
            provider="test",
            latency_ms=0,
        )


class GraphTests(unittest.TestCase):
    def test_parse_numbered_plan(self):
        self.assertEqual(parse_numbered_plan("1. Inspect\n2. Act"), ["Inspect", "Act"])

    def test_parse_python_action(self):
        code = parse_python_action('{"type": "python", "code": "print(2 + 2)"}')
        self.assertEqual(code, "print(2 + 2)")

    def test_stub_graph_reaches_done(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("test task", temp_dir)
            state = HarnessState(
                task="test task",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            graph = build_graph(Nodes(LMClient(provider="stub"), traces), backend="simple")
            final_state = graph.invoke(state)
            self.assertEqual(final_state.status, "done")
            self.assertTrue(final_state.final_answer)
            self.assertIn("Trace report", traces.render_report(run_id))

    def test_langgraph_backend_optional_dependency_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("hello", temp_dir)
            state = HarnessState(
                task="hello",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            nodes = Nodes(LMClient(provider="stub"), traces)

            if importlib.util.find_spec("langgraph") is None:
                with self.assertRaisesRegex(RuntimeError, "langgraph is not installed"):
                    build_graph(nodes, backend="langgraph")
                auto_graph = build_graph(nodes, backend="auto")
                self.assertTrue(hasattr(auto_graph, "invoke"))
                return

            graph = build_graph(nodes, backend="langgraph")
            final_state = graph.invoke(state)
            self.assertEqual(final_state.status, "done")

    @unittest.skipUnless(docker_available(), "Docker daemon is not available")
    def test_stub_graph_lists_workspace_through_sandbox(self):
        DockerREPL.build_image(image=IMAGE)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "alpha.txt").write_text("a", encoding="utf-8")
            (workspace / "beta.txt").write_text("b", encoding="utf-8")
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("List files in workspace", str(workspace))
            state = HarnessState(
                task="List files in workspace",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                sandbox_config=SandboxConfig(
                    image=IMAGE,
                    workspace=workspace,
                    default_timeout_s=5,
                ),
            )
            graph = build_graph(Nodes(LMClient(provider="stub"), traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertIn("alpha.txt", final_state.final_answer)
        self.assertIn("beta.txt", final_state.final_answer)
        self.assertIn("sandbox_execution", report)
        self.assertIn("alpha.txt", report)

    @unittest.skipUnless(docker_available(), "Docker daemon is not available")
    def test_act_retries_malformed_action_json(self):
        DockerREPL.build_image(image=IMAGE)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("Print a number", str(workspace))
            state = HarnessState(
                task="Print a number",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            client = BadThenGoodActionClient()
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                sandbox_config=SandboxConfig(
                    image=IMAGE,
                    workspace=workspace,
                    default_timeout_s=5,
                ),
                max_action_retries=1,
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 2)
        self.assertIn("123", final_state.final_answer)
        self.assertIn("action_parse_error", report)


if __name__ == "__main__":
    unittest.main()
