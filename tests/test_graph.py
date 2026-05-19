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
from rlm_harness.memory import Memory, MemoryPagingConfig
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


class CapturingClient(LMClient):
    def __init__(self):
        super().__init__(provider="stub")
        self.plan_prompt = ""

    def complete(self, messages, max_tokens=512, temperature=0.2):
        user_text = ""
        for message in reversed(list(messages)):
            if message.role == "user":
                user_text = message.content
                break
        if "Return a concise numbered plan" in user_text:
            self.plan_prompt = user_text
        return super().complete(messages, max_tokens=max_tokens, temperature=temperature)


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

    def test_memory_paging_writes_archival_summary_and_bounds_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            memory = Memory(path / "memory.db")
            traces = TraceStore(path / "traces.db")
            run_id = traces.start_run("explain alpha beta gamma workspace notes", temp_dir)
            state = HarnessState(
                task="explain alpha beta gamma workspace notes",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                memory=memory,
                memory_paging=MemoryPagingConfig(
                    max_history_tokens=8,
                    preserve_recent_steps=1,
                    recall_limit=3,
                    archival_limit=3,
                ),
            )
            try:
                graph = build_graph(
                    Nodes(LMClient(provider="stub"), traces, runtime),
                    backend="simple",
                )
                final_state = graph.invoke(state)
                archives = memory.archival_search(
                    "alpha beta gamma workspace notes",
                    k=5,
                    kind="episode",
                    source_thread=run_id,
                )
                recall = memory.recall_page(run_id, k=10)
                report = traces.render_report(run_id)
            finally:
                memory.close()

        self.assertEqual(final_state.status, "done")
        self.assertGreaterEqual(final_state.scratch.get("memory_pages_written", 0), 1)
        self.assertLessEqual(len(final_state.history), 1)
        self.assertTrue(archives)
        self.assertIn("Archived harness history summary", archives[0].memory.content)
        self.assertGreater(len(recall), len(final_state.history))
        self.assertIn("memory_paged", report)

    def test_memory_hydrates_existing_thread_on_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            thread_id = "thread-resume"
            memory = Memory(path / "memory.db")
            memory.recall_append(
                thread_id,
                "user",
                "previous decision: continue with sqlite vec only",
            )
            memory.archival_add(
                "episode",
                "archived episode: sqlite vec was selected as the only vector backend",
                source_thread=thread_id,
            )
            traces = TraceStore(path / "traces.db")
            run_id = traces.start_run("continue sqlite vec implementation", temp_dir, thread_id)
            state = HarnessState(
                task="continue sqlite vec implementation",
                workspace=temp_dir,
                thread_id=thread_id,
                run_id=run_id,
            )
            client = CapturingClient()
            runtime = GraphRuntimeConfig(
                memory=memory,
                memory_paging=MemoryPagingConfig(
                    max_history_tokens=1000,
                    recall_limit=5,
                    archival_limit=5,
                ),
            )
            try:
                graph = build_graph(Nodes(client, traces, runtime), backend="simple")
                final_state = graph.invoke(state)
                report = traces.render_report(run_id)
            finally:
                memory.close()

        self.assertEqual(final_state.status, "done")
        self.assertIn("previous decision", final_state.scratch["memory_context"])
        self.assertIn("sqlite vec", final_state.scratch["memory_context"])
        self.assertIn("Memory context", client.plan_prompt)
        self.assertIn("memory_hydrated", report)

    @unittest.skipIf(importlib.util.find_spec("langgraph") is None, "langgraph is not installed")
    def test_langgraph_backend_runs_explicit_memory_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            memory = Memory(path / "memory.db")
            traces = TraceStore(path / "traces.db")
            run_id = traces.start_run("langgraph memory paging alpha beta", temp_dir)
            state = HarnessState(
                task="langgraph memory paging alpha beta",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                memory=memory,
                memory_paging=MemoryPagingConfig(
                    max_history_tokens=8,
                    preserve_recent_steps=1,
                ),
            )
            try:
                graph = build_graph(
                    Nodes(LMClient(provider="stub"), traces, runtime),
                    backend="langgraph",
                )
                final_state = graph.invoke(state)
                archives = memory.archival_search(
                    "langgraph memory paging alpha beta",
                    k=3,
                    kind="episode",
                    source_thread=run_id,
                )
                report = traces.render_report(run_id)
            finally:
                memory.close()

        self.assertEqual(final_state.status, "done")
        self.assertTrue(archives)
        self.assertIn("memory_hydrated", report)
        self.assertIn("memory_paged", report)

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
