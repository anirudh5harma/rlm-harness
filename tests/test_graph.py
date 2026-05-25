import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import (
    GraphRuntimeConfig,
    Nodes,
    final_answer_from_action,
    is_informational_task,
    is_project_summary_task,
    looks_like_source_dump,
    normalize_user_output,
    parse_numbered_plan,
    parse_python_action,
    render_observation,
)
from rlm_harness.memory import Memory, MemoryPagingConfig
from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import DockerREPL, SandboxConfig
from rlm_harness.tracing import TraceStore
from rlm_harness.types import Completion, HarnessState

IMAGE = "rlm-harness-sandbox:test"


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


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


class ToolErrorThenGoodActionClient(LMClient):
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
            content = "1. Inspect workspace\n2. Summarize project"
        elif "Return only valid JSON" in user_text:
            self.action_calls += 1
            if self.action_calls == 1:
                content = '{"type":"python","code":"print(read_file(None))"}'
            else:
                self.assert_recent_tool_error_context(user_text)
                content = (
                    '{"type":"python","code":"'
                    "print('Project summary\\\\n- recovered from invalid path')"
                    '"}'
                )
        elif "Decide whether the task is complete" in user_text:
            content = "done"
        else:
            content = "done"

        return Completion(
            content=content,
            model="tool-error-then-good",
            provider="test",
            latency_ms=0,
        )

    def assert_recent_tool_error_context(self, user_text):
        if "path must be a non-empty string" not in user_text:
            raise AssertionError("retry prompt did not include the previous tool error")


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

    def test_parse_python_action_accepts_fenced_json(self):
        code = parse_python_action('```json\n{"type": "python", "code": "print(7)"}\n```')
        self.assertEqual(code, "print(7)")

    def test_final_answer_uses_sandbox_output_not_raw_observation(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": "Project summary\n- Uses LangGraph\n",
                    "stderr": "",
                    "code": "print('Project summary')",
                }
            )
        )

        self.assertEqual(answer, "Project summary\n- Uses LangGraph")
        self.assertNotIn('"code"', answer)

    def test_project_overview_dict_output_becomes_human_summary(self):
        overview = {
            "files": [
                "package.json",
                "tsconfig.json",
                "vite.config.ts",
                "src/router.tsx",
                "src/routes/index.tsx",
                "src/components/ui/button.tsx",
                "src/styles.css",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"dev":"vite dev","build":"vite build","lint":"eslint ."},'
                        '"dependencies":{"react":"^19.2.0","@tanstack/react-start":"^1.0.0",'
                        '"@tanstack/react-router":"^1.0.0","@tailwindcss/vite":"^4.0.0",'
                        '"@radix-ui/react-dialog":"^1.0.0"},'
                        '"devDependencies":{"typescript":"^5.0.0","vite":"^7.0.0"}}'
                    ),
                }
            ],
            "git_status": "?? .rlm_harness/\n",
            "git_log": "abc123 project update\n",
        }

        answer = normalize_user_output(str(overview), task="summarize this project")

        self.assertIn("Project Summary", answer)
        self.assertIn("Tech stack: Node.js, TypeScript, React", answer)
        self.assertIn("TanStack Start", answer)
        self.assertIn("Useful commands", answer)
        self.assertIn("npm run dev", answer)
        self.assertNotIn("{'files':", answer)

    def test_what_is_this_project_is_project_summary_task(self):
        self.assertTrue(is_project_summary_task("what is this project"))
        self.assertTrue(is_informational_task("what is this project"))

    def test_action_prompt_routes_project_summary_to_summary_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("what is this project", temp_dir)
            state = HarnessState(
                task="what is this project",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                plan=["inspect the project"],
            )
            messages = Nodes(LMClient(provider="stub"), traces)._action_messages(state)

        system_prompt = messages[0].content
        self.assertIn("project_summary()", system_prompt)
        self.assertIn("Never print raw source code", system_prompt)

    def test_project_summary_reflects_source_dump_as_incomplete(self):
        source_dump = "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "class Example:",
                "    def method(self):",
                "        if True:",
                "            return 1",
                "def other():",
                "    return Example()",
            ]
        )
        self.assertTrue(looks_like_source_dump(source_dump))
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("what is this project", temp_dir)
            state = HarnessState(
                task="what is this project",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "ok",
                                "stdout": source_dump,
                                "stderr": "",
                                "code": "print(read_file('app.py'))",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "continue")
        self.assertFalse(final_state.final_answer)

    def test_stopped_project_summary_source_dump_does_not_return_source(self):
        source_dump = "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "class Example:",
                "    def method(self):",
                "        if True:",
                "            return 1",
                "def other():",
                "    return Example()",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("what is this project", temp_dir)
            state = HarnessState(
                task="what is this project",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "ok",
                                "stdout": source_dump,
                                "stderr": "",
                                "code": "print(read_file('app.py'))",
                            }
                        ),
                    }
                ],
            )
            runtime = GraphRuntimeConfig(max_iterations=1)

            final_state = Nodes(LMClient(provider="stub"), traces, runtime).reflect(state)

        self.assertEqual(final_state.status, "stopped")
        self.assertIn("printed source code", final_state.final_answer)
        self.assertNotIn("class Example", final_state.final_answer)

    def test_final_answer_does_not_expose_empty_sandbox_observation(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": "",
                    "stderr": "",
                    "code": "def summarize_project(path):\n    return 'summary'",
                }
            )
        )

        self.assertIn("did not produce a user-facing response", answer)
        self.assertNotIn("summarize_project", answer)
        self.assertNotIn('"code"', answer)

    def test_reflect_sandbox_error_uses_stderr_not_raw_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("list files", temp_dir)
            state = HarnessState(
                task="list files",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "sandbox_error",
                                "stdout": "",
                                "stderr": "docker socket unavailable",
                                "code": "print('hello')",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "error")
        self.assertEqual(final_state.final_answer, "docker socket unavailable")
        self.assertNotIn('"code"', final_state.final_answer)

    def test_reflect_tool_error_is_retryable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("summarize this project", temp_dir)
            state = HarnessState(
                task="summarize this project",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "tool_error",
                                "stdout": "",
                                "stderr": "ToolError: path must be a non-empty string",
                                "code": "print(read_file(None))",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "continue")

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

    def test_reflect_continues_informational_task_when_action_prints_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("summarize this project", temp_dir)
            state = HarnessState(
                task="summarize this project",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "ok",
                                "stdout": "",
                                "stderr": "",
                                "code": "def summarize_project(path): return 'summary'",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "continue")

    @unittest.skipUnless(docker_available(), "Docker daemon is not available")
    def test_graph_recovers_from_tool_error_on_next_action(self):
        DockerREPL.build_image(image=IMAGE)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("summarize this project", str(workspace))
            state = HarnessState(
                task="summarize this project",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            client = ToolErrorThenGoodActionClient()
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                sandbox_config=SandboxConfig(
                    image=IMAGE,
                    workspace=workspace,
                    default_timeout_s=5,
                ),
                max_iterations=3,
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 2)
        self.assertIn("recovered from invalid path", final_state.final_answer)
        self.assertIn("tool_error", report)

    @unittest.skipIf(not module_available("langgraph"), "langgraph is not installed")
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

    @unittest.skipIf(not module_available("langgraph"), "langgraph is not installed")
    @unittest.skipIf(
        not module_available("langgraph.checkpoint.sqlite"),
        "langgraph SQLite checkpointer is not installed",
    )
    def test_langgraph_backend_writes_sqlite_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            traces = TraceStore(path / "traces.db")
            run_id = traces.start_run("checkpoint graph", temp_dir)
            state = HarnessState(
                task="checkpoint graph",
                workspace=temp_dir,
                thread_id="checkpoint-thread",
                run_id=run_id,
            )
            graph = build_graph(
                Nodes(LMClient(provider="stub"), traces),
                backend="langgraph",
                checkpoint_path=path / "checkpoints.db",
            )
            try:
                final_state = graph.invoke(state)
            finally:
                graph.close()

            import sqlite3

            with sqlite3.connect(path / "checkpoints.db") as connection:
                count = connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0]

        self.assertEqual(final_state.status, "done")
        self.assertGreater(count, 0)

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

            if not module_available("langgraph"):
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
    def test_stub_graph_fixes_failing_unittest_through_coding_tools(self):
        DockerREPL.build_image(image=IMAGE)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "mathlib.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            (workspace / "test_mathlib.py").write_text(
                "import unittest\n"
                "from mathlib import add\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("Fix failing test in mathlib.py", str(workspace))
            state = HarnessState(
                task="Fix failing test in mathlib.py",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                sandbox_config=SandboxConfig(
                    image=IMAGE,
                    workspace=workspace,
                    default_timeout_s=10,
                ),
            )
            graph = build_graph(Nodes(LMClient(provider="stub"), traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            mathlib_content = (workspace / "mathlib.py").read_text(encoding="utf-8")

        self.assertEqual(final_state.status, "done")
        self.assertIn("return a + b", mathlib_content)
        self.assertIn("OK", final_state.final_answer)

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
