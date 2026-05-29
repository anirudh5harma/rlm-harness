import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rlm_harness.actions import CompleteTaskAction, CompletionStatus, TextObservation
from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import (
    GraphRuntimeConfig,
    Nodes,
    build_final_answer,
    deterministic_typed_action_for_task,
    executable_tool_payload,
    fallback_project_answer_if_needed,
    final_answer_from_action,
    is_code_editing_task,
    is_informational_task,
    is_project_audit_task,
    is_project_summary_task,
    looks_like_code_edit_result,
    looks_like_file_inventory,
    looks_like_project_audit,
    looks_like_source_dump,
    normalize_user_output,
    parse_numbered_plan,
    parse_python_action,
    parse_typed_tool_action,
    render_observation,
    render_typed_observation,
    rlm_action_context,
)
from rlm_harness.graph.task_policy import (
    looks_like_legacy_project_summary,
    looks_like_project_summary,
    looks_like_project_summary_markup_noise,
)
from rlm_harness.graph.verification import VerificationGate
from rlm_harness.kernel import (
    ActionSelectedEvent,
    AutonomyMode,
    CompletionEvent,
    ObservationRecordedEvent,
)
from rlm_harness.memory import Memory, MemoryPagingConfig
from rlm_harness.memory.evolution import EvolutionProposalStore
from rlm_harness.memory.profile import TasteProfileStore
from rlm_harness.model_client import LMClient
from rlm_harness.project_style import scan_project_style
from rlm_harness.rlm.runtime import find_repl_blocks
from rlm_harness.sandbox import DockerREPL, SandboxConfig
from rlm_harness.sandbox import tools as sandbox_tools
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
                    "print('Project Summary\\\\n"
                    "This project recovered from invalid path handling.\\\\n\\\\n"
                    "What I would do next:\\\\n"
                    "- recovered from invalid path\\\\n\\\\n"
                    "Verification I would run:\\\\n"
                    "- pytest')"
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


class TypedProjectSummaryClient(LMClient):
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
            content = "1. Summarize the project"
        elif "Return one typed action JSON object" in user_text:
            self.action_calls += 1
            content = '{"kind":"project_summary","max_files":80}'
        elif "Decide whether the task is complete" in user_text:
            content = "done"
        else:
            content = "done"

        return Completion(
            content=content,
            model="typed-summary",
            provider="test",
            latency_ms=0,
        )


class NonJsonProjectSummaryClient(LMClient):
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
            content = "1. Summarize the project"
        elif "Return one typed action JSON object" in user_text:
            self.action_calls += 1
            content = "I should inspect the project first."
        elif "Decide whether the task is complete" in user_text:
            content = "done"
        else:
            content = "done"

        return Completion(
            content=content,
            model="non-json-summary",
            provider="test",
            latency_ms=0,
        )


class ActionKindProjectSummaryClient(LMClient):
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
            content = "1. Summarize the project"
        elif "Return one typed action JSON object" in user_text:
            self.action_calls += 1
            content = '{"action_kind":"project_summary","max_files":80}'
        elif "Decide whether the task is complete" in user_text:
            content = "done"
        else:
            content = "done"

        return Completion(
            content=content,
            model="action-kind-summary",
            provider="test",
            latency_ms=0,
        )


class ReasoningPartialClient(LMClient):
    def __init__(self):
        super().__init__(provider="stub")

    def complete(self, messages, max_tokens=512, temperature=0.2):
        return Completion(
            content=(
                "The user wants me to summarize the partial state. "
                "Budget: iteration 3/3. I need to explain what happened."
            ),
            model="reasoning-partial",
            provider="test",
            latency_ms=0,
        )


class BadThenGoodTypedActionClient(LMClient):
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
            content = "1. Complete the task"
        elif "Return one typed action JSON object" in user_text:
            self.action_calls += 1
            if self.action_calls == 1:
                content = '{"kind":"record_memory","content":"not executable here"}'
            else:
                content = (
                    '{"kind":"complete_task","summary":"Done via typed tools.",'
                    '"status":"success","verification":"not needed"}'
                )
        elif "Decide whether the task is complete" in user_text:
            content = "done"
        else:
            content = "done"

        return Completion(
            content=content,
            model="typed-retry",
            provider="test",
            latency_ms=0,
        )


class TypedWriteActionClient(LMClient):
    def __init__(self):
        super().__init__(provider="stub")

    def complete(self, messages, max_tokens=512, temperature=0.2):
        user_text = ""
        for message in reversed(list(messages)):
            if message.role == "user":
                user_text = message.content
                break

        if "Return a concise numbered plan" in user_text:
            content = "1. Try to write a file"
        elif "Return one typed action JSON object" in user_text:
            content = '{"kind":"write_file","path":"notes.txt","content":"nope\\n"}'
        else:
            content = "done"

        return Completion(
            content=content,
            model="typed-write",
            provider="test",
            latency_ms=0,
        )


class GraphTests(unittest.TestCase):
    def test_parse_numbered_plan(self):
        plan = parse_numbered_plan("1. Inspect\n2. Act")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].description, "Inspect")
        self.assertEqual(plan.steps[1].description, "Act")

    def test_parse_python_action(self):
        code = parse_python_action('{"type": "python", "code": "print(2 + 2)"}')
        self.assertEqual(code, "print(2 + 2)")

    def test_parse_python_action_accepts_fenced_json(self):
        code = parse_python_action('```json\n{"type": "python", "code": "print(7)"}\n```')
        self.assertEqual(code, "print(7)")

    def test_parse_typed_tool_action_accepts_direct_action_object(self):
        action = parse_typed_tool_action('{"kind": "project_summary", "max_files": 20}')

        self.assertEqual(action.kind, "project_summary")
        self.assertEqual(action.max_files, 20)

    def test_parse_typed_tool_action_accepts_jsonish_mapping(self):
        action = parse_typed_tool_action("{'kind': 'project_summary', 'max_files': 20,}")

        self.assertEqual(action.kind, "project_summary")
        self.assertEqual(action.max_files, 20)

    def test_parse_typed_tool_action_accepts_action_kind_alias(self):
        action = parse_typed_tool_action(
            '{"action_kind":"project_overview","max_files":50,"max_read_bytes":8192}'
        )

        self.assertEqual(action.kind, "project_overview")
        self.assertEqual(action.max_files, 50)

    def test_parse_typed_tool_action_accepts_tool_wrapper(self):
        action = parse_typed_tool_action(
            '{"type":"tool","name":"complete_task","summary":"Done","status":"success"}'
        )

        self.assertIsInstance(action, CompleteTaskAction)
        self.assertEqual(action.summary, "Done")

    def test_parse_typed_tool_action_accepts_mcp_actions(self):
        action = parse_typed_tool_action(
            '{"kind":"mcp_call_tool","server":"github","tool_name":"get_issue","arguments":{"id":"1"}}'
        )

        self.assertEqual(action.kind, "mcp_call_tool")
        self.assertEqual(action.server, "github")
        self.assertEqual(action.arguments, {"id": "1"})

    def test_parse_typed_tool_action_rejects_host_only_action(self):
        with self.assertRaisesRegex(Exception, "not executable"):
            parse_typed_tool_action('{"kind":"record_memory","content":"remember this"}')

    def test_ask_autonomy_catalog_is_read_only(self):
        names = {tool["name"] for tool in executable_tool_payload(AutonomyMode.ASK)}

        self.assertIn("project_summary", names)
        self.assertIn("mcp_list_tools", names)
        self.assertIn("complete_task", names)
        self.assertNotIn("mcp_call_tool", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("run_shell", names)
        self.assertNotIn("propose_file_change", names)

    def test_render_typed_observation_preserves_legacy_stdout_shape(self):
        rendered = render_typed_observation(
            TextObservation(
                action_id="act_1",
                text="Done via typed tools.",
                summary=CompletionStatus.SUCCESS.value,
            )
        )

        payload = json.loads(rendered)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["stdout"], "Done via typed tools.")
        self.assertEqual(payload["observation_kind"], "text")

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
        self.assertIn("It appears to use Node.js, TypeScript, React", answer)
        self.assertIn("TanStack Start", answer)
        self.assertIn("What I would do next", answer)
        self.assertIn("Verification I would run", answer)
        self.assertIn("npm run build", answer)
        self.assertNotIn("Files inspected", answer)
        self.assertNotIn("Working tree:", answer)
        self.assertNotIn("{'files':", answer)

    def test_normalize_list_files_payload_as_bullets(self):
        answer = normalize_user_output(
            '["alpha.txt", "beta.txt"]',
            task="List files in workspace",
        )

        self.assertEqual(answer, "- alpha.txt\n- beta.txt")
        self.assertNotIn("[", answer)

    def test_file_question_selects_read_file_action(self):
        action = deterministic_typed_action_for_task("Explain mathlib.py")

        self.assertEqual(action.kind, "read_file")
        self.assertEqual(action.path, "mathlib.py")

    def test_file_observation_renders_file_summary(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": "def add(a, b):\n    return a + b\n",
                    "stderr": "",
                    "observation_kind": "file",
                    "path": "mathlib.py",
                    "truncated": False,
                }
            ),
            task="Explain mathlib.py",
        )

        self.assertIn("File Summary: mathlib.py", answer)
        self.assertIn("Functions: `add`", answer)
        self.assertIn("focused test", answer)
        self.assertNotIn("return a + b", answer)

    def test_where_question_selects_search_action(self):
        action = deterministic_typed_action_for_task("Where is add defined?")

        self.assertEqual(action.kind, "search_code")
        self.assertEqual(action.pattern, "add")

    def test_search_observation_renders_location_summary(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": "mathlib.py:1:def add(a, b):\n",
                    "stderr": "",
                    "observation_kind": "text",
                }
            ),
            task="Where is add defined?",
        )

        self.assertIn("Search Results for `add`", answer)
        self.assertIn("mathlib.py:1 - def add(a, b):", answer)
        self.assertIn("Open the most relevant match", answer)
        self.assertNotIn('"stdout"', answer)

    def test_git_change_question_selects_git_status_action(self):
        action = deterministic_typed_action_for_task("What changed in this repo?")

        self.assertEqual(action.kind, "git_status")

    def test_git_status_observation_renders_change_summary(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": " M app.py\nM cli.py\n?? notes.md\n",
                    "stderr": "",
                    "observation_kind": "text",
                    "summary": "git status",
                }
            ),
            task="What changed in this repo?",
        )

        self.assertIn("Git Changes", answer)
        self.assertIn("- Modified: app.py", answer)
        self.assertIn("- Modified: cli.py", answer)
        self.assertIn("- Untracked: notes.md", answer)
        self.assertIn("Review the diff", answer)
        self.assertNotIn('"stdout"', answer)

    def test_verification_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("How do I run tests?")

        self.assertEqual(action.kind, "project_overview")

    def test_entrypoint_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Where is the CLI entrypoint?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_entrypoint_answer(self):
        overview = {
            "files": ["Cargo.toml", "crates/cli/src/main.rs", "README.md"],
            "documents": [
                {
                    "path": "Cargo.toml",
                    "content": (
                        "[workspace]\n"
                        "members = [\"crates/cli\"]\n\n"
                        "[[bin]]\n"
                        "name = \"sample\"\n"
                        "path = \"crates/cli/src/main.rs\"\n"
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="Where is the CLI entrypoint?",
        )

        self.assertIn("Entrypoints", answer)
        self.assertIn("`crates/cli/src/main.rs`", answer)
        self.assertIn("Cargo.toml is present.", answer)
        self.assertIn("Open the top candidate", answer)
        self.assertNotIn("Search Results for `the`", answer)
        self.assertNotIn("Project Summary", answer)

    def test_project_overview_renders_verification_commands_for_test_question(self):
        overview = {
            "files": ["Cargo.toml", "crates/cli/src/main.rs"],
            "documents": [{"path": "Cargo.toml", "content": "[workspace]\n"}],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(json.dumps(overview), task="How do I run tests?")

        self.assertIn("Verification Commands", answer)
        self.assertIn("`cargo test`", answer)
        self.assertIn("Cargo.toml is present.", answer)
        self.assertNotIn("Project Summary", answer)

    def test_run_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("How do I run this project?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_run_commands_for_run_question(self):
        overview = {
            "files": ["package.json", "pnpm-lock.yaml", "src/main.ts"],
            "documents": [
                {
                    "path": "package.json",
                    "content": '{"scripts":{"dev":"vite dev","build":"vite build"}}',
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="How do I run this project?",
        )

        self.assertIn("Run Commands", answer)
        self.assertIn("`pnpm dev`", answer)
        self.assertIn("package.json defines script(s): build, dev.", answer)
        self.assertIn("Run the first command", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_stack_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("What stack does this project use?")

        self.assertEqual(action.kind, "project_overview")

    def test_dependency_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("List package dependencies.")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_dependency_answer(self):
        overview = {
            "files": ["package.json", "pnpm-lock.yaml"],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"dependencies":{"react":"^19.0.0","vite":"^7.0.0"},'
                        '"devDependencies":{"typescript":"^5.0.0","vitest":"^3.0.0"}}'
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="List package dependencies.",
        )

        self.assertIn("Dependencies", answer)
        self.assertIn("runtime:", answer)
        self.assertIn("`react`", answer)
        self.assertIn("`vite`", answer)
        self.assertIn("development:", answer)
        self.assertIn("`typescript`", answer)
        self.assertIn("pnpm appears to be the package manager.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_project_overview_renders_stack_answer(self):
        overview = {
            "files": ["package.json", "src/App.tsx"],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"dependencies":{"react":"^19.0.0","vite":"^7.0.0",'
                        '"@tanstack/react-router":"^1.0.0"},'
                        '"devDependencies":{"typescript":"^5.0.0"}}'
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What stack does this project use?",
        )

        self.assertIn("Tech Stack", answer)
        self.assertIn("- Node.js", answer)
        self.assertIn("- TypeScript", answer)
        self.assertIn("- React", answer)
        self.assertIn("- Vite", answer)
        self.assertIn("Dependency Signals", answer)
        self.assertIn("package.json is present.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_test_location_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Where are the tests?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_test_location_answer(self):
        overview = {
            "files": ["pyproject.toml", "app.py", "tests/test_app.py"],
            "documents": [{"path": "pyproject.toml", "content": "[project]\nname='sample'\n"}],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="Where are the tests?",
        )

        self.assertIn("Test Files", answer)
        self.assertIn("`tests/test_app.py`", answer)
        self.assertIn("Likely Commands", answer)
        self.assertIn("`pytest`", answer)
        self.assertIn("A top-level tests/ directory is present.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_project_structure_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("How is this project structured?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_project_structure_answer(self):
        overview = {
            "files": [
                "README.md",
                "Cargo.toml",
                "crates/cli/src/main.rs",
                "tests/test_smoke.py",
                "docs/architecture.md",
            ],
            "documents": [{"path": "Cargo.toml", "content": "[workspace]\n"}],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="How is this project structured?",
        )

        self.assertIn("Project Structure", answer)
        self.assertIn("`crates` - Rust workspace crates", answer)
        self.assertIn("`tests` - test coverage", answer)
        self.assertIn("Start With", answer)
        self.assertIn("`Cargo.toml`", answer)
        self.assertIn("crates/ contains Rust workspace members.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_todo_question_selects_marker_search_action(self):
        action = deterministic_typed_action_for_task("Find TODOs and FIXME comments.")

        self.assertEqual(action.kind, "search_code")
        self.assertEqual(action.pattern, r"\b(TODO|FIXME|HACK|XXX)\b")

    def test_todo_search_observation_renders_task_markers(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": (
                        "src/app.py:2:    pass  # TODO: wire real implementation\n"
                        "src/client.ts:1:// FIXME: handle auth refresh\n"
                    ),
                    "stderr": "",
                    "observation_kind": "text",
                }
            ),
            task="What TODOs are in this project?",
        )

        self.assertIn("Task Markers", answer)
        self.assertIn("src/app.py:2 - pass  # TODO: wire real implementation", answer)
        self.assertIn("src/client.ts:1 - // FIXME: handle auth refresh", answer)
        self.assertIn("verify whether it is still current", answer)
        self.assertNotIn("Search Results for `TODOs`", answer)
        self.assertNotIn('"stdout"', answer)

    def test_setup_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task(
            "How do I install dependencies for this project?"
        )

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_setup_answer(self):
        overview = {
            "files": ["package.json", "pnpm-lock.yaml", "src/main.ts"],
            "documents": [
                {
                    "path": "package.json",
                    "content": '{"scripts":{"dev":"vite dev","test":"vitest"}}',
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="How do I install dependencies for this project?",
        )

        self.assertIn("Setup Commands", answer)
        self.assertIn("`pnpm install`", answer)
        self.assertIn("After Setup", answer)
        self.assertIn("`pnpm dev`", answer)
        self.assertIn("pnpm-lock.yaml indicates pnpm.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_project_commands_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("What scripts can I run in this project?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_project_commands_answer(self):
        overview = {
            "files": ["package.json", "pnpm-lock.yaml", "src/main.ts"],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"dev":"vite dev","build":"vite build",'
                        '"lint":"eslint .","test":"vitest"}}'
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What scripts can I run in this project?",
        )

        self.assertIn("Project Commands", answer)
        self.assertIn("`pnpm dev` - vite dev", answer)
        self.assertIn("`pnpm build` - vite build", answer)
        self.assertIn("`pnpm test` - vitest", answer)
        self.assertIn("package.json defines script(s): build, dev, lint, test.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_environment_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("What environment variables do I need?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_environment_answer_without_real_env_values(self):
        overview = {
            "files": ["package.json", ".env.example", ".env"],
            "documents": [
                {
                    "path": ".env.example",
                    "content": (
                        "OPENAI_API_KEY=\n"
                        "DATABASE_URL=postgres://example\n"
                        "LOG_LEVEL=info\n"
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What environment variables do I need?",
        )

        self.assertIn("Environment Variables", answer)
        self.assertIn("`OPENAI_API_KEY` from `.env.example` - required", answer)
        self.assertIn("`DATABASE_URL` from `.env.example`", answer)
        self.assertIn("Real `.env` files are present", answer)
        self.assertNotIn("secret-real-value", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_container_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("How do I run this in Docker?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_container_answer(self):
        overview = {
            "files": [
                "package.json",
                "Dockerfile",
                "docker-compose.yml",
                ".dockerignore",
            ],
            "documents": [
                {"path": "package.json", "content": '{"name":"webapp"}'},
                {
                    "path": "Dockerfile",
                    "content": "FROM node:22\nEXPOSE 3000\nCMD [\"pnpm\", \"start\"]\n",
                },
                {"path": "docker-compose.yml", "content": "services:\n  web:\n    build: .\n"},
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="How do I run this in Docker?",
        )

        self.assertIn("Container Runtime", answer)
        self.assertIn("`Dockerfile` - Docker image definition", answer)
        self.assertIn("`docker-compose.yml` - Docker Compose runtime config", answer)
        self.assertIn("`docker compose up --build`", answer)
        self.assertIn("`docker build -t webapp .`", answer)
        self.assertIn("`docker run --rm -p 3000:3000 webapp`", answer)
        self.assertIn("Dockerfile exposes port(s): 3000.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_deployment_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("How do I deploy this project?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_deployment_answer(self):
        overview = {
            "files": ["package.json", "pnpm-lock.yaml", "vercel.json"],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"build":"vite build",'
                        '"deploy":"vercel deploy --prod"}}'
                    ),
                },
                {"path": "vercel.json", "content": '{"buildCommand":"pnpm build"}'},
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="How do I deploy this project?",
        )

        self.assertIn("Deployment", answer)
        self.assertIn("Vercel (`vercel.json`)", answer)
        self.assertIn("`vercel.json` - Vercel deployment config", answer)
        self.assertIn("`pnpm deploy` - vercel deploy --prod", answer)
        self.assertIn("`pnpm build` - vite build", answer)
        self.assertIn("vercel.json was read for command hints.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_database_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("What database does this use?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_database_answer(self):
        overview = {
            "files": [
                "package.json",
                "prisma/schema.prisma",
                "prisma/migrations/001_init/migration.sql",
                ".env.example",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"db:migrate":"prisma migrate dev"},'
                        '"dependencies":{"@prisma/client":"^6.0.0","prisma":"^6.0.0"}}'
                    ),
                },
                {
                    "path": "prisma/schema.prisma",
                    "content": (
                        "datasource db { provider = \"postgresql\" "
                        "url = env(\"DATABASE_URL\") }\n"
                    ),
                },
                {"path": ".env.example", "content": "DATABASE_URL=postgres://example\n"},
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What database does this use?",
        )

        self.assertIn("Database / Schema", answer)
        self.assertIn("Prisma ORM", answer)
        self.assertIn("Prisma schema/migrations", answer)
        self.assertIn("`prisma/schema.prisma` - Prisma schema", answer)
        self.assertIn("`prisma/migrations/001_init/migration.sql` - Prisma migration", answer)
        self.assertIn("`npm run db:migrate` - prisma migrate dev", answer)
        self.assertIn("`DATABASE_URL` appears in `.env.example`.", answer)
        self.assertNotIn("postgres://example", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_auth_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Where is authentication handled?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_auth_answer(self):
        overview = {
            "files": [
                "package.json",
                "auth.ts",
                "app/api/auth/[...nextauth]/route.ts",
                "middleware.ts",
                "app/login/page.tsx",
                ".env.example",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"dev":"next dev"},'
                        '"dependencies":{"next-auth":"^5.0.0","jose":"^6.0.0"}}'
                    ),
                },
                {"path": "auth.ts", "content": "export const { auth } = NextAuth({})\n"},
                {
                    "path": ".env.example",
                    "content": "AUTH_SECRET=\nGITHUB_CLIENT_ID=\nGITHUB_CLIENT_SECRET=\n",
                },
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="Where is authentication handled?",
        )

        self.assertIn("Auth / Sessions", answer)
        self.assertIn("NextAuth/Auth.js", answer)
        self.assertIn("JOSE/JWT", answer)
        self.assertIn("`auth.ts` - auth config/module", answer)
        self.assertIn("`app/api/auth/[...nextauth]/route.ts` - NextAuth/Auth.js route", answer)
        self.assertIn("`middleware.ts` - request/auth middleware", answer)
        self.assertIn("`AUTH_SECRET` appears in `.env.example`.", answer)
        self.assertIn("`npm run dev`", answer)
        self.assertNotIn("GITHUB_CLIENT_SECRET=", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_frontend_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Where is the frontend code?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_frontend_answer(self):
        overview = {
            "files": [
                "package.json",
                "app/page.tsx",
                "app/layout.tsx",
                "components/ui/button.tsx",
                "src/components/Header.tsx",
                "app/globals.css",
                "tailwind.config.ts",
                "components.json",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"dev":"next dev"},'
                        '"dependencies":{"next":"^16.0.0","react":"^19.0.0",'
                        '"tailwindcss":"^4.0.0","lucide-react":"^0.500.0"}}'
                    ),
                },
                {"path": "components.json", "content": '{"style":"new-york"}'},
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="Where is the frontend code?",
        )

        self.assertIn("UI / Frontend", answer)
        self.assertIn("Next.js", answer)
        self.assertIn("React", answer)
        self.assertIn("Tailwind CSS", answer)
        self.assertIn("shadcn/ui", answer)
        self.assertIn("`app/page.tsx` - root app page", answer)
        self.assertIn("`app/layout.tsx` - app layout shell", answer)
        self.assertIn("`components/ui/button.tsx` - UI component", answer)
        self.assertIn("`app/globals.css` - global styles", answer)
        self.assertIn("`npm run dev`", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_edit_target_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task(
            "What file should I edit to change the homepage UI?"
        )

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_edit_target_answer(self):
        overview = {
            "files": [
                "package.json",
                "pnpm-lock.yaml",
                "app/page.tsx",
                "app/layout.tsx",
                "components/ui/button.tsx",
                "app/globals.css",
                "tests/homepage.test.tsx",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"test":"vitest"},'
                        '"dependencies":{"next":"^16.0.0","react":"^19.0.0"}}'
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What file should I edit to change the homepage UI?",
        )

        self.assertIn("Edit Targets", answer)
        self.assertIn("`app/page.tsx` - root app page", answer)
        self.assertIn("`components/ui/button.tsx` - UI component", answer)
        self.assertIn("`app/globals.css` - global styles", answer)
        self.assertIn("Matched request terms: homepage", answer)
        self.assertIn("Check After Editing", answer)
        self.assertIn("`pnpm test`", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_debugging_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task(
            "How should I debug failing tests in this project?"
        )

        self.assertEqual(action.kind, "project_overview")

    def test_debugging_question_does_not_intercept_fix_task(self):
        action = deterministic_typed_action_for_task("Fix the failing test in mathlib.py")

        self.assertNotEqual(getattr(action, "kind", None), "project_overview")

    def test_project_overview_renders_debugging_answer(self):
        overview = {
            "files": [
                "package.json",
                "pnpm-lock.yaml",
                "src/App.test.tsx",
                "vitest.config.ts",
                "playwright.config.ts",
                ".github/workflows/ci.yml",
                "app/error.tsx",
                "lib/logger.ts",
                "tsconfig.json",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"test":"vitest","lint":"eslint .",'
                        '"typecheck":"tsc --noEmit"}}'
                    ),
                },
                {
                    "path": ".github/workflows/ci.yml",
                    "content": "steps:\n  - run: pnpm test\n  - run: pnpm lint\n",
                },
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="How should I debug failing tests in this project?",
        )

        self.assertIn("Debugging Path", answer)
        self.assertIn("Start With", answer)
        self.assertIn("`pnpm test` - vitest", answer)
        self.assertIn("`pnpm lint` - eslint .", answer)
        self.assertIn("`pnpm typecheck` - tsc --noEmit", answer)
        self.assertIn("`src/App.test.tsx` - test file", answer)
        self.assertIn("`vitest.config.ts` - test runner config", answer)
        self.assertIn("`.github/workflows/ci.yml` - CI workflow", answer)
        self.assertIn("`app/error.tsx` - route error boundary", answer)
        self.assertIn("`lib/logger.ts` - logging helper", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_api_routes_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Where are the API routes?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_api_routes_answer(self):
        overview = {
            "files": [
                "package.json",
                "app/api/users/route.ts",
                "pages/api/health.ts",
                "src/server.ts",
            ],
            "documents": [
                {
                    "path": "package.json",
                    "content": (
                        '{"scripts":{"dev":"next dev"},'
                        '"dependencies":{"next":"^16.0.0","express":"^5.0.0"}}'
                    ),
                }
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="Where are the API routes?",
        )

        self.assertIn("API / Routes", answer)
        self.assertIn("Next.js", answer)
        self.assertIn("Express", answer)
        self.assertIn("`app/api/users/route.ts` - Next.js App Router API route", answer)
        self.assertIn("`pages/api/health.ts` - Next.js Pages API route", answer)
        self.assertIn("Likely Local Server", answer)
        self.assertIn("`npm run dev`", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_commit_readiness_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Am I ready to commit?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_commit_readiness_answer(self):
        overview = {
            "files": ["package.json", "pnpm-lock.yaml", "src/app.ts", "README.md"],
            "documents": [
                {
                    "path": "package.json",
                    "content": '{"scripts":{"test":"vitest","lint":"eslint ."}}',
                }
            ],
            "git_status": " M src/app.ts\n?? notes.md\n",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="Am I ready to commit?",
        )

        self.assertIn("Commit Readiness", answer)
        self.assertIn("Working Tree", answer)
        self.assertIn("- Modified: src/app.ts", answer)
        self.assertIn("- Untracked: notes.md", answer)
        self.assertIn("Recommended Checks", answer)
        self.assertIn("`pnpm test`", answer)
        self.assertIn("Not quite ready", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_config_files_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("Where is project config?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_config_files_answer(self):
        overview = {
            "files": [
                "package.json",
                "tsconfig.json",
                "vite.config.ts",
                "vercel.json",
                ".env",
            ],
            "documents": [{"path": "package.json", "content": '{"scripts":{"dev":"vite dev"}}'}],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What config files are in this repo?",
        )

        self.assertIn("Config Files", answer)
        self.assertIn("`package.json` - Node package/dependency config", answer)
        self.assertIn("`tsconfig.json` - TypeScript/JavaScript compiler config", answer)
        self.assertIn("`vite.config.ts` - Vite build/dev-server config", answer)
        self.assertIn("`vercel.json` - deployment/runtime config", answer)
        self.assertIn("Real `.env` files are present", answer)
        self.assertNotIn("secret-real-value", answer)
        self.assertNotIn("Search Results for `project`", answer)
        self.assertNotIn("Project Summary", answer)

    def test_ci_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("What CI checks run for this repo?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_ci_answer(self):
        overview = {
            "files": [
                "package.json",
                ".github/workflows/ci.yml",
            ],
            "documents": [
                {"path": "package.json", "content": '{"scripts":{"test":"vitest"}}'},
                {
                    "path": ".github/workflows/ci.yml",
                    "content": (
                        "name: CI\n"
                        "jobs:\n"
                        "  test:\n"
                        "    steps:\n"
                        "      - run: pnpm test\n"
                        "      - run: pnpm lint\n"
                    ),
                },
            ],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What CI checks run for this repo?",
        )

        self.assertIn("CI Checks", answer)
        self.assertIn("Workflow Files", answer)
        self.assertIn("`.github/workflows/ci.yml`", answer)
        self.assertIn("`pnpm test`", answer)
        self.assertIn("`pnpm lint`", answer)
        self.assertIn("workflow file was read", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_documentation_question_selects_project_overview_action(self):
        action = deterministic_typed_action_for_task("What docs exist in this repo?")

        self.assertEqual(action.kind, "project_overview")

    def test_project_overview_renders_documentation_answer(self):
        overview = {
            "files": [
                "README.md",
                "docs/architecture.md",
                "docs/install.md",
                "docs/api.md",
            ],
            "documents": [{"path": "README.md", "content": "# Project\n"}],
            "git_status": "",
            "git_log": "",
        }

        answer = normalize_user_output(
            json.dumps(overview),
            task="What docs exist in this repo?",
        )

        self.assertIn("Documentation", answer)
        self.assertIn("`README.md` - project overview", answer)
        self.assertIn("`docs/install.md` - setup/getting-started docs", answer)
        self.assertIn("`docs/architecture.md` - architecture/design notes", answer)
        self.assertIn("Read First", answer)
        self.assertIn("A README is present.", answer)
        self.assertNotIn("Return one typed action JSON object", answer)
        self.assertNotIn("Project Summary", answer)

    def test_what_is_this_project_is_project_summary_task(self):
        self.assertTrue(is_project_summary_task("what is this project"))
        self.assertTrue(is_project_summary_task("what is this porject about"))
        self.assertTrue(is_informational_task("what is this project"))

    def test_project_gap_prompt_is_audit_task(self):
        task = "find any logical and technical gaps in this project"

        self.assertTrue(is_project_audit_task(task))
        self.assertTrue(is_informational_task(task))

    def test_project_next_steps_prompt_is_audit_task(self):
        task = "what is this porject about and what must be done next"

        self.assertTrue(is_project_summary_task(task))
        self.assertTrue(is_project_audit_task(task))
        self.assertTrue(is_informational_task(task))
        self.assertFalse(
            looks_like_project_audit(
                "Project Summary\n"
                "Recent commits:\n"
                "2eefcc2 fixed technical gaps\n"
                "Notable source files:\n"
                "- pyproject.toml"
            )
        )

    def test_legacy_project_summary_is_not_accepted_as_good_answer(self):
        legacy_summary = (
            "Project Summary\n"
            "What it is: [![skills.sh](https://skills.sh/b/a3fckx/sansara)]"
            "(https://skills.sh/b/a3fckx/sansara)\n"
            "Files inspected: 300\n"
            "Key config/docs: README.md, Cargo.toml\n"
            "Working tree:\n"
            "M  crates/sansara-cli/src/main.rs\n"
            "Recent commits:\n"
            "3236378 chore: simplify sansara routing and docs\n"
            "Notable source files:\n"
            "- README.md"
        )

        self.assertTrue(looks_like_legacy_project_summary(legacy_summary))
        self.assertFalse(looks_like_project_summary(legacy_summary))

    def test_markdown_scrape_noise_is_not_accepted_as_project_summary(self):
        noisy_summary = (
            "Project Summary\n"
            "this project is vault-native autonomous agent OS: a **flat wiki** "
            "Obsidian vault is the world model. It appears to use Rust.\n"
            "What I would do next:\n"
            "- Read the README.\n"
            "Verification I would run:\n"
            "- cargo test\n"
        )

        self.assertTrue(looks_like_project_summary_markup_noise(noisy_summary))
        self.assertFalse(looks_like_project_summary(noisy_summary))

    def test_code_edit_task_requires_change_or_verification_evidence(self):
        self.assertTrue(is_code_editing_task("fix failing tests in mathlib.py"))
        self.assertFalse(looks_like_code_edit_result("done"))
        self.assertTrue(looks_like_code_edit_result("Changed mathlib.py\nVerification: pytest OK"))

    def test_rlm_runtime_accepts_python_fenced_cells(self):
        self.assertEqual(find_repl_blocks("```python\nprint(1)\n```"), ["print(1)"])

    def test_build_final_answer_appends_verification_for_code_edits(self):
        answer = build_final_answer(
            render_observation({"status": "ok", "stdout": "Done", "stderr": ""}),
            task="Fix failing test in mathlib.py",
            verification={
                "summary": "  [PASS] project_command: Ran 1 test | OK",
                "checks": [],
            },
        )

        self.assertIn("Completed the requested code change.", answer)
        self.assertIn("Verification", answer)
        self.assertIn("OK", answer)

    def test_reflect_honors_rlm_completion_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("List files in workspace", temp_dir)
            state = HarnessState(
                task="List files in workspace",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            state.scratch["action_completed"] = True
            state.history.append(
                {
                    "node": "observe",
                    "content": render_observation(
                        {
                            "status": "ok",
                            "stdout": "The workspace contains alpha.txt and beta.txt.",
                            "stderr": "",
                        }
                    ),
                }
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "done")
        self.assertIn("alpha.txt", final_state.final_answer)

    def test_reflect_synthesizes_code_edit_answer_after_verified_empty_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("Fix failing test in mathlib.py", temp_dir)
            state = HarnessState(
                task="Fix failing test in mathlib.py",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            state.scratch["action_completed"] = True
            state.scratch["verification_result"] = {
                "passed": True,
                "summary": "  [PASS] project_command: Ran 1 test | OK",
                "checks": [],
            }
            state.history.append(
                {
                    "node": "observe",
                    "content": render_observation(
                        {"status": "ok", "stdout": "", "stderr": ""}
                    ),
                }
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "done")
        self.assertIn("Completed the requested code change.", final_state.final_answer)
        self.assertIn("Verification", final_state.final_answer)
        self.assertIn("OK", final_state.final_answer)

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
        self.assertIn("plain, friendly English", system_prompt)

    def test_action_prompt_routes_project_audit_to_audit_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            task = "find any logical and technical gaps in this project"
            run_id = traces.start_run(task, temp_dir)
            state = HarnessState(
                task=task,
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                plan=["inspect the project"],
            )
            messages = Nodes(LMClient(provider="stub"), traces)._action_messages(state)

        system_prompt = messages[0].content
        self.assertIn("project_audit()", system_prompt)
        self.assertIn("evidence, impact, and recommendations", system_prompt)
        self.assertIn("ALL FILES inventory", system_prompt)

    def test_project_audit_reflects_file_inventory_as_incomplete(self):
        file_inventory = "\n".join(
            [
                "ALL FILES:",
                ".gitignore",
                "package.json",
                "tsconfig.json",
                "vite.config.ts",
                "src/content.ts",
                "src/router.tsx",
                "src/routes/index.tsx",
                "src/components/ui/button.tsx",
                "src/styles.css",
            ]
        )
        self.assertTrue(looks_like_file_inventory(file_inventory))
        self.assertFalse(looks_like_project_audit(file_inventory))
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            task = "find any logical and technical gaps in this project"
            run_id = traces.start_run(task, temp_dir)
            state = HarnessState(
                task=task,
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "ok",
                                "stdout": file_inventory,
                                "stderr": "",
                                "code": "print('\\n'.join(list_files()))",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "continue")
        self.assertFalse(final_state.final_answer)

    def test_project_audit_returns_evidence_backed_findings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir).resolve()
            (workspace / "package.json").write_text(
                (
                    '{"dependencies":{"react":"^19.0.0","typescript":"^5.0.0"},'
                    '"scripts":{"build":"vite build"}}'
                ),
                encoding="utf-8",
            )
            (workspace / "src" / "routes").mkdir(parents=True)
            (workspace / "src" / "routes" / "index.tsx").write_text(
                "export default function Index() { return <main>Hello</main> }\n",
                encoding="utf-8",
            )
            old_workspace = sandbox_tools.WORKSPACE
            sandbox_tools.WORKSPACE = workspace
            try:
                audit = sandbox_tools.project_audit()
            finally:
                sandbox_tools.WORKSPACE = old_workspace

        self.assertIn("Project Gap Analysis", audit)
        self.assertIn("What I would fix or clarify next", audit)
        self.assertIn("Evidence:", audit)
        self.assertIn("Next move:", audit)

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

    def test_project_summary_reflects_json_file_inventory_as_incomplete(self):
        file_inventory = json.dumps(
            [
                ".gitignore",
                "package.json",
                "src/routes/index.tsx",
                "src/router.tsx",
                "src/styles.css",
            ],
            indent=2,
        )
        self.assertTrue(looks_like_file_inventory(file_inventory))
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("what is this project about", temp_dir)
            state = HarnessState(
                task="what is this project about",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "ok",
                                "stdout": file_inventory,
                                "stderr": "",
                                "code": "print(json.dumps(list_files()))",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "continue")
        self.assertFalse(final_state.final_answer)

    def test_rlm_action_context_uses_container_workspace_not_host_path(self):
        state = HarnessState(
            task="explain this project",
            workspace="/Users/anirudhsharma/Documents/projects/who-am-i",
            thread_id="thread",
            run_id="run",
            plan=["inspect"],
        )

        context = rlm_action_context(state)

        self.assertEqual(context["workspace"], "/workspace")
        self.assertNotIn("/Users/anirudhsharma", json.dumps(context))
        self.assertIn("host paths are not available", context["workspace_note"])

    def test_project_summary_fallback_replaces_file_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Who Am I\n\nA personal portfolio app.\n",
                encoding="utf-8",
            )
            bad_answer = json.dumps(["package.json", "src/routes/index.tsx"], indent=2)

            answer = fallback_project_answer_if_needed(
                "what is this project about",
                bad_answer,
                workspace,
                "done",
            )

        self.assertIn("Project Summary", answer)
        self.assertIn("personal portfolio app", answer)
        self.assertIn("What I would do next", answer)
        self.assertNotEqual(answer, bad_answer)

    def test_project_summary_fallback_replaces_legacy_metric_summary(self):
        legacy_answer = (
            "Project Summary\n"
            "What it is: [![skills.sh](https://skills.sh/b/a3fckx/sansara)]"
            "(https://skills.sh/b/a3fckx/sansara)\n"
            "Files inspected: 300\n"
            "Key config/docs: README.md, Cargo.toml\n"
            "Working tree:\n"
            "M  crates/sansara-cli/src/main.rs\n"
            "Recent commits:\n"
            "3236378 chore: simplify sansara routing and docs\n"
            "Notable source files:\n"
            "- README.md"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Sansara\n\n"
                "[![skills.sh](https://skills.sh/b/a3fckx/sansara)]"
                "(https://skills.sh/b/a3fckx/sansara)\n\n"
                "Vault-native agent OS for coding assistants.\n",
                encoding="utf-8",
            )
            (workspace / "Cargo.toml").write_text(
                '[package]\nname = "sansara"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )

            answer = fallback_project_answer_if_needed(
                "what is this project about and what must I do next",
                legacy_answer,
                workspace,
                "done",
            )

        self.assertIn("Project Summary", answer)
        self.assertIn("sansara is vault-native agent OS for coding assistants.", answer)
        self.assertIn("What I would do next", answer)
        self.assertIn("Verification I would run", answer)
        self.assertNotIn("Files inspected", answer)
        self.assertNotIn("Working tree:", answer)
        self.assertNotIn("skills.sh", answer)

    def test_done_replaces_legacy_project_summary_before_final_answer(self):
        legacy_answer = (
            "Project Summary\n"
            "What it is: [![skills.sh](https://skills.sh/b/a3fckx/sansara)]"
            "(https://skills.sh/b/a3fckx/sansara)\n"
            "Files inspected: 300\n"
            "Key config/docs: README.md, Cargo.toml\n"
            "Working tree:\n"
            "M  crates/sansara-cli/src/main.rs\n"
            "Recent commits:\n"
            "3236378 chore: simplify sansara routing and docs\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Sansara\n\nVault-native agent OS for coding assistants.\n",
                encoding="utf-8",
            )
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run(
                "what is this project about and what must I do next",
                str(workspace),
            )
            state = HarnessState(
                task="what is this project about and what must I do next",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
                scratch={
                    "last_action": render_observation(
                        {
                            "status": "ok",
                            "stdout": legacy_answer,
                            "stderr": "",
                            "code": "print(project_summary())",
                        }
                    )
                },
            )

            final_state = Nodes(LMClient(provider="stub"), traces).done(state)

        self.assertIn("Project Summary", final_state.final_answer)
        self.assertIn("What I would do next", final_state.final_answer)
        self.assertNotIn("Files inspected", final_state.final_answer)

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
        self.assertIn("did not produce a project summary", final_state.final_answer)
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

    def test_taste_learning_from_run_is_injected_into_future_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            profile_memory = Memory(path / "profile.db")
            traces = TraceStore(path / "traces.db")
            try:
                first_run = traces.start_run(
                    "I prefer concise final answers and minimal diffs.",
                    temp_dir,
                    "taste-thread",
                )
                first_state = HarnessState(
                    task="I prefer concise final answers and minimal diffs.",
                    workspace=temp_dir,
                    thread_id="taste-thread",
                    run_id=first_run,
                )
                runtime = GraphRuntimeConfig(profile_memory=profile_memory)
                graph = build_graph(
                    Nodes(LMClient(provider="stub"), traces, runtime),
                    backend="simple",
                )
                first_final = graph.invoke(first_state)

                client = CapturingClient()
                second_run = traces.start_run(
                    "summarize this project",
                    temp_dir,
                    "taste-thread",
                )
                second_state = HarnessState(
                    task="summarize this project",
                    workspace=temp_dir,
                    thread_id="taste-thread",
                    run_id=second_run,
                )
                graph = build_graph(Nodes(client, traces, runtime), backend="simple")
                second_final = graph.invoke(second_state)
                proposals = EvolutionProposalStore(profile_memory).proposals()
                report = traces.render_report(first_run)
            finally:
                profile_memory.close()

        self.assertEqual(first_final.status, "done")
        self.assertEqual(second_final.status, "done")
        self.assertIn("Taste context", client.plan_prompt)
        self.assertIn("concise final answers", client.plan_prompt)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, "prompt_rule")
        self.assertIn("concise final answers", proposals[0].body)
        self.assertIn("taste_learned", report)
        self.assertIn("evolution_proposed", report)

    def test_scanned_project_style_is_injected_into_future_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "pyproject.toml").write_text(
                "[project]\n"
                "dependencies = ['pydantic>=2']\n"
                "[project.optional-dependencies]\n"
                "dev = ['pytest>=8']\n"
                "[tool.ruff]\n"
                "line-length = 100\n",
                encoding="utf-8",
            )
            profile_memory = Memory(path / "profile.db")
            project_memory = Memory(path / "memory.db")
            traces = TraceStore(path / "traces.db")
            try:
                store = TasteProfileStore(project_memory)
                for record in scan_project_style(path):
                    store.add(record)
                client = CapturingClient()
                run_id = traces.start_run("summarize this project", temp_dir, "style-thread")
                state = HarnessState(
                    task="summarize this project",
                    workspace=temp_dir,
                    thread_id="style-thread",
                    run_id=run_id,
                )
                runtime = GraphRuntimeConfig(
                    profile_memory=profile_memory,
                    memory=project_memory,
                )
                graph = build_graph(Nodes(client, traces, runtime), backend="simple")
                final = graph.invoke(state)
            finally:
                profile_memory.close()
                project_memory.close()

        self.assertEqual(final.status, "done")
        self.assertIn("Taste context", client.plan_prompt)
        self.assertIn("Project conventions", client.plan_prompt)
        self.assertIn("Keep Python line length at 100 characters", client.plan_prompt)

    def test_project_style_is_auto_scanned_before_planning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "pyproject.toml").write_text(
                "[project]\n"
                "[project.optional-dependencies]\n"
                "dev = ['pytest>=8']\n"
                "[tool.ruff]\n"
                "line-length = 100\n",
                encoding="utf-8",
            )
            profile_memory = Memory(path / "profile.db")
            project_memory = Memory(path / "memory.db")
            traces = TraceStore(path / "traces.db")
            try:
                client = CapturingClient()
                run_id = traces.start_run("summarize this project", temp_dir, "style-thread")
                state = HarnessState(
                    task="summarize this project",
                    workspace=temp_dir,
                    thread_id="style-thread",
                    run_id=run_id,
                )
                runtime = GraphRuntimeConfig(
                    profile_memory=profile_memory,
                    memory=project_memory,
                    auto_style_scan=True,
                )
                graph = build_graph(Nodes(client, traces, runtime), backend="simple")
                final = graph.invoke(state)
                records = TasteProfileStore(project_memory).records(scope="project")
                report = traces.render_report(run_id)
            finally:
                profile_memory.close()
                project_memory.close()

        self.assertEqual(final.status, "done")
        self.assertIn("Keep Python line length at 100 characters", client.plan_prompt)
        self.assertTrue(any(record.kind == "style" for record in records))
        self.assertIn("project_style_scanned", report)

    def test_project_style_auto_scan_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "pyproject.toml").write_text(
                "[tool.ruff]\nline-length = 100\n",
                encoding="utf-8",
            )
            profile_memory = Memory(path / "profile.db")
            project_memory = Memory(path / "memory.db")
            traces = TraceStore(path / "traces.db")
            try:
                client = CapturingClient()
                run_id = traces.start_run("summarize this project", temp_dir, "style-thread")
                state = HarnessState(
                    task="summarize this project",
                    workspace=temp_dir,
                    thread_id="style-thread",
                    run_id=run_id,
                )
                runtime = GraphRuntimeConfig(
                    profile_memory=profile_memory,
                    memory=project_memory,
                    auto_style_scan=False,
                )
                graph = build_graph(Nodes(client, traces, runtime), backend="simple")
                final = graph.invoke(state)
                records = TasteProfileStore(project_memory).records(scope="project")
            finally:
                profile_memory.close()
                project_memory.close()

        self.assertEqual(final.status, "done")
        self.assertEqual(records, [])
        self.assertNotIn("Keep Python line length at 100 characters", client.plan_prompt)

    def test_completion_marker_becomes_clean_user_output(self):
        answer = final_answer_from_action(
            render_observation(
                {
                    "status": "ok",
                    "stdout": (
                        '__RLM_FINAL_ANSWER__"Changed files: app.py'
                        '\\nVerification: pytest"\n'
                    ),
                    "stderr": "",
                }
            )
        )

        self.assertEqual(answer, "Changed files: app.py\nVerification: pytest")

    def test_verification_discovers_project_native_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "pyproject.toml").write_text(
                "[project]\nname = 'demo'\n[project.optional-dependencies]\ndev = ['pytest']\n",
                encoding="utf-8",
            )
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_ok.py").write_text(
                "def test_ok():\n    assert True\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=False)
            (workspace / "app.py").write_text("print('ok')\n", encoding="utf-8")

            result = VerificationGate(workspace).verify()

        commands = [check.command for check in result.checks]
        self.assertIn("python -m pytest -q", commands)
        self.assertTrue(result.passed)

    def test_verification_runs_root_unittest_files_without_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "test_mathlib.py").write_text(
                "import unittest\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(2 + 3, 5)\n",
                encoding="utf-8",
            )

            result = VerificationGate(workspace).verify()

        commands = [check.command for check in result.checks]
        self.assertIn("python -m unittest discover -v", commands)
        self.assertTrue(result.passed)
        self.assertIn("OK", result.summary)

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

    def test_reflect_continues_code_edit_when_output_lacks_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("fix failing tests in mathlib.py", temp_dir)
            state = HarnessState(
                task="fix failing tests in mathlib.py",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
                history=[
                    {
                        "node": "observe",
                        "content": render_observation(
                            {
                                "status": "ok",
                                "stdout": "done",
                                "stderr": "",
                                "code": "print('done')",
                            }
                        ),
                    }
                ],
            )

            final_state = Nodes(LMClient(provider="stub"), traces).reflect(state)

        self.assertEqual(final_state.status, "continue")
        self.assertFalse(final_state.final_answer)

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
                act_engine="json",
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

    def test_tool_action_engine_runs_project_summary_without_docker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Typed Harness\n\nProduction coding harness.\n",
                encoding="utf-8",
            )
            (workspace / "pyproject.toml").write_text(
                '[project]\nname = "typed-harness"\n',
                encoding="utf-8",
            )
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("what is this project", str(workspace))
            state = HarnessState(
                task="what is this project",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            client = TypedProjectSummaryClient()
            runtime = GraphRuntimeConfig(
                sandbox_enabled=False,
                act_engine="tool",
                max_iterations=3,
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            typed_events = traces.typed_events(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 0)
        self.assertIn("Project Summary", final_state.final_answer)
        self.assertTrue(any(isinstance(event, ActionSelectedEvent) for event in typed_events))
        self.assertTrue(any(isinstance(event, ObservationRecordedEvent) for event in typed_events))
        self.assertTrue(any(isinstance(event, CompletionEvent) for event in typed_events))

    def test_tool_action_engine_edits_files_when_docker_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "mathlib.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            (workspace / "test_mathlib.py").write_text(
                "import unittest\n\n"
                "from mathlib import add\n\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_adds_numbers(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            traces = TraceStore(workspace / "traces.db")
            task = "Fix the failing test in mathlib.py and report what changed."
            run_id = traces.start_run(task, str(workspace))
            state = HarnessState(
                task=task,
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                sandbox_enabled=False,
                act_engine="tool",
                max_iterations=6,
            )
            graph = build_graph(Nodes(LMClient(provider="stub"), traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            content = (workspace / "mathlib.py").read_text(encoding="utf-8")

        self.assertEqual(final_state.status, "done")
        self.assertIn("return a + b", content)
        self.assertIn("Changed files", final_state.final_answer)

    def test_tool_action_engine_retries_non_executable_action(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("finish with typed tool", str(workspace))
            state = HarnessState(
                task="finish with typed tool",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            client = BadThenGoodTypedActionClient()
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                act_engine="tool",
                max_action_retries=1,
                max_iterations=3,
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 2)
        self.assertIn("Done via typed tools.", final_state.final_answer)
        self.assertIn("action_parse_error", report)

    def test_tool_action_engine_falls_back_for_project_summary_parse_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Example\n\nA tiny project for testing fallback summaries.\n",
                encoding="utf-8",
            )
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("what is this project about", str(workspace))
            state = HarnessState(
                task="what is this project about",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            state.scratch["deterministic_project_action_used"] = True
            client = NonJsonProjectSummaryClient()
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                act_engine="tool",
                max_action_retries=0,
                max_iterations=3,
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 1)
        self.assertIn("Project Summary", final_state.final_answer)
        self.assertIn("fallback summaries", final_state.final_answer)
        self.assertIn("action_parse_recovered", report)

    def test_tool_action_engine_accepts_action_kind_alias_from_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Alias Project\n\nA tiny project for action kind alias tests.\n",
                encoding="utf-8",
            )
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("what is this project about", str(workspace))
            state = HarnessState(
                task="what is this project about",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            state.scratch["deterministic_project_action_used"] = True
            client = ActionKindProjectSummaryClient()
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                act_engine="tool",
                max_action_retries=0,
                max_iterations=3,
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 1)
        self.assertIn("Alias Project", final_state.final_answer)
        self.assertNotIn("action_parse_error", report)

    def test_tool_action_engine_enforces_ask_read_only_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            traces = TraceStore(workspace / "traces.db")
            run_id = traces.start_run("write a file", str(workspace))
            state = HarnessState(
                task="write a file",
                workspace=str(workspace),
                thread_id=run_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                sandbox_enabled=True,
                act_engine="tool",
                autonomy=AutonomyMode.ASK,
                max_iterations=3,
            )
            graph = build_graph(Nodes(TypedWriteActionClient(), traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            wrote_file = (workspace / "notes.txt").exists()

        self.assertEqual(final_state.status, "error")
        self.assertFalse(wrote_file)
        self.assertIn("read-only", final_state.final_answer)

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
        self.assertIn("rlm_runtime", report)
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
                act_engine="json",
            )
            graph = build_graph(Nodes(client, traces, runtime), backend="simple")
            final_state = graph.invoke(state)
            report = traces.render_report(run_id)

        self.assertEqual(final_state.status, "done")
        self.assertEqual(client.action_calls, 2)
        self.assertIn("123", final_state.final_answer)
        self.assertIn("action_parse_error", report)

    def test_checkpoint_save_and_resume_from_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            thread_id = "checkpoint-thread-1"
            memory = Memory(path / "memory.db")
            state = HarnessState(
                task="three step task: first summarize, then audit, then list files",
                workspace=temp_dir,
                thread_id=thread_id,
                run_id="checkpoint-run-1",
            )
            from rlm_harness.graph.checkpoint import CheckpointManager
            from rlm_harness.graph.planning import advance_to_next_step, parse_structured_plan

            state.plan = parse_structured_plan(
                "1. First step\n2. Second step\n3. Third step"
            )
            # Simulate completing step 1
            advance_to_next_step(state.plan)

            mgr = CheckpointManager(memory)
            cp = mgr.save(state)
            self.assertEqual(cp.step_id, "2")
            self.assertIn("1", cp.completed_step_ids)

            # Now simulate a resume
            loaded = mgr.load_latest(thread_id)
            self.assertIsNotNone(loaded)

            new_state = HarnessState(
                task="three step task: first summarize, then audit, then list files",
                workspace=temp_dir,
                thread_id=thread_id,
                run_id="checkpoint-run-2",
            )
            new_state.plan = parse_structured_plan(
                "1. First step\n2. Second step\n3. Third step"
            )
            new_state = CheckpointManager.resume_state(new_state, loaded)
            self.assertTrue(new_state.scratch.get("resumed_from_checkpoint"))
            self.assertEqual(new_state.plan.current_step_id, "2")
            self.assertEqual(
                new_state.plan.steps[0].status, "completed"
            )

            memory.close()

    def test_plan_resumes_from_checkpoint_when_memory_active(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            thread_id = "checkpoint-resume-graph"
            memory = Memory(path / "memory.db")
            traces = TraceStore(path / "traces.db")

            # Pre-seed a checkpoint
            from rlm_harness.graph.checkpoint import CheckpointManager
            from rlm_harness.graph.planning import advance_to_next_step, parse_structured_plan

            temp_state = HarnessState(
                task="resume me",
                workspace=temp_dir,
                thread_id=thread_id,
                run_id="seed-run",
            )
            temp_state.plan = parse_structured_plan(
                "1. Already done step\n2. Should start here\n3. Future step"
            )
            advance_to_next_step(temp_state.plan)
            mgr = CheckpointManager(memory)
            mgr.save(temp_state)

            # Now invoke the graph — plan() should detect the checkpoint and resume
            run_id = traces.start_run("resume me", temp_dir, thread_id)
            state = HarnessState(
                task="resume me",
                workspace=temp_dir,
                thread_id=thread_id,
                run_id=run_id,
            )
            runtime = GraphRuntimeConfig(
                memory=memory,
                memory_paging=MemoryPagingConfig(),
            )
            try:
                graph = build_graph(
                    Nodes(LMClient(provider="stub"), traces, runtime),
                    backend="simple",
                )
                final_state = graph.invoke(state)
            finally:
                close_getattr = getattr(graph, "close", None)
                if callable(close_getattr):
                    close_getattr()
                memory.close()

        self.assertTrue(final_state.scratch.get("resumed_from_checkpoint"))
        self.assertEqual(final_state.status, "done")

    def test_budget_exhaustion_synthesizes_partial_answer(self):
        from rlm_harness.graph.planning import parse_structured_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("multi-step task", temp_dir)
            state = HarnessState(
                task="multi-step task",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            state.plan = parse_structured_plan(
                "1. Check environment\n2. List files\n3. Run audit"
            )
            state.budget.iterations_used = 5
            state.budget.iteration_limit = 5
            state.budget.tokens_used = 99000
            state.budget.token_limit = 100000
            state.status = "continue"

            nodes = Nodes(LMClient(provider="stub"), traces)
            final_state = nodes.finalize_partial(state)

            self.assertEqual(final_state.status, "stopped")
            self.assertTrue(final_state.final_answer)
            self.assertIn("multi-step", final_state.final_answer)

    def test_partial_answer_hides_internal_reasoning(self):
        from rlm_harness.graph.planning import parse_structured_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            traces = TraceStore(Path(temp_dir) / "traces.db")
            run_id = traces.start_run("what is this project about", temp_dir)
            state = HarnessState(
                task="what is this project about",
                workspace=temp_dir,
                thread_id=run_id,
                run_id=run_id,
            )
            state.plan = parse_structured_plan(
                "1. Read README\n2. Inspect source\n3. Suggest next build"
            )
            state.history.append(
                {
                    "node": "observe",
                    "content": render_observation(
                        {
                            "status": "ok",
                            "stdout": "# Agent Kit\n\nRust CLI for local coding agents.",
                            "stderr": "",
                        }
                    ),
                }
            )

            nodes = Nodes(ReasoningPartialClient(), traces)
            final_state = nodes.finalize_partial(state)

        self.assertEqual(final_state.status, "stopped")
        self.assertIn("What I found", final_state.final_answer)
        self.assertIn("Rust CLI for local coding agents", final_state.final_answer)
        self.assertIn("What I would do next", final_state.final_answer)
        self.assertNotIn("The user wants", final_state.final_answer)
        self.assertNotIn("Budget:", final_state.final_answer)


if __name__ == "__main__":
    unittest.main()
