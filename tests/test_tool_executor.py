import contextlib
import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rlm_harness.actions import (
    ApplyPatchAction,
    ApplyPendingChangeAction,
    ClearPendingChangesAction,
    CommandObservation,
    CompleteTaskAction,
    DataObservation,
    FileObservation,
    MCPCallToolAction,
    MCPListToolsAction,
    ObservationStatus,
    PermissionObservation,
    ProjectSummaryAction,
    ProposeChangeAction,
    ReadFileAction,
    ReadFirstExistingAction,
    RunShellAction,
    TextObservation,
    WriteFileAction,
)
from rlm_harness.kernel import AutonomyMode
from rlm_harness.mcp_client import MCPClient, MCPClientError
from rlm_harness.mcp_config import MCPAuthConfig, MCPConfigStore, MCPServerConfig
from rlm_harness.sandbox import tools as sandbox_tools
from rlm_harness.tools import ToolExecutor


@contextlib.contextmanager
def mcp_test_server():
    session_id = "session-test"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.headers.get("authorization") != "Bearer secret-token":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"missing token")
                return
            if self.headers.get("MCP-Protocol-Version") != "2025-06-18":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"missing protocol version")
                return
            method = payload.get("method")
            if method != "initialize" and self.headers.get("Mcp-Session-Id") != session_id:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"missing session id")
                return
            if method == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            result = {}
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-mcp", "version": "1.0"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "lookup_doc",
                            "description": "Look up documentation",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            elif method == "tools/call":
                query = payload.get("params", {}).get("arguments", {}).get("query", "")
                result = {"content": [{"type": "text", "text": f"result for {query}"}]}

            response = {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
            self.send_response(200)
            self.send_header("content-type", "application/json")
            if method == "initialize":
                self.send_header("Mcp-Session-Id", session_id)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp"
    finally:
        server.shutdown()
        thread.join(timeout=1)


class ToolExecutorTests(unittest.TestCase):
    def test_read_file_action_returns_file_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text("# Harness\n", encoding="utf-8")

            observation = ToolExecutor(workspace).execute(ReadFileAction(path="README.md"))

        self.assertIsInstance(observation, FileObservation)
        self.assertEqual(observation.path, "README.md")
        self.assertEqual(observation.content, "# Harness\n")

    def test_write_file_requires_confirmation_before_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "notes.txt"
            action = WriteFileAction(path="notes.txt", content="hello\n")
            executor = ToolExecutor(workspace)

            denied = executor.execute(action)
            written = executor.execute(action, approved=True)
            content = target.read_text(encoding="utf-8")

        self.assertIsInstance(denied, PermissionObservation)
        self.assertEqual(denied.decision, "needs_confirmation")
        self.assertIsInstance(written, TextObservation)
        self.assertEqual(content, "hello\n")

    def test_ask_mode_denies_write_even_when_approved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            action = WriteFileAction(path="notes.txt", content="hello\n")

            observation = ToolExecutor(workspace, autonomy=AutonomyMode.ASK).execute(
                action,
                approved=True,
            )

        self.assertIsInstance(observation, PermissionObservation)
        self.assertEqual(observation.decision, "denied")
        self.assertIn("read-only", observation.reason)

    def test_proposal_and_apply_pending_change_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "app.py"
            target.write_text("print('old')\n", encoding="utf-8")
            executor = ToolExecutor(workspace)
            try:
                proposal = executor.execute(
                    ProposeChangeAction(
                        path="app.py",
                        content="print('new')\n",
                        reason="test",
                    )
                )
                change_id = proposal.data["id"]
                denied = executor.execute(ApplyPendingChangeAction(change_id=change_id))
                applied = executor.execute(
                    ApplyPendingChangeAction(change_id=change_id),
                    approved=True,
                )
                content = target.read_text(encoding="utf-8")
            finally:
                executor.execute(ClearPendingChangesAction())

        self.assertIsInstance(proposal, DataObservation)
        self.assertTrue(proposal.data["approval_required"])
        self.assertIsInstance(denied, PermissionObservation)
        self.assertIsInstance(applied, TextObservation)
        self.assertEqual(content, "print('new')\n")

    def test_run_shell_maps_return_codes_to_command_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ToolExecutor(Path(temp_dir))

            ok = executor.execute(RunShellAction(command="printf ok"))
            failed = executor.execute(RunShellAction(command="exit 7"))

        self.assertIsInstance(ok, CommandObservation)
        self.assertEqual(ok.status, ObservationStatus.OK)
        self.assertEqual(ok.stdout, "ok")
        self.assertEqual(failed.status, ObservationStatus.ERROR)
        self.assertEqual(failed.exit_code, 7)

    def test_apply_patch_reports_changed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "app.py").write_text("print('old')\n", encoding="utf-8")
            patch = (
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1 +1 @@\n"
                "-print('old')\n"
                "+print('new')\n"
            )

            observation = ToolExecutor(workspace).execute(ApplyPatchAction(diff=patch))

        self.assertEqual(observation.kind, "patch")
        self.assertEqual(observation.changed_files, ["app.py"])
        self.assertEqual(observation.diff_summary, "patch applied")

    def test_apply_patch_targets_nested_workspace_inside_parent_git_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            workspace = repo / "eval-work" / "case"
            workspace.mkdir(parents=True)
            target = workspace / "app.py"
            target.write_text("print('old')\n", encoding="utf-8")
            patch = (
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1 +1 @@\n"
                "-print('old')\n"
                "+print('new')\n"
            )

            observation = ToolExecutor(workspace).execute(ApplyPatchAction(diff=patch))
            content = target.read_text(encoding="utf-8")

        self.assertEqual(observation.kind, "patch")
        self.assertEqual(content, "print('new')\n")

    def test_destructive_shell_command_returns_error_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            observation = ToolExecutor(Path(temp_dir)).execute(
                RunShellAction(command="rm -rf .")
            )

        self.assertEqual(observation.kind, "error")
        self.assertIn("destructive", observation.message)

    def test_project_summary_action_uses_workspace_context_and_restores_it(self):
        old_workspace = sandbox_tools.WORKSPACE
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text(
                "# Example\n\nA tiny Python package.\n",
                encoding="utf-8",
            )
            (workspace / "pyproject.toml").write_text(
                '[project]\nname = "example"\ndescription = "Tiny package."\n',
                encoding="utf-8",
            )

            observation = ToolExecutor(workspace).execute(ProjectSummaryAction())

        self.assertIs(sandbox_tools.WORKSPACE, old_workspace)
        self.assertIsInstance(observation, TextObservation)
        self.assertIn("Project Summary", observation.text)
        self.assertIn("example", observation.text.lower())

    def test_read_first_existing_uses_matched_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            observation = ToolExecutor(workspace).execute(
                ReadFirstExistingAction(paths=["missing.txt", "pyproject.toml"])
            )

        self.assertIsInstance(observation, FileObservation)
        self.assertEqual(observation.path, "pyproject.toml")
        self.assertEqual(observation.content, "[project]\n")

    def test_complete_task_action_returns_text_observation(self):
        observation = ToolExecutor(Path.cwd()).execute(
            CompleteTaskAction(summary="All set.", verification="pytest")
        )

        self.assertIsInstance(observation, TextObservation)
        self.assertEqual(observation.text, "All set.")
        self.assertEqual(observation.summary, "success")

    def test_mcp_list_tools_uses_authenticated_http_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MCPConfigStore(Path(temp_dir) / "mcp.json")
            with mcp_test_server() as server_url:
                store.add(
                    MCPServerConfig(
                        name="docs",
                        transport="http",
                        url=server_url,
                        auth=MCPAuthConfig(type="bearer_env", token_env="HARNESS_TEST_MCP_TOKEN"),
                        purposes=["docs"],
                    )
                )

                with unittest.mock.patch.dict(
                    "os.environ",
                    {"HARNESS_TEST_MCP_TOKEN": "secret-token"},
                    clear=False,
                ):
                    observation = ToolExecutor(Path(temp_dir), mcp_store=store).execute(
                        MCPListToolsAction(purpose="docs")
                    )

        self.assertIsInstance(observation, DataObservation)
        self.assertEqual(observation.data["server"], "docs")
        self.assertEqual(observation.data["tools"][0]["name"], "lookup_doc")

    def test_mcp_list_tools_uses_single_enabled_server_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MCPConfigStore(Path(temp_dir) / "mcp.json")
            with mcp_test_server() as server_url:
                store.add(
                    MCPServerConfig(
                        name="docs",
                        transport="http",
                        url=server_url,
                        auth=MCPAuthConfig(type="bearer_env", token_env="HARNESS_TEST_MCP_TOKEN"),
                        purposes=["docs"],
                    )
                )

                with unittest.mock.patch.dict(
                    "os.environ",
                    {"HARNESS_TEST_MCP_TOKEN": "secret-token"},
                    clear=False,
                ):
                    observation = ToolExecutor(Path(temp_dir), mcp_store=store).execute(
                        MCPListToolsAction()
                    )

        self.assertIsInstance(observation, DataObservation)
        self.assertEqual(observation.data["server"], "docs")
        self.assertEqual(observation.data["tools"][0]["name"], "lookup_doc")

    def test_mcp_call_tool_requires_trusted_server_then_returns_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MCPConfigStore(Path(temp_dir) / "mcp.json")
            with mcp_test_server() as server_url:
                store.add(
                    MCPServerConfig(
                        name="docs",
                        transport="http",
                        url=server_url,
                        auth=MCPAuthConfig(type="bearer_env", token_env="HARNESS_TEST_MCP_TOKEN"),
                        purposes=["docs"],
                        trusted=False,
                    )
                )
                with unittest.mock.patch.dict(
                    "os.environ",
                    {"HARNESS_TEST_MCP_TOKEN": "secret-token"},
                    clear=False,
                ):
                    executor = ToolExecutor(Path(temp_dir), mcp_store=store)
                    gated = executor.execute(
                        MCPCallToolAction(
                            server="docs",
                            tool_name="lookup_doc",
                            arguments={"query": "auth"},
                        )
                    )
                    trusted = executor.execute(
                        MCPCallToolAction(
                            server="docs",
                            tool_name="lookup_doc",
                            arguments={"query": "auth"},
                        ),
                        approved=True,
                    )

        self.assertIsInstance(gated, PermissionObservation)
        self.assertEqual(gated.decision, "needs_confirmation")
        self.assertIsInstance(trusted, DataObservation)
        self.assertEqual(trusted.data["tool"], "lookup_doc")
        self.assertEqual(trusted.data["result"]["content"][0]["text"], "result for auth")

    def test_mcp_list_and_call_tool_uses_stdio_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            server_script = workspace / "stdio_mcp_server.py"
            server_script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    for line in sys.stdin:
                        payload = json.loads(line)
                        method = payload.get("method")
                        if method == "notifications/initialized":
                            continue
                        if method == "initialize":
                            result = {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "stdio-test", "version": "1.0"},
                            }
                        elif method == "tools/list":
                            result = {
                                "tools": [
                                    {
                                        "name": "lookup_doc",
                                        "description": "Look up local docs",
                                        "inputSchema": {"type": "object"},
                                    }
                                ]
                            }
                        elif method == "tools/call":
                            query = payload.get("params", {}).get("arguments", {}).get("query", "")
                            result = {
                                "content": [{"type": "text", "text": f"local result for {query}"}]
                            }
                        else:
                            result = {}
                        print(
                            json.dumps(
                                {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
                            ),
                            flush=True,
                        )
                    """
                ),
                encoding="utf-8",
            )
            store = MCPConfigStore(workspace / "mcp.json")
            store.add(
                MCPServerConfig(
                    name="local-docs",
                    transport="stdio",
                    command=sys.executable,
                    args=[str(server_script)],
                    purposes=["docs"],
                    trusted=True,
                )
            )
            executor = ToolExecutor(workspace, mcp_store=store)

            tools = executor.execute(MCPListToolsAction(purpose="docs"))
            result = executor.execute(
                MCPCallToolAction(
                    server="local-docs",
                    tool_name="lookup_doc",
                    arguments={"query": "stdio"},
                )
            )

        self.assertIsInstance(tools, DataObservation)
        self.assertEqual(tools.data["server"], "local-docs")
        self.assertEqual(tools.data["tools"][0]["name"], "lookup_doc")
        self.assertIsInstance(result, DataObservation)
        self.assertEqual(result.data["result"]["content"][0]["text"], "local result for stdio")

    def test_mcp_stdio_timeout_returns_action_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            server_script = workspace / "silent_mcp_server.py"
            server_script.write_text(
                "import sys, time\nfor _line in sys.stdin:\n    time.sleep(5)\n",
                encoding="utf-8",
            )
            client = MCPClient(
                MCPServerConfig(
                    name="silent",
                    transport="stdio",
                    command=sys.executable,
                    args=[str(server_script)],
                ),
                timeout_s=0.1,
            )

            with self.assertRaises(MCPClientError) as raised:
                client.list_tools()

        self.assertIn("timed out", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
