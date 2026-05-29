from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import tomllib

from rlm_harness import cli, config, mcp_cli
from rlm_harness.kernel import AutonomyMode
from rlm_harness.tracing import TraceStore


class CLIConfigTests(unittest.TestCase):
    def test_parser_uses_harness_as_user_facing_command(self):
        help_stdout = io.StringIO()
        with contextlib.redirect_stdout(help_stdout):
            exit_code = cli.main([])

        help_text = help_stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("usage: harness", help_text)
        self.assertIn('harness "fix tests"', help_text)
        self.assertNotIn("usage: rlm-harness", help_text)

    def test_default_task_alias_still_normalizes_to_run(self):
        self.assertEqual(cli.normalize_argv(["fix tests"]), ["run", "fix tests"])
        self.assertEqual(cli.normalize_argv(["run", "fix tests"]), ["run", "fix tests"])
        self.assertEqual(cli.normalize_argv(["ask", "what is this"]), ["ask", "what is this"])
        self.assertEqual(cli.normalize_argv(["plan", "fix this"]), ["plan", "fix this"])
        self.assertEqual(cli.normalize_argv(["work", "fix tests"]), ["work", "fix tests"])
        self.assertEqual(cli.normalize_argv(["commands"]), ["commands"])
        self.assertEqual(cli.normalize_argv(["/"]), ["palette"])
        self.assertEqual(cli.normalize_argv(["/help"]), ["palette"])
        self.assertEqual(cli.normalize_argv(["continue"]), ["continue"])
        self.assertEqual(cli.normalize_argv(["/continue"]), ["continue"])
        self.assertEqual(cli.normalize_argv(["--continue", "next"]), ["continue", "next"])
        self.assertEqual(cli.normalize_argv(["-c", "next"]), ["continue", "next"])
        self.assertEqual(cli.normalize_argv(["-p", "fix tests"]), ["run", "fix tests"])
        self.assertEqual(cli.normalize_argv(["--print", "fix tests"]), ["run", "fix tests"])
        self.assertEqual(cli.normalize_argv(["--plan", "fix tests"]), ["plan", "fix tests"])
        self.assertEqual(
            cli.normalize_argv(["--permission-mode", "standard", "fix tests"]),
            ["run", "fix tests", "--permission-mode", "standard"],
        )
        self.assertEqual(
            cli.normalize_argv(["--permission-mode", "plan", "fix tests"]),
            ["plan", "fix tests"],
        )
        self.assertEqual(
            cli.normalize_argv(["--auto-accept", "--model", "stub", "fix tests"]),
            ["run", "fix tests", "--auto-accept", "--model", "stub"],
        )
        self.assertEqual(cli.normalize_argv(["--list-models"]), ["model"])
        self.assertEqual(cli.normalize_argv(["tools"]), ["tools"])
        self.assertEqual(cli.normalize_argv(["status"]), ["status"])
        self.assertEqual(cli.normalize_argv(["/status"]), ["status"])
        self.assertEqual(cli.normalize_argv(["taste"]), ["taste"])
        self.assertEqual(cli.normalize_argv(["/taste", "list"]), ["taste", "list"])
        self.assertEqual(cli.normalize_argv(["update"]), ["update"])
        self.assertEqual(cli.normalize_argv(["/update"]), ["update"])
        self.assertEqual(cli.normalize_argv(["--help"]), ["--help"])
        self.assertEqual(cli.normalize_argv(["/model", "custom/coder"]), ["model", "custom/coder"])

    def test_commands_lists_clean_public_surface(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(["commands"]), 0)

        text = stdout.getvalue()
        self.assertIn("Harness commands", text)
        self.assertIn('harness ask "what is this project?"', text)
        self.assertIn('harness plan "how should we fix this?"', text)
        self.assertIn('harness "fix tests"', text)
        self.assertIn('harness work "fix tests"', text)
        self.assertIn("harness continue [task]", text)
        self.assertIn("harness trace list|report|events", text)
        self.assertIn("harness status", text)
        self.assertIn("harness tools", text)
        self.assertIn("harness /", text)
        self.assertIn("harness init [--provider name] [--api-key key]", text)
        self.assertIn("harness profile list|context|learn|scan|approve|reject", text)
        self.assertIn("harness taste list|context|learn|scan|approve|reject", text)
        self.assertIn("Tip:", text)

    def test_commands_json_is_agent_readable(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(["commands", "--json"]), 0)

        commands = json.loads(stdout.getvalue())
        names = {command["name"] for command in commands}

        self.assertIn("ask", names)
        self.assertIn("plan", names)
        self.assertIn("run", names)
        self.assertIn("work", names)
        self.assertIn("continue", names)
        self.assertIn("trace", names)
        self.assertIn("status", names)
        self.assertIn("mcp", names)
        self.assertIn("palette", names)
        self.assertIn("init", names)
        self.assertIn("profile", names)
        self.assertIn("taste", names)
        self.assertNotIn("sandbox", names)

    def test_trace_commands_are_registered_and_agent_readable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_db = Path(tmpdir) / "traces.db"
            traces = TraceStore(trace_db)
            run_id = traces.start_run("trace task", tmpdir, thread_id="thread-trace")
            traces.event(run_id, "final", {"final_answer": "done"}, node="done")
            traces.finish_run(run_id, "done")

            list_stdout = io.StringIO()
            with contextlib.redirect_stdout(list_stdout):
                list_exit = cli.main(
                    [
                        "trace",
                        "--trace-db",
                        str(trace_db),
                        "list",
                        "--thread-id",
                        "thread-trace",
                        "--json",
                    ]
                )
            report_stdout = io.StringIO()
            with contextlib.redirect_stdout(report_stdout):
                report_exit = cli.main(
                    ["trace", "--trace-db", str(trace_db), "report", run_id, "--json"]
                )
            events_stdout = io.StringIO()
            with contextlib.redirect_stdout(events_stdout):
                events_exit = cli.main(
                    ["trace", "--trace-db", str(trace_db), "events", run_id, "--json"]
                )

        runs = json.loads(list_stdout.getvalue())
        report = json.loads(report_stdout.getvalue())
        events = json.loads(events_stdout.getvalue())

        self.assertEqual(list_exit, 0)
        self.assertEqual(report_exit, 0)
        self.assertEqual(events_exit, 0)
        self.assertEqual(runs[0]["run_id"], run_id)
        self.assertEqual(report["thread_id"], "thread-trace")
        self.assertEqual(report["final_answer"], "done")
        self.assertEqual(events[0]["kind"], "final")

    def test_slash_palette_lists_commands_and_tools(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(["/"]), 0)

        text = stdout.getvalue()
        self.assertIn("Harness slash palette", text)
        self.assertIn('/ask "what is this project?"', text)
        self.assertIn('/plan "how should we fix this?"', text)
        self.assertIn('/run "fix tests"', text)
        self.assertIn("/tools", text)
        self.assertIn("/mcp list|setup|add|show|tools|trust|enable", text)
        self.assertIn("/init [--provider name] [--api-key key]", text)
        self.assertIn("/sandbox build|run", text)
        self.assertIn("Harness tools", text)
        self.assertIn("read_file [read]", text)
        self.assertIn("project_summary [read]", text)
        self.assertIn("complete_task [low]", text)
        self.assertIn("python_repl [medium]", text)

    def test_slash_palette_json_lists_commands_and_tools(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(["/help", "--json"]), 0)

        payload = json.loads(stdout.getvalue())
        command_names = {command["name"] for command in payload["commands"]}
        tool_names = {tool["name"] for tool in payload["tools"]}

        self.assertIn("ask", command_names)
        self.assertIn("mcp", command_names)
        self.assertIn("palette", command_names)
        self.assertIn("sandbox", command_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("project_summary", tool_names)
        self.assertIn("complete_task", tool_names)
        self.assertIn("python_repl", tool_names)

    def test_commands_and_palette_use_cyan_blue_when_color_is_forced(self):
        for command in (["commands"], ["/"]):
            with self.subTest(command=command):
                stdout = io.StringIO()
                with patch.dict(os.environ, {"HARNESS_COLOR": "on"}, clear=False), (
                    contextlib.redirect_stdout(stdout)
                ):
                    self.assertEqual(cli.main(command), 0)

                text = stdout.getvalue()
                self.assertIn("\033[96m", text)
                self.assertIn("\033[94m", text)

    def test_interactive_slash_prints_full_palette(self):
        stdout = io.StringIO()
        with patch("builtins.input", side_effect=["/", "/q"]), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.interactive_loop(), 0)

        text = stdout.getvalue()
        self.assertIn("Harness interactive mode", text)
        self.assertIn("Harness slash palette", text)
        self.assertIn("/mcp list|setup|add|show|tools|trust|enable", text)
        self.assertIn("python_repl [medium]", text)

    def test_interactive_slash_command_dispatches_with_arguments(self):
        with patch("builtins.input", side_effect=["/mcp list --json", "/q"]), patch.object(
            cli,
            "main",
            return_value=0,
        ) as main:
            self.assertEqual(cli.interactive_loop(), 0)

        main.assert_called_once_with(["mcp", "list", "--json"])

    def test_tools_lists_capability_catalog(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(["tools"]), 0)

        text = stdout.getvalue()
        self.assertIn("Harness tools", text)
        self.assertIn("project_summary [read]", text)
        self.assertIn("write_file [high] (confirmation)", text)
        self.assertNotIn("python_repl", text)

    def test_tools_json_is_agent_readable(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.main(["tools", "--json"]), 0)

        tools = json.loads(stdout.getvalue())
        by_name = {tool["name"]: tool for tool in tools}

        self.assertEqual(by_name["read_file"]["risk"], "read")
        self.assertTrue(by_name["write_file"]["requires_confirmation"])
        self.assertEqual(by_name["complete_task"]["side_effect"], "completion")
        self.assertNotIn("python_repl", by_name)

    def test_mcp_command_adds_and_lists_authenticated_purpose_server(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                add_exit = cli.main(
                    [
                        "mcp",
                        "--mcp-config",
                        str(config_path),
                        "add",
                        "github",
                        "--transport",
                        "http",
                        "--url",
                        "https://mcp.example/github",
                        "--auth",
                        "bearer_env",
                        "--token-env",
                        "HARNESS_TEST_GITHUB_TOKEN",
                        "--purpose",
                        "github",
                        "--trusted",
                    ]
                )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.dict(
                os.environ,
                {"HARNESS_TEST_GITHUB_TOKEN": ""},
                clear=False,
            ):
                list_exit = cli.main(
                    ["mcp", "--mcp-config", str(config_path), "list", "--json"]
                )
            servers = json.loads(stdout.getvalue())

        self.assertEqual(add_exit, 0)
        self.assertEqual(list_exit, 0)
        self.assertEqual(servers[0]["name"], "github")
        self.assertEqual(servers[0]["transport"], "http")
        self.assertEqual(servers[0]["purposes"], ["github"])
        self.assertEqual(
            servers[0]["auth"],
            "bearer_env via $HARNESS_TEST_GITHUB_TOKEN (missing)",
        )

    def test_mcp_command_adds_local_stdio_server_with_args_and_env(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                add_exit = cli.main(
                    [
                        "mcp",
                        "--mcp-config",
                        str(config_path),
                        "add",
                        "filesystem",
                        "--transport",
                        "stdio",
                        "--command",
                        "python",
                        "--env",
                        "ROOT=/tmp/project",
                        "--purpose",
                        "files",
                        "--args",
                        "-m",
                        "mcp_server_filesystem",
                    ]
                )
            server = cli.MCPConfigStore(config_path).get("filesystem")

        self.assertEqual(add_exit, 0)
        self.assertIsNotNone(server)
        assert server is not None
        self.assertEqual(server.transport, "stdio")
        self.assertEqual(server.command, "python")
        self.assertEqual(server.args, ["-m", "mcp_server_filesystem"])
        self.assertEqual(server.env, {"ROOT": "/tmp/project"})
        self.assertEqual(server.purposes, ["files"])

    def test_mcp_setup_guides_authenticated_purpose_server(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            setup_stdout = io.StringIO()
            with (
                patch("sys.stdin.isatty", return_value=True),
                patch(
                    "builtins.input",
                    side_effect=[
                        "",
                        "https://mcp.example/docs",
                        "",
                        "",
                        "",
                        "n",
                    ],
                ),
                contextlib.redirect_stdout(setup_stdout),
            ):
                setup_exit = cli.main(
                    [
                        "mcp",
                        "--mcp-config",
                        str(config_path),
                        "setup",
                        "docs",
                    ]
                )
            list_stdout = io.StringIO()
            with contextlib.redirect_stdout(list_stdout), patch.dict(os.environ, {}, clear=True):
                list_exit = cli.main(
                    ["mcp", "--mcp-config", str(config_path), "list", "--json"]
                )
            servers = json.loads(list_stdout.getvalue())

        self.assertEqual(setup_exit, 0)
        self.assertEqual(list_exit, 0)
        self.assertIn("Harness MCP setup", setup_stdout.getvalue())
        self.assertEqual(servers[0]["name"], "docs")
        self.assertEqual(servers[0]["transport"], "http")
        self.assertEqual(servers[0]["url"], "https://mcp.example/docs")
        self.assertEqual(servers[0]["purposes"], ["docs"])
        self.assertFalse(servers[0]["trusted"])
        self.assertEqual(servers[0]["auth"], "bearer_env via $DOCS_TOKEN (missing)")

    def test_mcp_setup_explains_non_interactive_fallback(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), patch(
                "sys.stdin.isatty",
                return_value=False,
            ):
                exit_code = cli.main(
                    [
                        "mcp",
                        "--mcp-config",
                        str(Path(tmpdir) / "mcp.json"),
                        "setup",
                        "docs",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("Use `harness mcp add ...`", stderr.getvalue())

    def test_mcp_command_updates_trust_and_enabled_state(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "mcp",
                            "--mcp-config",
                            str(config_path),
                            "add",
                            "docs",
                            "--transport",
                            "http",
                            "--url",
                            "https://mcp.example/docs",
                            "--purpose",
                            "docs",
                            "--disabled",
                        ]
                    ),
                    0,
                )
            trust_stdout = io.StringIO()
            with contextlib.redirect_stdout(trust_stdout):
                trust_exit = cli.main(
                    ["mcp", "--mcp-config", str(config_path), "trust", "docs", "--json"]
                )
            enable_stdout = io.StringIO()
            with contextlib.redirect_stdout(enable_stdout):
                enable_exit = cli.main(
                    ["mcp", "--mcp-config", str(config_path), "enable", "docs", "--json"]
                )
            trusted = json.loads(trust_stdout.getvalue())
            enabled = json.loads(enable_stdout.getvalue())

        self.assertEqual(trust_exit, 0)
        self.assertEqual(enable_exit, 0)
        self.assertTrue(trusted["trusted"])
        self.assertFalse(trusted["enabled"])
        self.assertTrue(enabled["trusted"])
        self.assertTrue(enabled["enabled"])

    def test_mcp_tools_command_lists_remote_tools_by_purpose(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "mcp",
                            "--mcp-config",
                            str(config_path),
                            "add",
                            "docs",
                            "--transport",
                            "http",
                            "--url",
                            "https://mcp.example/docs",
                            "--auth",
                            "bearer_env",
                            "--token-env",
                            "DOCS_TOKEN",
                            "--purpose",
                            "docs",
                        ]
                    ),
                    0,
                )
            stdout = io.StringIO()
            with patch.object(
                mcp_cli.MCPClient,
                "list_tools",
                return_value={
                    "tools": [
                        {
                            "name": "lookup_doc",
                            "description": "Look up documentation",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            ) as list_tools, contextlib.redirect_stdout(stdout), patch.dict(
                os.environ,
                {"DOCS_TOKEN": "secret-token"},
                clear=False,
            ):
                exit_code = cli.main(
                    [
                        "mcp",
                        "--mcp-config",
                        str(config_path),
                        "tools",
                        "--purpose",
                        "docs",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        list_tools.assert_called_once()
        self.assertEqual(payload["server"], "docs")
        self.assertEqual(payload["tools"][0]["name"], "lookup_doc")

    def test_matching_mcp_servers_are_loaded_into_runtime_context(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "mcp",
                            "--mcp-config",
                            str(config_path),
                            "add",
                            "linear",
                            "--transport",
                            "http",
                            "--url",
                            "https://mcp.example/linear",
                            "--auth",
                            "bearer_env",
                            "--token-env",
                            "LINEAR_TOKEN",
                            "--purpose",
                            "linear",
                        ]
                    ),
                    0,
                )
            parsed = cli.parser().parse_args(
                [
                    "run",
                    "summarize the linear issue",
                    "--mcp-config",
                    str(config_path),
                ]
            )
            with patch.dict(os.environ, {"LINEAR_TOKEN": "secret-token"}, clear=False):
                runtime = cli.build_runtime(
                    parsed,
                    Path(tmpdir),
                    None,
                    None,
                    task="summarize the linear issue",
                )

        self.assertIn("MCP servers selected for this task", runtime.mcp_context)
        self.assertIn("linear", runtime.mcp_context)
        self.assertIn("bearer_env via $LINEAR_TOKEN (present)", runtime.mcp_context)

    def test_enabled_mcp_purpose_routes_are_visible_without_exact_task_match(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "mcp",
                            "--mcp-config",
                            str(config_path),
                            "add",
                            "github",
                            "--transport",
                            "http",
                            "--url",
                            "https://mcp.example/github",
                            "--auth",
                            "bearer_env",
                            "--token-env",
                            "GITHUB_TOKEN",
                            "--purpose",
                            "github",
                        ]
                    ),
                    0,
                )
            parsed = cli.parser().parse_args(
                [
                    "run",
                    "review issue 42 and summarize the next action",
                    "--mcp-config",
                    str(config_path),
                ]
            )
            with patch.dict(os.environ, {"GITHUB_TOKEN": "secret-token"}, clear=False):
                runtime = cli.build_runtime(
                    parsed,
                    Path(tmpdir),
                    None,
                    None,
                    task="review issue 42 and summarize the next action",
                )

        self.assertIn("MCP servers available for designated workflow purposes", runtime.mcp_context)
        self.assertIn("github", runtime.mcp_context)
        self.assertIn("mcp_list_tools", runtime.mcp_context)
        self.assertIn("bearer_env via $GITHUB_TOKEN (present)", runtime.mcp_context)

    def test_taste_command_learns_and_lists_records(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_db = Path(tmpdir) / "profile.db"
            memory_db = Path(tmpdir) / "memory.db"
            learn_stdout = io.StringIO()
            with contextlib.redirect_stdout(learn_stdout):
                learn_exit = cli.main(
                    [
                        "taste",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "learn",
                        "Prefer small reviewable diffs.",
                        "--active",
                    ]
                )

            list_stdout = io.StringIO()
            with contextlib.redirect_stdout(list_stdout):
                list_exit = cli.main(
                    [
                        "/taste",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "list",
                        "--json",
                    ]
                )
            records = json.loads(list_stdout.getvalue())

        self.assertEqual(learn_exit, 0)
        self.assertEqual(list_exit, 0)
        self.assertIn("active", learn_stdout.getvalue())
        self.assertEqual(records[0]["text"], "Prefer small reviewable diffs.")
        self.assertEqual(records[0]["status"], "active")

    def test_taste_context_shows_prompt_context_for_active_preferences(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_db = Path(tmpdir) / "profile.db"
            memory_db = Path(tmpdir) / "memory.db"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "taste",
                            "--profile-db",
                            str(profile_db),
                            "--memory-db",
                            str(memory_db),
                            "learn",
                            "Prefer concise final answers.",
                            "--active",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    cli.main(
                        [
                            "taste",
                            "--profile-db",
                            str(profile_db),
                            "--memory-db",
                            str(memory_db),
                            "learn",
                            "Run `pytest` before final response.",
                            "--scope",
                            "project",
                            "--kind",
                            "verification_command",
                            "--active",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    cli.main(
                        [
                            "evolve",
                            "--profile-db",
                            str(profile_db),
                            "--memory-db",
                            str(memory_db),
                            "propose",
                            "--title",
                            "Concise summaries",
                            "--body",
                            "Keep final answers short unless asked for detail.",
                            "--rationale",
                            "taste context test",
                            "--approved",
                        ]
                    ),
                    0,
                )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "taste",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "context",
                    ]
                )
            text = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Taste context:", text)
        self.assertIn("User taste:", text)
        self.assertIn("Prefer concise final answers.", text)
        self.assertIn("Project conventions:", text)
        self.assertIn("Run `pytest` before final response.", text)
        self.assertIn("Approved user evolution:", text)
        self.assertIn("Keep final answers short unless asked for detail.", text)

    def test_taste_context_json_is_agent_readable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_db = Path(tmpdir) / "profile.db"
            memory_db = Path(tmpdir) / "memory.db"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "taste",
                            "--profile-db",
                            str(profile_db),
                            "--memory-db",
                            str(memory_db),
                            "learn",
                            "Prefer small reviewable diffs.",
                            "--active",
                        ]
                    ),
                    0,
                )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "profile",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "context",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertFalse(payload["empty"])
        self.assertIn("Taste context:", payload["context"])
        self.assertIn("Prefer small reviewable diffs.", payload["context"])

    def test_taste_scan_learns_project_style_records(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text(
                "[project]\n"
                "dependencies = ['pydantic>=2']\n"
                "[project.optional-dependencies]\n"
                "dev = ['pytest>=8', 'ruff>=0.6']\n"
                "[tool.ruff]\n"
                "line-length = 100\n",
                encoding="utf-8",
            )
            (root / "app.py").write_text(
                "def hello():\n"
                "    message = \"hello\"\n"
                "    return message\n",
                encoding="utf-8",
            )
            profile_db = root / "profile.db"
            memory_db = root / "memory.db"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "taste",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "scan",
                        "--workspace",
                        str(root),
                        "--json",
                    ]
                )
            records = json.loads(stdout.getvalue())
            reject_stdout = io.StringIO()
            with contextlib.redirect_stdout(reject_stdout):
                reject_exit = cli.main(
                    [
                        "taste",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "reject",
                        records[0]["id"],
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(reject_exit, 0)
        texts = {record["text"] for record in records}
        self.assertIn("Keep Python line length at 100 characters.", texts)
        self.assertIn("Run `pytest` for Python test verification.", texts)
        self.assertTrue(all(record["scope"] == "project" for record in records))
        self.assertIn("rejected", reject_stdout.getvalue())

    def test_taste_scan_learns_editorconfig_prettier_and_package_manager(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".editorconfig").write_text(
                "root = true\n"
                "[*]\n"
                "indent_style = space\n"
                "indent_size = 2\n"
                "max_line_length = 100\n"
                "end_of_line = lf\n",
                encoding="utf-8",
            )
            (root / ".prettierrc").write_text(
                json.dumps(
                    {
                        "printWidth": 100,
                        "singleQuote": True,
                        "semi": False,
                        "trailingComma": "all",
                    }
                ),
                encoding="utf-8",
            )
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "packageManager": "pnpm@9.0.0",
                        "scripts": {
                            "test": "vitest run",
                            "lint": "eslint .",
                            "typecheck": "tsc --noEmit",
                        },
                    }
                ),
                encoding="utf-8",
            )
            profile_db = root / "profile.db"
            memory_db = root / "memory.db"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "taste",
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "scan",
                        "--workspace",
                        str(root),
                        "--json",
                    ]
                )
            records = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        texts = {record["text"] for record in records}
        self.assertIn("Use 2-space indentation where .editorconfig applies.", texts)
        self.assertIn(
            "Keep line length at 100 characters where .editorconfig applies.",
            texts,
        )
        self.assertIn("Use LF line endings where .editorconfig applies.", texts)
        self.assertIn("Use pnpm for JavaScript package commands.", texts)
        self.assertIn("Run `pnpm test` for test verification.", texts)
        self.assertIn("Run `pnpm lint` for lint verification.", texts)
        self.assertIn("Use Prettier for JavaScript and TypeScript formatting.", texts)
        self.assertIn(
            "Keep JavaScript and TypeScript line length at 100 characters.",
            texts,
        )
        self.assertIn(
            "Prefer single quotes in JavaScript and TypeScript when either quote works.",
            texts,
        )
        self.assertIn(
            "Omit semicolons in JavaScript and TypeScript where optional.",
            texts,
        )

    def test_init_saves_provider_scans_style_and_reports_readiness_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            profile_db = root / "profile.db"
            memory_db = root / "memory.db"
            mcp_config = root / "mcp.json"
            (root / "pyproject.toml").write_text(
                "[project]\n"
                "dependencies = ['pydantic>=2']\n"
                "[project.optional-dependencies]\n"
                "dev = ['pytest>=8', 'ruff>=0.6']\n"
                "[tool.ruff]\n"
                "line-length = 100\n",
                encoding="utf-8",
            )
            (root / "app.py").write_text(
                "def hello():\n"
                "    message = \"hello\"\n"
                "    return message\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "mcp",
                            "--mcp-config",
                            str(mcp_config),
                            "add",
                            "github",
                            "--transport",
                            "http",
                            "--url",
                            "https://mcp.example/github",
                            "--auth",
                            "bearer_env",
                            "--token-env",
                            "GITHUB_TOKEN",
                            "--purpose",
                            "github",
                            "--trusted",
                        ]
                    ),
                    0,
                )

            stdout = io.StringIO()
            with (
                patch.object(config, "CONFIG_PATH", config_path),
                patch.object(cli, "CONFIG_PATH", config_path),
                patch("rlm_harness.readiness.CONFIG_PATH", config_path),
                patch.dict(os.environ, {"GITHUB_TOKEN": "secret-token"}, clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli.main(
                    [
                        "init",
                        "--workspace",
                        str(root),
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--memory-db",
                        str(memory_db),
                        "--profile-db",
                        str(profile_db),
                        "--mcp-config",
                        str(mcp_config),
                        "--no-docker",
                        "--json",
                    ]
            )
            payload = json.loads(stdout.getvalue())
            saved = config.load_user_config(config_path)
            memory_created = memory_db.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["config_updated"])
        self.assertEqual(saved["provider"], "stub")
        self.assertEqual(saved["model"], "stub")
        self.assertEqual(payload["config"]["provider"], "stub")
        self.assertEqual(payload["config"]["mcp_config_path"], str(mcp_config))
        self.assertEqual(payload["config"]["profile_path"], str(profile_db))
        self.assertEqual(payload["mcp"]["configured"], 1)
        self.assertEqual(payload["mcp"]["authenticated"], 1)
        self.assertEqual(payload["mcp"]["credentials_present"], 1)
        self.assertIn(payload["readiness"]["status"], {"ready", "degraded", "needs_setup"})
        texts = {record["text"] for record in payload["style_records"]}
        self.assertIn("Keep Python line length at 100 characters.", texts)
        self.assertIn("Run `pytest` for Python test verification.", texts)
        self.assertTrue(memory_created)

    def test_init_can_skip_style_scan(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            memory_db = root / "memory.db"
            stdout = io.StringIO()
            with (
                patch.object(config, "CONFIG_PATH", config_path),
                patch.object(cli, "CONFIG_PATH", config_path),
                patch("rlm_harness.readiness.CONFIG_PATH", config_path),
                patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli.main(
                    [
                        "init",
                        "--workspace",
                        str(root),
                        "--memory-db",
                        str(memory_db),
                        "--no-style-scan",
                        "--no-docker",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            memory_created = memory_db.exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["style_records"], [])
        self.assertFalse(memory_created)

    def test_init_rejects_unknown_provider(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli.main(
                    [
                        "init",
                        "--workspace",
                        tmpdir,
                        "--provider",
                        "not-a-provider",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("unknown provider", stderr.getvalue())

    def test_status_command_reports_latest_run_taste_and_evolution(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_db = root / "traces.db"
            profile_db = root / "profile.db"
            memory_db = root / "memory.db"
            traces = TraceStore(trace_db)
            run_id = traces.start_run(
                "latest task",
                str(root),
                thread_id="thread-status",
            )
            traces.finish_run(run_id, "done")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli.main(
                        [
                            "taste",
                            "--profile-db",
                            str(profile_db),
                            "learn",
                            "Prefer direct answers.",
                            "--active",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    cli.main(
                        [
                            "evolve",
                            "--profile-db",
                            str(profile_db),
                            "--memory-db",
                            str(memory_db),
                            "propose",
                            "--title",
                            "Direct answers",
                            "--body",
                            "Prefer direct answers.",
                            "--rationale",
                            "status test",
                        ]
                    ),
                    0,
                )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "status",
                        "--trace-db",
                        str(trace_db),
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["latest_run"]["thread_id"], "thread-status")
        self.assertEqual(payload["latest_run"]["task"], "latest task")
        self.assertEqual(payload["taste"]["active"], 1)
        self.assertEqual(payload["evolution"]["pending"], 1)
        self.assertIn("Run `harness continue`", "\n".join(payload["next"]))

    def test_status_recommends_daily_driver_next_actions(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            trace_db = root / "traces.db"
            profile_db = root / "profile.db"
            memory_db = root / "memory.db"
            mcp_config = root / "mcp.json"
            stdout = io.StringIO()
            with (
                patch.object(config, "CONFIG_PATH", config_path),
                patch.object(cli, "CONFIG_PATH", config_path),
                patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli.main(
                    [
                        "status",
                        "--trace-db",
                        str(trace_db),
                        "--profile-db",
                        str(profile_db),
                        "--memory-db",
                        str(memory_db),
                        "--mcp-config",
                        str(mcp_config),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        next_actions = "\n".join(payload["next"])
        self.assertIn("harness init --provider openrouter", next_actions)
        self.assertIn('harness ask "what is this project?"', next_actions)
        self.assertIn("harness taste scan", next_actions)
        self.assertIn("harness mcp setup", next_actions)

    def test_status_text_prints_next_actions(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.dict(os.environ, {}, clear=True):
                exit_code = cli.main(
                    [
                        "status",
                        "--trace-db",
                        str(Path(tmpdir) / "traces.db"),
                        "--profile-db",
                        str(Path(tmpdir) / "profile.db"),
                        "--memory-db",
                        str(Path(tmpdir) / "memory.db"),
                        "--mcp-config",
                        str(Path(tmpdir) / "mcp.json"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        text = stdout.getvalue()
        self.assertIn("Harness status", text)
        self.assertIn("next\n", text)
        self.assertIn("harness taste scan", text)

    def test_harness_env_vars_drive_provider_defaults(self):
        with patch.dict(
            os.environ,
            {
                "HARNESS_PROVIDER": "openrouter",
                "HARNESS_MODEL": "custom/coder",
                "HARNESS_BASE_URL": "https://example.test/v1",
                "HARNESS_API_KEY": "secret-key",
            },
            clear=True,
        ):
            parsed = cli.parser().parse_args(["run", "ship it"])
            client = cli.build_client(parsed)

        self.assertEqual(parsed.provider, "openrouter")
        self.assertEqual(parsed.model, "custom/coder")
        self.assertEqual(parsed.base_url, "https://example.test/v1")
        self.assertEqual(client.api_key, "secret-key")

    def test_run_defaults_to_tool_action_engine(self):
        parsed = cli.parser().parse_args(["run", "what is this project"])

        self.assertEqual(parsed.act_engine, "tool")

    def test_run_accepts_rlm_compatibility_engine(self):
        parsed = cli.parser().parse_args(["run", "what is this project", "--act-engine", "rlm"])

        self.assertEqual(parsed.act_engine, "rlm")

    def test_run_accepts_autonomy_mode(self):
        parsed = cli.parser().parse_args(["run", "what is this project", "--mode", "ask"])
        runtime = cli.build_runtime(parsed, Path(".").resolve(), None, None)

        self.assertEqual(parsed.autonomy, "ask")
        self.assertEqual(runtime.autonomy, AutonomyMode.ASK)

    def test_run_accepts_permission_mode_aliases(self):
        parsed = cli.parser().parse_args(
            ["run", "what is this project", "--permission-mode", "standard"]
        )
        cli.apply_permission_aliases(parsed)
        runtime = cli.build_runtime(parsed, Path(".").resolve(), None, None)

        self.assertEqual(parsed.autonomy, "sandbox")
        self.assertEqual(runtime.autonomy, AutonomyMode.SANDBOX)

    def test_run_can_disable_automatic_style_scan(self):
        parsed = cli.parser().parse_args(
            ["run", "what is this project", "--no-style-scan"]
        )
        runtime = cli.build_runtime(parsed, Path(".").resolve(), None, None)

        self.assertFalse(runtime.auto_style_scan)

    def test_trusted_permission_shortcuts_select_trusted_mode(self):
        for flag in ["--auto-accept", "--trust", "--yolo", "--dangerously-skip-permissions"]:
            with self.subTest(flag=flag):
                parsed = cli.parser().parse_args(["run", "fix tests", flag])
                cli.apply_permission_aliases(parsed)

                self.assertEqual(parsed.autonomy, "trusted")

    def test_run_plan_alias_uses_plan_only_execution(self):
        with patch.object(cli, "run_plan_task", return_value=0) as run_plan_task, patch.object(
            cli,
            "run_task",
        ) as run_task:
            exit_code = cli.main(["run", "fix this", "--plan", "--provider", "stub"])

        args = run_plan_task.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(args.autonomy, "plan")
        run_task.assert_not_called()

    def test_run_blocks_unconfigured_stub_provider_before_fake_work(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            stdout = io.StringIO()
            with (
                patch.object(config, "CONFIG_PATH", config_path),
                patch.object(cli, "CONFIG_PATH", config_path),
                patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli.main(
                    [
                        "ask",
                        "what is this project",
                        "--workspace",
                        tmpdir,
                        "--trace-db",
                        str(Path(tmpdir) / "traces.db"),
                        "--memory-db",
                        str(Path(tmpdir) / "memory.db"),
                        "--profile-db",
                        str(Path(tmpdir) / "profile.db"),
                        "--no-memory",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"], "provider_not_configured")
        self.assertIn("harness init --provider openrouter", "\n".join(payload["next"]))

    def test_run_allows_intentional_stub_smoke_test_when_provider_is_explicit(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Example\n\nA tiny project.\n", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "ask",
                        "what is this project",
                        "--workspace",
                        tmpdir,
                        "--trace-db",
                        str(root / "traces.db"),
                        "--memory-db",
                        str(root / "memory.db"),
                        "--profile-db",
                        str(root / "profile.db"),
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--no-memory",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "done")
        self.assertIn("Project Summary", payload["final_answer"])

    def test_default_run_uses_typed_tools_for_coding_task(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
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
            trace_db = workspace / "traces.db"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "Fix the failing test in mathlib.py and report what changed.",
                        "--workspace",
                        tmpdir,
                        "--trace-db",
                        str(trace_db),
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--no-memory",
                        "--graph-backend",
                        "simple",
                        "--json",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            mathlib_content = (workspace / "mathlib.py").read_text(encoding="utf-8")
            events = TraceStore(trace_db).events(payload["run_id"])
            action_kinds = [
                event["payload"].get("action", {}).get("kind")
                for event in events
                if event["kind"] == "action_selected"
            ]

        self.assertEqual(exit_code, 0)
        self.assertIn("return a + b", mathlib_content)
        self.assertIn("Changed files", payload["final_answer"])
        self.assertIn("mathlib.py", payload["final_answer"])
        self.assertIn("OK", payload["final_answer"])
        self.assertIn("apply_patch", action_kinds)

    def test_ask_command_forces_tool_engine_and_read_only_mode(self):
        with patch.object(cli, "run_task", return_value=0) as run_task:
            exit_code = cli.main(["ask", "what is this project", "--provider", "stub"])

        args = run_task.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(args.act_engine, "tool")
        self.assertEqual(args.autonomy, "ask")

    def test_plan_command_forces_tool_engine_and_plan_only_mode(self):
        with patch.object(cli, "run_plan_task", return_value=0) as run_plan_task, patch.object(
            cli,
            "run_task",
        ) as run_task:
            exit_code = cli.main(["plan", "fix this", "--provider", "stub"])

        args = run_plan_task.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(args.act_engine, "tool")
        self.assertEqual(args.autonomy, "plan")
        run_task.assert_not_called()

    def test_plan_command_stops_after_planning(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_db = Path(tmpdir) / "traces.db"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "plan",
                        "fix this",
                        "--provider",
                        "stub",
                        "--model",
                        "stub",
                        "--workspace",
                        tmpdir,
                        "--trace-db",
                        str(trace_db),
                        "--no-memory",
                        "--json",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            events = TraceStore(trace_db).events(payload["run_id"])
            event_kinds = {event["kind"] for event in events}

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "done")
        self.assertIn("Implementation Plan", payload["final_answer"])
        self.assertIn("Inspect the task.", payload["final_answer"])
        self.assertIn("plan_created", event_kinds)
        self.assertIn("completion", event_kinds)
        self.assertNotIn("action_selected", event_kinds)
        self.assertNotIn("observation_recorded", event_kinds)

    def test_work_command_forces_tool_engine_and_sandbox_mode(self):
        with patch.object(cli, "run_task", return_value=0) as run_task:
            exit_code = cli.main(["work", "fix tests", "--provider", "stub"])

        args = run_task.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(args.act_engine, "tool")
        self.assertEqual(args.autonomy, "sandbox")

    def test_config_accepts_common_api_key_fallbacks(self):
        with patch.dict(
            os.environ,
            {"HARNESS_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "openrouter-key"},
            clear=True,
        ):
            self.assertEqual(config.default_api_key(), "openrouter-key")

        with patch.dict(
            os.environ,
            {"HARNESS_PROVIDER": "openai", "OPENAI_API_KEY": "openai-key"},
            clear=True,
        ):
            self.assertEqual(config.default_api_key(), "openai-key")

    def test_provider_specific_defaults_do_not_reuse_saved_values(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.object(config, "CONFIG_PATH", config_path), patch.dict(
                os.environ,
                {"HARNESS_PROVIDER": "groq", "OPENAI_API_KEY": "openai-key"},
                clear=True,
            ):
                config.save_user_config(
                    {
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "saved-openai-key",
                    }
                )

                self.assertEqual(config.default_provider(), "groq")
                self.assertEqual(config.default_model(), "openai/gpt-oss-120b")
                self.assertEqual(config.default_base_url(), "https://api.groq.com/openai/v1")
                self.assertIsNone(config.default_api_key())

    def test_pyproject_exposes_harness_and_legacy_console_scripts(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('harness = "rlm_harness.cli:main"', pyproject)
        self.assertIn('rlm-harness = "rlm_harness.cli:main"', pyproject)

    def test_langgraph_is_packaged_for_default_installs(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        dependencies = pyproject["project"]["dependencies"]

        self.assertIn("langgraph>=0.2", dependencies)
        self.assertIn("langgraph-checkpoint-sqlite>=3.1", dependencies)
        self.assertEqual(pyproject["project"]["optional-dependencies"]["graph"], [])

    def test_build_client_uses_api_key_argument_before_environment(self):
        args = Namespace(
            provider="openrouter",
            model="custom/coder",
            base_url="https://example.test/v1",
            api_key="explicit-key",
            timeout=12,
        )
        with patch.dict(os.environ, {"HARNESS_API_KEY": "env-key"}, clear=True):
            client = cli.build_client(args)

        self.assertEqual(client.api_key, "explicit-key")

    def test_saved_config_drives_defaults_when_env_is_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.object(config, "CONFIG_PATH", config_path), patch.dict(
                os.environ,
                {},
                clear=True,
            ):
                config.save_user_config(
                    {
                        "provider": "openrouter",
                        "model": "saved/coder",
                        "base_url": "https://saved.example/v1",
                        "api_key": "saved-key",
                    }
                )
                self.assertEqual(config.default_provider(), "openrouter")
                self.assertEqual(config.default_model(), "saved/coder")
                self.assertEqual(config.default_base_url(), "https://saved.example/v1")
                self.assertEqual(config.default_api_key(), "saved-key")

    def test_model_and_provider_commands_persist_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.object(config, "CONFIG_PATH", config_path), patch.object(
                cli, "CONFIG_PATH", config_path
            ), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(cli.main(["/model", "saved/coder"]), 0)
                self.assertEqual(
                    cli.main(
                        [
                            "/provider",
                            "openrouter",
                            "--keep-model",
                            "--base-url",
                            "https://example.test/v1",
                            "--api-key",
                            "secret-key",
                        ]
                    ),
                    0,
                )
                data = config.load_user_config(config_path)

        self.assertEqual(data["model"], "saved/coder")
        self.assertEqual(data["provider"], "openrouter")
        self.assertEqual(data["base_url"], "https://example.test/v1")
        self.assertEqual(data["api_key"], "secret-key")

    def test_provider_without_name_lists_popular_options(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cli.main(["/provider", "--no-prompt"]), 0)
        text = stdout.getvalue()
        self.assertIn("openrouter", text)
        self.assertIn("opencode-go", text)
        self.assertNotIn("openai-compatible", text)

    def test_model_without_name_lists_provider_models(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cli.main(["/model", "--provider", "openrouter", "--offline"]), 0)
        text = stdout.getvalue()
        self.assertIn("Available models for openrouter", text)
        self.assertIn("qwen/qwen3.7-max", text)

    def test_openai_compatible_alias_normalizes_to_custom(self):
        with patch.dict(os.environ, {"HARNESS_PROVIDER": "openai-compatible"}, clear=True):
            self.assertEqual(config.default_provider(), "custom")

    def test_provider_accepts_opencode_go_as_two_words(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch.object(config, "CONFIG_PATH", config_path), patch.object(
                cli, "CONFIG_PATH", config_path
            ), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    cli.main(["/provider", "opencode", "go", "--api-key", "secret-key"]),
                    0,
                )
                data = config.load_user_config(config_path)

        self.assertEqual(data["provider"], "opencode-go")
        self.assertEqual(data["model"], "glm-5.1")

    def test_doctor_command_is_registered_and_reports_setup_json(self):
        stdout = io.StringIO()
        with (
            patch("rlm_harness.maintenance_cli.shutil.which", return_value=None),
            patch("rlm_harness.maintenance_cli.module_status", return_value="ok"),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = cli.main(["doctor", "--json"])
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["docker_cli"], "missing")
        self.assertEqual(payload["langgraph"], "ok")
        self.assertIn("profile_db", payload)

    def test_update_rebuilds_sandbox_from_managed_source_checkout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir) / "harness"
            src_dir = app_dir / "src"
            venv_bin = app_dir / "venv/bin"
            (src_dir / ".git").mkdir(parents=True)
            venv_bin.mkdir(parents=True)
            (venv_bin / "pip").write_text("", encoding="utf-8")
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="refs/heads/main\n",
                    stderr="",
                )

            args = Namespace(in_place=False, no_sandbox_rebuild=False)
            with patch.dict(os.environ, {"HARNESS_APP_DIR": str(app_dir)}, clear=True), patch(
                "shutil.which", return_value="/usr/local/bin/docker"
            ), patch("subprocess.run", side_effect=fake_run):
                self.assertEqual(cli.cmd_update(args), 0)

        self.assertIn(
            [
                str(venv_bin / "harness"),
                "sandbox",
                "build",
                "--dockerfile",
                str(src_dir / "docker/sandbox.Dockerfile"),
                "--context",
                str(src_dir),
            ],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
