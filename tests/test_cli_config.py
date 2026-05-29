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

from rlm_harness import cli, config
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
        self.assertEqual(cli.normalize_argv(["continue"]), ["continue"])
        self.assertEqual(cli.normalize_argv(["/continue"]), ["continue"])
        self.assertEqual(cli.normalize_argv(["--continue", "next"]), ["continue", "next"])
        self.assertEqual(cli.normalize_argv(["-c", "next"]), ["continue", "next"])
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
        self.assertIn("harness profile list|learn|approve|reject", text)
        self.assertIn("harness taste list|learn|approve|reject", text)
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
        self.assertIn("profile", names)
        self.assertIn("taste", names)
        self.assertNotIn("sandbox", names)

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

    def test_taste_command_learns_and_lists_records(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_db = Path(tmpdir) / "profile.db"
            learn_stdout = io.StringIO()
            with contextlib.redirect_stdout(learn_stdout):
                learn_exit = cli.main(
                    [
                        "taste",
                        "--profile-db",
                        str(profile_db),
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
