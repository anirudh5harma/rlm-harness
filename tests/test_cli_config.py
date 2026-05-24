from __future__ import annotations

import contextlib
import io
import os
import subprocess
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rlm_harness import cli, config


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
        self.assertEqual(cli.normalize_argv(["update"]), ["update"])
        self.assertEqual(cli.normalize_argv(["/update"]), ["update"])
        self.assertEqual(cli.normalize_argv(["--help"]), ["--help"])
        self.assertEqual(cli.normalize_argv(["/model", "custom/coder"]), ["model", "custom/coder"])

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
                return subprocess.CompletedProcess(command, 0, stdout="refs/heads/main\n", stderr="")

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
