from __future__ import annotations

import contextlib
import io
import os
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
        self.assertEqual(cli.normalize_argv(["--help"]), ["--help"])
        self.assertEqual(cli.normalize_argv(["/model", "custom/coder"]), ["model", "custom/coder"])

    def test_harness_env_vars_drive_openai_compatible_defaults(self):
        with patch.dict(
            os.environ,
            {
                "HARNESS_PROVIDER": "openai-compatible",
                "HARNESS_MODEL": "custom/coder",
                "HARNESS_BASE_URL": "https://example.test/v1",
                "HARNESS_API_KEY": "secret-key",
            },
            clear=True,
        ):
            parsed = cli.parser().parse_args(["run", "ship it"])
            client = cli.build_client(parsed)

        self.assertEqual(parsed.provider, "openai-compatible")
        self.assertEqual(parsed.model, "custom/coder")
        self.assertEqual(parsed.base_url, "https://example.test/v1")
        self.assertEqual(client.api_key, "secret-key")

    def test_config_accepts_common_api_key_fallbacks(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "openrouter-key"}, clear=True):
            self.assertEqual(config.default_api_key(), "openrouter-key")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            self.assertEqual(config.default_api_key(), "openai-key")

    def test_pyproject_exposes_harness_and_legacy_console_scripts(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('harness = "rlm_harness.cli:main"', pyproject)
        self.assertIn('rlm-harness = "rlm_harness.cli:main"', pyproject)

    def test_build_client_uses_api_key_argument_before_environment(self):
        args = Namespace(
            provider="openai-compatible",
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
                        "provider": "openai-compatible",
                        "model": "saved/coder",
                        "base_url": "https://saved.example/v1",
                        "api_key": "saved-key",
                    }
                )
                self.assertEqual(config.default_provider(), "openai-compatible")
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
                            "openai-compatible",
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
        self.assertEqual(data["provider"], "openai-compatible")
        self.assertEqual(data["base_url"], "https://example.test/v1")
        self.assertEqual(data["api_key"], "secret-key")


if __name__ == "__main__":
    unittest.main()
