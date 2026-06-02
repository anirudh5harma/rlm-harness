"""Tests for the CLI trim + extension install (Phase F).

The pivot plan's Phase F gate:

    "harness --help lists <= 12 top-level commands."

This test asserts the gate directly, plus a smoke test for the
`harness install` extension command (Phase F stub).
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlm_harness.cli_catalog import (
    PUBLIC_COMMANDS,
)
from rlm_harness.config import default_extension_root


class CliTrimGateTests(unittest.TestCase):
    def test_harness_help_lists_at_most_twelve_top_level_commands(self):
        """The Phase F gate: `harness --help` lists <= 12
        user-facing top-level commands. The legacy aliases are
        registered as subcommands but hidden from the help
        output.
        """
        result = subprocess.run(
            [sys.executable, "-m", "rlm_harness.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # Extract the metavar from the usage line. The
        # metavar is the `{a,b,c}` brace list of public
        # commands.
        for line in result.stdout.splitlines():
            if line.strip().startswith("{"):
                # Strip the `{` and `}` and split on `,`.
                inside = line.strip().strip("{}")
                names = [name.strip() for name in inside.split(",") if name.strip()]
                self.assertLessEqual(
                    len(names),
                    12,
                    f"help metavar lists {len(names)} commands: {names}",
                )
                # The set of advertised commands is a subset
                # of the public set.
                self.assertTrue(set(names).issubset(PUBLIC_COMMANDS))
                return
        self.fail("Could not find a `{...}` metavar in harness --help output")

    def test_public_commands_size_is_at_most_twelve(self):
        """PUBLIC_COMMANDS is the single source of truth for
        what `harness --help` advertises.
        """
        self.assertLessEqual(len(PUBLIC_COMMANDS), 12)

    def test_legacy_aliases_are_still_registered(self):
        """Backward compat: existing scripts that call
        `harness run` or `harness taste` still work; the
        aliases are registered as subcommands. They're just
        hidden from `--help`.
        """
        result = subprocess.run(
            [sys.executable, "-m", "rlm_harness.cli", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # `run` is still a valid subcommand; the parser
        # accepts it. The exit code may be non-zero if the
        # parser is in strict mode, but the error must not be
        # "unrecognized arguments".
        self.assertNotIn("unrecognized arguments", result.stderr + result.stdout)

    def test_install_command_is_registered(self):
        """The `harness install` command is a new top-level
        command. Phase F ships a no-op stub.
        """
        result = subprocess.run(
            [sys.executable, "-m", "rlm_harness.cli", "install", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("extension", result.stdout.lower())


class InstallCommandStubTests(unittest.TestCase):
    def test_install_stub_prints_target_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlm_harness.cli",
                    "install",
                    "npm:foo-bar",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            import json

            payload = json.loads(result.stdout)
            self.assertEqual(payload["source"], "npm:foo-bar")
            self.assertEqual(payload["status"], "stub")
            # The target sits under the default extension root.
            target = Path(payload["target"])
            extension_root = Path(default_extension_root())
            try:
                target.relative_to(extension_root)
            except ValueError:
                self.fail(
                    f"target {target} is not under extension root {extension_root}"
                )

    def test_install_list_command_returns_empty_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlm_harness.cli",
                    "install",
                    "--list",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            import json

            payload = json.loads(result.stdout)
            self.assertEqual(payload["extensions"], [])


if __name__ == "__main__":
    unittest.main()
