"""Tests for the install smoke: the harness binary works from a
fresh virtualenv (Phase H.4).

The pivot plan's Phase H gate:

    "A fresh user can install, configure, run readiness, ask
     project questions, perform a small edit, verify it, inspect
     the trace, and see learned preferences without using the
     repo checkout."

This test simulates the "fresh venv" path: a throwaway
venv is created, the harness package is installed from the
repo root, and the resulting `harness` binary runs `--help`
and `doctor --json` without needing the test runner.
"""
import json
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class InstallSmokeTests(unittest.TestCase):
    def test_harness_works_in_a_fresh_venv(self):
        """End-to-end smoke: a fresh venv + `pip install -e .` +
        `harness --help` works. This is the closest we can get
        to a real `curl ... | sh` install without networking.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "venv"
            venv.EnvBuilder(
                system_site_packages=False,
                clear=True,
                symlinks=False,
                with_pip=True,
            ).create(str(venv_dir))
            pip_bin = venv_dir / "bin" / "pip"
            harness_bin = venv_dir / "bin" / "harness"

            install = subprocess.run(
                [
                    str(pip_bin),
                    "install",
                    "--quiet",
                    str(REPO_ROOT),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(
                install.returncode,
                0,
                msg=f"pip install failed: {install.stderr[:2000]}",
            )
            self.assertTrue(
                harness_bin.exists(),
                f"harness binary not found at {harness_bin}",
            )

            help_result = subprocess.run(
                [str(harness_bin), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
            for command in ("ask", "work", "trace", "install", "eval"):
                self.assertIn(
                    command,
                    help_result.stdout,
                    f"harness --help did not list `{command}`",
                )

            doctor = subprocess.run(
                [str(harness_bin), "doctor", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            try:
                payload = json.loads(doctor.stdout)
            except json.JSONDecodeError:
                self.fail(
                    f"harness doctor did not return valid JSON: "
                    f"{doctor.stdout[:500]}"
                )
            self.assertIn("python", payload)


if __name__ == "__main__":
    unittest.main()
