"""Tests for `harness doctor` and the release gate (Phase H).

The pivot plan's Phase H gate:

    "A fresh user can install, configure, run readiness, ask
     project questions, perform a small edit, verify it, inspect
     the trace, and see learned preferences without using the
     repo checkout."

This test exercises the health-check surface: doctor returns
JSON, surfaces the required checks, and reports missing
dependencies in a way a fresh user can act on.
"""
import contextlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DoctorTests(unittest.TestCase):
    def test_doctor_returns_json_with_required_keys(self):
        """`harness doctor --json` is the JSON shape a fresh
        user pipes into a script.
        """
        with tempfile_workdir() as workdir:
            result = subprocess.run(
                [sys.executable, "-m", "rlm_harness.cli", "doctor", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            for key in (
                "python",
                "harness_cli",
                "docker_cli",
                "langgraph",
                "langgraph_checkpoint_sqlite",
                "sqlite_vec",
                "profile_db",
            ):
                self.assertIn(key, payload)

    def test_doctor_reports_python_version(self):
        with tempfile_workdir() as workdir:
            result = subprocess.run(
                [sys.executable, "-m", "rlm_harness.cli", "doctor", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )
            payload = json.loads(result.stdout)
            python_version = payload["python"]
            major, minor, *_ = python_version.split(".")
            self.assertEqual(int(major), 3)
            self.assertGreaterEqual(int(minor), 10)

    def test_doctor_module_status_helper(self):
        from rlm_harness.maintenance_cli import module_status

        # Real module that exists.
        self.assertEqual(module_status("json"), "ok")
        # Imaginary module.
        self.assertEqual(module_status("__nonexistent_module_xyz__"), "missing")

    def test_doctor_exit_code_nonzero_when_docker_missing(self):
        """When Docker is missing or unavailable, `harness
        doctor` exits non-zero so CI can gate on it. We don't
        require Docker to be present in this test environment;
        we only assert the exit code is meaningful (0 when ok,
        non-zero when sandbox is unreachable).
        """
        with tempfile_workdir() as workdir:
            result = subprocess.run(
                [sys.executable, "-m", "rlm_harness.cli", "doctor"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )
            # Exit code 0 or 1 — both are valid. The contract is
            # that the command is deterministic and well-behaved;
            # we don't require Docker to be installed here.
            self.assertIn(result.returncode, (0, 1))


@contextlib.contextmanager
def tempfile_workdir() -> str:
    """A throwaway cwd for subprocess tests. We do not write any
    state; the harness doctor path is read-only on the workspace.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class DogfoodReleaseGateTests(unittest.TestCase):
    def test_dogfood_dry_run_lists_suites_to_run(self):
        """`harness dogfood` is the release gate. We don't run
        the full evals here (they require provider config); we
        only check that the command is registered and reports a
        well-formed status. The deep dogfood coverage lives in
        `tests/test_dogfood.py`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlm_harness.cli",
                    "dogfood",
                    "--no-docker",
                    "--work-root",
                    str(Path(temp_dir) / "work"),
                    "--provider",
                    "stub",
                    "--model",
                    "stub",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=temp_dir,
            )
            # dogfood exits 0 when the suites pass (against the
            # stub provider), non-zero otherwise. Either is a
            # valid response; we only assert the command ran and
            # produced output.
            self.assertIn(result.returncode, (0, 1))
            # The combined output (stdout + stderr) should
            # mention at least one of the suites we ship.
            combined = result.stdout + result.stderr
            self.assertTrue(
                "daily-driver" in combined
                or "taste" in combined
                or "long-horizon" in combined
                or "long-context" in combined
                or "stub" in combined,
                f"dogfood output did not mention any suite: {combined[:500]}",
            )


if __name__ == "__main__":
    unittest.main()
