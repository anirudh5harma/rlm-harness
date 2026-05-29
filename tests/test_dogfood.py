import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rlm_harness import cli
from rlm_harness.dogfood import (
    DogfoodInstallResult,
    DogfoodSuiteResult,
    dogfood_status,
    render_dogfood_report,
    run_dogfood,
    run_feedback_smoke,
    run_install_smoke,
)
from rlm_harness.readiness import BLOCKED, READY, ReadinessCheck, ReadinessReport


class DogfoodTests(unittest.TestCase):
    def test_feedback_smoke_promotes_feedback_to_taste_and_proposal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_feedback_smoke(Path(temp_dir))

        self.assertTrue(result.passed)
        self.assertEqual(result.feedback_count, 1)
        self.assertEqual(result.taste_count, 1)
        self.assertEqual(result.proposal_count, 1)

    def test_dogfood_status_can_ignore_setup_gaps_for_local_proof(self):
        readiness = ReadinessReport(
            status="needs_setup",
            checks=[ReadinessCheck("provider", BLOCKED, "stub")],
        )
        suites = [DogfoodSuiteResult("taste-regression", True, 1.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback = run_feedback_smoke(Path(temp_dir))

        self.assertEqual(dogfood_status(readiness, suites, feedback, False), "passed")
        self.assertEqual(dogfood_status(readiness, suites, feedback, True), "failed")

    def test_run_dogfood_skips_daily_driver_when_docker_is_disabled(self):
        readiness = ReadinessReport(
            status=READY,
            checks=[ReadinessCheck("provider", READY, "stub")],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_dogfood(
                readiness=readiness,
                work_root=Path(temp_dir),
                sandbox_harness_command=["python", "-c", "print('unused')"],
                no_sandbox_harness_command=["python", "-c", "print('verification ok')"],
                timeout_s=30,
                no_docker=True,
            )

        self.assertEqual(report.status, "passed")
        self.assertEqual(report.suites[0].name, "taste-regression")
        self.assertTrue(report.suites[0].passed)
        self.assertTrue(report.suites[1].skipped)
        self.assertIn("feedback\tpassed", render_dogfood_report(report))

    def test_cli_dogfood_json_runs_no_docker_local_proof(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "dogfood",
                    "--work-root",
                    str(Path(temp_dir) / "dogfood"),
                    "--no-docker",
                    "--provider",
                    "stub",
                    "--model",
                    "stub",
                    "--eval-timeout",
                    "30",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["feedback"]["passed"])
        self.assertTrue(payload["install"]["skipped"])
        self.assertTrue(any(suite["skipped"] for suite in payload["suites"]))

    def test_dogfood_status_requires_install_smoke_when_present(self):
        readiness = ReadinessReport(
            status=READY,
            checks=[ReadinessCheck("provider", READY, "stub")],
        )
        suites = [DogfoodSuiteResult("taste-regression", True, 1.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback = run_feedback_smoke(Path(temp_dir))

        install = DogfoodInstallResult(False, False, "install failed")

        self.assertEqual(dogfood_status(readiness, suites, feedback, False, install), "failed")

    def test_install_smoke_runs_fresh_venv_and_bundled_eval(self):
        commands = []

        def fake_runner(command, **kwargs):
            commands.append(command)
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_install_smoke(
                Path(temp_dir) / "install",
                Path("/repo"),
                timeout_s=10,
                command_runner=fake_runner,
            )

        self.assertTrue(result.passed)
        self.assertFalse(result.skipped)
        self.assertEqual(len(commands), 4)
        self.assertIn("taste-regression", commands[-1])


if __name__ == "__main__":
    unittest.main()
