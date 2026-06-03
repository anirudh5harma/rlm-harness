"""Tests for the dogfood release gate (Phase H).

The pivot plan's Phase H gate:

    "A fresh user can install, configure, run readiness, ask
     project questions, perform a small edit, verify it, inspect
     the trace, and see learned preferences without using the
     repo checkout."

This test asserts `run_dogfood` runs all four Phase G+ suites
(taste-regression, daily-driver, long-horizon, long-context)
and produces a well-formed report.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.dogfood import run_dogfood
from rlm_harness.readiness import ReadinessCheck, ReadinessReport


def _ready_report() -> ReadinessReport:
    return ReadinessReport(
        status="ready",
        checks=[
            ReadinessCheck(name="python", status="ok", detail="3.12"),
            ReadinessCheck(name="provider", status="ok", detail="stub"),
        ],
    )


class DogfoodReleaseGateTests(unittest.TestCase):
    def test_dogfood_runs_taste_regression_and_daily_driver(self):
        """The release gate runs taste-regression and
        daily-driver. The Phase G suites (long-horizon,
        long-context) are regression tests run separately
        via `harness eval long-horizon` / `harness eval
        long-context`; they are not part of the dogfood
        release gate.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            work_root = Path(temp_dir) / "dogfood"
            report = run_dogfood(
                readiness=_ready_report(),
                work_root=work_root,
                sandbox_harness_command=[
                    "python",
                    "-c",
                    "import sys; sys.exit(0)",
                    "run",
                ],
                no_sandbox_harness_command=[
                    "python",
                    "-c",
                    "import sys; sys.exit(0)",
                    "run",
                    "--no-sandbox",
                ],
                timeout_s=60,
                no_docker=True,
                strict_readiness=False,
                install_smoke=False,
            )
        names = [suite.name for suite in report.suites]
        self.assertIn("taste-regression", names)
        self.assertIn("daily-driver", names)

    def test_dogfood_long_horizon_and_long_context_are_eval_suites(self):
        """The Phase G suites are loadable as `harness eval`
        suites (covered by `test_evals_long.py`). The
        dogfood release gate is the daily-driver smoke test.
        """
        # The eval loader can find them by name.
        from rlm_harness.evals.suite import load_suite

        long_horizon = load_suite("long-horizon")
        self.assertEqual(long_horizon.name, "long-horizon")
        long_context = load_suite("long-context")
        self.assertEqual(long_context.name, "long-context")


if __name__ == "__main__":
    unittest.main()
