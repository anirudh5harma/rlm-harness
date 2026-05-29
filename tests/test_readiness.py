import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlm_harness import cli
from rlm_harness.readiness import (
    BLOCKED,
    READY,
    WARNING,
    ReadinessCheck,
    build_readiness_report,
    overall_status,
    render_readiness_report,
)


class ReadinessTests(unittest.TestCase):
    def test_overall_status_distinguishes_blockers_from_warnings(self):
        self.assertEqual(overall_status([ReadinessCheck("a", READY, "ok")]), READY)
        self.assertEqual(
            overall_status(
                [
                    ReadinessCheck("a", READY, "ok"),
                    ReadinessCheck("b", WARNING, "soft issue"),
                ]
            ),
            "degraded",
        )
        self.assertEqual(
            overall_status(
                [
                    ReadinessCheck("a", WARNING, "soft issue"),
                    ReadinessCheck("b", BLOCKED, "hard issue"),
                ]
            ),
            "needs_setup",
        )

    def test_readiness_report_blocks_stub_provider_but_allows_no_docker_check(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            "os.environ",
            {
                "HARNESS_PROVIDER": "stub",
                "RLM_HARNESS_MEMORY_DB": "memory.db",
                "RLM_HARNESS_TRACE_DB": "traces.db",
                "RLM_HARNESS_PROFILE_DB": str(Path(temp_dir) / "profile.db"),
            },
            clear=True,
        ), patch(
            "rlm_harness.readiness.CONFIG_PATH",
            Path(temp_dir) / "config.json",
        ):
            report = build_readiness_report(Path(temp_dir), check_docker=False)

        self.assertEqual(report.status, "needs_setup")
        checks = {check.name: check for check in report.checks}
        self.assertEqual(checks["provider"].status, BLOCKED)
        self.assertEqual(checks["api_key"].status, BLOCKED)
        self.assertNotIn("docker_cli", checks)

    def test_readiness_render_includes_next_action(self):
        report = render_readiness_report(
            type(
                "Report",
                (),
                {
                    "status": "needs_setup",
                    "checks": [
                        ReadinessCheck(
                            "provider",
                            BLOCKED,
                            "provider=stub",
                            "Run `harness /provider`.",
                        )
                    ],
                },
            )()
        )

        self.assertIn("readiness\tneeds_setup", report)
        self.assertIn("Run `harness /provider`", report)
        self.assertIn("harness readiness", report)

    def test_cli_readiness_json_uses_report_status_as_exit_code(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            "os.environ",
            {
                "HARNESS_PROVIDER": "stub",
                "RLM_HARNESS_MEMORY_DB": str(Path(temp_dir) / "memory.db"),
                "RLM_HARNESS_TRACE_DB": str(Path(temp_dir) / "traces.db"),
                "RLM_HARNESS_PROFILE_DB": str(Path(temp_dir) / "profile.db"),
            },
            clear=True,
        ), patch(
            "rlm_harness.readiness.CONFIG_PATH",
            Path(temp_dir) / "config.json",
        ), patch(
            "rlm_harness.config.CONFIG_PATH",
            Path(temp_dir) / "config.json",
        ), contextlib.redirect_stdout(stdout):
            exit_code = cli.main(["readiness", "--json", "--no-docker"])

        self.assertEqual(exit_code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "needs_setup")
        self.assertIn("provider", {check["name"] for check in payload["checks"]})


if __name__ == "__main__":
    unittest.main()
