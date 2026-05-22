from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from rlm_harness import cli
from rlm_harness.evals.runner import EvalReport, EvalResult
from rlm_harness.evals.langsmith import (
    LangSmithExperimentUploader,
    LangSmithUploadConfig,
    build_external_experiment_payload,
    collect_run_metadata,
)
from rlm_harness.observability import maybe_traceable


class LangSmithIntegrationTests(unittest.TestCase):
    def sample_report(self) -> EvalReport:
        return EvalReport(
            run_id="run-123",
            suite="suite",
            results=[
                EvalResult(
                    case_id="case-1",
                    passed=True,
                    score=1.0,
                    status="ok",
                    latency_ms=42,
                    output="OK",
                    harness_stdout="fixed tests",
                    harness_stderr="",
                    workspace="/tmp/case-1",
                    metadata={"eval_type": "suite", "prompt": "Fix tests"},
                    started_at="2026-01-01T00:00:00Z",
                    finished_at="2026-01-01T00:00:01Z",
                )
            ],
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            metadata={"provider": "stub", "model": "stub"},
        )

    def test_eval_result_json_includes_timestamps_and_report_metadata(self):
        payload = json.loads(self.sample_report().to_json())

        self.assertEqual(payload["metadata"]["provider"], "stub")
        self.assertEqual(payload["started_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(payload["finished_at"], "2026-01-01T00:00:01Z")
        self.assertEqual(payload["results"][0]["started_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(payload["results"][0]["finished_at"], "2026-01-01T00:00:01Z")

    def test_build_external_experiment_payload_maps_harness_results_to_langsmith_schema(self):
        payload = build_external_experiment_payload(
            self.sample_report(),
            dataset_name="rlm-harness-suite",
            experiment_name="smoke-experiment",
            description="Harness eval upload",
        )

        self.assertEqual(payload["dataset_name"], "rlm-harness-suite")
        self.assertEqual(payload["experiment_name"], "smoke-experiment")
        self.assertEqual(payload["experiment_metadata"]["run_id"], "run-123")
        self.assertEqual(payload["summary_experiment_scores"][0]["key"], "pass_rate")
        self.assertEqual(payload["summary_experiment_scores"][0]["score"], 1.0)
        row = payload["results"][0]
        self.assertEqual(row["inputs"]["prompt"], "Fix tests")
        self.assertEqual(row["inputs"]["case_id"], "case-1")
        self.assertEqual(row["inputs"]["eval_type"], "suite")
        self.assertTrue(row["outputs"]["passed"])
        self.assertEqual(row["evaluation_scores"][0]["key"], "pass")
        self.assertEqual(row["run_metadata"]["latency_ms"], 42)
        self.assertEqual(row["start_time"], "2026-01-01T00:00:00Z")
        self.assertEqual(row["end_time"], "2026-01-01T00:00:01Z")

    def test_langsmith_uploader_posts_to_upload_experiment_endpoint(self):
        calls = []

        def fake_transport(url, body, headers, timeout_s):
            calls.append((url, body, headers, timeout_s))
            return {"experiment": {"id": "exp-1"}, "dataset": {"id": "ds-1"}}

        uploader = LangSmithExperimentUploader(
            LangSmithUploadConfig(
                api_key="key-123",
                endpoint="https://api.smith.langchain.com",
                dataset_name="rlm-harness",
                experiment_name="exp",
            ),
            transport=fake_transport,
        )
        response = uploader.upload(self.sample_report())

        self.assertEqual(response["experiment"]["id"], "exp-1")
        self.assertEqual(calls[0][0], "https://api.smith.langchain.com/datasets/upload-experiment")
        self.assertEqual(calls[0][2]["x-api-key"], "key-123")
        self.assertEqual(calls[0][3], 30)
        self.assertEqual(calls[0][1]["dataset_name"], "rlm-harness")

    def test_langsmith_uploader_skips_without_api_key_unless_required(self):
        uploader = LangSmithExperimentUploader(
            LangSmithUploadConfig(api_key=None, dataset_name="rlm-harness")
        )

        self.assertEqual(uploader.upload(self.sample_report()), {"skipped": "missing_api_key"})
        with self.assertRaises(RuntimeError):
            uploader.upload(self.sample_report(), required=True)

    def test_eval_parser_exposes_langsmith_upload_flags(self):
        parsed = cli.parser().parse_args(
            [
                "eval",
                "suite.yaml",
                "--langsmith-upload",
                "--langsmith-dataset",
                "rlm-harness",
                "--langsmith-experiment",
                "exp",
                "--langsmith-required",
            ]
        )

        self.assertTrue(parsed.langsmith_upload)
        self.assertTrue(parsed.langsmith_required)
        self.assertEqual(parsed.langsmith_dataset, "rlm-harness")
        self.assertEqual(parsed.langsmith_experiment, "exp")

    def test_collect_run_metadata_includes_eval_configuration_and_git_state(self):
        args = cli.parser().parse_args(
            [
                "eval",
                "suite.yaml",
                "--provider",
                "stub",
                "--model",
                "stub",
                "--no-sandbox",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = collect_run_metadata(args, Path(tmpdir))

        self.assertEqual(metadata["provider"], "stub")
        self.assertEqual(metadata["model"], "stub")
        self.assertTrue(metadata["no_sandbox"])
        self.assertIn("git_sha", metadata)
        self.assertIn("git_dirty", metadata)

    def test_maybe_traceable_is_noop_when_langsmith_package_or_tracing_is_unavailable(self):
        def target(value):
            return value + 1

        with patch.dict(os.environ, {}, clear=True):
            wrapped = maybe_traceable("unit", run_type="chain")(target)

        self.assertIs(wrapped, target)
        self.assertEqual(wrapped(2), 3)

    def test_cmd_eval_uses_importable_cli_path_for_case_workspaces(self):
        args = cli.parser().parse_args(["eval", "suite.yaml", "--provider", "stub", "--model", "stub"])

        command = cli.build_eval_harness_command(args)

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "-c")
        self.assertIn("sys.path.insert", command[2])
        self.assertIn("from rlm_harness.cli import main", command[2])
        self.assertEqual(command[3], "run")
        self.assertNotIn("-m", command)

    def test_eval_parser_rejects_removed_suite_and_benchmark_modes(self):
        parser = cli.parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["eval", "suite", "suite.yaml"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["eval", "long-horizon", "suite.yaml"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["eval", "swe-bench", "manifest.jsonl"])

    def test_now_iso_values_are_utc_iso_strings(self):
        report = self.sample_report()
        parsed = datetime.fromisoformat(report.started_at.replace("Z", "+00:00"))
        self.assertEqual(parsed.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()
