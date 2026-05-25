from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rlm_harness.evals.runner import EvalReport

LANGSMITH_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


@dataclass(frozen=True)
class LangSmithUploadConfig:
    dataset_name: str
    api_key: Optional[str] = None
    endpoint: str = LANGSMITH_DEFAULT_ENDPOINT
    experiment_name: Optional[str] = None
    description: str = "RLM Harness evaluation run"
    timeout_s: int = 30

    @classmethod
    def from_env(
        cls,
        dataset_name: str,
        experiment_name: Optional[str] = None,
        description: str = "RLM Harness evaluation run",
    ) -> LangSmithUploadConfig:
        return cls(
            api_key=os.environ.get("LANGSMITH_API_KEY"),
            endpoint=os.environ.get("LANGSMITH_ENDPOINT", LANGSMITH_DEFAULT_ENDPOINT),
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            description=description,
        )


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


def now_iso_from_timestamp(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def build_external_experiment_payload(
    report: EvalReport,
    dataset_name: str,
    experiment_name: Optional[str] = None,
    description: str = "RLM Harness evaluation run",
) -> dict[str, Any]:
    name = experiment_name or f"{report.suite}-{report.run_id}"
    return {
        "experiment_name": name,
        "experiment_description": description,
        "experiment_start_time": report.started_at,
        "experiment_end_time": report.finished_at,
        "dataset_name": dataset_name,
        "dataset_description": "Externally managed RLM Harness evaluation dataset",
        "experiment_metadata": {
            "run_id": report.run_id,
            "suite": report.suite,
            "source": "rlm-harness",
            **report.metadata,
        },
        "summary_experiment_scores": [
            {"key": "pass_rate", "score": report.pass_rate},
            {"key": "case_count", "score": len(report.results)},
        ],
        "results": [result_to_langsmith_row(result) for result in report.results],
    }


def result_to_langsmith_row(result) -> dict[str, Any]:
    prompt = result.metadata.get("prompt") or result.metadata.get("problem_statement") or ""
    eval_type = result.metadata.get("eval_type") or "unknown"
    return {
        "inputs": {
            "case_id": result.case_id,
            "prompt": prompt,
            "eval_type": eval_type,
        },
        "outputs": {
            "passed": result.passed,
            "score": result.score,
            "status": result.status,
            "grader_output": result.output,
            "harness_stdout": result.harness_stdout,
            "harness_stderr": result.harness_stderr,
        },
        "evaluation_scores": [
            {"key": "pass", "score": 1.0 if result.passed else 0.0},
            {"key": "score", "score": result.score},
        ],
        "start_time": result.started_at,
        "end_time": result.finished_at,
        "run_name": result.case_id,
        "run_metadata": {
            **result.metadata,
            "latency_ms": result.latency_ms,
            "workspace": result.workspace,
            "status": result.status,
            "source": "rlm-harness",
        },
    }


class LangSmithExperimentUploader:
    def __init__(
        self,
        config: LangSmithUploadConfig,
        transport: Optional[Transport] = None,
    ):
        self.config = config
        self.transport = transport or post_json

    def upload(self, report: EvalReport, required: bool = False) -> dict[str, Any]:
        if not self.config.api_key:
            if required:
                raise RuntimeError("LANGSMITH_API_KEY is required for --langsmith-required")
            return {"skipped": "missing_api_key"}
        url = self.config.endpoint.rstrip("/") + "/datasets/upload-experiment"
        payload = build_external_experiment_payload(
            report,
            dataset_name=self.config.dataset_name,
            experiment_name=self.config.experiment_name,
            description=self.config.description,
        )
        headers = {
            "content-type": "application/json",
            "x-api-key": self.config.api_key,
        }
        return self.transport(url, payload, headers, self.config.timeout_s)


def post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_s: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LangSmith upload failed: {exc}") from exc
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LangSmith upload response was not valid JSON") from exc
    return payload if isinstance(payload, dict) else {"response": payload}


def collect_run_metadata(args, repo_root: Path) -> dict[str, Any]:
    return {
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "eval_type": "suite",
        "eval_timeout": getattr(args, "eval_timeout", None),
        "no_sandbox": bool(getattr(args, "no_sandbox", False)),
        "git_sha": git_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_dirty": bool(git_output(["git", "status", "--porcelain"], repo_root)),
    }


def git_output(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()
