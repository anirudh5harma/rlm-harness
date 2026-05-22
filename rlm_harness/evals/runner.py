from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class GradeResult:
    passed: bool
    score: float
    output: str


class Grader(Protocol):
    def grade(self, workspace: Path) -> GradeResult: ...


@dataclass
class UnitTestGrader:
    command: str
    timeout_s: int = 300

    def grade(self, workspace: Path) -> GradeResult:
        completed = subprocess.run(
            self.command,
            cwd=workspace,
            shell=True,
            executable="/bin/sh",
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        output = completed.stdout + completed.stderr
        passed = completed.returncode == 0
        return GradeResult(passed=passed, score=1.0 if passed else 0.0, output=output)


@dataclass
class EvalCase:
    id: str
    prompt: str
    workspace: Path
    grader: Grader
    setup_commands: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalSuite:
    name: str
    cases: list[EvalCase]


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score: float
    status: str
    latency_ms: int
    output: str
    harness_stdout: str
    harness_stderr: str
    workspace: str
    metadata: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""


@dataclass
class EvalReport:
    run_id: str
    suite: str
    results: list[EvalResult]
    started_at: str = ""
    finished_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.passed) / len(self.results)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "suite": self.suite,
                "pass_rate": self.pass_rate,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "metadata": self.metadata,
                "results": [result.__dict__ for result in self.results],
            },
            indent=2,
            sort_keys=True,
        )


class EvalRunner:
    def __init__(
        self,
        harness_command: Optional[list[str]] = None,
        timeout_s: int = 900,
        clean_workspaces: bool = False,
    ):
        self.harness_command = harness_command or ["harness"]
        self.timeout_s = timeout_s
        self.clean_workspaces = clean_workspaces

    def run(self, suite: EvalSuite, metadata: Optional[dict] = None) -> EvalReport:
        run_id = str(uuid.uuid4())
        started_at = utc_now_iso()
        results = [self.run_case(case) for case in suite.cases]
        finished_at = utc_now_iso()
        return EvalReport(
            run_id=run_id,
            suite=suite.name,
            results=results,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata or {},
        )

    def run_case(self, case: EvalCase) -> EvalResult:
        workspace = Path(case.workspace)
        started_at = utc_now_iso()
        started = time.perf_counter()
        harness_stdout = ""
        harness_stderr = ""
        status = "ok"
        try:
            self.prepare_workspace(case)
            completed = subprocess.run(
                [*self.harness_command, case.prompt],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
            harness_stdout = completed.stdout
            harness_stderr = completed.stderr
            if completed.returncode != 0:
                status = "harness_error"
            grade = case.grader.grade(workspace)
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            grade = GradeResult(False, 0.0, (exc.stdout or "") + (exc.stderr or ""))
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            finished_at = utc_now_iso()
            if self.clean_workspaces and workspace.exists():
                shutil.rmtree(workspace)
        return EvalResult(
            case_id=case.id,
            passed=grade.passed,
            score=grade.score,
            status=status,
            latency_ms=latency_ms,
            output=grade.output,
            harness_stdout=harness_stdout,
            harness_stderr=harness_stderr,
            workspace=str(workspace),
            metadata=case.metadata,
            started_at=started_at,
            finished_at=finished_at,
        )

    def prepare_workspace(self, case: EvalCase) -> None:
        case.workspace.mkdir(parents=True, exist_ok=True)
        for relative, content in case.files.items():
            target = case.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for command in case.setup_commands:
            completed = subprocess.run(
                command,
                cwd=case.workspace,
                shell=True,
                executable="/bin/sh",
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"setup failed for {case.id}: {command}\n{completed.stdout}{completed.stderr}"
                )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
