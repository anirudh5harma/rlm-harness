from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from rlm_harness.memory import Memory
from rlm_harness.memory.evolution import EvolutionProposal, EvolutionProposalStore
from rlm_harness.memory.profile import TasteProfileStore, TasteRecord


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
        command = normalize_python_command(self.command)
        completed = subprocess.run(
            command,
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
    taste_records: list[dict] = field(default_factory=list)
    evolution_proposals: list[dict] = field(default_factory=list)
    output_contains: list[str] = field(default_factory=list)
    output_not_contains: list[str] = field(default_factory=list)
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
        self.harness_command = normalize_python_argv(harness_command or ["harness"])
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
        profile_db = workspace / ".rlm_harness" / "profile.db"
        memory_db = workspace / ".rlm_harness" / "memory.db"
        started_at = utc_now_iso()
        started = time.perf_counter()
        harness_stdout = ""
        harness_stderr = ""
        status = "ok"
        try:
            self.prepare_workspace(case)
            self.prepare_case_memory(case, profile_db, memory_db)
            completed = subprocess.run(
                [
                    *self.harness_command,
                    "--memory-db",
                    str(memory_db),
                    "--profile-db",
                    str(profile_db),
                    case.prompt,
                ],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
            harness_stdout = completed.stdout
            harness_stderr = completed.stderr
            harness_grade = GradeResult(True, 1.0, "")
            if completed.returncode != 0:
                status = "harness_error"
                harness_grade = GradeResult(
                    False,
                    0.0,
                    f"harness exited with {completed.returncode}",
                )
            grade = combine_grades(
                harness_grade,
                case.grader.grade(workspace),
                grade_output_expectations(case, harness_stdout, harness_stderr),
            )
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

    def prepare_case_memory(
        self,
        case: EvalCase,
        profile_db: Path,
        memory_db: Path,
    ) -> None:
        profile_db.parent.mkdir(parents=True, exist_ok=True)
        with Memory(profile_db) as profile_memory:
            store = TasteProfileStore(profile_memory)
            for raw in case.taste_records:
                store.add(
                    TasteRecord.create(
                        scope=str(raw.get("scope", "user")),
                        kind=str(raw.get("kind", "preference")),
                        text=str(raw["text"]),
                        confidence=float(raw.get("confidence", 0.95)),
                        status=str(raw.get("status", "active")),
                        evidence={"source": "eval", "case_id": case.id},
                    )
                )
            evolution_store = EvolutionProposalStore(profile_memory)
            for raw in case.evolution_proposals:
                if str(raw.get("scope", "user")) != "user":
                    continue
                evolution_store.add(proposal_from_eval(case, raw))

        with Memory(memory_db) as project_memory:
            evolution_store = EvolutionProposalStore(project_memory)
            for raw in case.evolution_proposals:
                if str(raw.get("scope", "user")) == "user":
                    continue
                evolution_store.add(proposal_from_eval(case, raw))


def proposal_from_eval(case: EvalCase, raw: dict) -> EvolutionProposal:
    return EvolutionProposal.create(
        scope=str(raw.get("scope", "user")),
        kind=str(raw.get("kind", "prompt_rule")),
        title=str(raw["title"]),
        body=str(raw["body"]),
        rationale=str(raw.get("rationale", f"Seeded by eval case {case.id}.")),
        status=str(raw.get("status", "approved")),
        evidence={"source": "eval", "case_id": case.id},
    )


def grade_output_expectations(
    case: EvalCase,
    harness_stdout: str,
    harness_stderr: str,
) -> GradeResult:
    combined = harness_stdout + harness_stderr
    normalized = combined.lower()
    failures = []
    for expected in case.output_contains:
        if expected.lower() not in normalized:
            failures.append(f"missing expected output: {expected}")
    for forbidden in case.output_not_contains:
        if forbidden.lower() in normalized:
            failures.append(f"forbidden output present: {forbidden}")
    if failures:
        return GradeResult(False, 0.0, "\n".join(failures))
    if case.output_contains or case.output_not_contains:
        return GradeResult(True, 1.0, "output expectations passed")
    return GradeResult(True, 1.0, "")


def combine_grades(*grades: GradeResult) -> GradeResult:
    failures = [grade for grade in grades if not grade.passed]
    output = "\n".join(grade.output for grade in grades if grade.output)
    if failures:
        return GradeResult(False, min(grade.score for grade in grades), output)
    return GradeResult(True, min(grade.score for grade in grades), output)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_python_argv(command: list[str]) -> list[str]:
    if command and command[0] == "python" and shutil.which("python") is None:
        return [sys.executable, *command[1:]]
    return command


def normalize_python_command(command: str) -> str:
    if command == "python" or command.startswith("python "):
        if shutil.which("python") is None:
            return sys.executable + command[len("python") :]
    return command
