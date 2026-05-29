from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rlm_harness.evals.runner import EvalReport, EvalRunner
from rlm_harness.evals.suite import EvalSuiteFileLoader
from rlm_harness.memory import Memory
from rlm_harness.memory.evolution import EvolutionProposalManager
from rlm_harness.memory.feedback import (
    FeedbackRecord,
    FeedbackStore,
    infer_evolution_from_feedback,
    infer_taste_from_feedback,
)
from rlm_harness.memory.profile import TasteProfileStore
from rlm_harness.readiness import ReadinessReport


@dataclass(frozen=True)
class DogfoodSuiteResult:
    name: str
    passed: bool
    pass_rate: float
    skipped: bool = False
    reason: str = ""
    report: dict | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "skipped": self.skipped,
            "reason": self.reason,
            "report": self.report,
        }


@dataclass(frozen=True)
class DogfoodFeedbackResult:
    passed: bool
    feedback_count: int
    taste_count: int
    proposal_count: int
    detail: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "feedback_count": self.feedback_count,
            "taste_count": self.taste_count,
            "proposal_count": self.proposal_count,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DogfoodInstallResult:
    passed: bool
    skipped: bool
    detail: str
    commands: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "skipped": self.skipped,
            "detail": self.detail,
            "commands": self.commands,
        }


@dataclass(frozen=True)
class DogfoodReport:
    status: str
    readiness: dict
    suites: list[DogfoodSuiteResult] = field(default_factory=list)
    feedback: DogfoodFeedbackResult | None = None
    install: DogfoodInstallResult | None = None
    strict_readiness: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "strict_readiness": self.strict_readiness,
            "readiness": self.readiness,
            "suites": [suite.to_dict() for suite in self.suites],
            "feedback": None if self.feedback is None else self.feedback.to_dict(),
            "install": None if self.install is None else self.install.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def run_dogfood(
    readiness: ReadinessReport,
    work_root: Path,
    sandbox_harness_command: list[str],
    no_sandbox_harness_command: list[str],
    timeout_s: int = 900,
    no_docker: bool = False,
    strict_readiness: bool = False,
    install_smoke: bool = False,
    repo_root: Path | None = None,
) -> DogfoodReport:
    work_root.mkdir(parents=True, exist_ok=True)
    suites = [
        run_suite(
            "taste-regression",
            work_root / "taste-regression",
            no_sandbox_harness_command,
            timeout_s,
        )
    ]
    if no_docker:
        suites.append(
            DogfoodSuiteResult(
                name="daily-driver",
                passed=True,
                pass_rate=0.0,
                skipped=True,
                reason="Docker checks were skipped; daily-driver needs sandboxed edits.",
            )
        )
    else:
        suites.append(
            run_suite(
                "daily-driver",
                work_root / "daily-driver",
                sandbox_harness_command,
                timeout_s,
            )
        )

    feedback = run_feedback_smoke(work_root / "feedback")
    install = (
        run_install_smoke(work_root / "install-smoke", repo_root or Path.cwd(), timeout_s)
        if install_smoke
        else DogfoodInstallResult(
            passed=True,
            skipped=True,
            detail="install smoke not requested",
        )
    )
    status = dogfood_status(readiness, suites, feedback, strict_readiness, install)
    return DogfoodReport(
        status=status,
        strict_readiness=strict_readiness,
        readiness=readiness.to_dict(),
        suites=suites,
        feedback=feedback,
        install=install,
    )


def run_suite(
    name: str,
    work_root: Path,
    harness_command: list[str],
    timeout_s: int,
) -> DogfoodSuiteResult:
    suite = EvalSuiteFileLoader().load_suite(name, work_root)
    report = EvalRunner(harness_command=harness_command, timeout_s=timeout_s).run(suite)
    return DogfoodSuiteResult(
        name=name,
        passed=all(result.passed for result in report.results),
        pass_rate=report.pass_rate,
        report=eval_report_to_dict(report),
    )


def run_feedback_smoke(work_root: Path) -> DogfoodFeedbackResult:
    work_root.mkdir(parents=True, exist_ok=True)
    profile_db = work_root / "profile.db"
    memory_db = work_root / "memory.db"
    with Memory(profile_db) as profile_memory, Memory(memory_db) as project_memory:
        feedback = FeedbackStore(profile_memory).add(
            FeedbackRecord.create(
                scope="user",
                rating="positive",
                comment="Liked concise summaries.",
                evidence={"source": "dogfood"},
            )
        )
        taste_store = TasteProfileStore(profile_memory)
        taste_records = [
            taste_store.add(record) for record in infer_taste_from_feedback(feedback)
        ]
        evolution = EvolutionProposalManager(profile_memory, project_memory)
        proposals = [
            proposal
            for proposal in (
                evolution.add(proposal)
                for proposal in infer_evolution_from_feedback(feedback)
            )
            if proposal is not None
        ]
        feedback_count = len(FeedbackStore(profile_memory).records())

    passed = feedback_count >= 1 and len(taste_records) >= 1 and len(proposals) >= 1
    return DogfoodFeedbackResult(
        passed=passed,
        feedback_count=feedback_count,
        taste_count=len(taste_records),
        proposal_count=len(proposals),
        detail=(
            "feedback promoted into taste and evolution proposal"
            if passed
            else "missing signal"
        ),
    )


def dogfood_status(
    readiness: ReadinessReport,
    suites: list[DogfoodSuiteResult],
    feedback: DogfoodFeedbackResult,
    strict_readiness: bool,
    install: DogfoodInstallResult | None = None,
) -> str:
    readiness_ok = readiness.status in {"ready", "degraded"} or not strict_readiness
    suites_ok = all(suite.passed for suite in suites if not suite.skipped)
    install_ok = install is None or install.passed
    if readiness_ok and suites_ok and feedback.passed and install_ok:
        return "passed"
    return "failed"


CommandRunner = Callable[..., subprocess.CompletedProcess]


def run_install_smoke(
    work_root: Path,
    repo_root: Path,
    timeout_s: int,
    command_runner: CommandRunner = subprocess.run,
) -> DogfoodInstallResult:
    work_root.mkdir(parents=True, exist_ok=True)
    venv_dir = work_root / "venv"
    python_bin = venv_dir / "bin" / "python"
    harness_bin = venv_dir / "bin" / "harness"
    commands = [
        [sys.executable, "-m", "venv", str(venv_dir)],
        [str(python_bin), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        [str(python_bin), "-m", "pip", "install", "--quiet", str(repo_root)],
        [
            str(harness_bin),
            "eval",
            "taste-regression",
            "--no-sandbox",
            "--provider",
            "stub",
            "--model",
            "stub",
            "--eval-timeout",
            "60",
        ],
    ]
    for command in commands:
        completed = command_runner(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            output = (completed.stdout or "") + (completed.stderr or "")
            return DogfoodInstallResult(
                passed=False,
                skipped=False,
                detail=f"install smoke failed: {' '.join(command)}\n{output[-1000:]}",
                commands=commands,
            )
    return DogfoodInstallResult(
        passed=True,
        skipped=False,
        detail="fresh venv install and built-in eval passed",
        commands=commands,
    )


def eval_report_to_dict(report: EvalReport) -> dict:
    return json.loads(report.to_json())


def render_dogfood_report(report: DogfoodReport) -> str:
    lines = [f"dogfood\t{report.status}"]
    lines.append(f"readiness\t{report.readiness.get('status')}")
    for suite in report.suites:
        if suite.skipped:
            lines.append(f"suite\t{suite.name}\tskipped\t{suite.reason}")
        else:
            state = "passed" if suite.passed else "failed"
            lines.append(f"suite\t{suite.name}\t{state}\tpass_rate={suite.pass_rate:.3f}")
    if report.feedback is not None:
        state = "passed" if report.feedback.passed else "failed"
        lines.append(
            f"feedback\t{state}\tfeedback={report.feedback.feedback_count}"
            f"\ttaste={report.feedback.taste_count}"
            f"\tproposals={report.feedback.proposal_count}"
        )
    if report.install is not None:
        if report.install.skipped:
            lines.append(f"install\tskipped\t{report.install.detail}")
        else:
            state = "passed" if report.install.passed else "failed"
            lines.append(f"install\t{state}\t{report.install.detail}")
    return "\n".join(lines)
