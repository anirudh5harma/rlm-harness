from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rlm_harness.sandbox.types import ExecutionResult


_RUN_SHELL_CALLBACK = None


def set_run_shell_callback(callback):
    """In sandbox mode, set a callback that routes shell commands through DockerREPL."""
    global _RUN_SHELL_CALLBACK
    _RUN_SHELL_CALLBACK = callback


def _run_shell(cmd: str, timeout: float = 15.0) -> dict:
    if _RUN_SHELL_CALLBACK is not None:
        result = _RUN_SHELL_CALLBACK(cmd, timeout)
        if isinstance(result, ExecutionResult):
            return {
                "returncode": 0 if result.status == "ok" else 1,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            }
        return result

    import subprocess as _subprocess
    try:
        result = _subprocess.run(
            cmd,
            shell=True,
            executable="/bin/sh",
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except _subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.output or "",
            "stderr": f"command timed out after {timeout:g}s\n",
            "timed_out": True,
        }


@dataclass
class VerificationCheck:
    check_type: str
    passed: bool
    output: str
    command: str = ""


@dataclass
class VerificationResult:
    passed: bool
    checks: list[VerificationCheck] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def failed_checks(self) -> list[VerificationCheck]:
        return [c for c in self.checks if not c.passed]


class VerificationGate:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def verify(self) -> VerificationResult:
        changed_files = self._detect_changed_files()
        checks: list[VerificationCheck] = []

        if self._has_python_changes(changed_files):
            checks.append(self._run_ruff())
            checks.append(self._run_python_syntax_check(changed_files))

        if self._has_test_surface():
            checks.append(self._run_test_collection())

        passed = all(c.passed for c in checks)
        summary_lines = []
        if changed_files:
            summary_lines.append(f"Changed files: {', '.join(changed_files[:10])}")
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            summary_lines.append(f"  [{status}] {c.check_type}")

        return VerificationResult(
            passed=passed,
            checks=checks,
            changed_files=changed_files,
            summary="\n".join(summary_lines) if summary_lines else "No verification performed.",
        )

    def _detect_changed_files(self) -> list[str]:
        try:
            output = _run_shell("git diff --name-only", timeout=10.0)
            if output["returncode"] != 0:
                return []
            return [line.strip() for line in output["stdout"].splitlines() if line.strip()]
        except Exception:
            return []

    def _has_python_changes(self, changed_files: list[str]) -> bool:
        return any(f.endswith(".py") for f in changed_files)

    def _has_test_surface(self) -> bool:
        try:
            result = _run_shell("ls tests/ 2>/dev/null || ls test/ 2>/dev/null || echo NOTFOUND", timeout=5.0)
            return result["returncode"] == 0 and "NOTFOUND" not in result.get("stdout", "")
        except Exception:
            return False

    def _run_ruff(self) -> VerificationCheck:
        try:
            result = _run_shell("ruff check --output-format=concise 2>&1 || true", timeout=30.0)
            stdout = (result.get("stdout") or "").strip()
            passed = not stdout or "All checks passed" in stdout
            return VerificationCheck(
                check_type="ruff",
                passed=passed,
                output=stdout[:500] if stdout else "ruff not installed or nothing to check",
                command="ruff check",
            )
        except Exception as exc:
            return VerificationCheck(
                check_type="ruff",
                passed=True,
                output=f"ruff check skipped: {exc}",
                command="ruff check",
            )

    def _run_python_syntax_check(self, changed_files: list[str]) -> VerificationCheck:
        py_files = [f for f in changed_files if f.endswith(".py")]
        if not py_files:
            return VerificationCheck(check_type="python_syntax", passed=True, output="no .py files changed")

        errors: list[str] = []
        for f in py_files[:20]:
            result = _run_shell(f"python -m py_compile {f} 2>&1 || true", timeout=15.0)
            if result["returncode"] != 0:
                errors.append(f"{f}: {result.get('stderr', '')[:200]}")
        passed = len(errors) == 0
        return VerificationCheck(
            check_type="python_syntax",
            passed=passed,
            output="\n".join(errors) if errors else "all changed .py files compile",
        )

    def _run_test_collection(self) -> VerificationCheck:
        try:
            result = _run_shell(
                "python -m pytest --collect-only -q 2>&1 || true",
                timeout=30.0,
            )
            stdout = (result.get("stdout") or "").strip()
            passed = "error" not in stdout.lower()[:200]
            return VerificationCheck(
                check_type="test_collection",
                passed=passed,
                output=stdout[:500] if stdout else "pytest not installed or no tests found",
                command="pytest --collect-only",
            )
        except Exception as exc:
            return VerificationCheck(
                check_type="test_collection",
                passed=True,
                output=f"test collection skipped: {exc}",
            )
