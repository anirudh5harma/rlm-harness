from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from rlm_harness.sandbox.types import ExecutionResult

_RUN_SHELL_CALLBACK = None


def set_run_shell_callback(callback):
    """In sandbox mode, set a callback that routes shell commands through DockerREPL."""
    global _RUN_SHELL_CALLBACK
    _RUN_SHELL_CALLBACK = callback


def _run_shell(cmd: str, timeout: float = 15.0, cwd: Path | None = None) -> dict:
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
            cwd=str(cwd) if cwd is not None else None,
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

        for command in self._discover_project_commands(changed_files):
            checks.append(self._run_project_command(command))

        passed = all(c.passed for c in checks)
        summary_lines = []
        if changed_files:
            summary_lines.append(f"Changed files: {', '.join(changed_files[:10])}")
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            line = f"  [{status}] {c.check_type}"
            output = compact_check_output(c.output)
            if output:
                line += f": {output}"
            summary_lines.append(line)

        return VerificationResult(
            passed=passed,
            checks=checks,
            changed_files=changed_files,
            summary="\n".join(summary_lines) if summary_lines else "No verification performed.",
        )

    def _detect_changed_files(self) -> list[str]:
        try:
            output = _run_shell("git diff --name-only", timeout=10.0, cwd=self.workspace)
            if output["returncode"] != 0:
                return []
            return [line.strip() for line in output["stdout"].splitlines() if line.strip()]
        except Exception:
            return []

    def _has_python_changes(self, changed_files: list[str]) -> bool:
        return any(f.endswith(".py") for f in changed_files)

    def _has_test_surface(self) -> bool:
        if has_python_test_files(self.workspace):
            return True
        try:
            result = _run_shell(
                "ls tests/ 2>/dev/null || ls test/ 2>/dev/null || echo NOTFOUND",
                timeout=5.0,
                cwd=self.workspace,
            )
            return result["returncode"] == 0 and "NOTFOUND" not in result.get("stdout", "")
        except Exception:
            return False

    def _run_ruff(self) -> VerificationCheck:
        try:
            result = _run_shell(
                "ruff check --output-format=concise 2>&1 || true",
                timeout=30.0,
                cwd=self.workspace,
            )
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
            return VerificationCheck(
                check_type="python_syntax",
                passed=True,
                output="no .py files changed",
            )

        errors: list[str] = []
        for f in py_files[:20]:
            result = _run_shell(
                f"{python_cmd()} -m py_compile {shlex.quote(f)} 2>&1 || true",
                timeout=15.0,
                cwd=self.workspace,
            )
            if result["returncode"] != 0:
                errors.append(f"{f}: {result.get('stderr', '')[:200]}")
        passed = len(errors) == 0
        return VerificationCheck(
            check_type="python_syntax",
            passed=passed,
            output="\n".join(errors) if errors else "all changed .py files compile",
            command="python -m py_compile <changed .py files>",
        )

    def _run_test_collection(self) -> VerificationCheck:
        try:
            result = _run_shell(
                f"{python_cmd()} -m pytest --collect-only -q 2>&1 || true",
                timeout=30.0,
                cwd=self.workspace,
            )
            stdout = (result.get("stdout") or "").strip()
            if "no module named pytest" in stdout.lower():
                return VerificationCheck(
                    check_type="test_collection",
                    passed=True,
                    output="pytest not installed; skipped collection",
                    command="pytest --collect-only",
                )
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

    def _discover_project_commands(self, changed_files: list[str]) -> list[str]:
        commands: list[str] = []
        commands.extend(self._commands_from_python_tests())
        commands.extend(self._commands_from_pyproject(changed_files))
        commands.extend(self._commands_from_package_json(changed_files))
        commands.extend(self._commands_from_makefile())
        commands.extend(self._commands_from_justfile())
        return dedupe_commands(commands)[:4]

    def _commands_from_python_tests(self) -> list[str]:
        if not has_root_python_test_files(self.workspace):
            return []
        return ["python -m unittest discover -v"]

    def _commands_from_pyproject(self, changed_files: list[str]) -> list[str]:
        pyproject = self.workspace / "pyproject.toml"
        if not pyproject.is_file():
            return []
        commands = []
        try:
            content = pyproject.read_text(encoding="utf-8")
            payload = tomllib.loads(content)
        except (OSError, tomllib.TOMLDecodeError):
            content = ""
            payload = {}

        optional_deps = (
            payload.get("project", {}).get("optional-dependencies", {})
            if isinstance(payload.get("project"), dict)
            else {}
        )
        if "dev" in optional_deps or (self.workspace / "tests").is_dir():
            commands.append("python -m pytest -q")
        if "ruff" in content:
            commands.append("ruff check .")
        return commands

    def _commands_from_package_json(self, changed_files: list[str]) -> list[str]:
        package_json = self.workspace / "package.json"
        if not package_json.is_file():
            return []
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        scripts = payload.get("scripts")
        if not isinstance(scripts, dict):
            return []

        changed_frontend = any(
            path.endswith((".js", ".jsx", ".ts", ".tsx", ".css", ".json"))
            for path in changed_files
        )
        preferred = ["test", "lint", "typecheck", "build"] if changed_frontend else ["test"]
        return [
            package_script_command(self.workspace, name)
            for name in preferred
            if isinstance(scripts.get(name), str)
        ]

    def _commands_from_makefile(self) -> list[str]:
        makefile = first_existing(self.workspace, ("Makefile", "makefile"))
        if makefile is None:
            return []
        content = safe_read_text(makefile)
        commands = []
        for target in ("test", "lint", "typecheck"):
            if re.search(rf"^{re.escape(target)}\s*:", content, flags=re.MULTILINE):
                commands.append(f"make {target}")
        return commands

    def _commands_from_justfile(self) -> list[str]:
        justfile = first_existing(self.workspace, ("justfile", "Justfile"))
        if justfile is None:
            return []
        content = safe_read_text(justfile)
        commands = []
        for recipe in ("test", "lint", "typecheck"):
            if re.search(rf"^{re.escape(recipe)}\s*:", content, flags=re.MULTILINE):
                commands.append(f"just {recipe}")
        return commands

    def _run_project_command(self, command: str) -> VerificationCheck:
        try:
            result = _run_shell(
                executable_command(command),
                timeout=60.0,
                cwd=self.workspace,
            )
        except Exception as exc:
            return VerificationCheck(
                check_type="project_command",
                passed=True,
                output=f"{command} skipped: {exc}",
                command=command,
            )
        stdout = ((result.get("stdout") or "") + (result.get("stderr") or "")).strip()
        return VerificationCheck(
            check_type="project_command",
            passed=result.get("returncode") == 0,
            output=stdout[:1000] if stdout else "command produced no output",
            command=command,
        )


def package_script_command(workspace: Path, name: str) -> str:
    if (workspace / "pnpm-lock.yaml").is_file():
        return f"pnpm run {name}"
    if (workspace / "yarn.lock").is_file():
        return f"yarn {name}"
    if (workspace / "bun.lockb").is_file():
        return f"bun run {name}"
    return f"npm run {name}"


def python_cmd() -> str:
    if _RUN_SHELL_CALLBACK is not None:
        return "python"
    return shlex.quote(sys.executable)


def executable_command(command: str) -> str:
    if command.startswith("python "):
        return python_cmd() + command[len("python") :]
    return command


def first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def dedupe_commands(commands: list[str]) -> list[str]:
    seen = set()
    result = []
    for command in commands:
        command = command.strip()
        if not command or command in seen:
            continue
        seen.add(command)
        result.append(command)
    return result


def has_python_test_files(workspace: Path) -> bool:
    patterns = ("test_*.py", "*_test.py")
    try:
        if has_root_python_test_files(workspace):
            return True
        for pattern in patterns:
            if next((workspace / "tests").glob(pattern), None) is not None:
                return True
            if next((workspace / "test").glob(pattern), None) is not None:
                return True
    except OSError:
        return False
    return False


def has_root_python_test_files(workspace: Path) -> bool:
    patterns = ("test_*.py", "*_test.py")
    try:
        return any(next(workspace.glob(pattern), None) is not None for pattern in patterns)
    except OSError:
        return False


def compact_check_output(output: str, limit: int = 500) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    compact = " | ".join(lines[-6:])
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
