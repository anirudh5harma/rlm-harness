from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

WORKSPACE = Path("/workspace")
DEFAULT_MAX_READ_BYTES = 200_000
DEFAULT_TIMEOUT_S = 30.0
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".rlm_harness",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "node_modules",
}
PROJECT_OVERVIEW_CANDIDATES = (
    "README.md",
    "README.rst",
    "README.txt",
    "readme.md",
    "readme.rst",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "tsconfig.json",
    "next.config.js",
    "vite.config.ts",
)


class ToolError(RuntimeError):
    pass


def read_file(path: str, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> str:
    target = workspace_path(path)
    if max_bytes <= 0:
        raise ToolError("max_bytes must be positive")
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    data = target.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def read_first_existing(
    paths: list[str],
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
) -> dict[str, Any]:
    if not isinstance(paths, list) or not paths:
        raise ToolError("paths must be a non-empty list of workspace-relative strings")
    for path in paths:
        if not isinstance(path, str) or not path.strip():
            raise ToolError("paths must contain only non-empty strings")
        target = workspace_path(path)
        if target.is_file():
            return {
                "path": str(target.relative_to(WORKSPACE)),
                "content": read_file(path, max_bytes),
            }
    return {"path": None, "content": ""}


def list_files(path: str = ".", max_depth: int = 4, max_count: int = 300) -> list[str]:
    if max_depth <= 0:
        raise ToolError("max_depth must be positive")
    if max_count <= 0:
        raise ToolError("max_count must be positive")
    root = workspace_path(path)
    if not root.exists():
        raise ToolError(f"path does not exist: {path}")
    if root.is_file():
        return [str(root.relative_to(WORKSPACE))]

    files: list[str] = []
    root_depth = len(root.relative_to(WORKSPACE).parts)
    for current, dirs, filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(WORKSPACE).parts) - root_depth
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        if depth >= max_depth:
            dirs[:] = []
        for filename in sorted(filenames):
            relative = str((current_path / filename).relative_to(WORKSPACE))
            files.append(relative)
            if len(files) >= max_count:
                return files
    return files


def project_overview(
    max_files: int = 300,
    max_read_bytes: int = 12_000,
) -> dict[str, Any]:
    files = list_files(".", max_depth=4, max_count=max_files)
    files_by_lower = {path.lower(): path for path in files}
    selected_paths = []
    for candidate in PROJECT_OVERVIEW_CANDIDATES:
        path = files_by_lower.get(candidate.lower())
        if path and path not in selected_paths:
            selected_paths.append(path)

    documents = []
    for path in selected_paths:
        try:
            documents.append({"path": path, "content": read_file(path, max_read_bytes)})
        except ToolError as exc:
            documents.append({"path": path, "error": str(exc)})

    return {
        "files": files,
        "documents": documents,
        "git_status": safe_git_status(),
        "git_log": safe_git_log(),
    }


def write_file(path: str, content: str) -> str:
    target = workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {target.relative_to(WORKSPACE)} ({len(content.encode('utf-8'))} bytes)"


def apply_patch(diff: str, timeout: float = DEFAULT_TIMEOUT_S) -> str:
    if not diff.strip():
        raise ToolError("diff must be non-empty")
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=diff,
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ToolError(render_command_failure(result))
    return "patch applied"


def run_shell(cmd: str, timeout: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    if not cmd.strip():
        raise ToolError("cmd must be non-empty")
    if timeout <= 0:
        raise ToolError("timeout must be positive")
    try:
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE,
            shell=True,
            executable="/bin/sh",
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\ncommand timed out after {timeout:g}s\n",
            "timed_out": True,
        }
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": False,
    }


def git_status() -> str:
    return run_git(["status", "--short"])


def git_diff(path: Optional[str] = None) -> str:
    command = ["diff", "--"]
    if path:
        command.append(str(workspace_path(path).relative_to(WORKSPACE)))
    return run_git(command)


def git_log(n: int = 10) -> str:
    if n <= 0:
        raise ToolError("n must be positive")
    return run_git(["log", f"-{n}", "--oneline"])


def safe_git_status() -> str:
    try:
        return git_status()
    except ToolError:
        return ""


def safe_git_log(n: int = 10) -> str:
    try:
        return git_log(n)
    except ToolError:
        return ""


def search_code(pattern: str, path: str = ".", max_count: int = 100) -> str:
    if not pattern:
        raise ToolError("pattern must be non-empty")
    if max_count <= 0:
        raise ToolError("max_count must be positive")
    search_root = workspace_path(path)
    result = subprocess.run(
        ["rg", "--line-number", "--max-count", str(max_count), pattern, str(search_root)],
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout
    if result.returncode == 1:
        return ""
    raise ToolError(render_command_failure(result))


def tool_names() -> list[str]:
    return [
        "read_file",
        "read_first_existing",
        "list_files",
        "project_overview",
        "write_file",
        "apply_patch",
        "run_shell",
        "git_status",
        "git_diff",
        "git_log",
        "search_code",
    ]


def tool_help() -> str:
    return json.dumps(TOOL_SCHEMAS, indent=2, sort_keys=True)


def run_git(args: list[str], timeout: float = DEFAULT_TIMEOUT_S) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ToolError(render_command_failure(result))
    return result.stdout


def workspace_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ToolError("path must be a non-empty string")
    raw = Path(path)
    if raw.is_absolute():
        try:
            target = raw.resolve(strict=False)
            target.relative_to(WORKSPACE)
        except ValueError as exc:
            raise ToolError(f"path escapes workspace: {path}") from exc
        return target

    target = (WORKSPACE / raw).resolve(strict=False)
    try:
        target.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ToolError(f"path escapes workspace: {path}") from exc
    return target


def render_command_failure(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the mounted workspace.",
        "parameters": {"path": "workspace-relative path", "max_bytes": "optional byte cap"},
    },
    {
        "name": "read_first_existing",
        "description": "Read the first existing file from a candidate path list.",
        "parameters": {"paths": "list of candidate paths", "max_bytes": "optional byte cap"},
    },
    {
        "name": "list_files",
        "description": "List workspace files while skipping common generated dependency dirs.",
        "parameters": {
            "path": "optional workspace-relative root",
            "max_depth": "optional traversal depth",
            "max_count": "optional result cap",
        },
    },
    {
        "name": "project_overview",
        "description": (
            "Return file list, common docs/config files, git status, and recent git log."
        ),
        "parameters": {"max_files": "optional file cap", "max_read_bytes": "optional per-file cap"},
    },
    {
        "name": "write_file",
        "description": "Write UTF-8 text to a workspace file, creating parent directories.",
        "parameters": {"path": "workspace-relative path", "content": "new file content"},
    },
    {
        "name": "apply_patch",
        "description": "Apply a unified diff to the workspace with git apply.",
        "parameters": {"diff": "unified diff"},
    },
    {
        "name": "run_shell",
        "description": "Run a shell command in /workspace inside the Docker sandbox.",
        "parameters": {"cmd": "shell command", "timeout": "optional seconds"},
    },
    {
        "name": "git_status",
        "description": "Return git status --short for the workspace.",
        "parameters": {},
    },
    {
        "name": "git_diff",
        "description": "Return git diff, optionally scoped to one workspace path.",
        "parameters": {"path": "optional workspace-relative path"},
    },
    {
        "name": "git_log",
        "description": "Return recent one-line git commits.",
        "parameters": {"n": "optional positive commit count"},
    },
    {
        "name": "search_code",
        "description": "Search workspace text with ripgrep.",
        "parameters": {"pattern": "regex pattern", "path": "optional path", "max_count": "cap"},
    },
]
