from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
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
_COMPLETION_SINK = None
_PENDING_CHANGES: dict[str, dict[str, Any]] = {}


class ToolError(RuntimeError):
    pass


def set_completion_sink(callback) -> None:
    global _COMPLETION_SINK
    _COMPLETION_SINK = callback


def complete_task(
    summary: str,
    status: str = "success",
    verification: Optional[str] = None,
) -> dict[str, Any]:
    if not isinstance(summary, str) or not summary.strip():
        raise ToolError("summary must be a non-empty string")
    if status not in {"success", "partial", "blocked"}:
        raise ToolError("status must be success, partial, or blocked")
    payload = {
        "summary": summary.strip(),
        "status": status,
        "verification": verification or "",
        "should_continue": False,
    }
    if _COMPLETION_SINK is not None:
        _COMPLETION_SINK(payload)
    return payload


def propose_file_change(path: str, content: str, reason: str = "") -> dict[str, Any]:
    target = workspace_path(path)
    if target.exists() and not target.is_file():
        raise ToolError(f"not a file: {path}")
    change_id = f"change-{uuid.uuid4().hex[:12]}"
    before = ""
    if target.is_file():
        before = target.read_text(encoding="utf-8", errors="replace")
    pending = {
        "id": change_id,
        "path": str(target.relative_to(WORKSPACE)),
        "reason": reason,
        "before": before,
        "content": content,
        "diff": unified_diff(str(target.relative_to(WORKSPACE)), before, content),
    }
    _PENDING_CHANGES[change_id] = pending
    return {
        "id": change_id,
        "path": pending["path"],
        "reason": reason,
        "diff": pending["diff"],
        "approval_required": True,
    }


def list_pending_changes() -> list[dict[str, Any]]:
    return [
        {
            "id": change["id"],
            "path": change["path"],
            "reason": change["reason"],
            "diff": change["diff"],
        }
        for change in _PENDING_CHANGES.values()
    ]


def apply_pending_change(change_id: str) -> str:
    change_id = change_id.strip()
    if not change_id:
        raise ToolError("change_id must be non-empty")
    change = _PENDING_CHANGES.pop(change_id, None)
    if change is None:
        raise ToolError(f"unknown pending change: {change_id}")
    return write_file(str(change["path"]), str(change["content"]))


def clear_pending_changes() -> str:
    count = len(_PENDING_CHANGES)
    _PENDING_CHANGES.clear()
    return f"cleared {count} pending change(s)"


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


def read_file_slice(
    path: str,
    start: int = 0,
    max_bytes: int = 12_000,
) -> dict[str, Any]:
    target = workspace_path(path)
    if start < 0:
        raise ToolError("start must be non-negative")
    if max_bytes <= 0:
        raise ToolError("max_bytes must be positive")
    if not target.is_file():
        raise ToolError(f"not a file: {path}")

    data = target.read_bytes()
    total = len(data)
    end = min(start + max_bytes, total)
    content = data[start:end].decode("utf-8", errors="replace") if start < total else ""
    return {
        "path": str(target.relative_to(WORKSPACE)),
        "start": start,
        "end": end,
        "total_bytes": total,
        "truncated": end < total,
        "content": content,
    }


def chunk_file(
    path: str,
    chunk_chars: int = 12_000,
    max_chunks: int = 40,
) -> list[dict[str, Any]]:
    if chunk_chars <= 0:
        raise ToolError("chunk_chars must be positive")
    if max_chunks <= 0:
        raise ToolError("max_chunks must be positive")

    max_bytes = chunk_chars * max_chunks * 4
    content = read_file(path, max_bytes=max_bytes)
    chunks = []
    for index, start in enumerate(range(0, len(content), chunk_chars)):
        if index >= max_chunks:
            break
        end = min(start + chunk_chars, len(content))
        chunks.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "chars": end - start,
                "content": content[start:end],
            }
        )
    return chunks


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


def project_summary(
    max_files: int = 300,
    max_read_bytes: int = 12_000,
) -> str:
    return render_project_overview_summary(
        project_overview(max_files=max_files, max_read_bytes=max_read_bytes)
    )


def project_audit(
    max_files: int = 500,
    max_read_bytes: int = 16_000,
) -> str:
    return render_project_audit(
        project_overview(max_files=max_files, max_read_bytes=max_read_bytes)
    )


def is_project_overview_payload(payload: dict) -> bool:
    return isinstance(payload.get("files"), list) and isinstance(payload.get("documents"), list)


def render_project_overview_summary(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    doc_paths = [str(doc.get("path")) for doc in documents if doc.get("path")]
    package = package_json_from_documents(documents)
    pyproject = pyproject_from_documents(documents)
    scripts = package.get("scripts", {}) if package else {}
    dependencies = package_dependencies(package)

    sections = ["Project Summary"]
    name = project_name(package, pyproject)
    description = project_description(package, pyproject, documents)
    if name or description:
        if name and description:
            sections.append(f"What it is: {name} - {description}")
        elif name:
            sections.append(f"What it is: {name}")
        else:
            sections.append(f"What it is: {description}")

    sections.append(f"Files inspected: {len(files)}")
    if doc_paths:
        sections.append("Key config/docs: " + ", ".join(doc_paths[:8]))

    stack = infer_project_stack(files, documents, dependencies)
    if stack:
        sections.append("Tech stack: " + ", ".join(stack))

    architecture = infer_project_architecture(files)
    if architecture:
        sections.append("Architecture: " + architecture)

    commands = render_scripts(scripts)
    if commands:
        sections.append("Useful commands:\n" + commands)

    git_status_text = str(payload.get("git_status") or "").strip()
    if git_status_text:
        sections.append("Working tree:\n" + git_status_text)

    git_log_text = str(payload.get("git_log") or "").strip()
    if git_log_text:
        sections.append("Recent commits:\n" + "\n".join(git_log_text.splitlines()[:5]))

    notable_files = notable_source_files(files)
    if notable_files:
        sections.append(
            "Notable source files:\n" + "\n".join(f"- {path}" for path in notable_files)
        )

    return "\n\n".join(sections)


def render_project_audit(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    package = package_json_from_documents(documents)
    pyproject = pyproject_from_documents(documents)
    dependencies = package_dependencies(package)
    scripts = package.get("scripts", {}) if package else {}
    stack = infer_project_stack(files, documents, dependencies)
    findings = project_audit_findings(files, documents, package, pyproject)

    sections = ["Project Gap Analysis"]
    summary = project_description(package, pyproject, documents)
    if summary:
        sections.append("Context: " + summary)
    if stack:
        sections.append("Detected stack: " + ", ".join(stack))

    reviewed = audit_reviewed_paths(files, documents)
    if reviewed:
        sections.append("Evidence reviewed:\n" + "\n".join(f"- {path}" for path in reviewed))

    if findings:
        rendered_findings = []
        for index, finding in enumerate(findings, start=1):
            rendered_findings.append(
                "\n".join(
                    [
                        f"{index}. [{finding['severity']}] {finding['title']}",
                        f"   Evidence: {finding['evidence']}",
                        f"   Impact: {finding['impact']}",
                        f"   Recommendation: {finding['recommendation']}",
                    ]
                )
            )
        sections.append("Findings:\n" + "\n\n".join(rendered_findings))
    else:
        sections.append(
            "Findings:\n"
            "No obvious structural gaps were detected from the high-level project files. "
            "A stronger audit should still inspect feature code, run the test suite, and "
            "exercise the main user workflows."
        )

    commands = audit_verification_commands(scripts, files)
    if commands:
        sections.append("Suggested verification:\n" + "\n".join(f"- {cmd}" for cmd in commands))

    return "\n\n".join(sections)


def project_audit_findings(
    files: list[str],
    documents: list[dict],
    package: dict,
    pyproject: dict,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    scripts = package.get("scripts", {}) if package else {}
    dependencies = package_dependencies(package)
    frontend = bool(package) or any(path.startswith("src/") for path in files)
    python_project = bool(pyproject) or "pyproject.toml" in files

    if frontend and not has_test_surface(files, scripts):
        findings.append(
            audit_finding(
                "high",
                "No visible automated test surface",
                "package.json does not expose a test script and no common test files were found.",
                "Logical regressions in routes, state, and UI behavior can ship without a fast "
                "feedback loop.",
                "Add focused unit/component tests for core behavior and expose them through "
                "npm run test or an equivalent CI command.",
            )
        )

    if python_project and not has_python_test_surface(files):
        findings.append(
            audit_finding(
                "high",
                "No visible Python test coverage",
                "pyproject.toml is present but no tests/ directory or test_*.py files were found.",
                "Harness, CLI, or runtime changes can regress without an executable safety net.",
                "Add tests for the main runtime paths and document the command in project "
                "metadata.",
            )
        )

    if frontend and "build" not in scripts:
        findings.append(
            audit_finding(
                "medium",
                "Build command is not discoverable",
                "package.json does not define a build script.",
                "Deployments and local verification depend on undocumented knowledge.",
                "Add a package.json build script that matches the deployment target.",
            )
        )

    if frontend and "lint" not in scripts:
        findings.append(
            audit_finding(
                "medium",
                "Static analysis command is not discoverable",
                "package.json does not define a lint script.",
                "Type, import, accessibility, and style regressions are harder to catch "
                "consistently.",
                "Expose linting or typechecking through package scripts and run it in CI.",
            )
        )

    if frontend and "typescript" in dependencies and not has_typecheck_script(scripts):
        findings.append(
            audit_finding(
                "medium",
                "TypeScript typechecking is not a first-class script",
                "TypeScript is used, but package.json has no typecheck script.",
                "Build tools can miss or delay some project-wide type errors depending on the "
                "framework pipeline.",
                "Add a typecheck script such as tsc --noEmit, or document why the build command "
                "is sufficient.",
            )
        )

    if frontend and "public/placeholder.svg" in files:
        findings.append(
            audit_finding(
                "low",
                "Placeholder asset remains in the public bundle",
                "public/placeholder.svg is present.",
                "Generated starter assets can leak into production or mask missing final assets.",
                "Remove the placeholder or replace it with an intentional product asset.",
            )
        )

    if frontend and has_many_ui_primitives(files) and not has_domain_component_layer(files):
        findings.append(
            audit_finding(
                "medium",
                "UI primitive layer is present without an obvious domain component layer",
                "Many src/components/ui/* files exist, but no non-primitive component directory "
                "was found.",
                "Application behavior may be concentrated in routes, making reuse and testing "
                "harder as the project grows.",
                "Extract domain components for repeated workflows and test them directly.",
            )
        )

    if frontend and "eslint.config.js" in files and "lint" not in scripts:
        findings.append(
            audit_finding(
                "low",
                "ESLint config exists but is not wired into scripts",
                "eslint.config.js exists while package.json has no lint script.",
                "Developers may skip static analysis because the expected command is unclear.",
                "Add a lint script that invokes ESLint over the source tree.",
            )
        )

    if not has_readme(documents):
        findings.append(
            audit_finding(
                "medium",
                "Project documentation is missing or not detected",
                "No README file was found among the inspected project documents.",
                "Setup, architecture, and operational assumptions become implicit and harder to "
                "audit.",
                "Add a README with purpose, setup, verification commands, and deployment notes.",
            )
        )

    if "vercel.json" in files and "build-vercel.mjs" in files and frontend:
        findings.append(
            audit_finding(
                "low",
                "Deployment has a custom build path that deserves verification",
                "vercel.json and build-vercel.mjs are both present.",
                "Custom deployment glue can diverge from local build behavior.",
                "Run the Vercel build command locally and document how it differs from npm run "
                "build.",
            )
        )

    return findings


def audit_finding(
    severity: str,
    title: str,
    evidence: str,
    impact: str,
    recommendation: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "impact": impact,
        "recommendation": recommendation,
    }


def audit_reviewed_paths(files: list[str], documents: list[dict]) -> list[str]:
    doc_paths = [str(doc.get("path")) for doc in documents if doc.get("path")]
    preferred = [
        *doc_paths,
        "src/routes/index.tsx",
        "src/routes/__root.tsx",
        "src/router.tsx",
        "src/content.ts",
        "src/styles.css",
        "rlm_harness/cli.py",
        "rlm_harness/graph/nodes.py",
        "rlm_harness/rlm/runtime.py",
        "rlm_harness/sandbox/tools.py",
    ]
    return [path for path in dedupe(preferred) if path in files][:12]


def audit_verification_commands(scripts: dict, files: list[str]) -> list[str]:
    commands = []
    if isinstance(scripts, dict):
        for name in ("test", "lint", "typecheck", "build"):
            if isinstance(scripts.get(name), str):
                commands.append(f"npm run {name}")
    if "pyproject.toml" in files and any(path.startswith("tests/") for path in files):
        commands.append("pytest")
    return commands


def package_json_from_documents(documents: list[dict]) -> dict:
    for doc in documents:
        if doc.get("path") == "package.json" and isinstance(doc.get("content"), str):
            try:
                payload = json.loads(str(doc["content"]))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def pyproject_from_documents(documents: list[dict]) -> dict:
    for doc in documents:
        if doc.get("path") == "pyproject.toml" and isinstance(doc.get("content"), str):
            return parse_pyproject_metadata(str(doc["content"]))
    return {}


def parse_pyproject_metadata(content: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        try:
            payload = tomllib.loads(content)
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            project = payload.get("project")
            return project if isinstance(project, dict) else {}

    metadata: dict[str, str] = {}
    in_project = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if not in_project or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key in {"name", "description"} and value:
            metadata[key] = value
    return metadata


def project_name(package: dict, pyproject: dict) -> str:
    for payload in (package, pyproject):
        name = payload.get("name") if isinstance(payload, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return ""


def project_description(package: dict, pyproject: dict, documents: list[dict]) -> str:
    for payload in (package, pyproject):
        description = payload.get("description") if isinstance(payload, dict) else None
        if isinstance(description, str) and description.strip():
            return one_line(description)

    readme = readme_content_from_documents(documents)
    if readme:
        return one_line(first_readme_paragraph(readme))
    return ""


def readme_content_from_documents(documents: list[dict]) -> str:
    for doc in documents:
        path = str(doc.get("path") or "").lower()
        content = doc.get("content")
        if path.startswith("readme") and isinstance(content, str):
            return content
    return ""


def first_readme_paragraph(content: str) -> str:
    lines = []
    seen_heading = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        if line.startswith("#"):
            seen_heading = True
            continue
        if seen_heading or not lines:
            lines.append(line)
    return " ".join(lines)


def one_line(text: str, max_chars: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def package_dependencies(package: dict) -> set[str]:
    names: set[str] = set()
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        values = package.get(section)
        if isinstance(values, dict):
            names.update(str(name) for name in values)
    return names


def infer_project_stack(
    files: list[str],
    documents: list[dict],
    dependencies: set[str],
) -> list[str]:
    stack = []
    if "package.json" in {doc.get("path") for doc in documents}:
        stack.append("Node.js")
    if "typescript" in dependencies or any(path.endswith((".ts", ".tsx")) for path in files):
        stack.append("TypeScript")
    if "react" in dependencies:
        stack.append("React")
    if "@tanstack/react-start" in dependencies:
        stack.append("TanStack Start")
    if "@tanstack/react-router" in dependencies:
        stack.append("TanStack Router")
    if "vite" in dependencies or "vite.config.ts" in files:
        stack.append("Vite")
    if "@tailwindcss/vite" in dependencies or "tailwindcss" in dependencies:
        stack.append("Tailwind CSS")
    if any(name.startswith("@radix-ui/") for name in dependencies):
        stack.append("Radix UI")
    if "pyproject.toml" in files:
        stack.append("Python")
    return dedupe(stack)


def infer_project_architecture(files: list[str]) -> str:
    details = []
    if any(path.startswith("rlm_harness/graph/") for path in files):
        details.append("graph orchestration under rlm_harness/graph")
    if any(path.startswith("rlm_harness/sandbox/") for path in files):
        details.append("sandbox execution under rlm_harness/sandbox")
    if any(path.startswith("rlm_harness/rlm/") for path in files):
        details.append("recursive runtime under rlm_harness/rlm")
    if any(path.startswith("src/routes/") for path in files):
        details.append("route modules under src/routes")
    if any(path.startswith("src/components/ui/") for path in files):
        details.append("shared UI primitives under src/components/ui")
    if "src/router.tsx" in files:
        details.append("router setup in src/router.tsx")
    if "src/styles.css" in files:
        details.append("global styles in src/styles.css")
    if any(path.startswith("public/") for path in files):
        details.append("static assets under public")
    if any(path.startswith("tests/") for path in files):
        details.append("test coverage under tests")
    return "; ".join(details)


def render_scripts(scripts: dict) -> str:
    if not isinstance(scripts, dict):
        return ""
    preferred = ["dev", "build", "preview", "lint", "test"]
    lines = []
    for name in preferred:
        command = scripts.get(name)
        if isinstance(command, str):
            lines.append(f"- npm run {name}: {command}")
    return "\n".join(lines)


def notable_source_files(files: list[str]) -> list[str]:
    preferred = [
        "package.json",
        "pyproject.toml",
        "README.md",
        "rlm_harness/cli.py",
        "rlm_harness/graph/nodes.py",
        "rlm_harness/rlm/runtime.py",
        "rlm_harness/sandbox/tools.py",
        "src/routes/index.tsx",
        "src/routes/__root.tsx",
        "src/router.tsx",
        "src/content.ts",
        "src/styles.css",
        "vite.config.ts",
        "tsconfig.json",
    ]
    return [path for path in preferred if path in files][:8]


def has_test_surface(files: list[str], scripts: dict) -> bool:
    if isinstance(scripts.get("test"), str):
        return True
    test_markers = (
        ".test.",
        ".spec.",
        "__tests__/",
        "tests/",
        "vitest.config.",
        "jest.config.",
        "playwright.config.",
        "cypress.config.",
    )
    return any(any(marker in path for marker in test_markers) for path in files)


def has_python_test_surface(files: list[str]) -> bool:
    return any(
        path.startswith("tests/")
        or path.endswith("_test.py")
        or Path(path).name.startswith("test_")
        for path in files
    )


def has_typecheck_script(scripts: dict) -> bool:
    for name, command in scripts.items():
        if not isinstance(name, str) or not isinstance(command, str):
            continue
        lowered = f"{name} {command}".lower()
        if "typecheck" in lowered or "tsc --noemit" in lowered or "tsc --no-emit" in lowered:
            return True
    return False


def has_many_ui_primitives(files: list[str]) -> bool:
    return sum(1 for path in files if path.startswith("src/components/ui/")) >= 8


def has_domain_component_layer(files: list[str]) -> bool:
    return any(
        path.startswith("src/components/")
        and not path.startswith("src/components/ui/")
        and path.endswith((".ts", ".tsx", ".js", ".jsx"))
        for path in files
    )


def has_readme(documents: list[dict]) -> bool:
    return any(str(doc.get("path") or "").lower().startswith("readme") for doc in documents)


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


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


def run_shell(
    cmd: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    allow_dangerous: bool = False,
) -> dict[str, Any]:
    if not cmd.strip():
        raise ToolError("cmd must be non-empty")
    if timeout <= 0:
        raise ToolError("timeout must be positive")
    if not allow_dangerous and looks_dangerous_command(cmd):
        raise ToolError(
            "command looks destructive; ask the user for approval or pass "
            "allow_dangerous=True only after explicit approval"
        )
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
        "propose_file_change",
        "list_pending_changes",
        "apply_pending_change",
        "clear_pending_changes",
        "read_file_slice",
        "chunk_file",
        "read_first_existing",
        "list_files",
        "project_overview",
        "project_summary",
        "project_audit",
        "write_file",
        "apply_patch",
        "run_shell",
        "git_status",
        "git_diff",
        "git_log",
        "search_code",
        "complete_task",
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


def looks_dangerous_command(cmd: str) -> bool:
    lowered = re.sub(r"\s+", " ", cmd.strip().lower())
    patterns = (
        r"\brm\s+(-[a-z]*r[a-z]*f|-rf|-fr)\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+(-[a-z]*f|-[a-z]*x|-[a-z]*d)",
        r"\bgit\s+checkout\s+--\b",
        r"\bsudo\b",
        r"\bdd\s+.*\bof=",
        r"\bmkfs(\.| )",
        r"\bcurl\b.*\|\s*(sh|bash)\b",
        r"\bwget\b.*\|\s*(sh|bash)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def unified_diff(path: str, before: str, after: str) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


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
        "name": "propose_file_change",
        "description": "Queue a file change and return a diff for review before applying.",
        "parameters": {
            "path": "workspace-relative path",
            "content": "proposed full file content",
            "reason": "optional reason for the proposal",
        },
    },
    {
        "name": "list_pending_changes",
        "description": "List queued file-change proposals with diffs.",
        "parameters": {},
    },
    {
        "name": "apply_pending_change",
        "description": "Apply one queued file-change proposal after approval.",
        "parameters": {"change_id": "pending change id"},
    },
    {
        "name": "clear_pending_changes",
        "description": "Discard all queued file-change proposals.",
        "parameters": {},
    },
    {
        "name": "read_file_slice",
        "description": "Read a bounded byte slice from a workspace text file.",
        "parameters": {
            "path": "workspace-relative path",
            "start": "optional zero-based byte offset",
            "max_bytes": "optional byte count",
        },
    },
    {
        "name": "chunk_file",
        "description": "Split a workspace text file into bounded character chunks.",
        "parameters": {
            "path": "workspace-relative path",
            "chunk_chars": "optional chunk size",
            "max_chunks": "optional chunk cap",
        },
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
        "name": "project_summary",
        "description": (
            "Return a concise human-readable summary of what the workspace project is."
        ),
        "parameters": {"max_files": "optional file cap", "max_read_bytes": "optional per-file cap"},
    },
    {
        "name": "project_audit",
        "description": (
            "Return an evidence-backed gap analysis for project review, audit, risk, or "
            "technical-debt questions."
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
    {
        "name": "complete_task",
        "description": "Signal that the requested task is complete, partial, or blocked.",
        "parameters": {
            "summary": "user-facing summary",
            "status": "success, partial, or blocked",
            "verification": "optional verification evidence",
        },
    },
]
