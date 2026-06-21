from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

WORKSPACE = Path("/workspace")


def set_workspace(path: Path | str) -> None:
    """Override the workspace root for the local REPL path.

    The sandbox REPL mounts the project at ``/workspace``; the local
    REPL runs in the user's process and uses the real workspace path.
    The tools resolve all relative paths against this module-level
    ``WORKSPACE`` constant, so the local REPL must point it at the
    runtime's workspace before any tool is called.
    """
    global WORKSPACE
    WORKSPACE = Path(path).resolve()
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
    "main.py",
    "app.py",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Makefile",
    "Dockerfile",
    "Dockerfile.dev",
    "Dockerfile.prod",
    ".dockerignore",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    ".env.example",
    ".env.sample",
    ".env.template",
    "env.example",
    "env.sample",
    "wrangler.toml",
    "vercel.json",
    "netlify.toml",
    "fly.toml",
    "railway.toml",
    "railway.json",
    "render.yaml",
    "render.yml",
    "Procfile",
    "amplify.yml",
    "prisma/schema.prisma",
    "drizzle.config.ts",
    "drizzle.config.js",
    "supabase/config.toml",
    "schema.prisma",
    "schema.sql",
    "db/schema.sql",
    "db/schema.ts",
    "database/schema.sql",
    "alembic.ini",
    "migrations/env.py",
    "knexfile.ts",
    "knexfile.js",
    "typeorm.config.ts",
    "typeorm.config.js",
    "sequelize.config.js",
    "routes/api.php",
    "routes/web.php",
    "urls.py",
    "auth.ts",
    "auth.js",
    "auth.config.ts",
    "auth.config.js",
    "middleware.ts",
    "middleware.js",
    "src/auth.ts",
    "src/auth.js",
    "src/middleware.ts",
    "src/middleware.js",
    "lib/auth.ts",
    "lib/auth.js",
    "lib/session.ts",
    "lib/session.js",
    "src/session.ts",
    "src/session.js",
    "app/api/auth/[...nextauth]/route.ts",
    "app/api/auth/[...nextauth]/route.js",
    "pages/api/auth/[...nextauth].ts",
    "pages/api/auth/[...nextauth].js",
    "app/login/page.tsx",
    "app/sign-in/page.tsx",
    "app/signup/page.tsx",
    "routes/auth.ts",
    "routes/auth.py",
    "src/routes/auth.ts",
    "src/routes/auth.py",
    "auth.py",
    "app/page.tsx",
    "app/page.ts",
    "app/layout.tsx",
    "app/layout.ts",
    "pages/index.tsx",
    "pages/index.ts",
    "src/App.tsx",
    "src/App.jsx",
    "src/main.tsx",
    "src/main.jsx",
    "components.json",
    "components/ui/button.tsx",
    "src/components/ui/button.tsx",
    "app/globals.css",
    "styles/globals.css",
    "src/index.css",
    "tailwind.config.ts",
    "tailwind.config.js",
    "tailwind.config.mjs",
    "postcss.config.js",
    "postcss.config.mjs",
    "vitest.config.ts",
    "vitest.config.js",
    "jest.config.ts",
    "jest.config.js",
    "playwright.config.ts",
    "playwright.config.js",
    "cypress.config.ts",
    "cypress.config.js",
    "eslint.config.js",
    "eslint.config.mjs",
    "biome.json",
    "app/error.tsx",
    "app/global-error.tsx",
    "src/error.tsx",
    "lib/logger.ts",
    "lib/logger.js",
    "src/logger.ts",
    "src/logger.js",
    "sentry.client.config.ts",
    "sentry.server.config.ts",
    "instrumentation.ts",
    "server.ts",
    "server.js",
    "src/server.ts",
    "src/server.js",
    "src/main.py",
    "src/main.ts",
    "src/app.ts",
    "tsconfig.json",
    "ruff.toml",
    "pytest.ini",
    "mypy.ini",
    ".github/workflows/ci.yml",
    ".github/workflows/ci.yaml",
    ".github/workflows/test.yml",
    ".github/workflows/tests.yml",
    ".github/workflows/lint.yml",
    ".github/workflows/build.yml",
    ".github/workflows/deploy.yml",
    ".github/workflows/deploy.yaml",
    ".github/workflows/release.yml",
    ".github/workflows/release.yaml",
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


def plan_orientation() -> dict[str, Any]:
    """Return workspace-grounded orientation for building an implementation plan.

    This is the tool the model calls when asked to plan a change.
    It inspects the workspace and returns the high-signal files to
    orient around plus the project-native verification command. The
    model formats these into a plan; the tool does not write anything.
    """
    overview = project_overview(max_files=300, max_read_bytes=12_000)
    files = [path for path in overview.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in overview.get("documents", []) if isinstance(doc, dict)]
    doc_paths = [str(doc.get("path")) for doc in documents if doc.get("path")]
    package = package_json_from_documents(documents)
    cargo = cargo_package_from_documents(documents)
    scripts = package.get("scripts", {}) if package else {}

    orientation = project_orientation_files(files, doc_paths)
    verification_commands = audit_verification_commands(scripts, files, cargo)
    verification = verification_commands[0] if verification_commands else "the project test command"

    return {
        "orientation_files": orientation[:5],
        "verification_command": verification,
        "all_files_count": len(files),
    }


def is_project_overview_payload(payload: dict) -> bool:
    return isinstance(payload.get("files"), list) and isinstance(payload.get("documents"), list)


def render_project_overview_summary(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    doc_paths = [str(doc.get("path")) for doc in documents if doc.get("path")]
    package = package_json_from_documents(documents)
    pyproject = pyproject_from_documents(documents)
    cargo = cargo_package_from_documents(documents)
    scripts = package.get("scripts", {}) if package else {}
    dependencies = package_dependencies(package)

    sections = ["Project Summary"]
    name = project_name(package, pyproject, cargo, documents)
    description = project_description(package, pyproject, cargo, documents)
    stack = infer_project_stack(files, documents, dependencies, cargo)
    architecture = infer_project_architecture(files)

    sections.append(render_project_identity(name, description, stack, architecture))

    orientation = project_orientation_files(files, doc_paths)
    if orientation:
        sections.append("I would orient around:\n" + "\n".join(f"- {line}" for line in orientation))

    next_steps = project_next_steps(files, scripts, payload, cargo)
    if next_steps:
        sections.append("What I would do next:\n" + "\n".join(f"- {step}" for step in next_steps))

    verification = project_verification_note(files, scripts, cargo)
    if verification:
        sections.append(verification)

    worktree_note = friendly_worktree_note(str(payload.get("git_status") or ""))
    if worktree_note:
        sections.append(worktree_note)

    return "\n\n".join(sections)


def render_project_audit(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    package = package_json_from_documents(documents)
    pyproject = pyproject_from_documents(documents)
    cargo = cargo_package_from_documents(documents)
    dependencies = package_dependencies(package)
    scripts = package.get("scripts", {}) if package else {}
    stack = infer_project_stack(files, documents, dependencies, cargo)
    findings = project_audit_findings(files, documents, package, pyproject)

    sections = ["Project Gap Analysis"]
    summary = project_description(package, pyproject, cargo, documents)
    if summary:
        sections.append("Here is my read: " + summary)
    if stack:
        sections.append("The working stack looks like " + human_join(stack) + ".")

    reviewed = audit_reviewed_paths(files, documents)
    if reviewed:
        sections.append(
            "I checked the high-signal files first:\n"
            + "\n".join(f"- {path}" for path in reviewed[:8])
        )

    if findings:
        rendered_findings = []
        for index, finding in enumerate(findings, start=1):
            rendered_findings.append(
                "\n".join(
                    [
                        f"{index}. {finding['title']}",
                        f"   Why it matters: {finding['impact']}",
                        f"   Evidence: {finding['evidence']}",
                        f"   Next move: {finding['recommendation']}",
                    ]
                )
            )
        sections.append("What I would fix or clarify next:\n" + "\n\n".join(rendered_findings))
    else:
        sections.append(
            "What I would do next:\n"
            "- I do not see an obvious structural red flag from the high-level files.\n"
            "- I would inspect the feature code behind the main CLI or app entrypoint next.\n"
            "- I would run the project verification command before making broad edits."
        )

    commands = audit_verification_commands(scripts, files, cargo)
    if commands:
        sections.append("Verification I would run:\n" + "\n".join(f"- {cmd}" for cmd in commands))

    return "\n\n".join(sections)


def render_project_identity(
    name: str,
    description: str,
    stack: list[str],
    architecture: str,
) -> str:
    subject = name or "this project"
    if description:
        sentence = f"{subject} is {sentence_fragment(description)}."
    else:
        sentence = f"{subject} is a software project with a few clear entrypoints."

    details = []
    if stack:
        details.append(f"It appears to use {human_join(stack)}.")
    if architecture:
        details.append(f"The code is organized around {architecture}.")
    if details:
        sentence = f"{sentence} {' '.join(details)}"
    return sentence


def project_orientation_files(files: list[str], doc_paths: list[str]) -> list[str]:
    lines = []
    for path in doc_paths[:4]:
        lines.append(orientation_description(path))
    for path in notable_source_files(files):
        if path in doc_paths:
            continue
        lines.append(orientation_description(path))
        if len(lines) >= 6:
            break
    return dedupe(lines)


def project_next_steps(
    files: list[str],
    scripts: dict,
    payload: dict,
    cargo: dict | None = None,
) -> list[str]:
    steps = []
    if "README.md" in files or "readme.md" in {path.lower() for path in files}:
        steps.append("Read the README once, then inspect the main entrypoint before editing.")
    else:
        steps.append("Add or find a short project overview so future work has a clear map.")

    git_status_text = str(payload.get("git_status") or "").strip()
    if git_status_text:
        steps.append("Review the existing uncommitted changes before making broad edits.")

    if any(path.startswith("tests/") for path in files):
        steps.append("Use the tests as the guardrail for focused code changes.")
    elif "Cargo.toml" in files:
        steps.append("Add or locate Rust tests around the behavior you want to change.")

    return dedupe(steps)[:4]


def project_verification_note(files: list[str], scripts: dict, cargo: dict | None = None) -> str:
    commands = audit_verification_commands(scripts, files, cargo)
    if commands:
        return "Verification I would run:\n" + "\n".join(f"- {cmd}" for cmd in commands[:3])
    return "Verification: I do not see a standard test command yet."


def friendly_worktree_note(git_status_text: str) -> str:
    changed = [line for line in git_status_text.splitlines() if line.strip()]
    if not changed:
        return ""
    return (
        "I also see uncommitted workspace changes; "
        "I would keep edits scoped until those are understood."
    )


def human_join(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def sentence_fragment(text: str) -> str:
    fragment = clean_markdown_inline(text).strip().rstrip(".")
    if not fragment:
        return "a software project"
    if fragment.startswith(("A ", "An ", "The ")):
        fragment = fragment[0].lower() + fragment[1:]
    elif len(fragment) > 1 and fragment[0].isupper() and fragment[1].islower():
        fragment = fragment[0].lower() + fragment[1:]
    return fragment


def orientation_description(path: str) -> str:
    if path.lower().startswith("readme"):
        return f"{path} for the purpose and setup story"
    if path == "Cargo.toml":
        return "Cargo.toml for Rust package layout and verification commands"
    if path == "package.json":
        return "package.json for scripts and frontend dependencies"
    if path == "pyproject.toml":
        return "pyproject.toml for Python packaging and tooling"
    if path.endswith("/src/main.rs") or path == "src/main.rs":
        return f"{path} for the CLI or binary entrypoint"
    if "graph/" in path:
        return f"{path} for agent orchestration"
    if "rlm/runtime" in path:
        return f"{path} for the recursive execution loop"
    if "sandbox/tools" in path:
        return f"{path} for workspace tools"
    return f"{path} for the main implementation shape"


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


def audit_verification_commands(
    scripts: dict,
    files: list[str],
    cargo: dict | None = None,
) -> list[str]:
    commands = []
    if isinstance(scripts, dict):
        for name in ("test", "lint", "typecheck", "build"):
            if isinstance(scripts.get(name), str):
                commands.append(f"npm run {name}")
    if "pyproject.toml" in files and any(path.startswith("tests/") for path in files):
        commands.append("pytest")
    if "Cargo.toml" in files:
        commands.append("cargo test")
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


def cargo_package_from_documents(documents: list[dict]) -> dict:
    for doc in documents:
        if doc.get("path") == "Cargo.toml" and isinstance(doc.get("content"), str):
            content = str(doc["content"])
            package = parse_toml_section(content, "package")
            if package:
                return package
            workspace = parse_toml_section(content, "workspace")
            workspace_package = parse_toml_section(content, "workspace.package")
            cargo: dict[str, object] = {}
            if workspace:
                cargo["workspace"] = True
            if workspace_package:
                cargo["workspace_package"] = workspace_package
            return cargo
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


def parse_toml_section(content: str, section_name: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        try:
            payload = tomllib.loads(content)
        except ValueError:
            payload = {}
        section = payload if isinstance(payload, dict) else None
        for part in section_name.split("."):
            if not isinstance(section, dict):
                section = None
                break
            section = section.get(part)
        return section if isinstance(section, dict) else {}

    metadata: dict[str, str] = {}
    in_section = False
    expected_header = f"[{section_name}]"
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == expected_header
            continue
        if not in_section or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:
            metadata[key.strip()] = value
    return metadata


def project_name(
    package: dict,
    pyproject: dict,
    cargo: dict | None = None,
    documents: list[dict] | None = None,
) -> str:
    for payload in (package, pyproject, cargo or {}):
        name = payload.get("name") if isinstance(payload, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    readme_title = readme_title_from_documents(documents or [])
    if readme_title:
        return readme_title
    return ""


def project_description(
    package: dict,
    pyproject: dict,
    cargo: dict | None,
    documents: list[dict],
) -> str:
    for payload in (package, pyproject, cargo or {}):
        description = payload.get("description") if isinstance(payload, dict) else None
        if isinstance(description, str) and description.strip():
            return one_line(clean_markdown_inline(description))

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


def readme_title_from_documents(documents: list[dict]) -> str:
    readme = readme_content_from_documents(documents)
    if not readme:
        return ""
    for raw_line in readme.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            title = clean_markdown_inline(line.lstrip("#").strip())
            return title if title and not is_readme_noise_line(title) else ""
    return ""


def first_readme_paragraph(content: str) -> str:
    lines = []
    seen_heading = False
    in_fence = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or is_readme_noise_line(line):
            continue
        if not line:
            if lines:
                break
            continue
        if line.startswith("#"):
            seen_heading = True
            continue
        if seen_heading or not lines:
            cleaned = clean_markdown_inline(line)
            if cleaned:
                lines.append(cleaned)
    return " ".join(lines)


def is_readme_noise_line(line: str) -> bool:
    if not line:
        return False
    lowered = line.lower()
    return (
        line.startswith("<!--")
        or "![" in line
        or "shields.io" in lowered
        or "skills.sh" in lowered
        or lowered.startswith("[![")
    )


def clean_markdown_inline(text: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"(?<!\w)(\*|_)([^*_]+)\1(?!\w)", r"\2", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = cleaned.strip("#*-_ \t")
    return re.sub(r"\s+", " ", cleaned).strip()


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
    cargo: dict | None = None,
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
    if cargo or "Cargo.toml" in files or any(path.endswith(".rs") for path in files):
        stack.append("Rust")
    if "go.mod" in files:
        stack.append("Go")
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
        "Cargo.toml",
        "pyproject.toml",
        "README.md",
        "crates/sansara-cli/src/main.rs",
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
    env = git_workspace_env(WORKSPACE)
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=diff,
        cwd=WORKSPACE,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ToolError(render_command_failure(result))
    return "patch applied"


def git_workspace_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    parent = workspace.resolve().parent
    existing = env.get("GIT_CEILING_DIRECTORIES")
    env["GIT_CEILING_DIRECTORIES"] = (
        str(parent) if not existing else f"{parent}{os.pathsep}{existing}"
    )
    return env


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
    search_arg = "."
    if search_root != WORKSPACE:
        search_arg = str(search_root.relative_to(WORKSPACE))
    result = subprocess.run(
        ["rg", "--line-number", "--max-count", str(max_count), pattern, search_arg],
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
        "plan_orientation",
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
        "name": "plan_orientation",
        "description": (
            "Return workspace-grounded orientation files and the verification command "
            "for building an implementation plan. Call this when asked to plan a change."
        ),
        "parameters": {},
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
