from __future__ import annotations

import json
import re
from pathlib import Path

from rlm_harness.graph.task_policy import is_code_editing_task
from rlm_harness.sandbox.tools import (
    cargo_package_from_documents,
    infer_project_stack,
    package_dependencies,
    package_json_from_documents,
    pyproject_from_documents,
)


def parse_search_matches(output: str) -> list[tuple[str, str, str]]:
    matches = []
    for raw_line in output.splitlines():
        path, line, text = split_rg_line(raw_line)
        if path and line:
            matches.append((path, line, text.strip()))
    return matches


def split_rg_line(raw_line: str) -> tuple[str, str, str]:
    parts = raw_line.split(":", 2)
    if len(parts) != 3:
        return "", "", ""
    path, line, text = parts
    if not line.isdigit():
        return "", "", ""
    return path.removeprefix("./"), line, text


def parse_git_status_lines(output: str) -> list[tuple[str, str]]:
    entries = []
    for raw_line in output.splitlines():
        raw = raw_line.rstrip()
        if not raw:
            continue
        if raw.startswith("?? "):
            code = "??"
            path = raw[3:].strip()
        elif len(raw) >= 3 and raw[2] == " ":
            code = raw[:2]
            path = raw[3:].strip()
        else:
            parts = raw.split(maxsplit=1)
            if len(parts) != 2:
                continue
            code, path = parts[0], parts[1].strip()
        if not path:
            continue
        entries.append((git_status_label(code), path))
    return entries


def git_status_label(code: str) -> str:
    if code == "??":
        return "Untracked"
    if "A" in code:
        return "Added"
    if "D" in code:
        return "Deleted"
    if "R" in code:
        return "Renamed"
    if "C" in code:
        return "Copied"
    if "M" in code:
        return "Modified"
    return "Changed"


def is_entrypoint_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "entrypoint",
            "entry point",
            "main file",
            "main module",
            "where does this start",
            "where should i start",
            "where do i start",
            "where is the cli",
            "where's the cli",
        )
    )


def is_edit_target_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what file should i edit",
            "which file should i edit",
            "what file should i change",
            "which file should i change",
            "where should i edit",
            "where should i make this change",
            "where should i make the change",
            "where do i make this change",
            "where do i make the change",
            "where would i change",
            "where would i edit",
            "where should this change go",
            "what should i edit to",
            "which files should i touch",
        )
    )


def render_edit_target_answer(payload: dict, task: str) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    candidates = edit_target_candidates(files, documents, task)
    commands = edit_target_verification_commands(payload)

    sections = ["Edit Targets"]
    if candidates:
        sections.append(
            "Likely Files:\n"
            + "\n".join(
                f"- `{path}` - {description}" for path, description in candidates[:10]
            )
        )
    else:
        sections.append("Likely Files:\n- I do not see an obvious edit target yet.")

    evidence = edit_target_evidence(task, candidates)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    if commands:
        sections.append(
            "Check After Editing:\n" + "\n".join(f"- `{command}`" for command in commands[:4])
        )
    sections.append(
        "What I would do next:\n"
        "- Open the top file, confirm the behavior is wired there, then make the smallest "
        "focused edit."
    )
    return "\n".join(sections)


def edit_target_verification_commands(payload: dict) -> list[str]:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    commands = verification_commands_from_project_overview(payload)
    has_python = "pyproject.toml" in files or any(path.endswith(".py") for path in files)
    if has_python:
        return commands
    return [
        command
        for command in commands
        if command not in {"pytest", "python -m pytest", "python -m unittest"}
    ]


def edit_target_candidates(
    files: list[str],
    documents: list[dict],
    task: str,
) -> list[tuple[str, str]]:
    lowered = task.lower()
    candidates: list[tuple[str, str]] = []

    if any(term in lowered for term in ("homepage", "home page", "landing page")):
        candidates.extend(homepage_edit_targets(files))
    if any(term in lowered for term in ("login", "sign in", "sign-in", "auth", "session")):
        candidates.extend(auth_related_files(files))
    if any(term in lowered for term in ("api", "endpoint", "route", "handler")):
        candidates.extend(api_route_files(files))
    if any(term in lowered for term in ("database", "schema", "migration", "model")):
        candidates.extend(database_schema_files(files))
    if any(term in lowered for term in ("component", "button", "ui", "style", "css", "frontend")):
        candidates.extend(frontend_related_files(files))
    if any(term in lowered for term in ("test", "spec", "coverage")):
        candidates.extend((path, "test file") for path in project_test_files(files))
    if any(term in lowered for term in ("config", "setting", "settings")):
        candidates.extend(project_config_files(files))
    if any(term in lowered for term in ("env", "secret", "environment variable")):
        candidates.extend((path, "safe env template") for path in safe_env_template_files(files))
    if any(term in lowered for term in ("cli", "command", "entrypoint", "start")):
        candidates.extend(
            (path, "entrypoint candidate")
            for path in entrypoint_candidates(files, documents)
        )

    if not candidates:
        candidates.extend(frontend_related_files(files)[:4])
        candidates.extend(api_route_files(files)[:4])
        candidates.extend(
            (path, "entrypoint candidate")
            for path in entrypoint_candidates(files, documents)[:4]
        )
        candidates.extend(project_config_files(files)[:4])

    return dedupe_candidate_pairs(candidates)[:12]


def homepage_edit_targets(files: list[str]) -> list[tuple[str, str]]:
    preferred = (
        ("app/page.tsx", "root app page"),
        ("app/page.ts", "root app page"),
        ("pages/index.tsx", "pages router homepage"),
        ("pages/index.ts", "pages router homepage"),
        ("src/App.tsx", "root app component"),
        ("src/App.jsx", "root app component"),
        ("src/main.tsx", "frontend mount entry"),
        ("src/main.jsx", "frontend mount entry"),
    )
    file_set = set(files)
    return [(path, description) for path, description in preferred if path in file_set]


def safe_env_template_files(files: list[str]) -> list[str]:
    result = []
    for path in files:
        lowered = path.lower()
        if lowered in {"env.example", "env.sample", ".env.example", ".env.sample", ".env.template"}:
            result.append(path)
        elif lowered.endswith((".env.example", ".env.sample", ".env.template")):
            result.append(path)
    return result


def dedupe_candidate_pairs(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    result = []
    for path, description in values:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append((path, description))
    return result


def edit_target_evidence(task: str, candidates: list[tuple[str, str]]) -> list[str]:
    evidence = []
    lowered = task.lower()
    matched_terms = [
        term
        for term in (
            "homepage",
            "login",
            "auth",
            "api",
            "database",
            "component",
            "style",
            "test",
            "config",
            "env",
            "cli",
        )
        if term in lowered
    ]
    if matched_terms:
        evidence.append("Matched request terms: " + ", ".join(matched_terms[:6]) + ".")
    if candidates:
        evidence.append("Ranked files by local project conventions and path names.")
    return evidence


def render_entrypoint_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    candidates = entrypoint_candidates(files, documents)

    sections = ["Entrypoints"]
    if candidates:
        sections.append("\n".join(f"- `{path}`" for path in candidates[:6]))
    else:
        sections.append("- I do not see an obvious app or CLI entrypoint yet.")

    evidence = entrypoint_evidence(files, candidates)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the top candidate, then follow the command or app wiring it calls."
    )
    return "\n".join(sections)


def entrypoint_candidates(files: list[str], documents: list[dict]) -> list[str]:
    ordered: list[str] = []
    file_set = set(files)
    for path in manifest_entrypoint_paths(documents):
        if path in file_set:
            ordered.append(path)

    ranked_patterns = (
        "crates/*/src/main.rs",
        "src/main.rs",
        "src/bin/*.rs",
        "cmd/*/main.go",
        "main.go",
        "src/main.ts",
        "src/main.tsx",
        "src/index.ts",
        "src/index.tsx",
        "src/App.tsx",
        "app/page.tsx",
        "pages/index.tsx",
        "main.py",
        "app.py",
        "src/*/__main__.py",
        "*/cli.py",
        "*_cli.py",
    )
    for pattern in ranked_patterns:
        ordered.extend(path for path in files if Path(path).match(pattern))
    return dedupe_strings(ordered)


def manifest_entrypoint_paths(documents: list[dict]) -> list[str]:
    paths: list[str] = []
    for doc in documents:
        path = str(doc.get("path") or "")
        content = str(doc.get("content") or "")
        if path == "Cargo.toml":
            paths.extend(cargo_manifest_paths(content))
        elif path == "package.json":
            paths.extend(package_manifest_paths(content))
    return paths


def cargo_manifest_paths(content: str) -> list[str]:
    paths = []
    for match in re.finditer(r'^\s*path\s*=\s*"([^"]+\.rs)"', content, flags=re.MULTILINE):
        paths.append(match.group(1))
    return paths


def package_manifest_paths(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    candidates = []
    for key in ("main", "module", "browser"):
        value = payload.get(key)
        if isinstance(value, str):
            candidates.append(value)
    bin_value = payload.get("bin")
    if isinstance(bin_value, str):
        candidates.append(bin_value)
    elif isinstance(bin_value, dict):
        candidates.extend(value for value in bin_value.values() if isinstance(value, str))
    return candidates


def entrypoint_evidence(files: list[str], candidates: list[str]) -> list[str]:
    evidence = []
    markers = {
        "Cargo.toml": "Cargo.toml is present.",
        "package.json": "package.json is present.",
        "pyproject.toml": "pyproject.toml is present.",
        "go.mod": "go.mod is present.",
    }
    for marker, line in markers.items():
        if marker in files:
            evidence.append(line)
    if any(path.endswith("/src/main.rs") or path == "src/main.rs" for path in candidates):
        evidence.append("Rust binaries conventionally start from src/main.rs.")
    if any(path.endswith("main.go") for path in candidates):
        evidence.append("Go commands conventionally start from main.go.")
    if any(path.endswith((".ts", ".tsx", ".js", ".jsx")) for path in candidates):
        evidence.append(
            "JavaScript and TypeScript apps commonly wire startup through index, "
            "main, or app files."
        )
    if any(path.endswith(".py") for path in candidates):
        evidence.append("Python CLIs commonly start from main.py, app.py, __main__.py, or cli.py.")
    return dedupe_strings(evidence)[:5]


def is_run_question(task: str) -> bool:
    if is_verification_question(task):
        return False
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how do i run this",
            "how do i run the",
            "how to run this",
            "how should i run this",
            "run this project",
            "run the project",
            "run this app",
            "run the app",
            "start this project",
            "start the project",
            "start this app",
            "start the app",
            "dev server",
            "local server",
            "launch this",
        )
    )


def render_run_answer(payload: dict) -> str:
    commands = run_commands_from_project_overview(payload)
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]

    sections = ["Run Commands"]
    if commands:
        sections.append("\n".join(f"- `{command}`" for command in commands))
    else:
        sections.append("- I do not see an obvious local run command yet.")

    evidence = run_command_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Run the first command, then inspect the error or startup URL before editing."
    )
    return "\n".join(sections)


def run_commands_from_project_overview(payload: dict) -> list[str]:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    scripts = package_scripts_from_documents(documents)
    package_manager = package_manager_for_files(files)
    commands = []

    if "dev" in scripts:
        commands.append(script_command(package_manager, "dev"))
    if "start" in scripts:
        commands.append(script_command(package_manager, "start"))
    if not commands and "serve" in scripts:
        commands.append(script_command(package_manager, "serve"))
    if ("Cargo.toml" in files or any(path.endswith(".rs") for path in files)) and any(
        path.endswith("/src/main.rs") or path == "src/main.rs" for path in files
    ):
        commands.append("cargo run")
    if "go.mod" in files or any(path.endswith("main.go") for path in files):
        commands.append("go run ./...")
    if "main.py" in files:
        commands.append("python main.py")
    elif "app.py" in files:
        commands.append("python app.py")

    if not commands and "build" in scripts:
        commands.append(script_command(package_manager, "build"))
    return dedupe_strings(commands)[:4]


def package_manager_for_files(files: list[str]) -> str:
    if "pnpm-lock.yaml" in files:
        return "pnpm"
    if "yarn.lock" in files:
        return "yarn"
    return "npm"


def script_command(package_manager: str, script: str) -> str:
    if package_manager == "npm" and script in {"start", "stop", "test"}:
        return f"npm {script}"
    if package_manager == "npm":
        return f"npm run {script}"
    return f"{package_manager} {script}"


def run_command_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    scripts = package_scripts_from_documents(documents)
    if scripts:
        names = ", ".join(sorted(str(name) for name in scripts)[:5])
        evidence.append(f"package.json defines script(s): {names}.")
    if "Cargo.toml" in files:
        evidence.append("Cargo.toml is present.")
    if "go.mod" in files:
        evidence.append("go.mod is present.")
    for path in ("main.py", "app.py", "src/main.rs", "main.go"):
        if path in files:
            evidence.append(f"{path} is present.")
    return dedupe_strings(evidence)[:5]


def is_stack_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what stack",
            "tech stack",
            "technology stack",
            "what language",
            "which language",
            "what framework",
            "which framework",
            "what dependencies",
            "what does this project use",
            "what is this built with",
            "what is it built with",
            "built with",
        )
    )


def is_dependency_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what dependencies",
            "which dependencies",
            "list dependencies",
            "list package dependencies",
            "package dependencies",
            "dependency list",
            "what packages",
            "which packages",
            "npm dependencies",
            "pnpm dependencies",
            "python dependencies",
            "rust dependencies",
        )
    )


def render_dependency_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    groups = dependency_groups_from_documents(documents)

    sections = ["Dependencies"]
    if groups:
        group_lines = []
        for title, names in groups[:8]:
            group_lines.append(f"{title}:")
            group_lines.extend(f"- `{name}`" for name in names[:10])
            if len(names) > 10:
                group_lines.append(f"- ...and {len(names) - 10} more.")
        sections.append("\n".join(group_lines))
    else:
        sections.append("- I do not see manifest-declared dependencies yet.")

    evidence = dependency_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the manifest before upgrading or removing dependencies."
    )
    return "\n".join(sections)


def dependency_groups_from_documents(documents: list[dict]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    package = package_json_from_documents(documents)
    for title, key in (
        ("runtime", "dependencies"),
        ("development", "devDependencies"),
        ("peer", "peerDependencies"),
    ):
        values = package.get(key) if isinstance(package, dict) else None
        if isinstance(values, dict) and values:
            groups.append((title, sorted(str(name) for name in values)))

    pyproject = pyproject_from_documents(documents)
    py_deps = pyproject.get("dependencies") if isinstance(pyproject, dict) else None
    if isinstance(py_deps, list) and py_deps:
        groups.append(("python", sorted(str(name).split(";", 1)[0].strip() for name in py_deps)))

    requirement_deps = requirements_dependencies_from_documents(documents)
    if requirement_deps:
        groups.append(("requirements.txt", requirement_deps))
    return groups


def requirements_dependencies_from_documents(documents: list[dict]) -> list[str]:
    for doc in documents:
        if doc.get("path") != "requirements.txt" or not isinstance(doc.get("content"), str):
            continue
        deps = []
        for raw_line in str(doc["content"]).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            deps.append(line)
        return deps
    return []


def dependency_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if "package.json" in files:
        evidence.append("package.json is present.")
        evidence.append(f"{package_manager_for_files(files)} appears to be the package manager.")
    if "requirements.txt" in files:
        evidence.append("requirements.txt is present.")
    if "pyproject.toml" in files:
        evidence.append("pyproject.toml is present.")
    if "Cargo.toml" in files:
        evidence.append("Cargo.toml is present.")
    if "go.mod" in files:
        evidence.append("go.mod is present.")
    doc_paths = {str(doc.get("path") or "") for doc in documents}
    if doc_paths:
        evidence.append("Manifest files were read for dependency names.")
    return dedupe_strings(evidence)[:6]


def render_stack_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    package = package_json_from_documents(documents)
    cargo = cargo_package_from_documents(documents)
    dependencies = package_dependencies(package)
    stack = infer_project_stack(files, documents, dependencies, cargo)

    sections = ["Tech Stack"]
    if stack:
        sections.append("\n".join(f"- {item}" for item in stack[:10]))
    else:
        sections.append("- I do not see enough manifest evidence to identify the stack yet.")

    dependency_lines = stack_dependency_highlights(dependencies)
    if dependency_lines:
        sections.append(
            "Dependency Signals:\n" + "\n".join(f"- {line}" for line in dependency_lines)
        )

    evidence = stack_evidence(files, package, cargo)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the manifest and the main entrypoint before making stack-specific changes."
    )
    return "\n".join(sections)


def stack_dependency_highlights(dependencies: set[str]) -> list[str]:
    if not dependencies:
        return []
    preferred = (
        "react",
        "@tanstack/react-start",
        "@tanstack/react-router",
        "vite",
        "typescript",
        "tailwindcss",
        "@tailwindcss/vite",
        "pytest",
        "ruff",
    )
    highlights = [name for name in preferred if name in dependencies]
    if not highlights:
        highlights = sorted(dependencies)[:6]
    return highlights[:6]


def stack_evidence(files: list[str], package: dict, cargo: dict) -> list[str]:
    evidence = []
    if package:
        evidence.append("package.json is present.")
    if cargo or "Cargo.toml" in files:
        evidence.append("Cargo.toml is present.")
    if "pyproject.toml" in files:
        evidence.append("pyproject.toml is present.")
    if "go.mod" in files:
        evidence.append("go.mod is present.")
    if any(path.endswith((".ts", ".tsx")) for path in files):
        evidence.append("TypeScript source files are present.")
    if any(path.endswith(".rs") for path in files):
        evidence.append("Rust source files are present.")
    if any(path.endswith(".py") for path in files):
        evidence.append("Python source files are present.")
    return dedupe_strings(evidence)[:6]


def is_test_location_question(task: str) -> bool:
    if is_verification_question(task):
        return False
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "where are the tests",
            "where are tests",
            "where is the test",
            "what tests exist",
            "what test files",
            "list tests",
            "list test files",
            "show tests",
            "show test files",
            "test files exist",
            "test coverage live",
        )
    )


def render_test_location_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    test_files = project_test_files(files)
    commands = verification_commands_from_project_overview(payload)

    sections = ["Test Files"]
    if test_files:
        sections.append("\n".join(f"- `{path}`" for path in test_files[:12]))
        if len(test_files) > 12:
            sections.append(f"- ...and {len(test_files) - 12} more test files.")
    else:
        sections.append("- I do not see obvious test files yet.")

    if commands:
        sections.append("Likely Commands:\n" + "\n".join(f"- `{command}`" for command in commands))

    evidence = test_location_evidence(files, test_files)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the closest test file for the code you plan to change, then run the "
        "narrow test command."
    )
    return "\n".join(sections)


def project_test_files(files: list[str]) -> list[str]:
    ranked = []
    for path in files:
        name = Path(path).name
        lowered = path.lower()
        if (
            "/tests/" in f"/{lowered}"
            or lowered.startswith("tests/")
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".test.ts")
            or name.endswith(".test.tsx")
            or name.endswith(".spec.ts")
            or name.endswith(".spec.tsx")
            or name.endswith("_test.go")
            or lowered.endswith("/test.rs")
            or "/__tests__/" in f"/{lowered}"
        ):
            ranked.append(path)
    return dedupe_strings(sorted(ranked))


def test_location_evidence(files: list[str], test_files: list[str]) -> list[str]:
    evidence = []
    if any(path.startswith("tests/") for path in files):
        evidence.append("A top-level tests/ directory is present.")
    if any("/tests/" in f"/{path.lower()}" for path in test_files):
        evidence.append("Nested tests/ directories are present.")
    if any(Path(path).name.startswith("test_") and path.endswith(".py") for path in test_files):
        evidence.append("Python test_*.py files are present.")
    if any(
        path.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"))
        for path in test_files
    ):
        evidence.append("JavaScript or TypeScript test/spec files are present.")
    if any(path.endswith("_test.go") for path in test_files):
        evidence.append("Go *_test.go files are present.")
    if any(path.endswith(".rs") for path in test_files):
        evidence.append("Rust test files are present.")
    return dedupe_strings(evidence)[:5]


def is_project_structure_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how is this project structured",
            "how is this repo structured",
            "project structure",
            "repo structure",
            "repository structure",
            "codebase structure",
            "quick map of this repo",
            "map of this repo",
            "map of the repo",
            "map the repo",
            "where should i look first",
            "where should i start reading",
            "orient me in this repo",
            "orient me in the repo",
        )
    )


def render_project_structure_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    directories = project_structure_directories(files)
    orientation = project_structure_orientation(files)

    sections = ["Project Structure"]
    if directories:
        sections.append(
            "\n".join(f"- `{path}` - {description}" for path, description in directories)
        )
    else:
        sections.append("- I only see files at the repository root so far.")

    if orientation:
        sections.append("Start With:\n" + "\n".join(f"- `{path}`" for path in orientation[:8]))

    evidence = project_structure_evidence(files)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Read the start-with files first, then follow the nearest entrypoint or test."
    )
    return "\n".join(sections)


def project_structure_directories(files: list[str]) -> list[tuple[str, str]]:
    counts: dict[str, int] = {}
    for path in files:
        parts = Path(path).parts
        if len(parts) > 1:
            root = parts[0]
            counts[root] = counts.get(root, 0) + 1
        if len(parts) > 2 and parts[0] in {"crates", "packages", "apps", "services"}:
            nested = f"{parts[0]}/{parts[1]}"
            counts[nested] = counts.get(nested, 0) + 1

    preferred = (
        "crates",
        "crates/cli",
        "packages",
        "apps",
        "services",
        "src",
        "tests",
        "docs",
        "public",
        "scripts",
        "rlm_harness",
    )
    ordered = [path for path in preferred if path in counts]
    ordered.extend(path for path in sorted(counts) if path not in ordered)

    result = []
    for path in ordered[:10]:
        result.append((path, directory_description(path, counts[path])))
    return result


def directory_description(path: str, count: int) -> str:
    name = Path(path).name
    if path == "src":
        return "application source"
    if path == "tests" or name == "tests":
        return "test coverage"
    if path == "docs":
        return "documentation"
    if path == "public":
        return "static assets"
    if path == "scripts":
        return "project scripts"
    if path == "crates":
        return "Rust workspace crates"
    if path.startswith("crates/"):
        return "Rust crate"
    if path == "packages":
        return "workspace packages"
    if path == "apps":
        return "application packages"
    if path == "services":
        return "service packages"
    if path == "rlm_harness":
        return "harness package source"
    return f"{count} file(s)"


def project_structure_orientation(files: list[str]) -> list[str]:
    preferred = (
        "README.md",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "docs/architecture.md",
        "src/main.rs",
        "src/main.ts",
        "src/main.tsx",
        "src/index.tsx",
        "crates/cli/src/main.rs",
        "tests/test_smoke.py",
    )
    selected = [path for path in preferred if path in files]
    for path in files:
        if path.endswith("/src/main.rs") and path not in selected:
            selected.append(path)
        elif path.startswith("tests/") and path not in selected:
            selected.append(path)
        elif path.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx")):
            selected.append(path)
    return dedupe_strings(selected)


def project_structure_evidence(files: list[str]) -> list[str]:
    evidence = []
    if "Cargo.toml" in files:
        evidence.append("Cargo.toml marks this as a Rust or mixed workspace.")
    if "package.json" in files:
        evidence.append("package.json marks this as a Node.js workspace or app.")
    if "pyproject.toml" in files:
        evidence.append("pyproject.toml marks this as a Python project.")
    if any(path.startswith("crates/") for path in files):
        evidence.append("crates/ contains Rust workspace members.")
    if any(path.startswith("src/routes/") for path in files):
        evidence.append("src/routes/ suggests route-based frontend structure.")
    if any(path.startswith("tests/") for path in files):
        evidence.append("tests/ contains test coverage.")
    if any(path.startswith("docs/") for path in files):
        evidence.append("docs/ contains project documentation.")
    return dedupe_strings(evidence)[:6]


def is_todo_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "todo",
            "todos",
            "fixme",
            "fixmes",
            "hack comments",
            "xxx comments",
            "technical debt markers",
            "debt markers",
            "unfinished comments",
        )
    )


def render_todo_answer(output: str) -> str:
    matches = parse_search_matches(output)
    if not matches:
        return (
            "Task Markers\n"
            "No TODO, FIXME, HACK, or XXX markers found.\n"
            "What I would do next:\n"
            "- Search issue trackers or docs if you expect tracked follow-up work."
        )

    lines = ["Task Markers"]
    for path, line, text in matches[:12]:
        lines.append(f"- {path}:{line} - {text}")
    if len(matches) > 12:
        lines.append(f"- ...and {len(matches) - 12} more markers.")
    lines.append(
        "What I would do next:\n"
        "- Pick the marker nearest the code you plan to touch and verify whether it "
        "is still current."
    )
    return "\n".join(lines)


def is_setup_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "install dependencies",
            "install deps",
            "install packages",
            "set this project up",
            "setup this project",
            "set up this project",
            "set up locally",
            "setup locally",
            "bootstrap this project",
            "bootstrap the project",
            "local setup",
            "development setup",
            "get this running locally",
            "prepare this project",
        )
    )


def render_setup_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    commands = setup_commands_from_project_overview(files, documents)
    next_commands = run_commands_from_project_overview(payload)
    verification = verification_commands_from_project_overview(payload)

    sections = ["Setup Commands"]
    if commands:
        sections.append("\n".join(f"- `{command}`" for command in commands))
    else:
        sections.append("- I do not see a standard dependency-install command yet.")

    follow_up = dedupe_strings(next_commands + verification)[:4]
    if follow_up:
        sections.append("After Setup:\n" + "\n".join(f"- `{command}`" for command in follow_up))

    evidence = setup_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Run setup first, then the narrowest run or verification command."
    )
    return "\n".join(sections)


def setup_commands_from_project_overview(files: list[str], documents: list[dict]) -> list[str]:
    commands = []
    if "package.json" in files:
        package_manager = package_manager_for_files(files)
        if package_manager == "pnpm":
            commands.append("pnpm install")
        elif package_manager == "yarn":
            commands.append("yarn install")
        else:
            commands.append("npm install")
    if "requirements.txt" in files:
        commands.append("python -m venv .venv")
        commands.append("python -m pip install -r requirements.txt")
    elif "pyproject.toml" in files:
        commands.append("python -m venv .venv")
        commands.append("python -m pip install -e .")
    if "Cargo.toml" in files:
        commands.append("cargo fetch")
    if "go.mod" in files:
        commands.append("go mod download")
    if "Gemfile" in files:
        commands.append("bundle install")
    return dedupe_strings(commands)[:6]


def setup_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if "package.json" in files:
        evidence.append("package.json is present.")
    if "pnpm-lock.yaml" in files:
        evidence.append("pnpm-lock.yaml indicates pnpm.")
    elif "yarn.lock" in files:
        evidence.append("yarn.lock indicates Yarn.")
    elif "package-lock.json" in files:
        evidence.append("package-lock.json indicates npm.")
    if "requirements.txt" in files:
        evidence.append("requirements.txt is present.")
    if "pyproject.toml" in files:
        evidence.append("pyproject.toml is present.")
    if "Cargo.toml" in files:
        evidence.append("Cargo.toml is present.")
    if "go.mod" in files:
        evidence.append("go.mod is present.")
    scripts = package_scripts_from_documents(documents)
    if scripts:
        names = ", ".join(sorted(str(name) for name in scripts)[:5])
        evidence.append(f"package.json defines script(s): {names}.")
    return dedupe_strings(evidence)[:6]


def is_project_commands_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what scripts can i run",
            "what scripts are available",
            "what npm scripts",
            "what pnpm scripts",
            "what yarn scripts",
            "available scripts",
            "project scripts",
            "what commands can i run",
            "what commands are available",
            "available commands",
            "repo commands",
            "project commands",
        )
    )


def render_project_commands_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    commands = project_commands_from_overview(files, documents)

    sections = ["Project Commands"]
    if commands:
        lines = []
        for command, detail in commands[:12]:
            suffix = f" - {detail}" if detail else ""
            lines.append(f"- `{command}`{suffix}")
        sections.append("\n".join(lines))
    else:
        sections.append("- I do not see obvious project commands yet.")

    evidence = project_command_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Run the command that matches your intent, then use the output to pick the next edit."
    )
    return "\n".join(sections)


def project_commands_from_overview(
    files: list[str],
    documents: list[dict],
) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    scripts = package_scripts_from_documents(documents)
    package_manager = package_manager_for_files(files)
    preferred = ("dev", "start", "build", "test", "lint", "preview", "serve")
    script_names = [name for name in preferred if name in scripts]
    script_names.extend(name for name in sorted(scripts) if name not in script_names)
    for name in script_names[:8]:
        detail = scripts.get(name)
        commands.append((script_command(package_manager, str(name)), str(detail or "")))

    if "Cargo.toml" in files:
        commands.append(("cargo build", "compile the Rust workspace"))
        commands.append(("cargo test", "run Rust tests"))
        if any(path.endswith("/src/main.rs") or path == "src/main.rs" for path in files):
            commands.append(("cargo run", "run the default binary"))
    if "pyproject.toml" in files or any(path.endswith(".py") for path in files):
        commands.append(("python -m pytest", "run Python tests"))
    if "go.mod" in files:
        commands.append(("go test ./...", "run Go tests"))
        commands.append(("go run ./...", "run Go packages"))
    if "Makefile" in files:
        commands.append(("make", "run the default Makefile target"))
    for path in files:
        if path.startswith("scripts/") and path.endswith(".sh"):
            commands.append((f"bash {path}", "run project script"))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for command, detail in commands:
        if command in seen:
            continue
        seen.add(command)
        deduped.append((command, detail))
    return deduped


def project_command_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    scripts = package_scripts_from_documents(documents)
    if scripts:
        names = ", ".join(sorted(str(name) for name in scripts)[:8])
        evidence.append(f"package.json defines script(s): {names}.")
    if "Cargo.toml" in files:
        evidence.append("Cargo.toml is present.")
    if "pyproject.toml" in files:
        evidence.append("pyproject.toml is present.")
    if "go.mod" in files:
        evidence.append("go.mod is present.")
    if "Makefile" in files:
        evidence.append("Makefile is present.")
    if any(path.startswith("scripts/") for path in files):
        evidence.append("scripts/ contains project scripts.")
    return dedupe_strings(evidence)[:6]


def is_environment_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "environment variables",
            "env vars",
            "env variables",
            "what env",
            "which env",
            "what secrets",
            "which secrets",
            "configure locally",
            ".env",
            "local config",
            "runtime config",
        )
    )


def render_environment_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    variables = environment_variables_from_documents(documents)

    sections = ["Environment Variables"]
    if variables:
        lines = []
        for name, source, detail in variables[:16]:
            suffix = f" - {detail}" if detail else ""
            lines.append(f"- `{name}` from `{source}`{suffix}")
        sections.append("\n".join(lines))
    else:
        sections.append("- I do not see an example env file or config-declared variables yet.")

    evidence = environment_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    if unsafe_env_files(files):
        sections.append(
            "Safety:\n"
            "- Real `.env` files are present, but I am not reading or echoing their values."
        )
    sections.append(
        "What I would do next:\n"
        "- Copy the example file to your local env file and fill in real secrets locally."
    )
    return "\n".join(sections)


def environment_variables_from_documents(documents: list[dict]) -> list[tuple[str, str, str]]:
    variables: list[tuple[str, str, str]] = []
    for doc in documents:
        path = str(doc.get("path") or "")
        content = doc.get("content")
        if not isinstance(content, str) or not safe_env_source(path):
            continue
        if path.endswith(".json"):
            continue
        if path.endswith(".toml"):
            variables.extend(toml_env_variables(path, content))
            continue
        variables.extend(env_file_variables(path, content))
    return dedupe_env_variables(variables)


def safe_env_source(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered in {"env.example", "env.sample", ".env.example", ".env.sample", ".env.template"}
        or lowered.endswith(".env.example")
        or lowered.endswith(".env.sample")
        or lowered.endswith(".env.template")
        or lowered in {"wrangler.toml", "vercel.json"}
    )


def env_file_variables(path: str, content: str) -> list[tuple[str, str, str]]:
    variables = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            continue
        detail = env_value_detail(value)
        variables.append((name, path, detail))
    return variables


def toml_env_variables(path: str, content: str) -> list[tuple[str, str, str]]:
    variables = []
    in_vars = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_vars = line in {"[vars]", "[env]", "[environment]"}
            continue
        if not in_vars or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            variables.append((name, path, env_value_detail(value)))
    return variables


def env_value_detail(value: str) -> str:
    cleaned = value.strip().strip("\"'")
    if not cleaned:
        return "required"
    if re.search(r"(secret|token|key|password|private)", cleaned, flags=re.IGNORECASE):
        return "set locally"
    return "example/default shown in template"


def dedupe_env_variables(values: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen = set()
    result = []
    for name, source, detail in values:
        if name in seen:
            continue
        seen.add(name)
        result.append((name, source, detail))
    return result


def environment_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    doc_paths = {str(doc.get("path") or "") for doc in documents}
    for path in sorted(doc_paths):
        if safe_env_source(path):
            evidence.append(f"{path} is included as a safe config template.")
    if "vercel.json" in files:
        evidence.append("vercel.json is present.")
    if "wrangler.toml" in files:
        evidence.append("wrangler.toml is present.")
    return dedupe_strings(evidence)[:6]


def unsafe_env_files(files: list[str]) -> list[str]:
    unsafe = []
    for path in files:
        lowered = path.lower()
        if lowered in {".env", ".env.local", ".env.production", ".env.development"}:
            unsafe.append(path)
    return unsafe


def is_container_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "docker",
            "dockerfile",
            "docker compose",
            "docker-compose",
            "container",
            "containers",
            "containerized",
            "compose up",
            "run in docker",
            "run with docker",
        )
    )


def render_container_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    container_files = project_container_files(files)
    commands = container_commands_from_project_overview(files, documents)

    sections = ["Container Runtime"]
    if container_files:
        sections.append(
            "Container Files:\n"
            + "\n".join(
                f"- `{path}` - {description}"
                for path, description in container_files[:10]
            )
        )
    else:
        sections.append("Container Files:\n- I do not see Docker/container files yet.")

    if commands:
        sections.append(
            "Likely Commands:\n" + "\n".join(f"- `{command}`" for command in commands)
        )

    evidence = container_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the container file first, then check environment variables and ports "
        "before running it."
    )
    return "\n".join(sections)


def project_container_files(files: list[str]) -> list[tuple[str, str]]:
    configs = []
    for path in files:
        description = container_file_description(path)
        if description:
            configs.append((path, description))
    return sorted(configs, key=lambda item: (container_rank(item[0]), item[0]))


def container_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "Docker image definition"
    if name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
        return "Docker Compose runtime config"
    if name == ".dockerignore":
        return "Docker build ignore rules"
    if lowered.startswith(("k8s/", "kubernetes/")) and lowered.endswith((".yml", ".yaml")):
        return "Kubernetes manifest"
    return ""


def container_rank(path: str) -> int:
    name = Path(path).name.lower()
    if name == "dockerfile":
        return 0
    if name.startswith("dockerfile."):
        return 1
    if "compose" in name:
        return 2
    if name == ".dockerignore":
        return 3
    return 4


def container_commands_from_project_overview(
    files: list[str],
    documents: list[dict],
) -> list[str]:
    commands = []
    compose_file = first_compose_file(files)
    if compose_file:
        commands.append("docker compose up --build")
        commands.append("docker compose down")

    dockerfile = first_dockerfile(files)
    if dockerfile:
        image_name = container_image_name(documents)
        build_context = "." if "/" not in dockerfile else str(Path(dockerfile).parent)
        dockerfile_flag = "" if Path(dockerfile).name == "Dockerfile" else f" -f {dockerfile}"
        commands.append(f"docker build{dockerfile_flag} -t {image_name} {build_context}")
        ports = docker_exposed_ports_from_documents(documents)
        if ports:
            commands.append(f"docker run --rm -p {ports[0]}:{ports[0]} {image_name}")
        else:
            commands.append(f"docker run --rm {image_name}")
    return dedupe_strings(commands)[:5]


def first_dockerfile(files: list[str]) -> str:
    candidates = [
        path
        for path in files
        if Path(path).name.lower() == "dockerfile"
        or Path(path).name.lower().startswith("dockerfile.")
    ]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda path: (container_rank(path), path))[0]


def first_compose_file(files: list[str]) -> str:
    compose_names = {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}
    candidates = [path for path in files if Path(path).name.lower() in compose_names]
    return sorted(candidates)[0] if candidates else ""


def container_image_name(documents: list[dict]) -> str:
    package = package_json_from_documents(documents)
    raw_name = package.get("name") if isinstance(package, dict) else ""
    if not isinstance(raw_name, str) or not raw_name.strip():
        return "app"
    name = raw_name.strip().lower().replace("@", "").replace("/", "-")
    name = re.sub(r"[^a-z0-9_.-]+", "-", name).strip("-")
    return name or "app"


def docker_exposed_ports_from_documents(documents: list[dict]) -> list[str]:
    ports = []
    for doc in documents:
        path = str(doc.get("path") or "")
        if Path(path).name.lower() != "dockerfile" or not isinstance(doc.get("content"), str):
            continue
        for raw_line in str(doc["content"]).splitlines():
            stripped = raw_line.strip()
            if not stripped.upper().startswith("EXPOSE "):
                continue
            for token in stripped.split()[1:]:
                port = token.split("/", 1)[0]
                if port.isdigit():
                    ports.append(port)
    return dedupe_strings(ports)


def container_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if first_dockerfile(files):
        evidence.append("Dockerfile is present.")
    if first_compose_file(files):
        evidence.append("Docker Compose config is present.")
    if ".dockerignore" in files:
        evidence.append(".dockerignore is present.")
    ports = docker_exposed_ports_from_documents(documents)
    if ports:
        evidence.append(f"Dockerfile exposes port(s): {', '.join(ports[:4])}.")
    if any(path.startswith(("k8s/", "kubernetes/")) for path in files):
        evidence.append("Kubernetes manifests are present.")
    return dedupe_strings(evidence)[:5]


def is_deployment_question(task: str) -> bool:
    if is_container_question(task):
        return False
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how do i deploy",
            "how to deploy",
            "deploy this",
            "deploy the",
            "deployment",
            "deployment config",
            "deploy config",
            "deploy command",
            "deploy commands",
            "hosting",
            "host this",
            "where is deploy",
            "where is deployment",
            "production deploy",
            "release workflow",
            "ship this",
            "publish this",
        )
    )


def render_deployment_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    configs = project_deployment_files(files)
    targets = deployment_targets(files)
    commands = deployment_commands_from_project_overview(files, documents)

    sections = ["Deployment"]
    if targets:
        sections.append(
            "Detected Targets:\n" + "\n".join(f"- {target}" for target in targets[:8])
        )
    else:
        sections.append("Detected Targets:\n- I do not see an obvious deployment target yet.")

    if configs:
        sections.append(
            "Deployment Files:\n"
            + "\n".join(
                f"- `{path}` - {description}" for path, description in configs[:12]
            )
        )

    if commands:
        sections.append(
            "Build/Deploy Commands:\n"
            + "\n".join(render_command_hint(command) for command in commands[:8])
        )

    evidence = deployment_evidence(files, documents, commands)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the deployment config, confirm required environment variables, then run "
        "the build or deploy command from this repo."
    )
    return "\n".join(sections)


def render_command_hint(command: str) -> str:
    if " - " not in command:
        return f"- `{command}`"
    executable, detail = command.split(" - ", 1)
    return f"- `{executable}` - {detail}"


def project_deployment_files(files: list[str]) -> list[tuple[str, str]]:
    configs = []
    for path in files:
        description = deployment_file_description(path)
        if description:
            configs.append((path, description))
    return sorted(configs, key=lambda item: (deployment_rank(item[0]), item[0]))


def deployment_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if lowered == "vercel.json":
        return "Vercel deployment config"
    if lowered == "wrangler.toml":
        return "Cloudflare Workers/Pages deployment config"
    if lowered == "netlify.toml":
        return "Netlify deployment config"
    if lowered == "fly.toml":
        return "Fly.io app deployment config"
    if lowered in {"railway.toml", "railway.json"}:
        return "Railway deployment config"
    if lowered in {"render.yaml", "render.yml"}:
        return "Render service deployment config"
    if name == "procfile":
        return "process declaration used by platforms like Heroku"
    if lowered == "amplify.yml":
        return "AWS Amplify deployment config"
    if lowered.startswith(".github/workflows/") and (
        "deploy" in name or "release" in name
    ):
        return "deployment workflow"
    if container_file_description(path):
        return container_file_description(path)
    if lowered.startswith(("k8s/", "kubernetes/")) and lowered.endswith((".yml", ".yaml")):
        return "Kubernetes deployment manifest"
    return ""


def deployment_rank(path: str) -> int:
    lowered = path.lower()
    order = (
        "vercel.json",
        "wrangler.toml",
        "netlify.toml",
        "fly.toml",
        "railway",
        "render.",
        "procfile",
        "amplify.yml",
        ".github/workflows",
        "docker",
        "k8s/",
        "kubernetes/",
    )
    for index, prefix in enumerate(order):
        if lowered.startswith(prefix) or prefix in lowered:
            return index
    return len(order)


def deployment_targets(files: list[str]) -> list[str]:
    targets = []
    lowered_files = {path.lower(): path for path in files}
    if "vercel.json" in lowered_files:
        targets.append("Vercel (`vercel.json`)")
    if "wrangler.toml" in lowered_files:
        targets.append("Cloudflare Workers/Pages (`wrangler.toml`)")
    if "netlify.toml" in lowered_files:
        targets.append("Netlify (`netlify.toml`)")
    if "fly.toml" in lowered_files:
        targets.append("Fly.io (`fly.toml`)")
    if "railway.toml" in lowered_files or "railway.json" in lowered_files:
        targets.append("Railway")
    if "render.yaml" in lowered_files or "render.yml" in lowered_files:
        targets.append("Render")
    if "procfile" in lowered_files:
        targets.append("Process-based hosting (`Procfile`)")
    if "amplify.yml" in lowered_files:
        targets.append("AWS Amplify (`amplify.yml`)")
    if any(
        path.startswith(".github/workflows/")
        and ("deploy" in Path(path).name.lower() or "release" in Path(path).name.lower())
        for path in files
    ):
        targets.append("GitHub Actions deployment workflow")
    if project_container_files(files):
        targets.append("Container deployment")
    if any(path.startswith(("k8s/", "kubernetes/")) for path in files):
        targets.append("Kubernetes")
    return dedupe_strings(targets)


def deployment_commands_from_project_overview(
    files: list[str],
    documents: list[dict],
) -> list[str]:
    package_manager = package_manager_for_files(files)
    scripts = package_scripts_from_documents(documents)
    commands = []

    for name, value in sorted(scripts.items(), key=lambda item: str(item[0])):
        if not isinstance(value, str):
            continue
        lowered_name = str(name).lower()
        lowered_value = value.lower()
        if "deploy" in lowered_name or any(
            marker in lowered_value
            for marker in ("vercel", "wrangler", "netlify", "flyctl", "railway")
        ):
            commands.append(f"{script_command(package_manager, str(name))} - {value}")

    build_script = scripts.get("build")
    if isinstance(build_script, str):
        commands.append(f"{script_command(package_manager, 'build')} - {build_script}")

    commands.extend(deployment_document_commands(files, documents))
    if any(path.lower() == "vercel.json" for path in files):
        commands.append("vercel deploy")
    if any(path.lower() == "wrangler.toml" for path in files):
        commands.append("wrangler deploy")
    if any(path.lower() == "netlify.toml" for path in files):
        commands.append("netlify deploy")
    if any(path.lower() == "fly.toml" for path in files):
        commands.append("flyctl deploy")
    commands.extend(container_commands_from_project_overview(files, documents)[:2])
    return dedupe_strings(commands)[:8]


def deployment_document_commands(files: list[str], documents: list[dict]) -> list[str]:
    commands = []
    for doc in documents:
        path = str(doc.get("path") or "")
        content = doc.get("content")
        if not isinstance(content, str):
            continue
        if path == "vercel.json":
            commands.extend(vercel_commands_from_content(content))
        if path.startswith(".github/workflows/") and (
            "deploy" in path.lower() or "release" in path.lower()
        ):
            commands.extend(
                command
                for command in ci_commands_from_documents([doc])
                if is_likely_deployment_command(command)
            )
    if "Procfile" in files:
        commands.append("check `Procfile` for the runtime process")
    return commands


def vercel_commands_from_content(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    commands = []
    for key in ("installCommand", "buildCommand"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            commands.append(value.strip())
    return commands


def is_likely_deployment_command(command: str) -> bool:
    lowered = command.lower()
    return any(
        marker in lowered
        for marker in (
            "deploy",
            "vercel",
            "wrangler",
            "netlify",
            "flyctl",
            "railway",
            "render",
        )
    )


def deployment_evidence(
    files: list[str],
    documents: list[dict],
    commands: list[str],
) -> list[str]:
    evidence = []
    targets = deployment_targets(files)
    if targets:
        evidence.append("Deployment target files are present.")
    if package_scripts_from_documents(documents):
        evidence.append("package.json scripts can provide build/deploy commands.")
    if any(str(doc.get("path") or "") == "vercel.json" for doc in documents):
        evidence.append("vercel.json was read for command hints.")
    if any(str(doc.get("path") or "") == "wrangler.toml" for doc in documents):
        evidence.append("wrangler.toml was read for Cloudflare deployment hints.")
    if commands:
        evidence.append("Deploy/build commands were inferred from local config.")
    return dedupe_strings(evidence)[:5]


def is_database_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what database",
            "which database",
            "database does",
            "database use",
            "db does",
            "db use",
            "data layer",
            "schema",
            "schemas",
            "migration",
            "migrations",
            "where is database",
            "where is db",
            "where is schema",
            "where are migrations",
            "storage layer",
        )
    )


def render_database_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    data_layer = database_stack_signals(files, documents)
    schema_files = database_schema_files(files)
    commands = database_commands_from_project_overview(files, documents)

    sections = ["Database / Schema"]
    if data_layer:
        sections.append("Detected Data Layer:\n" + "\n".join(f"- {item}" for item in data_layer))
    else:
        sections.append("Detected Data Layer:\n- I do not see an obvious database layer yet.")

    if schema_files:
        sections.append(
            "Schema and Migrations:\n"
            + "\n".join(
                f"- `{path}` - {description}"
                for path, description in schema_files[:14]
            )
        )
        if len(schema_files) > 14:
            sections.append(f"- ...and {len(schema_files) - 14} more database files.")

    if commands:
        sections.append(
            "Likely Commands:\n" + "\n".join(render_command_hint(command) for command in commands)
        )

    env_signals = database_env_signals(documents)
    if env_signals:
        sections.append(
            "Environment Hints:\n"
            + "\n".join(f"- `{name}` appears in `{source}`." for name, source in env_signals[:5])
        )

    evidence = database_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the schema or migration file, then run the safest local migration/status command."
    )
    return "\n".join(sections)


def database_stack_signals(files: list[str], documents: list[dict]) -> list[str]:
    package = package_json_from_documents(documents)
    dependencies = package_dependencies(package)
    pyproject = pyproject_from_documents(documents)
    python_dependencies = pyproject_dependencies(pyproject)
    signals = []

    node_database_signals = {
        "prisma": "Prisma ORM",
        "@prisma/client": "Prisma client",
        "drizzle-orm": "Drizzle ORM",
        "drizzle-kit": "Drizzle migrations",
        "@supabase/supabase-js": "Supabase client",
        "pg": "Postgres driver",
        "postgres": "Postgres client",
        "mysql2": "MySQL driver",
        "mongoose": "MongoDB via Mongoose",
        "mongodb": "MongoDB driver",
        "redis": "Redis client/cache",
        "ioredis": "Redis client/cache",
    }
    for dependency, label in node_database_signals.items():
        if dependency in dependencies:
            signals.append(label)

    python_database_signals = {
        "sqlalchemy": "SQLAlchemy ORM",
        "alembic": "Alembic migrations",
        "psycopg2": "Postgres driver",
        "psycopg": "Postgres driver",
        "asyncpg": "Postgres driver",
        "django": "Django ORM",
        "flask-sqlalchemy": "Flask SQLAlchemy",
        "pymongo": "MongoDB driver",
        "redis": "Redis client/cache",
    }
    for dependency, label in python_database_signals.items():
        if dependency in python_dependencies:
            signals.append(label)

    if any(path.startswith("prisma/") for path in files):
        signals.append("Prisma schema/migrations")
    if any(path.startswith("supabase/") for path in files):
        signals.append("Supabase local project")
    if any(path.startswith("drizzle/") for path in files) or any(
        Path(path).name.startswith("drizzle.config") for path in files
    ):
        signals.append("Drizzle schema/migrations")
    if any("migrations" in path.lower() and path.endswith(".sql") for path in files):
        signals.append("SQL migrations")
    if "schema.sql" in {Path(path).name.lower() for path in files}:
        signals.append("SQL schema file")
    return dedupe_strings(signals)[:8]


def pyproject_dependencies(pyproject: dict) -> set[str]:
    dependencies: set[str] = set()
    raw_dependencies = pyproject.get("dependencies") if isinstance(pyproject, dict) else None
    if isinstance(raw_dependencies, list):
        for raw_dependency in raw_dependencies:
            if not isinstance(raw_dependency, str):
                continue
            name = re.split(r"[<>=!~;\[\]\s]", raw_dependency.strip(), maxsplit=1)[0]
            if name:
                dependencies.add(name.lower())
    return dependencies


def database_schema_files(files: list[str]) -> list[tuple[str, str]]:
    schema_files = []
    for path in files:
        description = database_file_description(path)
        if description:
            schema_files.append((path, description))
    return sorted(schema_files, key=lambda item: (database_file_rank(item[0]), item[0]))


def database_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if lowered == "prisma/schema.prisma" or name == "schema.prisma":
        return "Prisma schema"
    if lowered.startswith("prisma/migrations/"):
        return "Prisma migration"
    if name.startswith("drizzle.config"):
        return "Drizzle config"
    if lowered.startswith("drizzle/") and lowered.endswith((".sql", ".ts", ".js")):
        return "Drizzle schema/migration"
    if lowered.startswith("supabase/migrations/") and lowered.endswith(".sql"):
        return "Supabase migration"
    if lowered == "supabase/config.toml":
        return "Supabase local config"
    if lowered.startswith("migrations/") and lowered.endswith((".sql", ".py")):
        return "database migration"
    if lowered.startswith("alembic/versions/") and lowered.endswith(".py"):
        return "Alembic migration"
    if lowered == "alembic.ini" or lowered == "migrations/env.py":
        return "Alembic migration config"
    if name in {"schema.sql", "structure.sql"}:
        return "SQL schema"
    if lowered in {"db/schema.ts", "db/schema.sql", "database/schema.sql"}:
        return "database schema"
    if lowered in {"knexfile.js", "knexfile.ts"}:
        return "Knex migration config"
    if lowered in {"typeorm.config.ts", "typeorm.config.js", "sequelize.config.js"}:
        return "ORM config"
    if name == "models.py" and (lowered.startswith(("app/", "src/")) or "/" not in lowered):
        return "Python data models"
    return ""


def database_file_rank(path: str) -> int:
    lowered = path.lower()
    order = (
        "prisma/schema.prisma",
        "drizzle.config",
        "supabase/config.toml",
        "schema.sql",
        "db/schema",
        "database/schema",
        "prisma/migrations",
        "supabase/migrations",
        "drizzle/",
        "migrations/",
        "alembic",
        "models.py",
    )
    for index, prefix in enumerate(order):
        if lowered.startswith(prefix) or prefix in lowered:
            return index
    return len(order)


def database_commands_from_project_overview(
    files: list[str],
    documents: list[dict],
) -> list[str]:
    package_manager = package_manager_for_files(files)
    scripts = package_scripts_from_documents(documents)
    commands = []
    for name, value in sorted(scripts.items(), key=lambda item: str(item[0])):
        if not isinstance(value, str):
            continue
        lowered_name = str(name).lower()
        lowered_value = value.lower()
        if any(marker in lowered_name for marker in ("db", "database", "migrate", "prisma")) or any(
            marker in lowered_value
            for marker in ("prisma", "drizzle", "migrate", "supabase db", "knex")
        ):
            commands.append(f"{script_command(package_manager, str(name))} - {value}")

    if "prisma/schema.prisma" in files:
        commands.append("npx prisma migrate dev")
        commands.append("npx prisma studio")
    if any(Path(path).name.startswith("drizzle.config") for path in files):
        commands.append("npx drizzle-kit migrate")
    if any(path.startswith("supabase/") for path in files):
        commands.append("supabase db push")
    if "alembic.ini" in files or any(path.startswith("alembic/") for path in files):
        commands.append("alembic upgrade head")
    return dedupe_strings(commands)[:8]


def database_env_signals(documents: list[dict]) -> list[tuple[str, str]]:
    signals = []
    database_names = {
        "DATABASE_URL",
        "POSTGRES_URL",
        "POSTGRES_PRISMA_URL",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "MONGODB_URI",
        "REDIS_URL",
        "MYSQL_URL",
    }
    for name, source, _detail in environment_variables_from_documents(documents):
        has_database_marker = any(
            marker in name for marker in ("DATABASE", "POSTGRES", "SUPABASE")
        )
        if name in database_names or has_database_marker:
            signals.append((name, source))
    return signals


def database_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if database_schema_files(files):
        evidence.append("Schema or migration files are present.")
    if database_stack_signals(files, documents):
        evidence.append("Database dependencies or framework files were detected.")
    if database_env_signals(documents):
        evidence.append("Safe env templates reference database-related variables.")
    return dedupe_strings(evidence)[:5]


def is_auth_question(task: str) -> bool:
    lowered = task.lower()
    if re.search(r"(\bauth\b|/auth|auth[_.-]|nextauth|oauth)", lowered):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "authentication",
            "authorization",
            "login",
            "log in",
            "sign in",
            "sign-in",
            "signin",
            "sign up",
            "sign-up",
            "signup",
            "session management",
            "session store",
            "session cookie",
            "where are sessions",
            "where is session",
            "jwt",
            "clerk",
            "supabase auth",
            "protected routes",
            "protected route",
        )
    )


def render_auth_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    stacks = auth_stack_signals(files, documents)
    auth_files = auth_related_files(files)
    env_signals = auth_env_signals(documents)
    commands = run_commands_from_project_overview(payload)

    sections = ["Auth / Sessions"]
    if stacks:
        sections.append("Detected Auth Stack:\n" + "\n".join(f"- {item}" for item in stacks))
    else:
        sections.append("Detected Auth Stack:\n- I do not see an obvious auth library yet.")

    if auth_files:
        sections.append(
            "Auth Files:\n"
            + "\n".join(
                f"- `{path}` - {description}" for path, description in auth_files[:16]
            )
        )
        if len(auth_files) > 16:
            sections.append(f"- ...and {len(auth_files) - 16} more auth-related files.")
    else:
        sections.append("Auth Files:\n- I do not see obvious auth/session files yet.")

    if env_signals:
        sections.append(
            "Environment Hints:\n"
            + "\n".join(f"- `{name}` appears in `{source}`." for name, source in env_signals[:8])
        )

    if commands:
        sections.append(
            "Likely Local Server:\n" + "\n".join(f"- `{command}`" for command in commands[:4])
        )

    evidence = auth_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the auth config or route first, then trace session reads into protected "
        "UI/API paths."
    )
    return "\n".join(sections)


def auth_stack_signals(files: list[str], documents: list[dict]) -> list[str]:
    package = package_json_from_documents(documents)
    dependencies = package_dependencies(package)
    pyproject = pyproject_from_documents(documents)
    python_dependencies = pyproject_dependencies(pyproject)
    signals = []

    node_auth_signals = {
        "next-auth": "NextAuth/Auth.js",
        "@auth/core": "Auth.js",
        "@auth/nextjs": "Auth.js for Next.js",
        "@clerk/nextjs": "Clerk",
        "@supabase/supabase-js": "Supabase Auth",
        "better-auth": "Better Auth",
        "lucia": "Lucia Auth",
        "passport": "Passport.js",
        "jsonwebtoken": "JWT",
        "jose": "JOSE/JWT",
        "@auth0/nextjs-auth0": "Auth0",
        "firebase": "Firebase Auth",
        "firebase-admin": "Firebase Admin Auth",
        "iron-session": "cookie sessions",
    }
    for dependency, label in node_auth_signals.items():
        if dependency in dependencies:
            signals.append(label)

    python_auth_signals = {
        "django": "Django auth",
        "django-allauth": "Django allauth",
        "djangorestframework-simplejwt": "Django REST Framework JWT",
        "fastapi-users": "FastAPI Users",
        "flask-login": "Flask-Login",
        "authlib": "Authlib OAuth",
        "python-jose": "JOSE/JWT",
        "pyjwt": "JWT",
        "passlib": "password hashing",
    }
    for dependency, label in python_auth_signals.items():
        if dependency in python_dependencies:
            signals.append(label)

    if any("/api/auth/" in f"/{path.lower()}" for path in files):
        signals.append("API auth route")
    if any(Path(path).name.lower().startswith("middleware.") for path in files):
        signals.append("request middleware")
    if any("login" in path.lower() or "sign-in" in path.lower() for path in files):
        signals.append("login/sign-in UI")
    if auth_env_signals(documents):
        signals.append("auth-related environment variables")
    return dedupe_strings(signals)[:8]


def auth_related_files(files: list[str]) -> list[tuple[str, str]]:
    auth_files = []
    for path in files:
        description = auth_file_description(path)
        if description:
            auth_files.append((path, description))
    return sorted(auth_files, key=lambda item: (auth_file_rank(item[0]), item[0]))


def auth_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    auth_module_files = {
        "auth.ts",
        "auth.js",
        "src/auth.ts",
        "src/auth.js",
        "lib/auth.ts",
        "lib/auth.js",
    }
    if lowered in auth_module_files:
        return "auth config/module"
    if name.startswith("auth.config."):
        return "Auth.js config"
    if "/api/auth/" in f"/{lowered}":
        if "nextauth" in lowered:
            return "NextAuth/Auth.js route"
        return "auth API route"
    if name.startswith("middleware.") and lowered.endswith((".ts", ".js", ".py")):
        return "request/auth middleware"
    if lowered in {"lib/session.ts", "lib/session.js", "src/session.ts", "src/session.js"}:
        return "session helper"
    auth_ui_suffixes = (
        "/login/page.tsx",
        "/login/page.ts",
        "/sign-in/page.tsx",
        "/sign-in/page.ts",
    )
    if lowered.endswith(auth_ui_suffixes):
        return "auth UI route"
    if lowered.endswith(("/signup/page.tsx", "/sign-up/page.tsx", "/register/page.tsx")):
        return "registration UI route"
    if lowered.startswith(("src/auth/", "lib/auth/", "app/auth/")):
        return "auth module"
    if lowered.startswith(("routes/auth.", "src/routes/auth.")):
        return "auth route module"
    if name in {"auth.py", "sessions.py", "session.py"}:
        return "Python auth/session module"
    return ""


def auth_file_rank(path: str) -> int:
    lowered = path.lower()
    order = (
        "auth.",
        "auth.config",
        "src/auth.",
        "lib/auth.",
        "app/api/auth/",
        "pages/api/auth/",
        "middleware.",
        "src/middleware.",
        "lib/session.",
        "src/session.",
        "app/auth/",
        "src/auth/",
        "lib/auth/",
        "routes/auth.",
        "src/routes/auth.",
        "login/",
        "sign-in/",
    )
    for index, prefix in enumerate(order):
        if lowered.startswith(prefix) or prefix in lowered:
            return index
    return len(order)


def auth_env_signals(documents: list[dict]) -> list[tuple[str, str]]:
    exact_names = {
        "AUTH_SECRET",
        "NEXTAUTH_SECRET",
        "NEXTAUTH_URL",
        "CLERK_SECRET_KEY",
        "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "JWT_SECRET",
        "SESSION_SECRET",
        "AUTH0_SECRET",
        "AUTH0_BASE_URL",
        "AUTH0_ISSUER_BASE_URL",
        "AUTH0_CLIENT_ID",
        "AUTH0_CLIENT_SECRET",
    }
    marker_names = (
        "OAUTH",
        "AUTH_",
        "CLERK_",
        "NEXTAUTH_",
        "GOOGLE_CLIENT",
        "GITHUB_CLIENT",
        "SESSION_",
        "JWT_",
    )
    signals = []
    for name, source, _detail in environment_variables_from_documents(documents):
        if name in exact_names or any(marker in name for marker in marker_names):
            signals.append((name, source))
    return signals


def auth_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if auth_related_files(files):
        evidence.append("Auth/session file names or routes are present.")
    if auth_stack_signals(files, documents):
        evidence.append("Auth dependencies, conventions, or env names were detected.")
    if auth_env_signals(documents):
        evidence.append("Safe env templates reference auth-related variables.")
    return dedupe_strings(evidence)[:5]


def is_frontend_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "frontend",
            "front end",
            "front-end",
            "ui code",
            "user interface",
            "where are components",
            "where are the components",
            "where is components",
            "component files",
            "react components",
            "page components",
            "where are pages",
            "where is the app shell",
            "where is layout",
            "where are styles",
            "styling files",
            "tailwind",
            "design system",
        )
    )


def render_frontend_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    stacks = frontend_stack_signals(files, documents)
    frontend_files = frontend_related_files(files)
    commands = run_commands_from_project_overview(payload)

    sections = ["UI / Frontend"]
    if stacks:
        sections.append("Detected Frontend Stack:\n" + "\n".join(f"- {item}" for item in stacks))
    else:
        sections.append("Detected Frontend Stack:\n- I do not see an obvious frontend stack yet.")

    if frontend_files:
        sections.append(
            "Frontend Files:\n"
            + "\n".join(
                f"- `{path}` - {description}"
                for path, description in frontend_files[:18]
            )
        )
        if len(frontend_files) > 18:
            sections.append(f"- ...and {len(frontend_files) - 18} more frontend files.")
    else:
        sections.append("Frontend Files:\n- I do not see obvious frontend files yet.")

    if commands:
        sections.append(
            "Likely Local Server:\n" + "\n".join(f"- `{command}`" for command in commands[:4])
        )

    evidence = frontend_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the root page or app component first, then follow imported components and styles."
    )
    return "\n".join(sections)


def frontend_stack_signals(files: list[str], documents: list[dict]) -> list[str]:
    package = package_json_from_documents(documents)
    dependencies = package_dependencies(package)
    signals = []
    dependency_signals = {
        "next": "Next.js",
        "react": "React",
        "vue": "Vue",
        "svelte": "Svelte",
        "solid-js": "Solid",
        "astro": "Astro",
        "vite": "Vite",
        "@tanstack/react-router": "TanStack Router",
        "@tanstack/react-start": "TanStack Start",
        "tailwindcss": "Tailwind CSS",
        "@tailwindcss/vite": "Tailwind CSS",
        "styled-components": "styled-components",
        "@emotion/react": "Emotion",
        "lucide-react": "Lucide React",
        "framer-motion": "Framer Motion",
    }
    for dependency, label in dependency_signals.items():
        if dependency in dependencies:
            signals.append(label)
    if "components.json" in files:
        signals.append("shadcn/ui")
    if any(path.startswith("app/") for path in files):
        signals.append("app directory routing")
    if any(path.startswith("pages/") for path in files):
        signals.append("pages directory routing")
    if any("tailwind.config" in Path(path).name for path in files):
        signals.append("Tailwind config")
    return dedupe_strings(signals)[:10]


def frontend_related_files(files: list[str]) -> list[tuple[str, str]]:
    frontend_files = []
    for path in files:
        description = frontend_file_description(path)
        if description:
            frontend_files.append((path, description))
    return sorted(frontend_files, key=lambda item: (frontend_file_rank(item[0]), item[0]))


def frontend_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if lowered in {"app/page.tsx", "app/page.ts", "app/page.jsx", "app/page.js"}:
        return "root app page"
    if lowered in {"app/layout.tsx", "app/layout.ts", "app/layout.jsx", "app/layout.js"}:
        return "app layout shell"
    if lowered in {"src/app.tsx", "src/app.jsx", "src/app.ts", "src/app.js"}:
        return "root app component"
    if lowered in {"src/main.tsx", "src/main.jsx", "src/main.ts", "src/main.js"}:
        return "frontend mount entry"
    if lowered in {"pages/index.tsx", "pages/index.jsx", "pages/index.ts", "pages/index.js"}:
        return "pages router homepage"
    if lowered.startswith(("components/", "src/components/")) and lowered.endswith(
        (".tsx", ".jsx", ".ts", ".js")
    ):
        if "/ui/" in f"/{lowered}":
            return "UI component"
        return "component"
    if lowered.startswith(("app/", "pages/")) and lowered.endswith(
        ("/page.tsx", "/page.jsx", ".tsx", ".jsx")
    ):
        return "page route"
    if lowered in {"app/globals.css", "src/index.css", "src/app.css", "styles/globals.css"}:
        return "global styles"
    if name.startswith("tailwind.config."):
        return "Tailwind config"
    if name == "components.json":
        return "component registry config"
    if name.startswith("vite.config."):
        return "Vite frontend config"
    if name.startswith("next.config."):
        return "Next.js frontend config"
    return ""


def frontend_file_rank(path: str) -> int:
    lowered = path.lower()
    order = (
        "app/page.",
        "app/layout.",
        "src/app.",
        "src/main.",
        "pages/index.",
        "components/ui/",
        "src/components/ui/",
        "components/",
        "src/components/",
        "app/",
        "pages/",
        "app/globals.css",
        "styles/globals.css",
        "src/index.css",
        "tailwind.config",
        "components.json",
        "vite.config",
        "next.config",
    )
    for index, prefix in enumerate(order):
        if lowered.startswith(prefix) or prefix in lowered:
            return index
    return len(order)


def frontend_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if frontend_related_files(files):
        evidence.append("Frontend entry, page, component, or style files are present.")
    if frontend_stack_signals(files, documents):
        evidence.append("Frontend dependencies or framework conventions were detected.")
    if package_scripts_from_documents(documents):
        evidence.append("package.json scripts can provide a local frontend server command.")
    return dedupe_strings(evidence)[:5]


def is_debugging_question(task: str) -> bool:
    if is_code_editing_task(task):
        return False
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how should i debug",
            "how do i debug",
            "how to debug",
            "debug this",
            "debug failures",
            "debug failing",
            "investigate failure",
            "investigate failures",
            "investigate failing",
            "why is this failing",
            "why are tests failing",
            "why are the tests failing",
            "where are errors handled",
            "where is error handling",
            "error handling",
            "logging setup",
            "where are logs",
            "where is logging",
        )
    )


def render_debugging_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    commands = debug_commands_from_project_overview(files, documents)
    debug_files = debugging_files(files)

    sections = ["Debugging Path"]
    if commands:
        sections.append(
            "Start With:\n" + "\n".join(render_command_hint(command) for command in commands[:8])
        )
    else:
        sections.append("Start With:\n- I do not see an obvious debug/test command yet.")

    if debug_files:
        sections.append(
            "Useful Files:\n"
            + "\n".join(f"- `{path}` - {description}" for path, description in debug_files[:16])
        )
        if len(debug_files) > 16:
            sections.append(f"- ...and {len(debug_files) - 16} more investigation files.")
    else:
        sections.append("Useful Files:\n- I do not see obvious test, CI, or logging files yet.")

    evidence = debugging_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Run the narrowest failing check, read the first error frame, then open the closest "
        "test or handler file."
    )
    return "\n".join(sections)


def debug_commands_from_project_overview(
    files: list[str],
    documents: list[dict],
) -> list[str]:
    package_manager = package_manager_for_files(files)
    scripts = package_scripts_from_documents(documents)
    preferred = (
        "test",
        "test:unit",
        "test:e2e",
        "lint",
        "typecheck",
        "type-check",
        "check",
        "build",
        "e2e",
    )
    commands = []
    for name in preferred:
        detail = scripts.get(name)
        if isinstance(detail, str):
            commands.append(f"{script_command(package_manager, name)} - {detail}")
    for name, detail in sorted(scripts.items(), key=lambda item: str(item[0])):
        if not isinstance(detail, str):
            continue
        lowered_name = str(name).lower()
        lowered_detail = detail.lower()
        name_matches = any(
            marker in lowered_name for marker in ("test", "lint", "type", "check", "e2e")
        )
        detail_matches = any(
            marker in lowered_detail
            for marker in ("vitest", "jest", "playwright", "cypress", "eslint", "tsc")
        )
        if name_matches or detail_matches:
            commands.append(f"{script_command(package_manager, str(name))} - {detail}")

    commands.extend(
        verification_commands_from_project_overview(
            {"files": files, "documents": documents}
        )
    )
    if "Cargo.toml" in files:
        commands.append("cargo clippy")
    if "go.mod" in files:
        commands.append("go test ./...")
    if "Makefile" in files:
        commands.append("make test")

    deduped = []
    seen_executables = set()
    for command in commands:
        executable = command.split(" - ", 1)[0]
        if executable in seen_executables:
            continue
        seen_executables.add(executable)
        deduped.append(command)
    return deduped[:10]


def debugging_files(files: list[str]) -> list[tuple[str, str]]:
    candidates = []
    for path in files:
        description = debugging_file_description(path)
        if description:
            candidates.append((path, description))
    return sorted(candidates, key=lambda item: (debugging_file_rank(item[0]), item[0]))


def debugging_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if path in project_test_files([path]):
        return "test file"
    if lowered.startswith(".github/workflows/") and lowered.endswith((".yml", ".yaml")):
        return "CI workflow"
    if name.startswith(("vitest.config", "jest.config", "playwright.config", "cypress.config")):
        return "test runner config"
    if name in {"pytest.ini", "ruff.toml", "mypy.ini"}:
        return "Python check config"
    if name in {"tsconfig.json", "eslint.config.js", "eslint.config.mjs", "biome.json"}:
        return "JavaScript/TypeScript check config"
    if lowered.endswith(("/error.tsx", "/error.jsx", "/error.ts", "/error.js")):
        return "route error boundary"
    if lowered.endswith(("/global-error.tsx", "/global-error.jsx")):
        return "global error boundary"
    if name in {"logger.ts", "logger.js", "logging.py"} or "/logger." in lowered:
        return "logging helper"
    if name.startswith("sentry.") or name.startswith("instrumentation."):
        return "observability/error instrumentation"
    return ""


def debugging_file_rank(path: str) -> int:
    lowered = path.lower()
    name = Path(path).name.lower()
    if path in project_test_files([path]):
        return 0
    if name.startswith(("vitest.config", "jest.config", "playwright.config", "cypress.config")):
        return 1
    if lowered.startswith(".github/workflows/"):
        return 2
    if name in {"pytest.ini", "ruff.toml", "mypy.ini", "tsconfig.json"}:
        return 3
    if "error." in name or "global-error" in name:
        return 4
    if "logger" in name or "logging" in name:
        return 5
    return 6


def debugging_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    scripts = package_scripts_from_documents(documents)
    if scripts:
        names = ", ".join(sorted(str(name) for name in scripts)[:8])
        evidence.append(f"package.json defines script(s): {names}.")
    if project_test_files(files):
        evidence.append("Test files are present.")
    if ci_workflow_files(files):
        evidence.append("CI workflow files are present.")
    if debugging_files(files):
        evidence.append("Test, CI, error-boundary, or logging files were detected.")
    return dedupe_strings(evidence)[:5]


def is_api_routes_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "api routes",
            "api endpoints",
            "endpoints",
            "http routes",
            "server routes",
            "where are routes",
            "where is routes",
            "where are the routes",
            "where is the api",
            "route files",
            "routing files",
            "backend routes",
            "web routes",
        )
    )


def render_api_routes_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    route_files = api_route_files(files)
    frameworks = api_framework_signals(files, documents)
    commands = run_commands_from_project_overview(payload)

    sections = ["API / Routes"]
    if frameworks:
        sections.append("Detected Routing Stack:\n" + "\n".join(f"- {item}" for item in frameworks))
    else:
        sections.append("Detected Routing Stack:\n- I do not see an obvious routing framework yet.")

    if route_files:
        sections.append(
            "Route Files:\n"
            + "\n".join(
                f"- `{path}` - {description}" for path, description in route_files[:16]
            )
        )
        if len(route_files) > 16:
            sections.append(f"- ...and {len(route_files) - 16} more route files.")
    else:
        sections.append("Route Files:\n- I do not see obvious API route files yet.")

    if commands:
        sections.append(
            "Likely Local Server:\n" + "\n".join(f"- `{command}`" for command in commands[:4])
        )

    evidence = api_route_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the top route file, then trace the handler into services or database calls."
    )
    return "\n".join(sections)


def api_route_files(files: list[str]) -> list[tuple[str, str]]:
    routes = []
    for path in files:
        description = api_route_file_description(path)
        if description:
            routes.append((path, description))
    return sorted(routes, key=lambda item: (api_route_rank(item[0]), item[0]))


def api_route_file_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if lowered.startswith("app/api/") and name in {"route.ts", "route.js", "route.tsx"}:
        return "Next.js App Router API route"
    if lowered.startswith("pages/api/") and lowered.endswith((".ts", ".tsx", ".js", ".jsx")):
        return "Next.js Pages API route"
    if lowered.startswith(("src/routes/", "routes/")) and lowered.endswith(
        (".ts", ".js", ".py", ".rb")
    ):
        return "route module"
    if lowered in {"routes/api.php", "routes/web.php"}:
        return "Laravel route file"
    if lowered.endswith("urls.py") and (lowered == "urls.py" or "/urls.py" in lowered):
        return "Django URL routes"
    if lowered in {"server.ts", "server.js", "src/server.ts", "src/server.js"}:
        return "HTTP server entry"
    if lowered in {"app.py", "main.py", "src/main.py"}:
        return "Python app/server entry"
    if lowered.endswith(("/router.ts", "/router.js", "/routes.ts", "/routes.js")):
        return "router module"
    if "/api/" in lowered and lowered.endswith((".ts", ".tsx", ".js", ".jsx", ".py", ".rb")):
        return "API module"
    return ""


def api_route_rank(path: str) -> int:
    lowered = path.lower()
    order = (
        "app/api/",
        "pages/api/",
        "src/routes/",
        "routes/api.php",
        "routes/web.php",
        "routes/",
        "server.",
        "src/server.",
        "app.py",
        "main.py",
        "urls.py",
    )
    for index, prefix in enumerate(order):
        if lowered.startswith(prefix) or prefix in lowered:
            return index
    return len(order)


def api_framework_signals(files: list[str], documents: list[dict]) -> list[str]:
    package = package_json_from_documents(documents)
    dependencies = package_dependencies(package)
    pyproject = pyproject_from_documents(documents)
    python_dependencies = pyproject_dependencies(pyproject)
    signals = []
    node_frameworks = {
        "next": "Next.js",
        "express": "Express",
        "fastify": "Fastify",
        "hono": "Hono",
        "koa": "Koa",
        "@nestjs/core": "NestJS",
        "@remix-run/node": "Remix",
    }
    for dependency, label in node_frameworks.items():
        if dependency in dependencies:
            signals.append(label)
    python_frameworks = {
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "starlette": "Starlette",
    }
    for dependency, label in python_frameworks.items():
        if dependency in python_dependencies:
            signals.append(label)
    if any(path.startswith("app/api/") for path in files):
        signals.append("Next.js App Router")
    if any(path.startswith("pages/api/") for path in files):
        signals.append("Next.js Pages Router")
    if any(path.startswith(("src/routes/", "routes/")) for path in files):
        signals.append("route modules")
    if "routes/api.php" in files:
        signals.append("Laravel")
    return dedupe_strings(signals)[:8]


def api_route_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    if api_route_files(files):
        evidence.append("Route-like files are present.")
    if api_framework_signals(files, documents):
        evidence.append("Routing framework dependencies or conventions were detected.")
    if package_scripts_from_documents(documents):
        evidence.append("package.json scripts can provide a local server command.")
    return dedupe_strings(evidence)[:5]


def is_commit_readiness_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "ready to commit",
            "can i commit",
            "should i commit",
            "before committing",
            "ready to push",
            "can i push",
            "before pushing",
            "ready for pr",
            "ready for a pr",
            "ready for pull request",
            "pull request ready",
            "open a pr",
            "commit checklist",
            "push checklist",
            "pr checklist",
        )
    )


def render_commit_readiness_answer(payload: dict) -> str:
    git_status = str(payload.get("git_status") or "")
    changes = parse_git_status_lines(git_status)
    commands = verification_commands_from_project_overview(payload)

    sections = ["Commit Readiness"]
    if changes:
        sections.append(
            "Working Tree:\n"
            + "\n".join(f"- {label}: {path}" for label, path in changes[:12])
        )
        if len(changes) > 12:
            sections.append(f"- ...and {len(changes) - 12} more paths.")
    else:
        sections.append("Working Tree:\n- No local changes detected.")

    if commands:
        sections.append(
            "Recommended Checks:\n" + "\n".join(f"- `{command}`" for command in commands)
        )
    else:
        sections.append("Recommended Checks:\n- I do not see an obvious test command yet.")

    sections.append("Assessment:\n" + commit_readiness_assessment(changes, commands))
    sections.append(
        "What I would do next:\n"
        "- Review the diff, run the recommended checks, then commit only the intended files."
    )
    return "\n".join(sections)


def commit_readiness_assessment(
    changes: list[tuple[str, str]],
    commands: list[str],
) -> str:
    if not changes and commands:
        return "- Working tree is clean; rerun checks if you are validating a pushed branch."
    if not changes:
        return "- Working tree is clean, but I cannot verify tests from status alone."
    if any(label == "Untracked" for label, _path in changes):
        return "- Not quite ready: decide whether each untracked file should be committed."
    if commands:
        return "- Close: review the diff and commit after the recommended checks pass."
    return "- Review the diff first; add a project-specific verification command before pushing."


def is_config_files_question(task: str) -> bool:
    lowered = task.lower()
    if is_environment_question(task):
        return False
    return any(
        phrase in lowered
        for phrase in (
            "config files",
            "configuration files",
            "project config",
            "repo config",
            "repository config",
            "where is config",
            "where is project config",
            "where are configs",
            "what config",
            "which config",
        )
    )


def render_config_files_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    configs = project_config_files(files)

    sections = ["Config Files"]
    if configs:
        sections.append(
            "\n".join(f"- `{path}` - {description}" for path, description in configs[:16])
        )
        if len(configs) > 16:
            sections.append(f"- ...and {len(configs) - 16} more config files.")
    else:
        sections.append("- I do not see obvious project config files yet.")

    evidence = config_file_evidence(configs)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    if unsafe_env_files(files):
        sections.append(
            "Safety:\n"
            "- Real `.env` files are present, but I am only listing the file names."
        )
    sections.append(
        "What I would do next:\n"
        "- Open the config file tied to the behavior you want to change."
    )
    return "\n".join(sections)


def project_config_files(files: list[str]) -> list[tuple[str, str]]:
    configs = []
    for path in files:
        description = config_file_description(path)
        if description:
            configs.append((path, description))
    return sorted(configs, key=lambda item: (config_rank(item[0]), item[0]))


def config_file_description(path: str) -> str:
    name = Path(path).name
    lowered = path.lower()
    if lowered in {"package.json", "pnpm-lock.yaml", "yarn.lock", "package-lock.json"}:
        return "Node package/dependency config"
    if lowered in {"pyproject.toml", "setup.py", "requirements.txt"}:
        return "Python package/dependency config"
    if lowered in {"cargo.toml", "cargo.lock"}:
        return "Rust package/workspace config"
    if lowered == "go.mod":
        return "Go module config"
    if lowered in {"tsconfig.json", "jsconfig.json"}:
        return "TypeScript/JavaScript compiler config"
    if name in {"vite.config.ts", "vite.config.js"}:
        return "Vite build/dev-server config"
    if name.startswith("next.config."):
        return "Next.js app config"
    if lowered in {"vercel.json", "wrangler.toml"}:
        return "deployment/runtime config"
    if lowered in {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}:
        return "container config"
    if lowered in {"ruff.toml", "mypy.ini", "pytest.ini"}:
        return "Python tooling config"
    if lowered.startswith(".env"):
        return "environment config file"
    if lowered.startswith(".github/workflows/"):
        return "CI workflow config"
    return ""


def config_rank(path: str) -> int:
    lowered = path.lower()
    order = (
        "package.json",
        "pyproject.toml",
        "cargo.toml",
        "go.mod",
        "tsconfig.json",
        "vite.config",
        "next.config",
        "vercel.json",
        "wrangler.toml",
        ".env",
        "docker",
        ".github/workflows",
    )
    for index, prefix in enumerate(order):
        if lowered.startswith(prefix) or prefix in lowered:
            return index
    return len(order)


def config_file_evidence(configs: list[tuple[str, str]]) -> list[str]:
    categories = dedupe_strings(description for _, description in configs)
    return [f"{category} detected." for category in categories[:6]]


def is_ci_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "ci checks",
            "ci run",
            "ci runs",
            "ci workflow",
            "ci workflows",
            "continuous integration",
            "github actions",
            "actions configured",
            "workflow files",
            "what checks run",
            "checks run for this repo",
            "pull request checks",
            "pr checks",
        )
    )


def render_ci_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    workflows = ci_workflow_files(files)
    workflow_commands = ci_commands_from_documents(documents)
    likely_commands = verification_commands_from_project_overview(payload)
    commands = workflow_commands or likely_commands

    sections = ["CI Checks"]
    if workflows:
        sections.append("Workflow Files:\n" + "\n".join(f"- `{path}`" for path in workflows[:10]))
    else:
        sections.append("Workflow Files:\n- I do not see GitHub Actions workflows yet.")

    if commands:
        sections.append(
            "Likely Checks:\n" + "\n".join(f"- `{command}`" for command in commands[:8])
        )

    evidence = ci_evidence(files, documents)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Open the workflow file, then run the matching local command before pushing."
    )
    return "\n".join(sections)


def ci_workflow_files(files: list[str]) -> list[str]:
    return sorted(
        path
        for path in files
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml"))
    )


def ci_commands_from_documents(documents: list[dict]) -> list[str]:
    commands = []
    for doc in documents:
        path = str(doc.get("path") or "")
        content = doc.get("content")
        if not path.startswith(".github/workflows/") or not isinstance(content, str):
            continue
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("- run:"):
                stripped = stripped[2:].strip()
            if not stripped.startswith("run:"):
                continue
            command = stripped.removeprefix("run:").strip().strip("'\"")
            if command:
                commands.append(command)
    return dedupe_strings(commands)


def ci_evidence(files: list[str], documents: list[dict]) -> list[str]:
    evidence = []
    workflows = ci_workflow_files(files)
    if workflows:
        evidence.append(".github/workflows/ contains CI workflow files.")
    if any(str(doc.get("path") or "").startswith(".github/workflows/") for doc in documents):
        evidence.append("At least one workflow file was read for command hints.")
    if "package.json" in files:
        evidence.append("package.json can provide local equivalents for CI checks.")
    if "Cargo.toml" in files:
        evidence.append("Cargo.toml can provide Rust check/test commands.")
    return dedupe_strings(evidence)[:5]


def is_documentation_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what docs",
            "which docs",
            "docs exist",
            "documentation exists",
            "documentation files",
            "where are docs",
            "where is documentation",
            "read first",
            "start reading",
            "onboarding docs",
            "developer docs",
        )
    )


def render_documentation_answer(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    docs = documentation_files(files)
    read_first = documentation_reading_order(docs)

    sections = ["Documentation"]
    if docs:
        sections.append(
            "\n".join(f"- `{path}` - {description}" for path, description in docs[:16])
        )
        if len(docs) > 16:
            sections.append(f"- ...and {len(docs) - 16} more docs.")
    else:
        sections.append("- I do not see obvious documentation files yet.")

    if read_first:
        sections.append("Read First:\n" + "\n".join(f"- `{path}`" for path in read_first[:6]))

    evidence = documentation_evidence(docs)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Read the first doc, then follow links or referenced config files from there."
    )
    return "\n".join(sections)


def documentation_files(files: list[str]) -> list[tuple[str, str]]:
    docs = []
    for path in files:
        description = documentation_description(path)
        if description:
            docs.append((path, description))
    return sorted(docs, key=lambda item: (documentation_rank(item[0]), item[0]))


def documentation_description(path: str) -> str:
    lowered = path.lower()
    name = Path(path).name.lower()
    if name.startswith("readme."):
        return "project overview"
    if lowered.startswith("docs/") and lowered.endswith((".md", ".rst", ".txt")):
        if "architecture" in lowered or "design" in lowered:
            return "architecture/design notes"
        if "install" in lowered or "setup" in lowered or "getting-started" in lowered:
            return "setup/getting-started docs"
        if "api" in lowered:
            return "API docs"
        if "contributing" in lowered:
            return "contribution docs"
        return "project documentation"
    if name in {"contributing.md", "changelog.md", "license.md"}:
        return "project metadata docs"
    return ""


def documentation_rank(path: str) -> int:
    lowered = path.lower()
    if Path(path).name.lower().startswith("readme."):
        return 0
    if "getting-started" in lowered or "install" in lowered or "setup" in lowered:
        return 1
    if "architecture" in lowered or "design" in lowered:
        return 2
    if "api" in lowered:
        return 3
    if "contributing" in lowered:
        return 4
    return 5


def documentation_reading_order(docs: list[tuple[str, str]]) -> list[str]:
    return [path for path, _ in docs[:8]]


def documentation_evidence(docs: list[tuple[str, str]]) -> list[str]:
    evidence = []
    if any(Path(path).name.lower().startswith("readme.") for path, _ in docs):
        evidence.append("A README is present.")
    if any(path.startswith("docs/") for path, _ in docs):
        evidence.append("docs/ contains project documentation.")
    if any("architecture" in path.lower() or "design" in path.lower() for path, _ in docs):
        evidence.append("Architecture or design docs are present.")
    if any("install" in path.lower() or "setup" in path.lower() for path, _ in docs):
        evidence.append("Setup-oriented docs are present.")
    return dedupe_strings(evidence)[:5]


def is_verification_question(task: str) -> bool:
    lowered = task.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how do i run tests",
            "how to run tests",
            "how should i run tests",
            "what tests should i run",
            "what test command",
            "test command",
            "verification command",
            "how do i verify",
            "how should i verify",
            "how to verify",
        )
    )


def render_verification_answer(payload: dict) -> str:
    commands = verification_commands_from_project_overview(payload)
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    sections = ["Verification Commands"]
    if commands:
        sections.append("\n".join(f"- `{command}`" for command in commands))
    else:
        sections.append("- I do not see a standard test command yet.")

    evidence = verification_evidence(files)
    if evidence:
        sections.append("Why:\n" + "\n".join(f"- {line}" for line in evidence))
    sections.append(
        "What I would do next:\n"
        "- Run the narrowest command first, then broaden if it passes."
    )
    return "\n".join(sections)


def verification_commands_from_project_overview(payload: dict) -> list[str]:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    scripts = package_scripts_from_documents(documents)
    commands = []
    if "test" in scripts:
        if "pnpm-lock.yaml" in files:
            commands.append("pnpm test")
        elif "yarn.lock" in files:
            commands.append("yarn test")
        else:
            commands.append("npm test")
    if "Cargo.toml" in files or any(path.endswith(".rs") for path in files):
        commands.append("cargo test")
    if any(path.startswith("tests/") for path in files):
        commands.append("pytest")
    elif any(Path(path).name.startswith("test_") and path.endswith(".py") for path in files):
        commands.append("python -m unittest")
    elif "pyproject.toml" in files or any(path.endswith(".py") for path in files):
        commands.append("python -m pytest")
    return dedupe_strings(commands)[:3]


def package_scripts_from_documents(documents: list[dict]) -> dict:
    for doc in documents:
        if doc.get("path") != "package.json" or not isinstance(doc.get("content"), str):
            continue
        try:
            payload = json.loads(str(doc["content"]))
        except json.JSONDecodeError:
            return {}
        scripts = payload.get("scripts")
        return scripts if isinstance(scripts, dict) else {}
    return {}


def verification_evidence(files: list[str]) -> list[str]:
    evidence = []
    for marker in ("Cargo.toml", "pyproject.toml", "package.json"):
        if marker in files:
            evidence.append(f"{marker} is present.")
    if any(path.startswith("tests/") for path in files):
        evidence.append("A tests/ directory is present.")
    elif any(Path(path).name.startswith("test_") and path.endswith(".py") for path in files):
        evidence.append("Python unittest-style test files are present.")
    return evidence[:4]


def dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


PROJECT_OVERVIEW_QUESTION_CHECKS = (
    is_debugging_question,
    is_edit_target_question,
    is_entrypoint_question,
    is_verification_question,
    is_container_question,
    is_deployment_question,
    is_database_question,
    is_auth_question,
    is_frontend_question,
    is_api_routes_question,
    is_commit_readiness_question,
    is_run_question,
    is_dependency_question,
    is_stack_question,
    is_test_location_question,
    is_project_structure_question,
    is_setup_question,
    is_project_commands_question,
    is_environment_question,
    is_config_files_question,
    is_ci_question,
    is_documentation_question,
)


def is_grounded_project_overview_question(task: str) -> bool:
    return any(check(task) for check in PROJECT_OVERVIEW_QUESTION_CHECKS)


def render_grounded_project_overview_answer(payload: dict, task: str) -> str | None:
    if is_entrypoint_question(task):
        return render_entrypoint_answer(payload)
    if is_verification_question(task):
        return render_verification_answer(payload)
    if is_container_question(task):
        return render_container_answer(payload)
    if is_deployment_question(task):
        return render_deployment_answer(payload)
    if is_database_question(task):
        return render_database_answer(payload)
    if is_auth_question(task):
        return render_auth_answer(payload)
    if is_frontend_question(task):
        return render_frontend_answer(payload)
    if is_debugging_question(task):
        return render_debugging_answer(payload)
    if is_edit_target_question(task):
        return render_edit_target_answer(payload, task)
    if is_api_routes_question(task):
        return render_api_routes_answer(payload)
    if is_commit_readiness_question(task):
        return render_commit_readiness_answer(payload)
    if is_run_question(task):
        return render_run_answer(payload)
    if is_dependency_question(task):
        return render_dependency_answer(payload)
    if is_stack_question(task):
        return render_stack_answer(payload)
    if is_test_location_question(task):
        return render_test_location_answer(payload)
    if is_project_structure_question(task):
        return render_project_structure_answer(payload)
    if is_setup_question(task):
        return render_setup_answer(payload)
    if is_project_commands_question(task):
        return render_project_commands_answer(payload)
    if is_environment_question(task):
        return render_environment_answer(payload)
    if is_config_files_question(task):
        return render_config_files_answer(payload)
    if is_ci_question(task):
        return render_ci_answer(payload)
    if is_documentation_question(task):
        return render_documentation_answer(payload)
    return None
