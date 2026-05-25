from __future__ import annotations

import re
from pathlib import Path


def is_informational_task(task: str) -> bool:
    if is_project_summary_task(task) or is_project_audit_task(task):
        return True
    terms = (
        "summarize",
        "summary",
        "explain",
        "describe",
        "list",
        "report",
        "inspect",
        "analyze",
        "analyse",
        "find",
        "identify",
        "audit",
        "review",
        "evaluate",
        "assess",
    )
    lowered = task.lower()
    return any(term in lowered for term in terms)


def is_project_summary_task(task: str) -> bool:
    lowered = task.lower()
    has_project_subject = bool(
        re.search(r"\b(project|repo|repository|codebase|workspace|application|app)\b", lowered)
    )
    has_summary_intent = any(
        term in lowered
        for term in (
            "what is",
            "what's",
            "tell me about",
            "summarize",
            "summary",
            "overview",
            "explain",
            "describe",
        )
    )
    return has_project_subject and has_summary_intent


def is_project_audit_task(task: str) -> bool:
    lowered = task.lower()
    has_project_subject = bool(
        re.search(r"\b(project|repo|repository|codebase|workspace|application|app)\b", lowered)
    )
    has_audit_intent = any(
        term in lowered
        for term in (
            "gap",
            "gaps",
            "risk",
            "risks",
            "issue",
            "issues",
            "problem",
            "problems",
            "bug",
            "bugs",
            "flaw",
            "flaws",
            "weakness",
            "weaknesses",
            "technical debt",
            "logical",
            "audit",
            "review",
            "critique",
            "evaluate",
            "assess",
            "find any",
            "identify",
        )
    )
    return has_project_subject and has_audit_intent


def is_code_editing_task(task: str) -> bool:
    lowered = task.lower()
    terms = (
        "fix",
        "implement",
        "change",
        "modify",
        "update",
        "add",
        "remove",
        "delete",
        "refactor",
        "rewrite",
        "create",
        "edit",
        "patch",
        "make",
    )
    code_subjects = (
        "code",
        "test",
        "tests",
        "bug",
        "file",
        "function",
        "class",
        "module",
        "cli",
        "api",
        "runtime",
        "harness",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".json",
        ".toml",
        ".md",
    )
    return any(term in lowered for term in terms) and any(
        subject in lowered for subject in code_subjects
    )


def looks_like_code_edit_result(output: str) -> bool:
    lowered = output.lower()
    evidence_terms = (
        "changed",
        "modified",
        "updated",
        "wrote ",
        "created",
        "deleted",
        "patch applied",
        "diff",
        "verification",
        "verified",
        "test",
        "tests",
        "pytest",
        "unittest",
        "ruff",
        "passed",
        "failed",
        " ok",
        "\nok",
    )
    has_path = bool(
        re.search(r"\b[\w./-]+\.(py|ts|tsx|js|jsx|json|toml|md|css|yml|yaml)\b", output)
    )
    return has_path or any(term in lowered for term in evidence_terms)


def looks_like_source_dump(output: str) -> bool:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    if output.count("```") >= 2:
        return True
    source_markers = (
        "from __future__ import ",
        "def ",
        "class ",
        "import ",
        "return ",
        "if __name__ == ",
        "function ",
        "const ",
        "export ",
    )
    marker_hits = sum(
        1
        for line in lines
        if line.lstrip().startswith(source_markers)
        or re.match(r"^\s{2,}(if|for|while|return|try|except|with)\b", line)
    )
    return marker_hits >= 5 and marker_hits / len(lines) >= 0.25


def looks_like_file_inventory(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    if any(line.upper() in {"ALL FILES:", "FILES:"} for line in lines[:3]):
        return True
    path_like = 0
    for line in lines:
        if len(line) > 180 or " " in line or "\t" in line:
            continue
        if re.match(r"^[A-Za-z0-9_./@:+-]+$", line) and (
            "/" in line or "." in Path(line).name
        ):
            path_like += 1
    return path_like >= 8 and path_like / len(lines) >= 0.75


def looks_like_project_summary(output: str) -> bool:
    lowered = output.lower()
    return (
        "project summary" in lowered
        or "what it is:" in lowered
        or ("tech stack:" in lowered and "files inspected:" in lowered)
    )


def looks_like_project_audit(output: str) -> bool:
    lowered = output.lower()
    has_audit_language = any(
        term in lowered
        for term in (
            "finding",
            "findings",
            "gap",
            "gaps",
            "risk",
            "risks",
            "issue",
            "issues",
            "impact:",
            "recommendation:",
        )
    )
    has_evidence = "evidence:" in lowered or bool(
        re.search(r"\b[\w./-]+\.(py|ts|tsx|js|jsx|json|toml|md|css|yml|yaml)\b", output)
    )
    return has_audit_language and has_evidence and not looks_like_file_inventory(output)
