from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import tomllib

from rlm_harness.memory.profile import ACTIVE, TasteRecord

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}


def scan_project_style(workspace: Path, *, max_files: int = 400) -> list[TasteRecord]:
    root = workspace.resolve()
    records: list[TasteRecord] = []
    records.extend(scan_editorconfig(root))
    records.extend(scan_pyproject(root))
    records.extend(scan_package_json(root))
    records.extend(scan_prettier(root))
    records.extend(scan_source_format(root, max_files=max_files))
    return dedupe_records(records)


def scan_editorconfig(root: Path) -> list[TasteRecord]:
    editorconfig = root / ".editorconfig"
    if not editorconfig.exists():
        return []
    settings = parse_editorconfig(editorconfig)
    records: list[TasteRecord] = []
    indent_style = settings.get("indent_style", "").lower()
    indent_size = settings.get("indent_size", "")
    if indent_style == "space" and indent_size.isdigit():
        records.append(
            style_record(
                f"Use {indent_size}-space indentation where .editorconfig applies.",
                evidence={"file": ".editorconfig", "source": "indent_size"},
                confidence=0.9,
            )
        )
    max_line_length = settings.get("max_line_length", "")
    if max_line_length.isdigit():
        records.append(
            style_record(
                f"Keep line length at {max_line_length} characters where .editorconfig applies.",
                evidence={"file": ".editorconfig", "source": "max_line_length"},
                confidence=0.86,
            )
        )
    end_of_line = settings.get("end_of_line", "").lower()
    if end_of_line in {"lf", "crlf"}:
        records.append(
            style_record(
                f"Use {end_of_line.upper()} line endings where .editorconfig applies.",
                evidence={"file": ".editorconfig", "source": "end_of_line"},
                confidence=0.82,
            )
        )
    return records


def scan_pyproject(root: Path) -> list[TasteRecord]:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    records: list[TasteRecord] = []
    tool = data.get("tool") if isinstance(data, dict) else {}
    tool = tool if isinstance(tool, dict) else {}
    project = data.get("project") if isinstance(data, dict) else {}
    project = project if isinstance(project, dict) else {}
    ruff = tool.get("ruff") if isinstance(tool.get("ruff"), dict) else {}
    pytest_config = tool.get("pytest") if isinstance(tool.get("pytest"), dict) else {}

    line_length = ruff.get("line-length")
    if isinstance(line_length, int):
        records.append(
            style_record(
                f"Keep Python line length at {line_length} characters.",
                evidence={"file": "pyproject.toml", "source": "tool.ruff.line-length"},
                confidence=0.88,
            )
        )
    if "ruff" in dependency_names(project):
        records.append(
            convention_record(
                "Use Ruff for Python linting and formatting checks.",
                evidence={"file": "pyproject.toml", "source": "project.dependencies"},
                confidence=0.82,
            )
        )
    if pytest_config or "pytest" in dependency_names(project):
        records.append(
            verification_record(
                "Run `pytest` for Python test verification.",
                evidence={"file": "pyproject.toml", "source": "pytest configuration"},
                confidence=0.86,
            )
        )
    return records


def scan_package_json(root: Path) -> list[TasteRecord]:
    package_json = root / "package.json"
    if not package_json.exists():
        return []
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scripts = data.get("scripts") if isinstance(data, dict) else {}
    if not isinstance(scripts, dict):
        scripts = {}

    records: list[TasteRecord] = []
    package_manager = package_manager_name(data)
    if package_manager:
        records.append(
            convention_record(
                f"Use {package_manager} for JavaScript package commands.",
                evidence={"file": "package.json", "source": "packageManager"},
                confidence=0.84,
            )
        )
    for name, purpose in (
        ("test", "test verification"),
        ("lint", "lint verification"),
        ("typecheck", "type checking"),
        ("build", "build verification"),
    ):
        if name in scripts:
            records.append(
                verification_record(
                    f"Run `{package_script_command(name, package_manager)}` for {purpose}.",
                    evidence={"file": "package.json", "script": name},
                    confidence=0.84,
                )
            )
    prettier = data.get("prettier") if isinstance(data, dict) else None
    if isinstance(prettier, dict):
        records.extend(prettier_records(prettier, "package.json", "prettier"))
    return records


def scan_prettier(root: Path) -> list[TasteRecord]:
    for name in (".prettierrc", ".prettierrc.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [
                convention_record(
                    "Use Prettier for JavaScript and TypeScript formatting.",
                    evidence={"file": name, "source": "config file"},
                    confidence=0.78,
                )
            ]
        if isinstance(data, dict):
            return prettier_records(data, name, "config file")
    for name in ("prettier.config.js", "prettier.config.cjs", "prettier.config.mjs"):
        if (root / name).exists():
            return [
                convention_record(
                    "Use Prettier for JavaScript and TypeScript formatting.",
                    evidence={"file": name, "source": "config file"},
                    confidence=0.78,
                )
            ]
    return []


def scan_source_format(root: Path, *, max_files: int) -> list[TasteRecord]:
    records: list[TasteRecord] = []
    files = list(source_files(root, max_files=max_files))
    if not files:
        return records

    python_files = [path for path in files if path.suffix == ".py"]
    if python_files:
        indent = common_python_indent(python_files)
        if indent:
            records.append(
                style_record(
                    f"Use {indent}-space indentation for Python files.",
                    evidence={"sampled_files": relative_paths(root, python_files[:20])},
                    confidence=0.78,
                )
            )
        quotes = common_python_quote_style(python_files)
        if quotes:
            records.append(
                style_record(
                    f"Prefer {quotes} quotes in Python code when either quote works.",
                    evidence={"sampled_files": relative_paths(root, python_files[:20])},
                    confidence=0.66,
                )
            )
    js_files = [path for path in files if path.suffix in {".js", ".jsx", ".ts", ".tsx"}]
    if js_files:
        quotes = common_js_quote_style(js_files)
        if quotes:
            records.append(
                style_record(
                    f"Prefer {quotes} quotes in JavaScript and TypeScript when either quote works.",
                    evidence={"sampled_files": relative_paths(root, js_files[:20])},
                    confidence=0.66,
                )
            )
    return records


def source_files(root: Path, *, max_files: int):
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            continue
        count += 1
        yield path


def common_python_indent(paths: list[Path]) -> int | None:
    counts: Counter[int] = Counter()
    for path in paths[:80]:
        for line in safe_lines(path):
            if not line.startswith(" "):
                continue
            stripped = line.lstrip(" ")
            if not stripped or stripped.startswith("#"):
                continue
            leading = len(line) - len(stripped)
            if leading in {2, 4}:
                counts[leading] += 1
    if not counts:
        return None
    indent, count = counts.most_common(1)[0]
    return indent if count >= 3 else None


def common_python_quote_style(paths: list[Path]) -> str | None:
    counts = Counter({"double": 0, "single": 0})
    string_re = re.compile(r"(?<![A-Za-z0-9_])([\"'])(?:\\.|[^\\])*?\1")
    for path in paths[:80]:
        for line in safe_lines(path):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for match in string_re.finditer(line):
                counts["double" if match.group(1) == '"' else "single"] += 1
    total = counts["double"] + counts["single"]
    if total < 12:
        return None
    winner, count = counts.most_common(1)[0]
    return winner if count / total >= 0.62 else None


def common_js_quote_style(paths: list[Path]) -> str | None:
    counts = Counter({"double": 0, "single": 0})
    string_re = re.compile(r"(?<![A-Za-z0-9_])([\"'])(?:\\.|[^\\])*?\1")
    for path in paths[:80]:
        for line in safe_lines(path):
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", "*")):
                continue
            for match in string_re.finditer(line):
                counts["double" if match.group(1) == '"' else "single"] += 1
    total = counts["double"] + counts["single"]
    if total < 12:
        return None
    winner, count = counts.most_common(1)[0]
    return winner if count / total >= 0.62 else None


def safe_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    except OSError:
        return []


def dependency_names(project: dict) -> set[str]:
    deps = []
    raw_dependencies = project.get("dependencies")
    if isinstance(raw_dependencies, list):
        deps.extend(str(item) for item in raw_dependencies)
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                deps.extend(str(item) for item in values)
    names = set()
    for dep in deps:
        match = re.match(r"([A-Za-z0-9_.-]+)", dep)
        if match:
            names.add(match.group(1).lower())
    return names


def parse_editorconfig(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    for line in safe_lines(path):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or stripped.startswith("["):
            continue
        key, separator, value = stripped.partition("=")
        if separator != "=":
            continue
        settings[key.strip().lower()] = value.strip()
    return settings


def package_manager_name(data: dict) -> str:
    raw = data.get("packageManager")
    if not isinstance(raw, str) or not raw.strip():
        dev_dependencies = data.get("devDependencies")
        if isinstance(dev_dependencies, dict) and dev_dependencies.get("pnpm"):
            return "pnpm"
        return "npm"
    return raw.split("@", 1)[0].strip() or "npm"


def package_script_command(script: str, package_manager: str) -> str:
    if package_manager == "yarn":
        return f"yarn {script}"
    if package_manager == "pnpm":
        return f"pnpm {script}"
    return f"npm run {script}"


def prettier_records(data: dict, file_name: str, source: str) -> list[TasteRecord]:
    records = [
        convention_record(
            "Use Prettier for JavaScript and TypeScript formatting.",
            evidence={"file": file_name, "source": source},
            confidence=0.8,
        )
    ]
    print_width = data.get("printWidth")
    if isinstance(print_width, int):
        records.append(
            style_record(
                f"Keep JavaScript and TypeScript line length at {print_width} characters.",
                evidence={"file": file_name, "source": "printWidth"},
                confidence=0.86,
            )
        )
    single_quote = data.get("singleQuote")
    if isinstance(single_quote, bool):
        quote = "single" if single_quote else "double"
        records.append(
            style_record(
                f"Prefer {quote} quotes in JavaScript and TypeScript when either quote works.",
                evidence={"file": file_name, "source": "singleQuote"},
                confidence=0.84,
            )
        )
    semi = data.get("semi")
    if isinstance(semi, bool):
        text = "Use semicolons in JavaScript and TypeScript." if semi else (
            "Omit semicolons in JavaScript and TypeScript where optional."
        )
        records.append(
            style_record(
                text,
                evidence={"file": file_name, "source": "semi"},
                confidence=0.82,
            )
        )
    trailing_comma = data.get("trailingComma")
    if isinstance(trailing_comma, str):
        records.append(
            style_record(
                f"Use Prettier trailingComma={trailing_comma}.",
                evidence={"file": file_name, "source": "trailingComma"},
                confidence=0.76,
            )
        )
    return records


def style_record(text: str, *, evidence: dict, confidence: float) -> TasteRecord:
    return TasteRecord.create(
        scope="project",
        kind="style",
        text=text,
        confidence=confidence,
        status=ACTIVE,
        evidence={"source": "taste scan", **evidence},
    )


def convention_record(text: str, *, evidence: dict, confidence: float) -> TasteRecord:
    return TasteRecord.create(
        scope="project",
        kind="convention",
        text=text,
        confidence=confidence,
        status=ACTIVE,
        evidence={"source": "taste scan", **evidence},
    )


def verification_record(text: str, *, evidence: dict, confidence: float) -> TasteRecord:
    return TasteRecord.create(
        scope="project",
        kind="verification_command",
        text=text,
        confidence=confidence,
        status=ACTIVE,
        evidence={"source": "taste scan", **evidence},
    )


def relative_paths(root: Path, paths: list[Path]) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]


def dedupe_records(records: list[TasteRecord]) -> list[TasteRecord]:
    result: list[TasteRecord] = []
    seen = set()
    for record in records:
        if record.dedupe_key in seen:
            continue
        seen.add(record.dedupe_key)
        result.append(record)
    return result
