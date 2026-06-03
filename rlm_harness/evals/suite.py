from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path
from typing import Any

from rlm_harness.evals.runner import EvalCase, EvalSuite, UnitTestGrader

BUILTIN_SUITES = {
    "daily-driver",
    "taste-regression",
    # Phase G
    "long-horizon",
    "long-context",
}


class EvalSuiteFileLoader:
    """Load local deterministic Harness eval suites from JSON or simple YAML."""

    def load_suite(self, path: Path | str, work_root: Path) -> EvalSuite:
        text = read_suite_text(path)
        data = parse_simple_suite(text)
        cases = []
        for raw in data.get("cases", []):
            case_id = str(raw["id"])
            prompt = str(raw["prompt"])
            metadata = {
                "eval_type": "suite",
                "prompt": prompt,
                **{
                    str(k): v
                    for k, v in raw.get("metadata", {}).items()
                },
            }
            cases.append(
                EvalCase(
                    id=case_id,
                    prompt=prompt,
                    workspace=work_root / case_id,
                    harness_args=[str(arg) for arg in raw.get("harness_args", [])],
                    files={str(k): str(v) for k, v in raw.get("files", {}).items()},
                    setup_commands=[str(cmd) for cmd in raw.get("setup_commands", [])],
                    taste_records=[dict(item) for item in raw.get("taste_records", [])],
                    evolution_proposals=[
                        dict(item) for item in raw.get("evolution_proposals", [])
                    ],
                    output_contains=[str(item) for item in raw.get("output_contains", [])],
                    output_not_contains=[
                        str(item) for item in raw.get("output_not_contains", [])
                    ],
                    grader=UnitTestGrader(str(raw.get("test_command", "python -m unittest"))),
                    metadata=metadata,
                )
            )
        fallback_name = normalize_builtin_suite_name(str(path))
        return EvalSuite(name=str(data.get("name", fallback_name)), cases=cases)


def read_suite_text(path: Path | str) -> str:
    path_or_name = str(path)
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    suite_name = normalize_builtin_suite_name(path_or_name)
    if suite_name in BUILTIN_SUITES:
        return (
            resources.files("rlm_harness.evals.suites")
            .joinpath(f"{suite_name}.json")
            .read_text(encoding="utf-8")
        )

    raise FileNotFoundError(
        f"eval suite not found: {path_or_name}. "
        f"Built-in suites: {', '.join(sorted(BUILTIN_SUITES))}"
    )


def load_suite(path: Path | str, work_root: Path | None = None) -> EvalSuite:
    """Convenience wrapper around `EvalSuiteFileLoader().load_suite`.

    If `work_root` is None, a fresh temp directory is used for
    each case's `workspace`. Pass an explicit `work_root` to
    reuse a directory across cases.
    """
    import tempfile

    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="rlm-eval-"))
    return EvalSuiteFileLoader().load_suite(path, work_root)


def normalize_builtin_suite_name(value: str) -> str:
    name = Path(value).name
    for suffix in (".json", ".yaml", ".yml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip().lower()


def parse_simple_suite(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        import json

        return json.loads(stripped)

    result: dict = {"cases": []}
    current_case: dict | None = None
    in_files = False
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        line = raw_line.rstrip("\n")
        stripped_line = line.strip()
        if stripped_line.startswith("name:"):
            result["name"] = stripped_line.split(":", 1)[1].strip()
            continue
        if stripped_line == "cases:":
            continue
        if stripped_line.startswith("- id:"):
            current_case = {"id": stripped_line.split(":", 1)[1].strip(), "files": {}}
            result["cases"].append(current_case)
            in_files = False
            continue
        if current_case is None:
            continue
        if stripped_line == "files:":
            in_files = True
            continue
        if in_files and ":" in stripped_line:
            key, value = stripped_line.split(":", 1)
            current_case.setdefault("files", {})[key.strip()] = parse_scalar(value.strip())
            continue
        if ":" in stripped_line:
            key, value = stripped_line.split(":", 1)
            current_case[key.strip()] = parse_scalar(value.strip())
            in_files = False
    return result


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
