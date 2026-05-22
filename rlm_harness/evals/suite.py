from __future__ import annotations

import ast
from pathlib import Path

from rlm_harness.evals.runner import EvalCase, EvalSuite, UnitTestGrader


class EvalSuiteFileLoader:
    """Load local deterministic Harness eval suites from JSON or simple YAML."""

    def load_suite(self, path: Path, work_root: Path) -> EvalSuite:
        text = path.read_text(encoding="utf-8")
        data = parse_simple_suite(text)
        cases = []
        for raw in data.get("cases", []):
            case_id = str(raw["id"])
            prompt = str(raw["prompt"])
            cases.append(
                EvalCase(
                    id=case_id,
                    prompt=prompt,
                    workspace=work_root / case_id,
                    files={str(k): str(v) for k, v in raw.get("files", {}).items()},
                    setup_commands=[str(cmd) for cmd in raw.get("setup_commands", [])],
                    grader=UnitTestGrader(str(raw.get("test_command", "python -m unittest"))),
                    metadata={"eval_type": "suite", "prompt": prompt},
                )
            )
        return EvalSuite(name=str(data.get("name", path.stem)), cases=cases)


def parse_simple_suite(text: str) -> dict:
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


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
