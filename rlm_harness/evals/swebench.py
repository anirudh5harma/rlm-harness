from __future__ import annotations

from pathlib import Path

from rlm_harness.evals.runner import EvalCase, UnitTestGrader


class SWEBenchAdapter:
    """Adapter for SWE-bench-style manifest records.

    This intentionally does not vendor or download SWE-bench. It converts already
    acquired records into harness eval cases. A higher-level runner can clone the
    repo/base commit in setup commands or pre-materialize the workspace.
    """

    def case_from_record(self, record: dict, work_root: Path) -> EvalCase:
        instance_id = str(record.get("instance_id") or record.get("id") or "swebench-case")
        repo = str(record.get("repo") or "")
        base_commit = str(record.get("base_commit") or "")
        problem = str(record.get("problem_statement") or record.get("prompt") or "")
        test_command = str(
            record.get("test_command") or record.get("test_cmd") or "python -m pytest"
        )
        prompt = (
            "Resolve this SWE-bench task. Modify the checked-out repository, "
            "run relevant tests, and leave the final patch in the workspace.\n\n"
            f"Repository: {repo}\n"
            f"Base commit: {base_commit}\n\n"
            f"Issue:\n{problem}\n\n"
            f"Validation command: {test_command}"
        )
        setup_commands = []
        clone_url = record.get("clone_url")
        if clone_url:
            setup_commands.append(f"git clone {clone_url} .")
            if base_commit:
                setup_commands.append(f"git checkout {base_commit}")
        return EvalCase(
            id=instance_id,
            prompt=prompt,
            workspace=work_root / instance_id,
            setup_commands=setup_commands,
            grader=UnitTestGrader(test_command),
            metadata={"benchmark": "swe-bench", "repo": repo, "base_commit": base_commit},
        )
