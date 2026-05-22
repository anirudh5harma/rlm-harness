import tempfile
import unittest
from pathlib import Path

from rlm_harness.evals.long_horizon import LongHorizonAdapter
from rlm_harness.evals.runner import EvalCase, EvalRunner, EvalSuite, UnitTestGrader
from rlm_harness.evals.swebench import SWEBenchAdapter


class EvalSystemTests(unittest.TestCase):
    def test_unit_test_grader_passes_when_command_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "test_ok.py").write_text(
                (
                    "import unittest\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_ok(self): self.assertTrue(True)\n"
                ),
                encoding="utf-8",
            )
            result = UnitTestGrader("python -m unittest").grade(workspace)
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertIn("OK", result.output)

    def test_eval_runner_records_case_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = EvalSuite(
                name="smoke",
                cases=[
                    EvalCase(
                        id="pass-python-tests",
                        prompt="run tests",
                        workspace=root / "case",
                        setup_commands=[
                            (
                                "cat > test_ok.py <<'PY'\n"
                                "import unittest\n"
                                "class T(unittest.TestCase):\n"
                                "    def test_ok(self): self.assertEqual(2+2, 4)\n"
                                "PY"
                            )
                        ],
                        grader=UnitTestGrader("python -m unittest"),
                    )
                ],
            )
            runner = EvalRunner(harness_command=["python", "-c", "print('noop harness')"])
            report = runner.run(suite)
        self.assertEqual(report.suite, "smoke")
        self.assertEqual(len(report.results), 1)
        self.assertTrue(report.results[0].passed)

    def test_swebench_adapter_builds_case_from_manifest(self):
        payload = {
            "instance_id": "repo__issue-1",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "test_command": "python -m pytest tests/test_bug.py",
        }
        case = SWEBenchAdapter().case_from_record(payload, Path("/tmp/work"))
        self.assertEqual(case.id, "repo__issue-1")
        self.assertIn("Fix the bug", case.prompt)
        self.assertIn("owner/repo", case.prompt)
        self.assertIsInstance(case.grader, UnitTestGrader)

    def test_long_horizon_adapter_loads_yaml_suite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_path = Path(temp_dir) / "suite.yaml"
            suite_path.write_text(
                "name: long\n"
                "cases:\n"
                "  - id: multi-step\n"
                "    prompt: Fix bug and document it\n"
                "    test_command: python -m unittest\n"
                "    files:\n"
                "      app.py: 'def add(a,b): return a-b\\n'\n",
                encoding="utf-8",
            )
            suite = LongHorizonAdapter().load_suite(suite_path, Path(temp_dir) / "work")
        self.assertEqual(suite.name, "long")
        self.assertEqual(suite.cases[0].id, "multi-step")
        self.assertEqual(suite.cases[0].files["app.py"], "def add(a,b): return a-b\n")


if __name__ == "__main__":
    unittest.main()
