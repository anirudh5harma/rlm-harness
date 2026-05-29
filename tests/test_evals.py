import tempfile
import unittest
from pathlib import Path

from rlm_harness import cli
from rlm_harness.evals.runner import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
    EvalSuite,
    UnitTestGrader,
    grade_output_expectations,
)
from rlm_harness.evals.suite import EvalSuiteFileLoader, read_suite_text
from rlm_harness.memory import Memory
from rlm_harness.memory.evolution import EvolutionProposalStore
from rlm_harness.memory.profile import TasteProfileManager, TasteProfileStore


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

    def test_eval_suite_file_loader_loads_yaml_suite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_path = Path(temp_dir) / "suite.yaml"
            suite_path.write_text(
                "name: smoke\n"
                "cases:\n"
                "  - id: local-case\n"
                "    prompt: Fix bug\n"
                "    test_command: python -m unittest\n"
                "    files:\n"
                "      app.py: 'def add(a,b): return a-b\\n'\n",
                encoding="utf-8",
            )
            suite = EvalSuiteFileLoader().load_suite(suite_path, Path(temp_dir) / "work")
        self.assertEqual(suite.name, "smoke")
        self.assertEqual(suite.cases[0].id, "local-case")
        self.assertEqual(suite.cases[0].files["app.py"], "def add(a,b): return a-b\n")
        self.assertEqual(suite.cases[0].metadata["eval_type"], "suite")

    def test_eval_runner_seeds_taste_and_grades_output_expectations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = EvalSuite(
                name="taste",
                cases=[
                    EvalCase(
                        id="concise-style",
                        prompt="summarize",
                        workspace=root / "case",
                        grader=UnitTestGrader("python -c 'print(\"ok\")'"),
                        taste_records=[
                            {
                                "scope": "user",
                                "kind": "preference",
                                "text": "Prefer compact summaries.",
                            }
                        ],
                        evolution_proposals=[
                            {
                                "scope": "user",
                                "kind": "prompt_rule",
                                "title": "Compact summaries",
                                "body": "Prefer compact summaries.",
                                "rationale": "Seeded by taste regression.",
                                "status": "approved",
                            }
                        ],
                        output_contains=["compact"],
                    )
                ],
            )
            runner = EvalRunner(
                harness_command=["python", "-c", "print('compact response')"],
            )
            report = runner.run(suite)

            profile_db = root / "case" / ".rlm_harness" / "profile.db"
            with Memory(profile_db) as memory:
                taste_records = TasteProfileStore(memory).records()
                proposals = EvolutionProposalStore(memory).proposals()

        self.assertTrue(report.results[0].passed)
        self.assertEqual(len(taste_records), 1)
        self.assertEqual(len(proposals), 1)
        self.assertIn("compact summaries", taste_records[0].text)

    def test_eval_runner_routes_project_taste_and_explicit_harness_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = EvalSuite(
                name="cli",
                cases=[
                    EvalCase(
                        id="taste-context",
                        prompt="show taste context",
                        workspace=root / "case",
                        harness_args=[
                            "taste",
                            "--profile-db",
                            "{profile_db}",
                            "--memory-db",
                            "{memory_db}",
                            "context",
                        ],
                        grader=UnitTestGrader("python -c 'print(\"ok\")'"),
                        taste_records=[
                            {
                                "scope": "user",
                                "kind": "preference",
                                "text": "Prefer direct answers.",
                            },
                            {
                                "scope": "project",
                                "kind": "verification_command",
                                "text": "Run `pytest` for verification.",
                            },
                        ],
                        output_contains=["User taste", "Project conventions", "pytest"],
                    )
                ],
            )
            runner = EvalRunner(
                harness_command=[
                    "python",
                    "-c",
                    (
                        "import sys; "
                        "from rlm_harness.cli import main; "
                        "raise SystemExit(main())"
                    ),
                    "run",
                    "--provider",
                    "stub",
                    "--model",
                    "stub",
                ],
            )
            report = runner.run(suite)

            profile_db = root / "case" / ".rlm_harness" / "profile.db"
            memory_db = root / "case" / ".rlm_harness" / "memory.db"
            with Memory(profile_db) as profile_memory, Memory(memory_db) as project_memory:
                context = TasteProfileManager(profile_memory, project_memory).render_context()

        self.assertTrue(report.results[0].passed, report.results[0].output)
        self.assertIn("User taste", report.results[0].harness_stdout)
        self.assertIn("Project conventions", report.results[0].harness_stdout)
        self.assertIn("Run `pytest` for verification.", context)

    def test_eval_suite_file_loader_loads_taste_regression_fields_from_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_path = Path(temp_dir) / "suite.json"
            suite_path.write_text(
                (
                    '{"name":"taste","cases":[{"id":"style","prompt":"Summarize",'
                    '"test_command":"python -c \\"print(1)\\"",'
                    '"taste_records":[{"scope":"user","kind":"preference",'
                    '"text":"Prefer direct answers."}],'
                    '"harness_args":["taste","context"],'
                    '"evolution_proposals":[{"scope":"user","kind":"prompt_rule",'
                    '"title":"Direct","body":"Prefer direct answers.",'
                    '"rationale":"Seeded by eval.","status":"approved"}],'
                    '"output_contains":["direct"],'
                    '"output_not_contains":["verbose"]}]}'
                ),
                encoding="utf-8",
            )
            suite = EvalSuiteFileLoader().load_suite(suite_path, Path(temp_dir) / "work")

        case = suite.cases[0]
        self.assertEqual(case.taste_records[0]["text"], "Prefer direct answers.")
        self.assertEqual(case.harness_args, ["taste", "context"])
        self.assertEqual(case.evolution_proposals[0]["title"], "Direct")
        self.assertEqual(case.output_contains, ["direct"])
        self.assertEqual(case.output_not_contains, ["verbose"])

    def test_eval_suite_file_loader_loads_built_in_daily_driver_suite(self):
        suite = EvalSuiteFileLoader().load_suite("daily-driver", Path("/tmp/work"))

        self.assertEqual(suite.name, "daily-driver")
        self.assertGreaterEqual(len(suite.cases), 3)
        self.assertIn("fix-python-unittest", {case.id for case in suite.cases})
        self.assertIn("slash-palette-cli", {case.id for case in suite.cases})
        self.assertIn('"name": "daily-driver"', read_suite_text("daily-driver"))

    def test_output_expectations_are_case_insensitive(self):
        case = EvalCase(
            id="style",
            prompt="summarize",
            workspace=Path("/tmp/work"),
            grader=UnitTestGrader("python -c 'print(1)'"),
            output_contains=["Verification"],
            output_not_contains=["__RLM_FINAL_ANSWER__"],
        )

        grade = grade_output_expectations(case, "verification: ok", "")

        self.assertTrue(grade.passed)

    def test_eval_runner_fails_case_when_harness_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = EvalSuite(
                name="harness-error",
                cases=[
                    EvalCase(
                        id="bad-run",
                        prompt="run",
                        workspace=Path(temp_dir) / "case",
                        grader=UnitTestGrader("python -c 'print(\"ok\")'"),
                    )
                ],
            )
            runner = EvalRunner(
                harness_command=["python", "-c", "import sys; sys.exit(7)"],
            )
            report = runner.run(suite)

        self.assertFalse(report.results[0].passed)
        self.assertEqual(report.results[0].status, "harness_error")
        self.assertIn("harness exited with 7", report.results[0].output)

    def test_eval_failure_proposals_are_recorded_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.db"
            report = EvalReport(
                run_id="run",
                suite="taste-regression",
                results=[
                    EvalResult(
                        case_id="style",
                        passed=False,
                        score=0.0,
                        status="harness_error",
                        latency_ms=1,
                        output="missing expected output",
                        harness_stdout="",
                        harness_stderr="",
                        workspace=temp_dir,
                        metadata={"prompt": "Summarize with learned style."},
                    )
                ],
            )

            recorded = cli.record_eval_failure_proposals(report, memory_path)
            with Memory(memory_path) as memory:
                proposals = EvolutionProposalStore(memory).proposals()

        self.assertEqual(recorded, 1)
        self.assertEqual(proposals[0].kind, "eval_case")
        self.assertIn("taste-regression/style", proposals[0].body)


if __name__ == "__main__":
    unittest.main()
