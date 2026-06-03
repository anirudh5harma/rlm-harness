"""Tests for the long-horizon + long-context eval suites (Phase G).

The pivot plan's Phase G gate:

    "Harness beats a hand-rolled mini-agent baseline on the new
     suites while not regressing on the daily-driver suite."

These tests assert the suite format and the grader. The actual
end-to-end evaluation against a real model is gated by
`HARNESS_EVAL_REAL=1`; the offline tests use the stub provider.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.evals.runner import (
    GradeResult,
    RLMContextEfficiencyGrader,
    combine_grades,
)
from rlm_harness.evals.suite import (
    load_suite,
)


class LoadSuiteTests(unittest.TestCase):
    def test_load_long_horizon_suite(self):
        path = Path("rlm_harness/evals/suites/long-horizon.json")
        suite = load_suite(path)
        self.assertEqual(suite.name, "long-horizon")
        self.assertGreater(len(suite.cases), 0)
        for case in suite.cases:
            self.assertTrue(case.prompt)
            self.assertTrue(case.grader is not None)

    def test_load_long_context_suite(self):
        path = Path("rlm_harness/evals/suites/long-context.json")
        suite = load_suite(path)
        self.assertEqual(suite.name, "long-context")
        self.assertGreater(len(suite.cases), 0)
        for case in suite.cases:
            self.assertTrue(case.prompt)
            # Long-context cases ship a manifest budget.
            self.assertIn("manifest_budget_tokens", case.metadata)


class CombineGradesTests(unittest.TestCase):
    def test_combine_all_passed(self):
        combined = combine_grades(
            GradeResult(passed=True, score=1.0, output="ok"),
            GradeResult(passed=True, score=1.0, output="ok"),
            GradeResult(passed=True, score=1.0, output="ok"),
        )
        self.assertTrue(combined.passed)
        self.assertEqual(combined.score, 1.0)

    def test_combine_one_fails(self):
        combined = combine_grades(
            GradeResult(passed=True, score=1.0, output="ok"),
            GradeResult(passed=False, score=0.0, output="boom"),
            GradeResult(passed=True, score=1.0, output="ok"),
        )
        self.assertFalse(combined.passed)
        self.assertEqual(combined.score, 0.0)


class RLMContextEfficiencyGraderTests(unittest.TestCase):
    def test_grader_passes_when_under_budget(self):
        """A run that used < N turns and stayed under the
        manifest budget is graded as `passed`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            # Write a tiny trace db that the grader can read.
            from rlm_harness.tracing import TraceStore

            store = TraceStore(workspace / "trace.db")
            run_id = store.start_run("x", str(workspace), thread_id="t")
            store.event(
                run_id,
                "turn_started",
                {"turn_index": 0, "context_preview": "small"},
                node="supervisor",
            )
            store.event(
                run_id,
                "turn_finished",
                {
                    "turn_index": 0,
                    "status": "done",
                    "iterations": 1,
                    "subcalls": 0,
                    "tokens_used": 100,
                },
                node="supervisor",
            )
            store.finish_run(run_id, "done")

            grader = RLMContextEfficiencyGrader(
                max_turns=5, max_manifest_tokens=20_000
            )
            result = grader.grade(workspace, run_id=run_id)
            self.assertTrue(result.passed, msg=result.output)

    def test_grader_fails_when_manifest_exceeds_budget(self):
        """A run whose manifest was over budget is graded
        `failed`. The grader surfaces the over-budget amount
        in the output.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            from rlm_harness.tracing import TraceStore

            store = TraceStore(workspace / "trace.db")
            run_id = store.start_run("x", str(workspace), thread_id="t")
            # An over-budget context preview (chars / 4 ~ tokens).
            big_preview = "x" * 100_000
            store.event(
                run_id,
                "turn_started",
                {
                    "turn_index": 0,
                    "context_preview": big_preview,
                },
                node="supervisor",
            )
            store.finish_run(run_id, "done")

            grader = RLMContextEfficiencyGrader(
                max_turns=5, max_manifest_tokens=20_000
            )
            result = grader.grade(workspace, run_id=run_id)
            self.assertFalse(result.passed)
            self.assertIn("over budget", result.output.lower())


class LongHorizonSuiteShapeTests(unittest.TestCase):
    def test_long_horizon_cases_have_turn_budget_metadata(self):
        """Long-horizon cases declare their expected turn
        budget in metadata. The grader and the harness both
        read this.
        """
        path = Path("rlm_harness/evals/suites/long-horizon.json")
        suite = load_suite(path)
        for case in suite.cases:
            self.assertIn(
                "turn_budget",
                case.metadata,
                f"long-horizon case {case.id} missing turn_budget",
            )
            self.assertGreaterEqual(case.metadata["turn_budget"], 1)

    def test_long_horizon_cases_require_persistent_state(self):
        """Long-horizon cases explicitly say what state must
        persist across turns. The grader checks the trace
        for state-carrying events.
        """
        path = Path("rlm_harness/evals/suites/long-horizon.json")
        suite = load_suite(path)
        for case in suite.cases:
            self.assertIn(
                "persistent_state_requirement",
                case.metadata,
                f"long-horizon case {case.id} missing persistent_state_requirement",
            )


if __name__ == "__main__":
    unittest.main()
