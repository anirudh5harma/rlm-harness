import unittest

from pydantic import ValidationError

from rlm_harness.actions import (
    CommandObservation,
    CompleteTaskAction,
    CompletionStatus,
    ReadFileAction,
    VerificationStatus,
    WriteFileAction,
    parse_action,
    parse_observation,
)
from rlm_harness.kernel import (
    ActionSelectedEvent,
    CompletionEvent,
    RunPhase,
    RunStartedEvent,
    RunState,
    VerificationEvent,
    parse_event,
)
from rlm_harness.types import HarnessState, TaskPlan


class KernelContractTests(unittest.TestCase):
    def test_action_round_trip_uses_discriminated_kind(self):
        action = ReadFileAction(path="README.md", reason="orient on the project")

        payload = action.model_dump(mode="json")
        parsed = parse_action(payload)

        self.assertIsInstance(parsed, ReadFileAction)
        self.assertEqual(parsed.path, "README.md")
        self.assertEqual(parsed.reason, "orient on the project")
        self.assertEqual(parsed.risk, action.risk)

    def test_write_file_requires_confirmation_by_default(self):
        action = WriteFileAction(path="README.md", content="# Harness\n")

        self.assertTrue(action.requires_confirmation)
        self.assertEqual(action.risk.value, "high")

    def test_observation_round_trip_preserves_command_result(self):
        observation = CommandObservation(
            action_id="act_1",
            command="pytest",
            exit_code=1,
            stdout="",
            stderr="failed",
            duration_ms=25,
        )

        parsed = parse_observation(observation.model_dump(mode="json"))

        self.assertIsInstance(parsed, CommandObservation)
        self.assertEqual(parsed.command, "pytest")
        self.assertEqual(parsed.exit_code, 1)
        self.assertEqual(parsed.stderr, "failed")

    def test_event_round_trip_preserves_nested_action(self):
        event = ActionSelectedEvent(
            run_id="run_1",
            sequence=2,
            node="select_action",
            action=ReadFileAction(path="pyproject.toml"),
        )

        parsed = parse_event(event.model_dump(mode="json"))

        self.assertIsInstance(parsed, ActionSelectedEvent)
        self.assertIsInstance(parsed.action, ReadFileAction)
        self.assertEqual(parsed.action.path, "pyproject.toml")

    def test_completion_event_must_come_from_explicit_completion_action(self):
        action = CompleteTaskAction(
            summary="Implemented the contract layer.",
            status=CompletionStatus.SUCCESS,
            verification="pytest tests/test_kernel_contracts.py",
        )

        event = CompletionEvent.from_action(
            run_id="run_1",
            sequence=4,
            node="finalize",
            action=action,
        )

        self.assertEqual(event.final_answer, "Implemented the contract layer.")
        self.assertEqual(event.status, CompletionStatus.SUCCESS)
        self.assertEqual(event.payload["action_id"], action.action_id)

    def test_unknown_action_kind_is_rejected(self):
        with self.assertRaises(ValidationError):
            parse_action({"kind": "invented_tool", "path": "README.md"})

    def test_run_state_adapts_from_existing_harness_state(self):
        harness_state = HarnessState(
            task="what is this project",
            workspace="/tmp/project",
            thread_id="thread_1",
            run_id="run_1",
            plan=TaskPlan.from_flat_steps(["Inspect", "Summarize"]),
            status="done",
            final_answer="A local coding harness.",
        )

        run_state = RunState.from_harness_state(harness_state)
        restored = run_state.to_harness_state()

        self.assertEqual(run_state.phase, RunPhase.DONE)
        self.assertEqual(run_state.request.task, "what is this project")
        self.assertEqual(restored.status, "done")
        self.assertEqual(restored.plan.total_count(), 2)
        self.assertEqual(restored.final_answer, "A local coding harness.")

    def test_run_events_cover_start_and_verification_statuses(self):
        started = RunStartedEvent(
            run_id="run_1",
            sequence=1,
            task="fix tests",
            workspace="/tmp/project",
            thread_id="thread_1",
        )
        verified = VerificationEvent(
            run_id="run_1",
            sequence=3,
            status=VerificationStatus.UNVERIFIED,
            checks=[{"name": "pytest", "status": "skipped"}],
        )

        parsed_started = parse_event(started.model_dump(mode="json"))
        parsed_verified = parse_event(verified.model_dump(mode="json"))

        self.assertIsInstance(parsed_started, RunStartedEvent)
        self.assertEqual(parsed_verified.status, VerificationStatus.UNVERIFIED)


if __name__ == "__main__":
    unittest.main()
