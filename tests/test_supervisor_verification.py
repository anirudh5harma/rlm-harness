"""Tests for the supervisor's strict-verification enforcement (Phase D.3).

The supervisor consumes a `verifier` hook that runs after the
last RLM turn. The hook returns a `VerificationResult`; the
`VerificationPolicy` classifies it; the run's terminal phase
respects the classification:

* `verified`            → `done` (only when the work was code-related)
* `not_applicable`      → `done` for non-code work
* `unverified`          → `unverified` (not `done`)
* `failed`              → `failed` (not `done`)

This is the Phase D gate: a code edit with passing checks
exits `done`; the same code edit with skipped checks exits
`unverified`; the same code edit with a failing check exits
`failed`.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.graph.verification import (
    VerificationCheck,
    VerificationResult,
)
from rlm_harness.kernel.state import (
    AutonomyMode,
    RunPhase,
    RunRequest,
    RunState,
)
from rlm_harness.kernel.supervisor import (
    Supervisor,
    SupervisorConfig,
)
from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import RLMRuntime
from rlm_harness.tracing import TraceStore


class ScriptedStreamClient(LMClient):
    def __init__(self, scripted):
        super().__init__(provider="stub", model="scripted")
        self._queue = list(scripted)

    def stream(self, messages, max_tokens=512, temperature=0.2):
        from rlm_harness.types import TokenEvent

        deltas = self._queue.pop(0) if self._queue else ["ok"]
        yield TokenEvent(type="start", model=self.model, provider=self.provider)
        for d in deltas:
            yield TokenEvent(
                type="delta", delta=d, model=self.model, provider=self.provider
            )
        yield TokenEvent(
            type="finish",
            model=self.model,
            provider=self.provider,
            usage={"prompt_tokens": 1, "completion_tokens": len(deltas)},
            finish_reason="stop",
        )

    def complete(self, messages, max_tokens=512, temperature=0.2):
        chunks: list[str] = []
        for event in self.stream(messages, max_tokens=max_tokens, temperature=temperature):
            if event.type == "delta":
                chunks.append(event.delta)
            if event.type == "finish":
                from rlm_harness.types import Completion

                return Completion(
                    content="".join(chunks),
                    model=event.model or self.model,
                    provider=event.provider or self.provider,
                    latency_ms=0,
                )
        from rlm_harness.types import Completion

        return Completion(
            content="".join(chunks), model=self.model, provider=self.provider, latency_ms=0
        )


def _check(*, check_type: str, passed: bool, output: str = "") -> VerificationCheck:
    return VerificationCheck(check_type=check_type, passed=passed, output=output)


def _make_state(run_id: str, workspace: Path) -> RunState:
    return RunState(
        request=RunRequest(
            task="edit a file",
            workspace=str(workspace),
            run_id=run_id,
            thread_id="t",
            autonomy=AutonomyMode.SANDBOX,
        )
    )


def state_workspace(run_id: str) -> str:
    """Build a workspace path for the test state. The state
    itself does not need to exist on disk; the supervisor only
    uses `state.request.workspace` to thread it to the verifier.
    """
    return f"/tmp/rlm-harness-test-{run_id}"


class SupervisorVerificationEnforcementTests(unittest.TestCase):
    def _build(
        self,
        verifier_result: VerificationResult,
        scripted: list[list[str]],
        max_turns: int = 3,
    ):
        # Build the supervisor inside a managed temp dir; return
        # the run_id, the workspace path, and a cleanup callback
        # so the test can keep the dir alive until `supervisor.run`
        # returns.
        temp_dir_cm = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_cm.__enter__()
        workspace = Path(temp_dir)
        trace_db = workspace / "trace.db"
        traces = TraceStore(trace_db)
        run_id = traces.start_run("edit a file", str(workspace), thread_id="t")
        client = ScriptedStreamClient(scripted)
        runtime = RLMRuntime(
            client, workspace=workspace, max_iterations=1, sandbox_enabled=False
        )
        supervisor = Supervisor(
            runtime=runtime,
            traces=traces,
            config=SupervisorConfig(max_turns=max_turns, max_subcalls_per_turn=2),
            verifier=lambda workspace, state: verifier_result,
        )

        def cleanup() -> None:
            temp_dir_cm.__exit__(None, None, None)

        return supervisor, run_id, workspace, cleanup

    def test_verified_status_allows_done(self):
        """`verified` is the only status that allows a `done`
        exit for a code-editing run. The supervisor exits
        `done` and the trace's run status is `done`.
        """
        verifier_result = VerificationResult(
            passed=True,
            checks=[_check(check_type="ruff", passed=True)],
            changed_files=["foo.py"],
            summary="[PASS] ruff",
        )
        supervisor, run_id, _, cleanup = self._build(
            verifier_result,
            [
                [
                    "```repl\n",
                    "answer['content'] = 'ok'\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ],
        )
        try:
            state = _make_state(run_id, Path(state_workspace(run_id)))
            final = supervisor.run(state)
        finally:
            cleanup()
        self.assertEqual(final.phase, RunPhase.DONE)

    def test_unverified_status_does_not_allow_done(self):
        """`unverified` exits as `unverified`, not `done`."""
        verifier_result = VerificationResult(
            passed=True,
            checks=[
                _check(
                    check_type="pytest",
                    passed=True,
                    output="pytest skipped: not installed",
                )
            ],
            changed_files=["foo.py"],
            summary="[PASS] pytest (skipped)",
        )
        supervisor, run_id, _, cleanup = self._build(
            verifier_result,
            [
                [
                    "```repl\n",
                    "answer['content'] = 'ok'\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ],
        )
        try:
            state = _make_state(run_id, Path(state_workspace(run_id)))
            final = supervisor.run(state)
        finally:
            cleanup()
        self.assertEqual(final.phase.value, "unverified")

    def test_failed_status_does_not_allow_done(self):
        """`failed` exits as `failed`, not `done`."""
        verifier_result = VerificationResult(
            passed=False,
            checks=[_check(check_type="ruff", passed=False, output="syntax error")],
            changed_files=["foo.py"],
            summary="[FAIL] ruff",
        )
        supervisor, run_id, _, cleanup = self._build(
            verifier_result,
            [
                [
                    "```repl\n",
                    "answer['content'] = 'ok'\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ],
        )
        try:
            state = _make_state(run_id, Path(state_workspace(run_id)))
            final = supervisor.run(state)
        finally:
            cleanup()
        self.assertEqual(final.phase.value, "failed")

    def test_not_applicable_status_with_no_changes_allows_done(self):
        """A non-code task with no changed files is
        `not_applicable`. The supervisor exits `done` because
        the work did not require verification.
        """
        verifier_result = VerificationResult(
            passed=True,
            checks=[],
            changed_files=[],
            summary="No verification performed.",
        )
        supervisor, run_id, _, cleanup = self._build(
            verifier_result,
            [
                [
                    "```repl\n",
                    "answer['content'] = 'summary'\n",
                    "answer['ready'] = True\n",
                    "```",
                ]
            ],
        )
        try:
            state = _make_state(run_id, Path(state_workspace(run_id)))
            final = supervisor.run(state)
        finally:
            cleanup()
        self.assertEqual(final.phase.value, "done")


if __name__ == "__main__":
    unittest.main()
