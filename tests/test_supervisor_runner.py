"""End-to-end test for the supervisor-driven run path (Phase A.5).

The supervisor replaces the LangGraph-based loop as the default
control plane. The existing `HarnessGraph` is preserved as the
`--graph-backend simple` path; `task_runtime` and the `harness` CLI
route to the supervisor by default.

This test exercises the supervisor end-to-end against the project's
trace store, model client, and memory store. It is the integration
test the Phase A gate calls out in `AGENTS.md` §6.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.kernel.state import (
    AutonomyMode,
    RunRequest,
    RunState,
)
from rlm_harness.kernel.supervisor import Supervisor, SupervisorConfig
from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import RLMRuntime
from rlm_harness.tracing import TraceStore


class ScriptedStreamClient(LMClient):
    def __init__(self, scripted):
        super().__init__(provider="stub", model="scripted")
        self._queue = list(scripted)
        self.calls: list[list[str]] = []

    def stream(self, messages, max_tokens=512, temperature=0.2):
        from rlm_harness.types import TokenEvent

        deltas = self._queue.pop(0) if self._queue else ["ok"]
        self.calls.append(deltas)
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


class SupervisorRunnerTests(unittest.TestCase):
    def test_supervisor_runner_drives_a_long_task(self):
        """The supervisor drives a multi-turn run end-to-end and
        the trace records one row per turn. This is the Phase A
        gate: 'trace from a long task should show many RLM turns
        per run, not 6'.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            run_id = traces.start_run("add docstring", str(workspace), thread_id="t")

            # 5 turns: each iterates without completing, except the
            # last one which calls complete_task.
            client = ScriptedStreamClient(
                [
                    ["```repl\nprint('a')\n```"],
                    ["```repl\nprint('b')\n```"],
                    ["```repl\nprint('c')\n```"],
                    ["```repl\nprint('d')\n```"],
                    [
                        "```repl\n",
                        "answer['content'] = 'done'\n",
                        "answer['ready'] = True\n",
                        "```",
                    ],
                ]
            )
            runtime = RLMRuntime(
                client,
                workspace=workspace,
                max_iterations=1,
                sandbox_enabled=False,
            )
            state = RunState(
                request=RunRequest(
                    task="add docstring",
                    workspace=str(workspace),
                    run_id=run_id,
                    thread_id="t",
                    autonomy=AutonomyMode.SANDBOX,
                )
            )
            supervisor = Supervisor(
                runtime=runtime,
                traces=traces,
                config=SupervisorConfig(max_turns=10, max_subcalls_per_turn=2),
            )
            final = supervisor.run(state)
            self.assertEqual(final.phase.value, "done")
            self.assertEqual(final.final_answer, "done")

            # The trace must show one row per turn, not a fixed 6
            # iterations. 5 turns → at least 5 `turn_started` rows
            # and 5 `turn_finished` rows.
            events = traces.events(run_id)
            turn_started = [e for e in events if e["kind"] == "turn_started"]
            turn_finished = [e for e in events if e["kind"] == "turn_finished"]
            self.assertEqual(len(turn_started), 5)
            self.assertEqual(len(turn_finished), 5)

    def test_supervisor_runner_surfaces_unverified_partial(self):
        """When the runtime returns `stopped` after a non-terminal
        turn and the supervisor hits `max_turns`, the run is
        marked `stopped` (the kernel equivalent of partial). The
        trace's `run_summary` reflects this.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            run_id = traces.start_run("x", str(workspace), thread_id="t")

            client = ScriptedStreamClient(
                [
                    ["```repl\nprint('a')\n```"],
                    ["```repl\nprint('a')\n```"],
                ]
            )
            runtime = RLMRuntime(
                client, workspace=workspace, max_iterations=1, sandbox_enabled=False
            )
            state = RunState(
                request=RunRequest(
                    task="x",
                    workspace=str(workspace),
                    run_id=run_id,
                    thread_id="t",
                    autonomy=AutonomyMode.SANDBOX,
                )
            )
            supervisor = Supervisor(
                runtime=runtime,
                traces=traces,
                config=SupervisorConfig(max_turns=2, max_subcalls_per_turn=2),
            )
            final = supervisor.run(state)
            self.assertEqual(final.phase.value, "stopped")
            summary = traces.run_summary(run_id)
            self.assertEqual(summary["status"], "stopped")

    def test_supervisor_runner_emits_completion_with_typed_event(self):
        """A `done` run records a typed `completion` event whose
        payload includes the final answer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            run_id = traces.start_run("x", str(workspace), thread_id="t")

            client = ScriptedStreamClient(
                [
                    [
                        "```repl\n",
                        "answer['content'] = 'final answer'\n",
                        "answer['ready'] = True\n",
                        "```",
                    ]
                ]
            )
            runtime = RLMRuntime(
                client, workspace=workspace, max_iterations=1, sandbox_enabled=False
            )
            state = RunState(
                request=RunRequest(
                    task="x",
                    workspace=str(workspace),
                    run_id=run_id,
                    thread_id="t",
                    autonomy=AutonomyMode.SANDBOX,
                )
            )
            supervisor = Supervisor(
                runtime=runtime,
                traces=traces,
                config=SupervisorConfig(max_turns=3, max_subcalls_per_turn=2),
            )
            supervisor.run(state)
            typed = traces.typed_events(run_id)
            completion_events = [e for e in typed if e.kind == "completion"]
            self.assertEqual(len(completion_events), 1)
            self.assertEqual(completion_events[0].final_answer, "final answer")


if __name__ == "__main__":
    unittest.main()
