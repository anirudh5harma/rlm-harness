"""Tests for the per-turn supervisor (Phase A.3).

The supervisor is the new control plane. It:

* takes a `RunState` plus a runtime config;
* runs up to `max_turns` RLM turns, each turn consuming up to
  `max_subcalls_per_turn` recursive sub-calls;
* emits typed events to the `TraceStore` between turns;
* returns a `RunState` whose `status` is one of `done`, `stopped`,
  `error`, or `unverified`.

The non-streaming `HarnessGraph` (graph/build.py) is preserved for
backward compatibility; the supervisor is the new path. This test
exercises the supervisor directly, not through the graph.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.kernel.state import (
    AutonomyMode,
    RunPhase,
    RunRequest,
    RunState,
)
from rlm_harness.kernel.supervisor import Supervisor, SupervisorConfig
from rlm_harness.model_client import LMClient
from rlm_harness.rlm.runtime import (
    RLMRuntime,
)
from rlm_harness.tracing import TraceStore


class ScriptedStreamClient(LMClient):
    """Stream scripted responses one chunk at a time."""

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
        # Fallback path: build a Completion from one full delta.
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
                    prompt_tokens=(event.usage or {}).get("prompt_tokens"),
                    completion_tokens=(event.usage or {}).get("completion_tokens"),
                )
        from rlm_harness.types import Completion

        return Completion(
            content="".join(chunks),
            model=self.model,
            provider=self.provider,
            latency_ms=0,
        )


class SupervisorTests(unittest.TestCase):
    def _build_state(self, temp_dir: Path, run_id: str = "run-test") -> RunState:
        return RunState(
            request=RunRequest(
                task="summarize",
                workspace=str(temp_dir),
                run_id=run_id,
                thread_id="thread-test",
                autonomy=AutonomyMode.SANDBOX,
            )
        )

    def test_supervisor_runs_multiple_turns_and_emits_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text("# Project\nA test project.\n")
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)

            # Scripted: 1st turn answers directly. 1 model call = 1 turn.
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
                client,
                workspace=workspace,
                max_iterations=1,
                sandbox_enabled=False,
            )
            config = SupervisorConfig(
                max_turns=5,
                max_subcalls_per_turn=4,
            )
            supervisor = Supervisor(
                runtime=runtime,
                traces=traces,
                config=config,
            )
            state = self._build_state(workspace)
            final = supervisor.run(state)

            self.assertEqual(final.phase, RunPhase.DONE)
            self.assertEqual(final.final_answer, "final answer")
            events = traces.events("run-test")
            # The supervisor records one event per phase: run_started,
            # turn_started, turn_finished, completion. The shape is
            # *per-turn* not *per-iteration* — exactly the change the
            # pivot plan calls for.
            self.assertGreaterEqual(len(events), 4)
            kinds = {event["kind"] for event in events}
            self.assertIn("run_started", kinds)
            self.assertIn("turn_started", kinds)
            self.assertIn("turn_finished", kinds)
            self.assertIn("completion", kinds)

    def test_supervisor_respects_max_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)

            # Each turn emits a repl block that does NOT set
            # answer['ready'], so the runtime iterates until
            # `max_iterations=1` and returns `status='stopped'`.
            # The supervisor must stop after `max_turns` and surface
            # a `stopped` phase.
            client = ScriptedStreamClient(
                [
                    ["```repl\nprint('still thinking')\n```"],
                    ["```repl\nprint('still thinking')\n```"],
                ]
            )
            runtime = RLMRuntime(
                client,
                workspace=workspace,
                max_iterations=1,
                sandbox_enabled=False,
            )
            config = SupervisorConfig(
                max_turns=2,
                max_subcalls_per_turn=4,
            )
            supervisor = Supervisor(runtime=runtime, traces=traces, config=config)
            state = self._build_state(workspace)
            final = supervisor.run(state)
            self.assertEqual(final.phase, RunPhase.STOPPED)
            self.assertEqual(len(client.calls), 2)

    def test_supervisor_emits_typed_run_started_and_completion_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            # The CLI owns run lifecycle; the supervisor only
            # forwards events and finishes the run. Register the
            # run before invoking the supervisor.
            run_id = traces.start_run(
                "summarize", str(workspace), thread_id="thread-test"
            )

            client = ScriptedStreamClient(
                [
                    [
                        "```repl\n",
                        "answer['content'] = 'final'\n",
                        "answer['ready'] = True\n",
                        "```",
                    ]
                ]
            )
            runtime = RLMRuntime(
                client, workspace=workspace, max_iterations=1, sandbox_enabled=False
            )
            supervisor = Supervisor(
                runtime=runtime,
                traces=traces,
                config=SupervisorConfig(max_turns=3, max_subcalls_per_turn=2),
            )
            state = self._build_state(workspace)
            state.request.run_id = run_id
            final = supervisor.run(state)
            self.assertEqual(final.phase, RunPhase.DONE)
            run_summary = traces.run_summary(run_id)
            self.assertEqual(run_summary["status"], "done")

    def test_default_max_turns_is_50_and_max_subcalls_is_8(self):
        """Per the pivot plan, the new defaults are `max_turns=50` and
        `max_subcalls_per_turn=8`. Old `max_iterations=6` default is
        deprecated but still accepted on `RLMRuntime` for backward
        compatibility.
        """
        self.assertEqual(SupervisorConfig().max_turns, 50)
        self.assertEqual(SupervisorConfig().max_subcalls_per_turn, 8)

    def test_supervisor_persists_pagination_hook_after_each_turn(self):
        """The supervisor must invoke the `between_turns` hook after
        every turn so the memory pager (Phase A.4) can compact
        state. The hook is "after turn N" — it fires once per turn,
        including the last one.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            trace_db = workspace / "trace.db"
            traces = TraceStore(trace_db)
            client = ScriptedStreamClient(
                [
                    ["```repl\nprint('still thinking')\n```"],
                    ["```repl\nprint('still thinking')\n```"],
                    ["```repl\nprint('still thinking')\n```"],
                ]
            )
            runtime = RLMRuntime(
                client, workspace=workspace, max_iterations=1, sandbox_enabled=False
            )
            hook_calls: list[int] = []

            def hook(state: RunState, turn_index: int) -> None:
                hook_calls.append(turn_index)

            supervisor = Supervisor(
                runtime=runtime,
                traces=traces,
                config=SupervisorConfig(
                    max_turns=3, max_subcalls_per_turn=2, between_turns=hook
                ),
            )
            state = self._build_state(workspace)
            final = supervisor.run(state)
            self.assertEqual(final.phase, RunPhase.STOPPED)
            self.assertEqual(hook_calls, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
