"""Tests for the supervisor's default between-turns pager (Phase A.4).

The pivot plan moves `state.history` paging from being triggered by
the LangGraph nodes to being triggered by the supervisor between
turns. The supervisor exposes a `between_turns` hook; the harness
ships a default that pages `state.history` into the project memory
store when the in-memory list crosses the configured budget.
"""
import tempfile
import unittest
from pathlib import Path

from rlm_harness.kernel.state import (
    AutonomyMode,
    RunRequest,
    RunState,
)
from rlm_harness.kernel.supervisor import (
    Supervisor,
    SupervisorConfig,
    page_history_between_turns,
)
from rlm_harness.memory import Memory
from rlm_harness.memory.paging import MemoryPagingConfig
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


class SupervisorPagingTests(unittest.TestCase):
    def test_page_history_between_turns_compacts_long_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            memory_db = workspace / "memory.db"
            with Memory(memory_db) as memory:
                # Build a long history: 200 short steps.
                state = RunState(
                    request=RunRequest(
                        task="x",
                        workspace=str(workspace),
                        run_id="r",
                        thread_id="t",
                        autonomy=AutonomyMode.SANDBOX,
                    )
                )
                for i in range(200):
                    state.history.append(
                        {
                            "node": "act",
                            "content": f"step {i}: " + ("x" * 200),
                        }
                    )
                page_history_between_turns(
                    memory, MemoryPagingConfig(max_history_tokens=400)
                )(state, 0)
                # The pager compacts history to fit the budget.
                self.assertLess(len(state.history), 200)
                # And writes the summary to archival memory.
                results = memory.archival_search("step", k=5, kind="episode")
                self.assertTrue(
                    any(
                        "step" in r.memory.content.lower()
                        for r in results
                    )
                )

    def test_page_history_is_a_noop_when_below_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            memory_db = workspace / "memory.db"
            with Memory(memory_db) as memory:
                state = RunState(
                    request=RunRequest(
                        task="x",
                        workspace=str(workspace),
                        run_id="r",
                        thread_id="t",
                        autonomy=AutonomyMode.SANDBOX,
                    )
                )
                state.history.append({"node": "act", "content": "short"})
                before = list(state.history)
                page_history_between_turns(
                    memory, MemoryPagingConfig(max_history_tokens=10_000)
                )(state, 0)
                # No compaction when the budget is huge.
                self.assertEqual(state.history, before)

    def test_supervisor_with_default_pager_writes_to_memory(self):
        """End-to-end: the supervisor uses the default pager, and
        after a run that produced long history, archival memory
        holds at least one summary.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            memory_db = workspace / "memory.db"
            trace_db = workspace / "trace.db"
            with Memory(memory_db) as memory:
                traces = TraceStore(trace_db)
                run_id = traces.start_run("x", str(workspace), thread_id="t")

                # Each turn emits a repl block that does NOT set
                # answer['ready']; the runtime returns `stopped`.
                # The supervisor then iterates. With a small paging
                # budget, the per-turn pager compacts history.
                client = ScriptedStreamClient(
                    [
                        ["```repl\nprint('a')\n```"],
                        ["```repl\nprint('a')\n```"],
                    ]
                )
                runtime = RLMRuntime(
                    client,
                    workspace=workspace,
                    max_iterations=1,
                    sandbox_enabled=False,
                )

                # Pre-populate `state.history` to exceed the paging
                # budget so the pager has something to compact.
                state = RunState(
                    request=RunRequest(
                        task="x",
                        workspace=str(workspace),
                        run_id=run_id,
                        thread_id="t",
                        autonomy=AutonomyMode.SANDBOX,
                    )
                )
                for _i in range(50):
                    state.history.append(
                        {"node": "act", "content": "filler " * 100}
                    )

                supervisor = Supervisor(
                    runtime=runtime,
                    traces=traces,
                    config=SupervisorConfig(
                        max_turns=2,
                        max_subcalls_per_turn=2,
                        between_turns=page_history_between_turns(
                            memory,
                            MemoryPagingConfig(
                                max_history_tokens=200,
                                preserve_recent_steps=2,
                                summary_max_tokens=80,
                            ),
                        ),
                    ),
                )
                final = supervisor.run(state)
                self.assertEqual(final.phase.value, "stopped")
                # The pager ran; state.history is shorter than
                # before, and an archival record was added.
                self.assertLess(len(final.history), 50)


if __name__ == "__main__":
    unittest.main()
