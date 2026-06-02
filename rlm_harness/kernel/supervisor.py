"""Per-turn supervisor (Phase A.3).

The supervisor is the new control plane. It runs up to `max_turns`
RLM turns, each turn consuming up to `max_subcalls_per_turn` recursive
sub-calls, and emits typed events to the `TraceStore` between turns.
The non-streaming `HarnessGraph` (graph/build.py) is preserved for
backward compatibility; the supervisor is the new path used by the
`harness` CLI in Phase F.

The supervisor is intentionally narrow:

* it does not know about plans, tools, memory, or verification;
  those are owned by separate services and composed by the call site;
* it does not interpret the RLM's response; it only forwards the
  final answer and the per-iteration state to the trace;
* it provides a `between_turns` hook so the memory pager (Phase A.4)
  and any future cross-turn service can run between turns without
  the supervisor growing a new responsibility.

The `RLMRuntime` is the unit of work: one supervisor run is N turns,
each turn is one `RLMRuntime.stream_turn()` consumption. The
streaming entry point is what makes per-turn checkpointing
possible.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from rlm_harness.kernel.events import (
    CompletionEvent,
    RunStartedEvent,
)
from rlm_harness.kernel.state import (
    RunPhase,
    RunState,
)
from rlm_harness.memory import Memory
from rlm_harness.memory.paging import MemoryPager, MemoryPagingConfig
from rlm_harness.rlm.runtime import (
    RLMResult,
    RLMRuntime,
    TurnFinished,
    TurnStarted,
)
from rlm_harness.tracing import TraceStore

# New defaults per the pivot plan. Old `max_iterations=6` default is
# deprecated but still accepted on `RLMRuntime` for backward
# compatibility with the existing tests.
DEFAULT_MAX_TURNS = 50
DEFAULT_MAX_SUBCALLS_PER_TURN = 8


@dataclass(frozen=True)
class SupervisorConfig:
    """Per-run supervisor configuration.

    `max_turns` is the number of RLM turns the supervisor is willing
    to run. A turn is one streaming run of the RLM runtime, including
    all repl blocks and sub-calls the model emits inside the turn.

    `max_subcalls_per_turn` bounds the recursion depth inside a turn.
    It is forwarded to the RLM runtime's `subcall_config`.

    `between_turns` is a hook called after every turn with the current
    `RunState` and the turn index. The memory pager uses it; future
    cross-turn services will too.
    """

    max_turns: int = DEFAULT_MAX_TURNS
    max_subcalls_per_turn: int = DEFAULT_MAX_SUBCALLS_PER_TURN
    between_turns: Optional[Callable[[RunState, int], None]] = None


class Supervisor:
    """Per-run, per-turn RLM supervisor."""

    def __init__(
        self,
        runtime: RLMRuntime,
        traces: TraceStore,
        config: Optional[SupervisorConfig] = None,
        *,
        between_turns: Optional[Callable[[RunState, int], None]] = None,
    ):
        self.runtime = runtime
        self.traces = traces
        if config is not None and between_turns is not None:
            raise ValueError(
                "pass `between_turns` either via SupervisorConfig or as a "
                "keyword argument, not both"
            )
        if between_turns is not None:
            config = SupervisorConfig(
                max_turns=config.max_turns if config else DEFAULT_MAX_TURNS,
                max_subcalls_per_turn=(
                    config.max_subcalls_per_turn if config else DEFAULT_MAX_SUBCALLS_PER_TURN
                ),
                between_turns=between_turns,
            )
        self.config = config or SupervisorConfig()
        if self.config.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.config.max_subcalls_per_turn <= 0:
            raise ValueError("max_subcalls_per_turn must be positive")
        # Cap the runtime's per-turn iteration count so the
        # subcall budget is enforced from below.
        self.runtime.subcall_config = self.runtime.subcall_config.__class__(
            max_depth=self.runtime.subcall_config.max_depth,
            max_subcalls=self.config.max_subcalls_per_turn,
            token_budget=self.runtime.subcall_config.token_budget,
            max_query_chars=self.runtime.subcall_config.max_query_chars,
            max_context_chars=self.runtime.subcall_config.max_context_chars,
            max_tokens=self.runtime.subcall_config.max_tokens,
            temperature=self.runtime.subcall_config.temperature,
        )

    def run(self, state: RunState) -> RunState:
        """Run the supervisor loop. Mutates and returns `state`."""
        run_id = state.request.run_id or ""
        self._record_run_started(run_id, state)
        state.phase = RunPhase.PLAN

        # The supervisor runs N turns. A turn is one streaming call
        # to the RLM runtime. The runtime already enforces per-turn
        # iteration limits; the supervisor enforces the per-run
        # `max_turns` ceiling.
        last_result: Optional[RLMResult] = None
        for turn_index in range(self.config.max_turns):
            context = self._build_turn_context(state)
            events = self.runtime.stream_turn(state.request.task, context=context)
            for event in events:
                if isinstance(event, TurnStarted):
                    self._record_turn_started(run_id, turn_index, event)
                elif isinstance(event, TurnFinished):
                    last_result = event.result
                    self._record_turn_finished(run_id, turn_index, event)
                    if last_result.status == "done":
                        state.phase = RunPhase.DONE
                        state.final_answer = last_result.final_answer
                        self._record_completion(run_id, state, last_result)
                        self._finish_run(run_id, "done")
                        return state
                    if last_result.status == "error":
                        state.phase = RunPhase.FAILED
                        state.final_answer = last_result.final_answer
                        self._finish_run(run_id, "error")
                        return state
            # Per-turn paging hook (Phase A.4). Caller may mutate
            # `state.history` or call out to the memory pager.
            if self.config.between_turns is not None:
                self.config.between_turns(state, turn_index)

        # Out of turns without `done` or `error`. Surface a `stopped`
        # status; the caller is expected to treat `stopped` as a
        # partial answer and surface it to the user.
        state.phase = RunPhase.STOPPED
        if last_result is not None and not state.final_answer:
            state.final_answer = last_result.final_answer
        self._record_completion(run_id, state, last_result)
        self._finish_run(run_id, "stopped")
        return state

    def _build_turn_context(self, state: RunState) -> dict:
        """Build the context the runtime hands to the model.

        Today this is a thin pass-through. Phase B will replace it
        with a content-addressed manifest of the working context.
        """
        return {
            "task": state.request.task,
            "history_summary": state.context.memory_context,
        }

    def _record_run_started(self, run_id: str, state: RunState) -> None:
        self.traces.record_typed_event(
            RunStartedEvent(
                run_id=run_id,
                sequence=self.traces.next_sequence(run_id),
                node="supervisor",
                task=state.request.task,
                workspace=state.request.workspace,
                thread_id=state.request.thread_id or run_id,
                payload={
                    "autonomy": state.request.autonomy.value,
                    "max_turns": self.config.max_turns,
                    "max_subcalls_per_turn": self.config.max_subcalls_per_turn,
                },
            )
        )

    def _record_turn_started(
        self, run_id: str, turn_index: int, event: TurnStarted
    ) -> None:
        self.traces.event(
            run_id,
            "turn_started",
            {
                "turn_index": turn_index,
                "query": event.query,
                "context_preview_chars": len(event.context_preview),
                "iteration_limit": event.iteration_limit,
            },
            node="supervisor",
        )

    def _record_turn_finished(
        self, run_id: str, turn_index: int, event: TurnFinished
    ) -> None:
        result = event.result
        self.traces.event(
            run_id,
            "turn_finished",
            {
                "turn_index": turn_index,
                "status": result.status,
                "iterations": result.iterations,
                "observations": len(result.observations),
                "subcalls": result.subcalls,
                "tokens_used": result.tokens_used,
            },
            node="supervisor",
        )

    def _record_completion(
        self, run_id: str, state: RunState, result: Optional[RLMResult]
    ) -> None:
        if not state.final_answer:
            return
        self.traces.record_typed_event(
            CompletionEvent.from_action(
                run_id=run_id,
                sequence=self.traces.next_sequence(run_id),
                node="supervisor",
                action=_completion_action(state, result),
            )
        )

    def _finish_run(self, run_id: str, status: str) -> None:
        self.traces.finish_run(run_id, status)


def _completion_action(state: RunState, result: Optional[RLMResult]):
    """Build a CompleteTaskAction from the final state.

    Imported here to avoid a cycle with `rlm_harness.actions.base`.
    """
    from rlm_harness.actions.base import CompleteTaskAction, CompletionStatus

    status = CompletionStatus.SUCCESS
    if state.phase == RunPhase.STOPPED:
        status = CompletionStatus.PARTIAL
    elif state.phase == RunPhase.FAILED:
        status = CompletionStatus.FAILED
    elif state.phase == RunPhase.BLOCKED:
        status = CompletionStatus.BLOCKED
    return CompleteTaskAction(
        summary=state.final_answer or "",
        status=status,
        verification=(
            (f"subcalls={result.subcalls}, tokens={result.tokens_used}")
            if result is not None
            else None
        ),
    )


__all__ = [
    "DEFAULT_MAX_SUBCALLS_PER_TURN",
    "DEFAULT_MAX_TURNS",
    "Supervisor",
    "SupervisorConfig",
    "page_history_between_turns",
]


def page_history_between_turns(
    memory: Memory,
    config: Optional[MemoryPagingConfig] = None,
    *,
    client: Optional[object] = None,
    traces: Optional[TraceStore] = None,
) -> Callable[[RunState, int], None]:
    """Build a `between_turns` hook that pages `state.history`.

    The hook is a thin wrapper over the existing `MemoryPager`. It
    is parameterised by a `Memory` instance (the project memory
    store) and an optional paging config. The optional `client` and
    `traces` are forwarded to the pager so per-step summaries show
    up in the trace.

    Usage:

        memory = Memory(Path("project.db"))
        supervisor = Supervisor(
            runtime=runtime,
            traces=traces,
            config=SupervisorConfig(
                max_turns=50,
                max_subcalls_per_turn=8,
                between_turns=page_history_between_turns(memory),
            ),
        )

    The hook is idempotent: if the in-memory `state.history` is
    below the budget, it is a no-op. It is also safe to call when
    `state.history` is empty.
    """
    pager = MemoryPager(
        memory,
        client=client if client is not None else _NoopSummaryClient(),
        traces=traces if traces is not None else _NoopTraces(),
        config=config or MemoryPagingConfig(),
    )

    def hook(state: RunState, turn_index: int) -> None:
        pager.persist_new_history(state)
        pager.page_if_needed(state)

    return hook


class _NoopSummaryClient:
    """Stand-in for an LMClient used by the pager when no client
    was supplied. Returns a deterministic, non-LLM summary derived
    from the truncated history, so the pager's behaviour is
    exercised end-to-end without spinning up a real model.
    """

    def complete(self, messages, max_tokens=512, temperature=0.0):
        from rlm_harness.types import Completion

        # Pull the user message; summarise the first 500 chars.
        text = ""
        for message in reversed(messages):
            if message.role == "user":
                text = message.content
                break
        summary = (
            "Compacted harness history (" + str(len(text)) + " chars): "
            + text[:500].replace("\n", " ")
        )
        return Completion(
            content=summary,
            model="noop",
            provider="noop",
            latency_ms=0,
        )


class _NoopTraces:
    def event(self, *args, **kwargs) -> None:
        return None

    def record_typed_event(self, *args, **kwargs) -> None:
        return None
