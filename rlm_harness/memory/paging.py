from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from rlm_harness.memory.store import Memory, MemoryValidationError, count_tokens
from rlm_harness.model_client import LMClient
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg


@dataclass(frozen=True)
class MemoryPagingConfig:
    max_history_tokens: int = 1600
    preserve_recent_steps: int = 4
    recall_limit: int = 6
    archival_limit: int = 3
    summary_max_tokens: int = 300

    def __post_init__(self) -> None:
        values = {
            "max_history_tokens": self.max_history_tokens,
            "preserve_recent_steps": self.preserve_recent_steps,
            "recall_limit": self.recall_limit,
            "archival_limit": self.archival_limit,
            "summary_max_tokens": self.summary_max_tokens,
        }
        for name, value in values.items():
            if value <= 0:
                raise MemoryValidationError(f"{name} must be positive")


class MemoryPager:
    def __init__(
        self,
        memory: Memory,
        client: LMClient,
        traces: TraceStore,
        config: Optional[MemoryPagingConfig] = None,
    ):
        self.memory = memory
        self.client = client
        self.traces = traces
        self.config = config or MemoryPagingConfig()

    def hydrate(self, state: HarnessState) -> HarnessState:
        if state.scratch.get("memory_hydrated"):
            return state

        recall_events = self.memory.recall_page(
            state.thread_id,
            query=state.task,
            k=self.config.recall_limit,
        )
        archival_results = self.memory.archival_search(
            state.task,
            k=self.config.archival_limit,
            kind="episode",
            source_thread=state.thread_id,
        )
        context = render_memory_context(recall_events, archival_results)
        state.scratch["memory_hydrated"] = True
        state.scratch["memory_context"] = context
        state.scratch["memory_recall_ids"] = [event.id for event in recall_events]
        state.scratch["memory_archival_ids"] = [
            result.memory.id for result in archival_results
        ]
        self.traces.event(
            state.run_id,
            "memory_hydrated",
            {
                "recall_count": len(recall_events),
                "archival_count": len(archival_results),
                "context_tokens": count_tokens(context) if context else 0,
            },
            node="memory",
        )
        return state

    def persist_new_history(self, state: HarnessState) -> HarnessState:
        persisted_count = int(state.scratch.get("memory_persisted_history_count", 0))
        if persisted_count < 0:
            persisted_count = 0
        if persisted_count > len(state.history):
            persisted_count = len(state.history)

        for step in state.history[persisted_count:]:
            node = str(step.get("node", "unknown"))
            self.memory.recall_append(
                state.thread_id,
                recall_role_for_node(node),
                format_step(step),
                metadata={"run_id": state.run_id, "node": node},
            )
        state.scratch["memory_persisted_history_count"] = len(state.history)
        return self.page_if_needed(state)

    def page_if_needed(self, state: HarnessState) -> HarnessState:
        history_tokens = history_token_count(state.history)
        if history_tokens <= self.config.max_history_tokens:
            return state
        if len(state.history) <= self.config.preserve_recent_steps:
            return state

        page_count = len(state.history) - self.config.preserve_recent_steps
        paged_steps = state.history[:page_count]
        retained_steps = state.history[page_count:]
        paged_content = "\n\n".join(format_step(step) for step in paged_steps)
        summary = self._summarize(state, paged_content)
        archive = self.memory.archival_add(
            "episode",
            summary,
            source_thread=state.thread_id,
            metadata={
                "run_id": state.run_id,
                "paged_step_count": len(paged_steps),
                "history_tokens_before": history_tokens,
                "retained_step_count": len(retained_steps),
            },
        )
        state.history = retained_steps
        state.scratch["memory_persisted_history_count"] = len(retained_steps)
        state.scratch["memory_pages_written"] = (
            int(state.scratch.get("memory_pages_written", 0)) + 1
        )
        state.scratch["last_memory_archive_id"] = archive.id
        self.traces.event(
            state.run_id,
            "memory_paged",
            {
                "archival_id": archive.id,
                "paged_step_count": len(paged_steps),
                "retained_step_count": len(retained_steps),
                "history_tokens_before": history_tokens,
                "summary_tokens": archive.tokens,
            },
            node="memory",
        )
        return state

    def _summarize(self, state: HarnessState, paged_content: str) -> str:
        completion = self.client.complete(
            [
                Msg(
                    role="system",
                    content=(
                        "Summarize older harness history for durable archival memory. "
                        "Preserve task intent, decisions, tool outcomes, errors, and facts "
                        "needed to resume later."
                    ),
                ),
                Msg(
                    role="user",
                    content=(
                        "Summarize older harness history.\n"
                        f"Task: {state.task}\n\nHistory:\n{paged_content}"
                    ),
                ),
            ],
            max_tokens=self.config.summary_max_tokens,
            temperature=0,
        )
        summary = completion.content.strip()
        self.traces.event(
            state.run_id,
            "model_completion",
            {
                "model": completion.model,
                "provider": completion.provider,
                "latency_ms": completion.latency_ms,
                "content": summary,
            },
            node="memory",
        )
        return summary or paged_content[:2000]


def render_memory_context(recall_events: list[Any], archival_results: list[Any]) -> str:
    sections = []
    if recall_events:
        recall_lines = [
            f"- [{event.role} #{event.id}] {event.content}" for event in recall_events
        ]
        sections.append("Recent recall:\n" + "\n".join(recall_lines))
    if archival_results:
        archival_lines = [
            f"- [episode #{result.memory.id}] {result.memory.content}"
            for result in archival_results
        ]
        sections.append("Relevant archival memory:\n" + "\n".join(archival_lines))
    return "\n\n".join(sections)


def format_step(step: dict[str, Any]) -> str:
    node = str(step.get("node", "unknown"))
    content = step.get("content", "")
    if isinstance(content, str):
        rendered = content
    else:
        rendered = json.dumps(content, sort_keys=True)
    return f"{node}: {rendered}"


def history_token_count(history: list[dict[str, Any]]) -> int:
    if not history:
        return 0
    return count_tokens("\n\n".join(format_step(step) for step in history))


def recall_role_for_node(node: str) -> str:
    if node == "observe":
        return "tool"
    if node == "reflect":
        return "reflection"
    if node == "plan":
        return "assistant"
    return "assistant"
