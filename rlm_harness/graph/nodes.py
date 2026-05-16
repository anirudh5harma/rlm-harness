from __future__ import annotations

from rlm_harness.model_client import LMClient
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg


def parse_numbered_plan(text: str) -> list[str]:
    plan = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "." in line:
            head, tail = line.split(".", 1)
            if head.strip().isdigit() and tail.strip():
                plan.append(tail.strip())
                continue
        plan.append(line.lstrip("-* ").strip())
    return plan or [text.strip()]


class Nodes:
    def __init__(self, client: LMClient, traces: TraceStore):
        self.client = client
        self.traces = traces

    def plan(self, state: HarnessState) -> HarnessState:
        messages = [
            Msg(role="system", content="You are the planning node for a coding-agent harness."),
            Msg(
                role="user",
                content=(
                    "Return a concise numbered plan for this task. "
                    f"Do not use tools yet.\n\nTask: {state.task}"
                ),
            ),
        ]
        completion = self.client.complete(messages, max_tokens=300, temperature=0.1)
        state.plan = parse_numbered_plan(completion.content)
        state.history.append({"node": "plan", "content": completion.content})
        self.traces.event(
            state.run_id,
            "model_completion",
            {
                "model": completion.model,
                "provider": completion.provider,
                "latency_ms": completion.latency_ms,
                "content": completion.content,
                "plan": state.plan,
            },
            node="plan",
        )
        return state

    def act(self, state: HarnessState) -> HarnessState:
        messages = [
            Msg(
                role="system",
                content=(
                    "You are the act node for a coding-agent harness. "
                    "This first implementation has no tools. Produce a direct, useful response."
                ),
            ),
            Msg(role="user", content=f"Task: {state.task}\nPlan: {state.plan}"),
        ]
        completion = self.client.complete(messages, max_tokens=700, temperature=0.2)
        state.scratch["last_action"] = completion.content
        state.history.append({"node": "act", "content": completion.content})
        self.traces.event(
            state.run_id,
            "model_completion",
            {
                "model": completion.model,
                "provider": completion.provider,
                "latency_ms": completion.latency_ms,
                "content": completion.content,
            },
            node="act",
        )
        return state

    def observe(self, state: HarnessState) -> HarnessState:
        observation = state.scratch.get("last_action", "")
        state.history.append({"node": "observe", "content": observation})
        self.traces.event(
            state.run_id,
            "observation",
            {"content": observation},
            node="observe",
        )
        return state

    def reflect(self, state: HarnessState) -> HarnessState:
        last_observation = state.history[-1]["content"] if state.history else ""
        messages = [
            Msg(
                role="system",
                content="You are the reflection node. Decide whether the task is complete.",
            ),
            Msg(
                role="user",
                content=(
                    "Decide whether the task is complete. Reply with only done or continue.\n\n"
                    f"Task: {state.task}\nLast observation: {last_observation}"
                ),
            ),
        ]
        completion = self.client.complete(messages, max_tokens=20, temperature=0)
        decision = "done" if "done" in completion.content.lower() else "continue"
        state.status = decision
        self.traces.event(
            state.run_id,
            "reflection",
            {"decision": decision, "content": completion.content},
            node="reflect",
        )
        return state

    def done(self, state: HarnessState) -> HarnessState:
        state.status = "done"
        state.final_answer = state.scratch.get("last_action", "")
        self.traces.event(
            state.run_id,
            "final",
            {"final_answer": state.final_answer},
            node="done",
        )
        return state
