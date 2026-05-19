from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class GraphRuntimeConfig:
    sandbox_enabled: bool = False
    sandbox_config: Optional[SandboxConfig] = None
    subcall_config: Optional[RLMSubcallConfig] = None
    max_action_retries: int = 1


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


def parse_python_action(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError("action was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ActionParseError("action must be a JSON object")
    if payload.get("type") != "python":
        raise ActionParseError("action type must be 'python'")

    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ActionParseError("action code must be a non-empty string")
    return code


def render_observation(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2)


class Nodes:
    def __init__(
        self,
        client: LMClient,
        traces: TraceStore,
        runtime: Optional[GraphRuntimeConfig] = None,
    ):
        self.client = client
        self.traces = traces
        self.runtime = runtime or GraphRuntimeConfig()

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
        if self.runtime.sandbox_enabled:
            return self._act_with_sandbox(state)

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

    def _act_with_sandbox(self, state: HarnessState) -> HarnessState:
        code = ""
        action_content = ""
        parse_error = ""
        completion = None

        for attempt in range(self.runtime.max_action_retries + 1):
            messages = self._action_messages(state, parse_error=parse_error)
            completion = self.client.complete(messages, max_tokens=900, temperature=0.1)
            action_content = completion.content
            self.traces.event(
                state.run_id,
                "model_completion",
                {
                    "model": completion.model,
                    "provider": completion.provider,
                    "latency_ms": completion.latency_ms,
                    "content": completion.content,
                    "attempt": attempt,
                },
                node="act",
            )
            try:
                code = parse_python_action(completion.content)
                break
            except ActionParseError as exc:
                parse_error = str(exc)
                self.traces.event(
                    state.run_id,
                    "action_parse_error",
                    {"message": parse_error, "content": completion.content, "attempt": attempt},
                    node="act",
                )
        else:
            observation = {
                "status": "action_parse_error",
                "stdout": "",
                "stderr": parse_error,
                "action": action_content,
            }
            rendered = render_observation(observation)
            state.scratch["last_action"] = rendered
            state.history.append({"node": "act", "content": rendered})
            return state

        sandbox_config = self.runtime.sandbox_config or SandboxConfig(
            workspace=Path(state.workspace)
        )

        try:
            with DockerREPL(
                sandbox_config,
                completion_client=self.client,
                subcall_config=self.runtime.subcall_config,
            ) as repl:
                result = repl.execute(code, timeout_s=sandbox_config.default_timeout_s)
            observation = {
                "status": result.status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "elapsed_ms": result.elapsed_ms,
                "subcalls": result.subcalls,
                "tokens_used": result.tokens_used,
                "code": code,
            }
        except SandboxError as exc:
            observation = {
                "status": "sandbox_error",
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "elapsed_ms": 0,
                "subcalls": 0,
                "tokens_used": 0,
                "code": code,
            }

        rendered = render_observation(observation)
        state.scratch["last_action"] = rendered
        state.history.append({"node": "act", "content": rendered})
        self.traces.event(
            state.run_id,
            "sandbox_execution",
            observation,
            node="act",
        )
        return state

    def _action_messages(self, state: HarnessState, parse_error: str = "") -> list[Msg]:
        retry = ""
        if parse_error:
            retry = (
                "\n\nYour previous response could not be parsed: "
                f"{parse_error}. Return only valid JSON now."
            )
        return [
            Msg(
                role="system",
                content=(
                    "You are the act node for a coding-agent harness. "
                    "Return exactly one JSON object and no markdown. "
                    'The schema is: {"type":"python","code":"..."} . '
                    "The code runs in a Docker sandbox with the workspace mounted at /workspace. "
                    "Use Python to inspect or manipulate the workspace. "
                    "You may call rlm.completion(query, context, depth_hint=-1) for a recursive "
                    "host-mediated model sub-call."
                ),
            ),
            Msg(
                role="user",
                content=(
                    "Return only valid JSON for this action.\n"
                    f"Task: {state.task}\nPlan: {state.plan}{retry}"
                ),
            ),
        ]

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
        observation_status = parse_observation_status(last_observation)
        if observation_status and observation_status != "ok":
            state.status = "error"
            state.final_answer = last_observation
            self.traces.event(
                state.run_id,
                "reflection",
                {
                    "decision": "error",
                    "content": f"sandbox observation status was {observation_status}",
                },
                node="reflect",
            )
            return state

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


def parse_observation_status(observation: str) -> Optional[str]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None
