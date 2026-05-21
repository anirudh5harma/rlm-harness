from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rlm_harness.memory import Memory
from rlm_harness.memory.paging import MemoryPager, MemoryPagingConfig
from rlm_harness.model_client import LMClient
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.sandbox.tools import TOOL_SCHEMAS
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
    max_iterations: int = 3
    memory: Optional[Memory] = None
    memory_paging: MemoryPagingConfig = MemoryPagingConfig()


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
    payload = parse_action_payload(text)
    if payload.get("type") != "python":
        raise ActionParseError("action type must be 'python'")

    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ActionParseError("action code must be a non-empty string")
    return code


def parse_action_payload(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = parse_embedded_json_object(text, exc)

    if not isinstance(payload, dict):
        raise ActionParseError("action must be a JSON object")
    return payload


def parse_embedded_json_object(text: str, original: json.JSONDecodeError) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1]).strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ActionParseError("action was not valid JSON") from original


def render_observation(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2)


def parse_observation_payload(observation: str) -> Optional[dict]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def observation_user_output(payload: dict) -> str:
    parts = []
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    if isinstance(stdout, str) and stdout.strip():
        parts.append(stdout.strip())
    if isinstance(stderr, str) and stderr.strip():
        parts.append(stderr.strip())
    return "\n".join(parts).strip()


def is_informational_task(task: str) -> bool:
    terms = (
        "summarize",
        "summary",
        "explain",
        "describe",
        "list",
        "report",
        "inspect",
        "analyze",
        "analyse",
    )
    lowered = task.lower()
    return any(term in lowered for term in terms)


def is_retryable_observation(status: Optional[str]) -> bool:
    return status in {"error", "tool_error", "timeout"}


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
        self.memory = (
            MemoryPager(
                self.runtime.memory,
                self.client,
                self.traces,
                self.runtime.memory_paging,
            )
            if self.runtime.memory is not None
            else None
        )

    def plan(self, state: HarnessState) -> HarnessState:
        memory_context = self._memory_context(state)
        messages = [
            Msg(role="system", content="You are the planning node for a coding-agent harness."),
            Msg(
                role="user",
                content=(
                    "Return a concise numbered plan for this task. "
                    f"Do not use tools yet.\n\nTask: {state.task}"
                    f"{memory_context}"
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
            return self._select_sandbox_action(state)

        messages = [
            Msg(
                role="system",
                content=(
                    "You are the act node for a coding-agent harness. "
                    "This first implementation has no tools. Produce a direct, useful response."
                ),
            ),
            Msg(
                role="user",
                content=f"Task: {state.task}\nPlan: {state.plan}{self._memory_context(state)}",
            ),
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

    def _select_sandbox_action(self, state: HarnessState) -> HarnessState:
        action_content = ""
        parse_error = ""

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
                state.scratch["pending_action_code"] = code
                state.scratch["pending_action_content"] = completion.content
                state.history.append({"node": "act", "content": completion.content})
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

    def execute_action(self, state: HarnessState) -> HarnessState:
        code = state.scratch.pop("pending_action_code", None)
        if not code:
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
        state.history.append({"node": "execute_action", "content": rendered})
        self.traces.event(
            state.run_id,
            "sandbox_execution",
            observation,
            node="execute_action",
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
                    "The Python code must perform the requested work immediately at top level; "
                    "do not only define unused functions, classes, or data structures. "
                    "Only call the listed tool functions; do not invent helper APIs. "
                    "README.md is optional; never assume it exists. For project summaries, "
                    "call project_overview() first and summarize from its files, documents, "
                    "git_status, and git_log. "
                    "When calling read_file, write_file, search_code, or git_diff, pass literal "
                    "non-empty workspace-relative string paths such as '.', 'pyproject.toml', "
                    "or 'src/app.py'. If you do not know the path, discover it first with "
                    "project_overview(), list_files(), or search_code. "
                    "For informational tasks, inspect the workspace and print the final "
                    "user-facing answer to stdout. For code-editing tasks, make the change, "
                    "run focused verification when possible, and print the changed files and "
                    "verification result. "
                    "Use the provided Python tool functions to inspect or manipulate "
                    "the workspace: "
                    f"{render_tool_list()}. "
                    "You may call rlm.completion(query, context, depth_hint=-1) for a recursive "
                    "host-mediated model sub-call."
                ),
            ),
            Msg(
                role="user",
                content=(
                    "Return only valid JSON for this action.\n"
                    f"Task: {state.task}\nPlan: {state.plan}"
                    f"{self._recent_history_context(state)}"
                    f"{self._memory_context(state)}{retry}"
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
        attempt = int(state.scratch.get("graph_iterations", 0)) + 1
        state.scratch["graph_iterations"] = attempt
        last_observation = state.history[-1]["content"] if state.history else ""
        observation_payload = parse_observation_payload(last_observation)
        if (
            observation_payload
            and observation_payload.get("status") == "ok"
            and not observation_user_output(observation_payload)
            and is_informational_task(state.task)
        ):
            state.status = self._continue_or_stop(
                state,
                last_observation,
                "informational task produced no user-facing output",
            )
            self.traces.event(
                state.run_id,
                "reflection",
                {
                    "decision": state.status,
                    "content": "informational task produced no user-facing output",
                },
                node="reflect",
            )
            return state

        observation_status = parse_observation_status(last_observation)
        if is_retryable_observation(observation_status):
            message = f"sandbox observation status was {observation_status}"
            state.status = self._continue_or_stop(state, last_observation, message)
            self.traces.event(
                state.run_id,
                "reflection",
                {"decision": state.status, "content": message},
                node="reflect",
            )
            return state

        if observation_status and observation_status != "ok":
            state.status = "error"
            state.final_answer = final_answer_from_action(last_observation, task=state.task)
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
                    f"{self._memory_context(state)}"
                ),
            ),
        ]
        completion = self.client.complete(messages, max_tokens=20, temperature=0)
        decision = "done" if "done" in completion.content.lower() else "continue"
        if decision == "continue" and attempt >= self.runtime.max_iterations:
            decision = "stopped"
            state.final_answer = final_answer_from_action(last_observation, task=state.task)
        state.status = decision
        self.traces.event(
            state.run_id,
            "reflection",
            {"decision": decision, "content": completion.content},
            node="reflect",
        )
        return state

    def _continue_or_stop(
        self,
        state: HarnessState,
        last_observation: str,
        reason: str,
    ) -> str:
        if int(state.scratch.get("graph_iterations", 0)) >= self.runtime.max_iterations:
            state.final_answer = (
                f"Stopped after {self.runtime.max_iterations} attempts. "
                f"Last issue: {reason}\n\n"
                f"{final_answer_from_action(last_observation, task=state.task)}"
            )
            return "stopped"
        return "continue"

    def memory_read(self, state: HarnessState) -> HarnessState:
        if self.memory is None:
            return state
        return self.memory.hydrate(state)

    def memory_write(self, state: HarnessState) -> HarnessState:
        if self.memory is None:
            return state
        return self.memory.persist_new_history(state)

    @staticmethod
    def _memory_context(state: HarnessState) -> str:
        context = state.scratch.get("memory_context", "")
        if not isinstance(context, str) or not context.strip():
            return ""
        return f"\n\nMemory context:\n{context}"

    @staticmethod
    def _recent_history_context(state: HarnessState) -> str:
        if not state.history:
            return ""
        recent = state.history[-4:]
        lines = []
        for item in recent:
            node = item.get("node", "unknown")
            content = str(item.get("content", "")).strip()
            if len(content) > 1800:
                content = content[:1800] + "\n..."
            lines.append(f"{node}: {content}")
        return "\n\nRecent graph history:\n" + "\n\n".join(lines)

    def done(self, state: HarnessState) -> HarnessState:
        state.status = "done"
        state.final_answer = final_answer_from_action(
            state.scratch.get("last_action", ""),
            task=state.task,
        )
        self.traces.event(
            state.run_id,
            "final",
            {"final_answer": state.final_answer},
            node="done",
        )
        return state


def parse_observation_status(observation: str) -> Optional[str]:
    payload = parse_observation_payload(observation)
    if payload is None:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def final_answer_from_action(action: str, task: str = "") -> str:
    payload = parse_observation_payload(action)
    if payload is None:
        return normalize_user_output(action, task=task)

    output = observation_user_output(payload)
    if output:
        return normalize_user_output(output, task=task)

    status = payload.get("status")
    if status == "ok":
        return (
            "The sandbox action completed successfully, but it did not produce a "
            "user-facing response."
        )
    if isinstance(status, str):
        return f"The sandbox action finished with status {status}, but did not produce output."
    return "The sandbox action finished without producing a user-facing response."


def normalize_user_output(output: str, task: str = "") -> str:
    structured = parse_structured_output(output)
    if structured is None:
        return output
    if isinstance(structured, dict) and is_project_overview_payload(structured):
        return render_project_overview_summary(structured)
    if is_informational_task(task):
        return json.dumps(structured, indent=2, sort_keys=True)
    return output


def parse_structured_output(output: str):
    stripped = output.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None


def is_project_overview_payload(payload: dict) -> bool:
    return isinstance(payload.get("files"), list) and isinstance(payload.get("documents"), list)


def render_project_overview_summary(payload: dict) -> str:
    files = [path for path in payload.get("files", []) if isinstance(path, str)]
    documents = [doc for doc in payload.get("documents", []) if isinstance(doc, dict)]
    doc_paths = [str(doc.get("path")) for doc in documents if doc.get("path")]
    package = package_json_from_documents(documents)
    scripts = package.get("scripts", {}) if package else {}
    dependencies = package_dependencies(package)

    sections = ["Project Summary"]
    sections.append(f"Files inspected: {len(files)}")
    if doc_paths:
        sections.append("Key config/docs: " + ", ".join(doc_paths[:8]))

    stack = infer_project_stack(files, documents, dependencies)
    if stack:
        sections.append("Tech stack: " + ", ".join(stack))

    architecture = infer_project_architecture(files)
    if architecture:
        sections.append("Architecture: " + architecture)

    commands = render_scripts(scripts)
    if commands:
        sections.append("Useful commands:\n" + commands)

    git_status = str(payload.get("git_status") or "").strip()
    if git_status:
        sections.append("Working tree:\n" + git_status)

    git_log = str(payload.get("git_log") or "").strip()
    if git_log:
        sections.append("Recent commits:\n" + "\n".join(git_log.splitlines()[:5]))

    notable_files = notable_source_files(files)
    if notable_files:
        sections.append(
            "Notable source files:\n" + "\n".join(f"- {path}" for path in notable_files)
        )

    return "\n\n".join(sections)


def package_json_from_documents(documents: list[dict]) -> dict:
    for doc in documents:
        if doc.get("path") == "package.json" and isinstance(doc.get("content"), str):
            try:
                payload = json.loads(str(doc["content"]))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def package_dependencies(package: dict) -> set[str]:
    names: set[str] = set()
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        values = package.get(section)
        if isinstance(values, dict):
            names.update(str(name) for name in values)
    return names


def infer_project_stack(
    files: list[str],
    documents: list[dict],
    dependencies: set[str],
) -> list[str]:
    stack = []
    if "package.json" in {doc.get("path") for doc in documents}:
        stack.append("Node.js")
    if "typescript" in dependencies or any(path.endswith((".ts", ".tsx")) for path in files):
        stack.append("TypeScript")
    if "react" in dependencies:
        stack.append("React")
    if "@tanstack/react-start" in dependencies:
        stack.append("TanStack Start")
    if "@tanstack/react-router" in dependencies:
        stack.append("TanStack Router")
    if "vite" in dependencies or "vite.config.ts" in files:
        stack.append("Vite")
    if "@tailwindcss/vite" in dependencies or "tailwindcss" in dependencies:
        stack.append("Tailwind CSS")
    if any(name.startswith("@radix-ui/") for name in dependencies):
        stack.append("Radix UI")
    if "pyproject.toml" in files:
        stack.append("Python")
    return dedupe(stack)


def infer_project_architecture(files: list[str]) -> str:
    details = []
    if any(path.startswith("src/routes/") for path in files):
        details.append("route modules under src/routes")
    if any(path.startswith("src/components/ui/") for path in files):
        details.append("shared UI primitives under src/components/ui")
    if "src/router.tsx" in files:
        details.append("router setup in src/router.tsx")
    if "src/styles.css" in files:
        details.append("global styles in src/styles.css")
    if any(path.startswith("public/") for path in files):
        details.append("static assets under public")
    return "; ".join(details)


def render_scripts(scripts: dict) -> str:
    if not isinstance(scripts, dict):
        return ""
    preferred = ["dev", "build", "preview", "lint", "test"]
    lines = []
    for name in preferred:
        command = scripts.get(name)
        if isinstance(command, str):
            lines.append(f"- npm run {name}: {command}")
    return "\n".join(lines)


def notable_source_files(files: list[str]) -> list[str]:
    preferred = [
        "package.json",
        "pyproject.toml",
        "src/routes/index.tsx",
        "src/routes/__root.tsx",
        "src/router.tsx",
        "src/content.ts",
        "src/styles.css",
        "vite.config.ts",
        "tsconfig.json",
    ]
    return [path for path in preferred if path in files][:8]


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def render_tool_list() -> str:
    return ", ".join(str(schema["name"]) for schema in TOOL_SCHEMAS)
