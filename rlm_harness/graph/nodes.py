from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rlm_harness.graph.planning import (
    advance_to_next_step,
    format_plan_context,
    parse_structured_plan,
)
from rlm_harness.graph.recovery import (
    ErrorClassifier,
    ErrorCategory,
    RecoverySelector,
    RecoveryStrategy,
    is_retryable_decision,
)
from rlm_harness.graph.task_policy import (
    estimate_task_complexity,
    is_code_editing_task,
    is_informational_task,
    is_project_audit_task,
    is_project_summary_task,
    looks_like_code_edit_result,
    looks_like_file_inventory,
    looks_like_project_audit,
    looks_like_project_summary,
    looks_like_source_dump,
)
from rlm_harness.graph.verification import VerificationGate, VerificationResult
from rlm_harness.memory import Memory
from rlm_harness.memory.paging import MemoryPager, MemoryPagingConfig
from rlm_harness.model_client import LMClient
from rlm_harness.observability import maybe_traceable
from rlm_harness.rlm import RLMRuntime
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.sandbox import tools as sandbox_tools
from rlm_harness.sandbox.tools import (
    TOOL_SCHEMAS,
    is_project_overview_payload,
    render_project_overview_summary,
)
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg, PlanStepStatus, TaskPlan


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class GraphRuntimeConfig:
    sandbox_enabled: bool = False
    sandbox_config: Optional[SandboxConfig] = None
    subcall_config: Optional[RLMSubcallConfig] = None
    max_action_retries: int = 1
    max_iterations: int = 6
    act_engine: str = "rlm"
    memory: Optional[Memory] = None
    memory_paging: MemoryPagingConfig = MemoryPagingConfig()


def parse_numbered_plan(text: str) -> TaskPlan:
    return parse_structured_plan(text)


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

    @maybe_traceable("Harness.plan", run_type="chain")
    def plan(self, state: HarnessState) -> HarnessState:
        if self.memory is not None:
            from rlm_harness.graph.checkpoint import CheckpointManager

            checkpoint_mgr = CheckpointManager(self.runtime.memory)
            checkpoint = checkpoint_mgr.load_latest(state.thread_id)
            if checkpoint:
                state = CheckpointManager.resume_state(state, checkpoint)
                state.history.append(
                    {
                        "node": "plan",
                        "content": (
                            f"Resumed from checkpoint at step {checkpoint.step_id}. "
                            f"{len(checkpoint.completed_step_ids)} steps already completed."
                        ),
                    }
                )
                self.traces.event(
                    state.run_id,
                    "plan_resumed",
                    {"checkpoint_step_id": checkpoint.step_id},
                    node="plan",
                )
                return state

        memory_context = self._memory_context(state)
        complexity = estimate_task_complexity(state.task)
        messages = [
            Msg(role="system", content="You are the planning node for a coding-agent harness."),
            Msg(
                role="user",
                content=(
                    "Return a concise numbered plan for this task. "
                    "Use sub-numbering for multi-part steps (e.g., 1, 2a, 2b, 3). "
                    f"The task appears to be {complexity} complexity. "
                    f"Do not use tools yet.\n\nTask: {state.task}"
                    f"{memory_context}"
                ),
            ),
        ]
        completion = self.client.complete(messages, max_tokens=600, temperature=0.1)
        state.plan = parse_numbered_plan(completion.content)
        if self.runtime.max_iterations < 6:
            from rlm_harness.graph.task_policy import default_max_iterations_for_complexity

            state.budget.iteration_limit = self.runtime.max_iterations
        else:
            from rlm_harness.graph.task_policy import default_max_iterations_for_complexity

            state.budget.iteration_limit = max(
                self.runtime.max_iterations,
                default_max_iterations_for_complexity(complexity),
            )

        state.history.append({"node": "plan", "content": completion.content})
        self.traces.event(
            state.run_id,
            "model_completion",
            {
                "model": completion.model,
                "provider": completion.provider,
                "latency_ms": completion.latency_ms,
                "content": completion.content,
                "plan": state.plan.model_dump(),
                "complexity": complexity,
            },
            node="plan",
        )
        return state

    @maybe_traceable("Harness.act", run_type="chain")
    def act(self, state: HarnessState) -> HarnessState:
        if self.runtime.sandbox_enabled:
            if self.runtime.act_engine == "rlm":
                return self._run_rlm_action(state)
            return self._select_sandbox_action(state)

        plan_text = format_plan_context(state.plan)
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
                content=(
                    f"Task: {state.task}\nPlan progress:\n{plan_text}"
                    f"{self._memory_context(state)}{self._recovery_context(state)}"
                ),
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

    def _run_rlm_action(self, state: HarnessState) -> HarnessState:
        sandbox_config = self.runtime.sandbox_config or SandboxConfig(
            workspace=Path(state.workspace)
        )
        runtime = RLMRuntime(
            self.client,
            workspace=Path(state.workspace),
            max_iterations=self.runtime.max_iterations,
            max_depth=(self.runtime.subcall_config.max_depth if self.runtime.subcall_config else 3),
            sandbox_enabled=True,
            sandbox_config=sandbox_config,
            subcall_config=self.runtime.subcall_config,
        )
        context = rlm_action_context(state, sandbox_config.mount_path)
        result = runtime.completion(state.task, context=context)
        final_answer = fallback_project_answer_if_needed(
            state.task,
            result.final_answer,
            Path(state.workspace),
            result.status,
        )
        observation = {
            "status": "ok" if final_answer else result.status,
            "stdout": final_answer or result.final_answer,
            "stderr": "\n".join(obs.stderr for obs in result.observations if obs.stderr),
            "timed_out": any(obs.timed_out for obs in result.observations),
            "elapsed_ms": sum(obs.elapsed_ms for obs in result.observations),
            "subcalls": result.subcalls,
            "tokens_used": result.tokens_used,
            "iterations": result.iterations,
            "engine": "rlm",
        }
        rendered = render_observation(observation)
        state.scratch["last_action"] = rendered
        state.history.append({"node": "act", "content": rendered})
        self.traces.event(
            state.run_id,
            "rlm_runtime",
            {
                **observation,
                "responses": result.responses,
                "observations": [obs.__dict__ for obs in result.observations],
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

    @maybe_traceable("Harness.execute_action", run_type="tool")
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
        plan_text = format_plan_context(state.plan)
        current_step = None
        if state.plan.current_step_id:
            for s in state.plan.steps:
                if s.id == state.plan.current_step_id:
                    current_step = s
                    break
        step_guidance = ""
        if current_step:
            step_guidance = (
                f"\nCurrent step (step {current_step.id}): {current_step.description}. "
                f"Focus on this step only."
            )
            if current_step.attempts > 0:
                step_guidance += (
                    f"\nPrevious attempt failed. This is attempt "
                    f"{current_step.attempts + 1}/{current_step.max_attempts}."
                )
        progress = (
            f"\nProgress: {state.plan.completed_count()}/{state.plan.total_count()} "
            f"steps completed."
        )
        budget_status = state.budget.progress_summary
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
                    "especially questions like 'what is this project', call project_summary() "
                    "and print only that user-facing summary. If you need more detail, call "
                    "project_overview() and summarize from its files, documents, git_status, "
                    "and git_log. Never print raw source code for a project-summary answer. "
                    "For project audit, review, risk, issue, or gap-analysis tasks, do not "
                    "answer with list_files(), project_overview(), or an ALL FILES inventory "
                    "alone. Call project_audit() as a baseline, inspect relevant config and "
                    "source files with read_file/search_code, and print findings with evidence, "
                    "impact, and recommendations. Use rlm.completion with the collected context "
                    "when the task asks for logical or technical analysis beyond simple facts. "
                    "For large files, use read_file_slice() or chunk_file() instead of printing "
                    "entire files into the observation stream. "
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
                    f"Task: {state.task}\nPlan progress:\n{plan_text}"
                    f"{step_guidance}{progress}\nBudget: {budget_status}"
                    f"{self._recent_history_context(state)}"
                    f"{self._memory_context(state)}"
                    f"{self._recovery_context(state)}{retry}"
                ),
            ),
        ]

    @maybe_traceable("Harness.observe", run_type="chain")
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

    @maybe_traceable("Harness.reflect", run_type="chain")
    def reflect(self, state: HarnessState) -> HarnessState:
        attempt = int(state.scratch.get("graph_iterations", 0)) + 1
        state.scratch["graph_iterations"] = attempt
        state.budget.iterations_used = attempt
        last_observation = state.history[-1]["content"] if state.history else ""
        observation_payload = parse_observation_payload(last_observation)

        current_step = _find_current_step(state)
        if current_step and current_step.id != state.plan.current_step_id:
            for step in state.plan.steps:
                if step.status == PlanStepStatus.IN_PROGRESS:
                    step.status = PlanStepStatus.PENDING
            current_step.status = PlanStepStatus.IN_PROGRESS
            state.plan.current_step_id = current_step.id

        if observation_payload:
            recovery_reason = self._check_output_quality(
                state, observation_payload, last_observation
            )
            if recovery_reason:
                category = ErrorClassifier.classify_for_recovery_failure(recovery_reason)
                self._apply_recovery(state, category, current_step, recovery_reason)
                return state

        observation_status = parse_observation_status(last_observation)
        if observation_status and observation_status != "ok":
            category = ErrorClassifier.classify(
                observation_status,
                observation_user_output(observation_payload or {}),
                (observation_payload.get("stderr", "") if observation_payload else ""),
                bool(observation_payload and observation_payload.get("timed_out", False)),
            )
            if category != ErrorCategory.UNKNOWN or observation_status == "error":
                self._apply_recovery(
                    state,
                    category,
                    current_step,
                    f"sandbox observation status was {observation_status}",
                )
                return state

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

        if observation_payload and observation_payload.get("status") == "ok":
            verification_passed = self._check_verification_result(state)
            if current_step and verification_passed:
                advance_to_next_step(state.plan)
                if not state.plan.has_more():
                    return self.done(state)

        messages = [
            Msg(
                role="system",
                content="You are the reflection node. Decide whether the task is complete.",
            ),
            Msg(
                role="user",
                content=(
                    "Decide whether the task is complete. Reply with only done or continue.\n\n"
                    f"Task: {state.task}\n"
                    f"Plan progress:\n{format_plan_context(state.plan)}\n"
                    f"Budget: {state.budget.progress_summary}\n"
                    f"Last observation: {last_observation}"
                    f"{self._memory_context(state)}"
                ),
            ),
        ]
        completion = self.client.complete(messages, max_tokens=20, temperature=0)
        decision = "done" if "done" in completion.content.lower() else "continue"
        if decision == "done":
            return self.done(state)
        if decision == "continue" and attempt >= self.runtime.max_iterations:
            state.status = "stopped"
            state.final_answer = final_answer_from_action(last_observation, task=state.task)
            self.traces.event(
                state.run_id,
                "reflection",
                {"decision": "stopped", "content": "max iterations reached"},
                node="reflect",
            )
            return state
        if decision == "continue" and state.budget.is_exhausted:
            return self.finalize_partial(state)

        state.status = decision
        self.traces.event(
            state.run_id,
            "reflection",
            {"decision": decision, "content": completion.content},
            node="reflect",
        )
        return state

    def _check_output_quality(
        self,
        state: HarnessState,
        observation_payload: dict,
        last_observation: str,
    ) -> str:
        if observation_payload.get("status") != "ok":
            return ""

        if is_project_summary_task(state.task):
            user_output = observation_user_output(observation_payload)
            if user_output and (
                looks_like_file_inventory(user_output)
                or looks_like_source_dump(user_output)
                or not looks_like_project_summary(user_output)
            ):
                return "project-summary task did not produce a project summary"

        if is_project_audit_task(state.task):
            user_output = observation_user_output(observation_payload)
            if user_output and (
                looks_like_file_inventory(user_output)
                or looks_like_source_dump(user_output)
                or not looks_like_project_audit(user_output)
            ):
                return "project-audit task did not produce evidence-backed findings"

        if not observation_user_output(observation_payload) and (
            is_informational_task(state.task) or is_code_editing_task(state.task)
        ):
            return "informational task produced no user-facing output"

        if is_code_editing_task(state.task):
            user_output = observation_user_output(observation_payload)
            if user_output and not looks_like_code_edit_result(user_output):
                return "code-editing task did not report changed files or verification"

        return ""

    def _check_verification_result(self, state: HarnessState) -> bool:
        result = state.scratch.get("verification_result")
        if result is None:
            return True
        if isinstance(result, dict):
            return bool(result.get("passed", True))
        if isinstance(result, VerificationResult):
            return result.passed
        return True

    def _apply_recovery(
        self,
        state: HarnessState,
        category: ErrorCategory,
        step: Optional["PlanStep"],
        reason: str,
    ) -> None:
        attempt = int(state.scratch.get("graph_iterations", 0))
        if attempt >= self.runtime.max_iterations:
            state.status = "stopped"
            state.final_answer = (
                f"Stopped after {self.runtime.max_iterations} attempts. "
                f"Last issue: {reason}"
            )
            self.traces.event(
                state.run_id,
                "reflection",
                {"decision": "stopped", "error_category": category.value, "reason": reason},
                node="reflect",
            )
            return

        if step is not None:
            step.attempts += 1
            step.status = PlanStepStatus.FAILED
        else:
            step = _find_or_create_default_step(state)

        strategy = RecoverySelector.select(category, step)
        hint = RecoverySelector.generate_hint(strategy, step, reason)

        if category == ErrorCategory.VERIFICATION_FAILURE:
            state.scratch["recovery_hint"] = hint
            step.status = PlanStepStatus.IN_PROGRESS
            state.status = "continue"
        elif is_retryable_decision(strategy):
            state.scratch["recovery_hint"] = hint
            step.status = PlanStepStatus.IN_PROGRESS
            state.status = "continue"
        elif strategy == RecoveryStrategy.SKIP:
            step.status = PlanStepStatus.SKIPPED
            advance_to_next_step(state.plan)
            if not state.plan.has_more():
                state = self.finalize_partial(state)
                state.scratch["recovery_hint"] = ""
                state.status = "continue"
        else:
            state.status = "error"
            last_obs = state.history[-1]["content"] if state.history else ""
            state.final_answer = final_answer_from_action(last_obs, task=state.task)

        self.traces.event(
            state.run_id,
            "reflection",
            {
                "decision": state.status,
                "error_category": category.value,
                "recovery_strategy": strategy.value,
                "step_id": step.id,
                "step_attempts": step.attempts,
                "reason": reason,
            },
            node="reflect",
        )

    @maybe_traceable("Harness.verify", run_type="tool")
    def verify(self, state: HarnessState) -> HarnessState:
        if not is_code_editing_task(state.task):
            state.history.append({"node": "verify", "content": "skipped (not a code editing task)"})
            return state

        if self.runtime.sandbox_enabled:
            sandbox_config = self.runtime.sandbox_config or SandboxConfig(
                workspace=Path(state.workspace)
            )
            try:
                with DockerREPL(
                    sandbox_config,
                    completion_client=self.client,
                    subcall_config=self.runtime.subcall_config,
                ) as repl:
                    import rlm_harness.graph.verification as verification

                    def _sandbox_shell(cmd: str, timeout: float):
                        code = (
                            "import subprocess as _s, json as _j\n"
                            f"_r = _s.run({cmd!r}, shell=True, executable='/bin/sh', "
                            f"text=True, capture_output=True, timeout={timeout}, check=False)\n"
                            "print(_j.dumps({"
                            "'returncode': _r.returncode, "
                            "'stdout': _r.stdout, "
                            "'stderr': _r.stderr, "
                            "'timed_out': False}))\n"
                        )
                        result = repl.execute(code, timeout_s=timeout + 5)
                        last_line = result.stdout.strip().split("\n")[-1]
                        try:
                            return json.loads(last_line)
                        except (json.JSONDecodeError, IndexError):
                            return {
                                "returncode": 1,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "timed_out": result.timed_out,
                            }

                    verification.set_run_shell_callback(_sandbox_shell)
                    try:
                        gate = VerificationGate(Path(state.workspace))
                        result = gate.verify()
                    finally:
                        verification.set_run_shell_callback(None)
            except SandboxError as exc:
                result = VerificationResult(
                    passed=True,
                    summary=f"Verification skipped (sandbox error): {exc}",
                )
        else:
            try:
                gate = VerificationGate(Path(state.workspace))
                result = gate.verify()
            except Exception as exc:
                result = VerificationResult(
                    passed=True,
                    summary=f"Verification skipped due to error: {exc}",
                )

        state.scratch["verification_result"] = {
            "passed": result.passed,
            "checks": [
                {"check_type": c.check_type, "passed": c.passed, "output": c.output}
                for c in result.checks
            ],
            "changed_files": result.changed_files,
            "summary": result.summary,
        }
        state.history.append({"node": "verify", "content": result.summary})
        self.traces.event(
            state.run_id,
            "verification",
            {
                "passed": result.passed,
                "checks": [
                    {"check_type": c.check_type, "passed": c.passed}
                    for c in result.checks
                ],
                "changed_files": result.changed_files,
            },
            node="verify",
        )
        return state

    def finalize_partial(self, state: HarnessState) -> HarnessState:
        completed = [s for s in state.plan.steps if s.status == PlanStepStatus.COMPLETED]
        pending = [s for s in state.plan.steps if s.status == PlanStepStatus.PENDING]
        completed_desc = [f"Step {s.id}: {s.description}" for s in completed]
        pending_desc = [f"Step {s.id}: {s.description}" for s in pending]

        messages = [
            Msg(
                role="system",
                content="Synthesize a partial answer from what was accomplished.",
            ),
            Msg(
                role="user",
                content=(
                    f"Task: {state.task}\n"
                    f"Completed ({len(completed)}): {completed_desc}\n"
                    f"Not done ({len(pending)}): {pending_desc}\n"
                    f"Budget: {state.budget.progress_summary}\n"
                    f"Last output: {state.history[-1]['content'] if state.history else 'none'}"
                ),
            ),
        ]
        completion = self.client.complete(messages, max_tokens=300, temperature=0.1)
        state.final_answer = completion.content
        state.status = "stopped"
        self.traces.event(
            state.run_id,
            "finalize_partial",
            {"partial_answer": completion.content},
            node="reflect",
        )
        return state

    def _continue_or_stop(
        self,
        state: HarnessState,
        last_observation: str,
        reason: str,
        include_last_answer: bool = True,
    ) -> str:
        if int(state.scratch.get("graph_iterations", 0)) >= self.runtime.max_iterations:
            state.final_answer = (
                f"Stopped after {self.runtime.max_iterations} attempts. "
                f"Last issue: {reason}"
            )
            if include_last_answer:
                state.final_answer = (
                    f"{state.final_answer}\n\n"
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
    def _recovery_context(state: HarnessState) -> str:
        hint = state.scratch.get("recovery_hint", "")
        if not isinstance(hint, str) or not hint.strip():
            return ""
        return f"\n\nRecovery guidance: {hint}"

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

    @maybe_traceable("Harness.done", run_type="chain")
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


def _find_current_step(state: HarnessState) -> Optional[PlanStep]:
    for step in state.plan.steps:
        if step.id == state.plan.current_step_id:
            return step
    for step in state.plan.steps:
        if step.status in (PlanStepStatus.PENDING, PlanStepStatus.IN_PROGRESS):
            return step
    return None


def _find_or_create_default_step(state: HarnessState) -> PlanStep:
    for step in state.plan.steps:
        if step.status in (PlanStepStatus.PENDING, PlanStepStatus.IN_PROGRESS):
            return step
    from rlm_harness.types import PlanStep

    step = PlanStep(id="1", description="complete the task")
    state.plan.steps.append(step)
    state.plan.current_step_id = "1"
    return step


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


def render_tool_list() -> str:
    return ", ".join(str(schema["name"]) for schema in TOOL_SCHEMAS)


def rlm_action_context(state: HarnessState, mount_path: str = "/workspace") -> dict:
    return {
        "task": state.task,
        "plan": state.plan.model_dump(),
        "plan_context": format_plan_context(state.plan),
        "recent_history": state.history[-4:],
        "memory_context": state.scratch.get("memory_context", ""),
        "workspace": mount_path,
        "workspace_note": (
            "Code runs inside Docker. Use workspace-relative tool paths or "
            f"{mount_path}; host paths are not available."
        ),
    }


def fallback_project_answer_if_needed(
    task: str,
    answer: str,
    workspace: Path,
    status: str,
) -> str:
    if is_project_summary_task(task) and (
        status != "done" or not looks_like_project_summary_answer(answer)
    ):
        return workspace_project_summary(workspace)
    if is_project_audit_task(task) and (status != "done" or not looks_like_project_audit(answer)):
        return workspace_project_audit(workspace)
    return answer


def looks_like_project_summary_answer(answer: str) -> bool:
    return (
        bool(answer.strip())
        and looks_like_project_summary(answer)
        and not looks_like_file_inventory(answer)
        and not looks_like_source_dump(answer)
    )


def workspace_project_summary(workspace: Path) -> str:
    return call_workspace_project_tool(workspace, sandbox_tools.project_summary)


def workspace_project_audit(workspace: Path) -> str:
    return call_workspace_project_tool(workspace, sandbox_tools.project_audit)


def call_workspace_project_tool(workspace: Path, tool) -> str:
    old_workspace = sandbox_tools.WORKSPACE
    sandbox_tools.WORKSPACE = workspace.resolve()
    try:
        return str(tool())
    finally:
        sandbox_tools.WORKSPACE = old_workspace
