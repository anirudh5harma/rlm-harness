from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional, TextIO

from rlm_harness.actions import CompleteTaskAction, CompletionStatus
from rlm_harness.cli_catalog import (
    ANSI_CYAN,
    should_use_color,
    stream_is_tty,
    style_text,
    terminal_flag,
)
from rlm_harness.config import default_provider
from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import GraphRuntimeConfig, Nodes
from rlm_harness.kernel import AutonomyMode, CompletionEvent, RunStartedEvent
from rlm_harness.mcp_config import MCP_CONFIG_PATH, MCPConfigStore, mcp_workflow_context
from rlm_harness.memory import Memory, MemoryError, MemoryPagingConfig
from rlm_harness.model_client import LMClientError
from rlm_harness.providers import normalize_provider
from rlm_harness.runtime_cli import build_client
from rlm_harness.sandbox import RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, TaskPlan

try:
    from rich.console import Console as RichConsole
    from rich.status import Status as RichStatus
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    RichConsole = None
    RichStatus = None

PERMISSION_MODE_ALIASES = {
    AutonomyMode.ASK.value: AutonomyMode.ASK.value,
    AutonomyMode.PLAN.value: AutonomyMode.PLAN.value,
    AutonomyMode.PROPOSE.value: AutonomyMode.PROPOSE.value,
    AutonomyMode.SANDBOX.value: AutonomyMode.SANDBOX.value,
    AutonomyMode.TRUSTED.value: AutonomyMode.TRUSTED.value,
    "standard": AutonomyMode.SANDBOX.value,
    "auto-accept": AutonomyMode.TRUSTED.value,
}

GRAPH_NODE_MARKERS = {
    "memory_read": ("memory", "loading relevant memory"),
    "plan": ("plan", "building the run plan"),
    "act": ("act", "choosing the next step"),
    "execute_action": ("work", "using workspace tools"),
    "verify": ("check", "running focused verification"),
    "observe": ("read", "reviewing tool output"),
    "reflect": ("think", "deciding whether to continue"),
    "done": ("final", "preparing the response"),
    "learn": ("learn", "saving useful preferences"),
}


def build_runtime(
    args: argparse.Namespace,
    workspace: Path,
    memory: Optional[Memory],
    profile_memory: Optional[Memory],
    task: str = "",
) -> GraphRuntimeConfig:
    return GraphRuntimeConfig(
        sandbox_enabled=not args.no_sandbox,
        sandbox_config=SandboxConfig(
            image=args.sandbox_image,
            workspace=workspace,
            memory=args.sandbox_memory,
            cpus=args.sandbox_cpus,
            default_timeout_s=args.sandbox_timeout,
        ),
        subcall_config=RLMSubcallConfig(
            max_depth=args.max_depth,
            max_subcalls=args.max_subcalls,
            token_budget=args.token_budget,
            max_tokens=args.subcall_max_tokens,
        ),
        max_action_retries=args.max_action_retries,
        max_iterations=args.max_iterations,
        act_engine=args.act_engine,
        autonomy=AutonomyMode(args.autonomy),
        memory=memory,
        profile_memory=profile_memory,
        mcp_config_path=Path(getattr(args, "mcp_config", str(MCP_CONFIG_PATH))),
        auto_style_scan=not getattr(args, "no_style_scan", False),
        style_scan_max_files=getattr(args, "style_scan_max_files", 400),
        mcp_context=(
            ""
            if getattr(args, "no_mcp", False)
            else mcp_workflow_context(
                task or getattr(args, "task", ""),
                MCPConfigStore(Path(getattr(args, "mcp_config", str(MCP_CONFIG_PATH)))),
            )
        ),
        memory_paging=MemoryPagingConfig(
            max_history_tokens=args.max_history_tokens,
            preserve_recent_steps=args.preserve_recent_steps,
            recall_limit=args.recall_limit,
            archival_limit=args.archival_limit,
            summary_max_tokens=args.summary_max_tokens,
        ),
    )


def should_emit_run_markers(args: argparse.Namespace, stream: TextIO) -> bool:
    if (
        getattr(args, "json_output", False)
        or getattr(args, "quiet", False)
        or getattr(args, "stream", False)
    ):
        return False
    progress = terminal_flag(os.environ.get("HARNESS_PROGRESS"))
    if progress is not None:
        return progress
    return stream_is_tty(stream)


def compact_marker_text(text: str, max_chars: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3].rstrip()}..."


def run_command_label(task: str) -> str:
    return f"harness run {shlex.quote(compact_marker_text(task))}"


class RunConsole:
    def __init__(self, args: argparse.Namespace, stream: Optional[TextIO] = None):
        self.stream = stream or sys.stderr
        self.enabled = should_emit_run_markers(args, self.stream)
        self.color = should_use_color(self.stream)
        self.memory_enabled = not getattr(args, "no_memory", False)
        self._rich_status: Any = None
        self._line_active = False
        self._use_rich = (
            self.enabled
            and stream_is_tty(self.stream)
            and RichConsole is not None
            and RichStatus is not None
        )

    def start(self, args: argparse.Namespace, task: str, workspace: Path) -> None:
        mode = [
            f"provider={args.provider}",
            f"model={args.model}",
            f"sandbox={'off' if args.no_sandbox else 'on'}",
            f"mode={args.autonomy}",
        ]
        self.update(f"{run_command_label(task)} • {workspace.name} • {', '.join(mode)}")

    def marker(
        self,
        label: str,
        message: str,
        *,
        important: bool = False,
        label_color: str = ANSI_CYAN,
    ) -> None:
        self.update(f"{label}: {message}", important=important, label_color=label_color)

    def update(
        self,
        message: str,
        *,
        important: bool = False,
        label_color: str = ANSI_CYAN,
    ) -> None:
        if not self.enabled:
            return
        rendered = style_text(message, label_color if important else ANSI_CYAN, self.color)
        if self._use_rich:
            if self._rich_status is None:
                console = RichConsole(file=self.stream, force_terminal=self.color)
                self._rich_status = RichStatus(
                    rendered,
                    console=console,
                    spinner="dots",
                    spinner_style="cyan",
                )
                self._rich_status.start()
                return
            self._rich_status.update(rendered)
            return

        self.stream.write("\r\033[K" + rendered)
        self.stream.flush()
        self._line_active = True

    def graph_update(self, update: object) -> None:
        for node in graph_update_nodes(update):
            if not self.memory_enabled and node in {"memory_read", "learn"}:
                continue
            marker = GRAPH_NODE_MARKERS.get(node)
            if marker is None:
                continue
            label, message = marker
            self.marker(label, message)

    def finish(self, status: str, run_id: str) -> None:
        self.clear()

    def error(self, message: str) -> None:
        self.clear()

    def clear(self) -> None:
        if not self.enabled:
            return
        if self._rich_status is not None:
            self._rich_status.stop()
            self._rich_status = None
            return
        if self._line_active:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._line_active = False


def apply_permission_aliases(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "plan_only", False):
        args.autonomy = AutonomyMode.PLAN.value
    permission_mode = getattr(args, "permission_mode", None)
    if permission_mode:
        args.autonomy = PERMISSION_MODE_ALIASES[permission_mode]
    if (
        getattr(args, "auto_accept", False)
        or getattr(args, "trust", False)
        or getattr(args, "yolo", False)
        or getattr(args, "dangerously_skip_permissions", False)
    ):
        args.autonomy = AutonomyMode.TRUSTED.value
    return args


def run_preflight_failure(args: argparse.Namespace) -> Optional[dict[str, object]]:
    provider = normalize_provider(str(getattr(args, "provider", default_provider())))
    if provider != "stub":
        return None
    if getattr(args, "provider_explicit", False) or getattr(args, "skip_onboarding", False):
        return None
    return {
        "error": "provider_not_configured",
        "message": (
            "Harness is still using the stub provider, which is for tests and smoke checks. "
            "Configure a real provider before running coding tasks."
        ),
        "next": [
            "Run `harness init --provider openrouter --api-key <key>`.",
            "Or run `harness /provider <name> --api-key <key>` and `harness /model <model>`.",
            "For an intentional stub smoke test, pass `--provider stub`.",
        ],
    }


def emit_preflight_failure(args: argparse.Namespace, failure: dict[str, object]) -> None:
    if getattr(args, "json_output", False):
        print(json.dumps(failure, sort_keys=True))
        return
    print(str(failure["message"]), file=sys.stderr)
    next_actions = failure.get("next")
    if isinstance(next_actions, list):
        print("next", file=sys.stderr)
        for action in next_actions:
            print(f"  {action}", file=sys.stderr)


def run_task(
    args: argparse.Namespace,
    task: str,
    thread_id: Optional[str],
    workspace: Path,
) -> int:
    preflight = run_preflight_failure(args)
    if preflight:
        emit_preflight_failure(args, preflight)
        return 1

    console = RunConsole(args)
    traces = TraceStore(Path(args.trace_db))
    run_id = traces.start_run(task, str(workspace), thread_id=thread_id)
    console.start(args, task, workspace)
    state = HarnessState(
        task=task,
        workspace=str(workspace),
        thread_id=thread_id or run_id,
        run_id=run_id,
        token_budget=args.token_budget,
    )
    record_run_started_event(args, traces, run_id, task, workspace, thread_id)

    try:
        with ExitStack() as stack:
            console.marker(
                "setup",
                "loading memory and profile" if not args.no_memory else "memory disabled",
            )
            memory = (
                None
                if args.no_memory
                else stack.enter_context(Memory(Path(args.memory_db)))
            )
            profile_memory = (
                None
                if args.no_memory
                else stack.enter_context(Memory(Path(args.profile_db)))
            )
            runtime = build_runtime(args, workspace, memory, profile_memory, task)
            console.marker("graph", "preparing runtime")
            graph = build_graph(
                Nodes(build_client(args), traces, runtime),
                backend=args.graph_backend,
                checkpoint_path=checkpoint_path(args),
            )
            if args.stream:
                final_state = run_streaming_graph(graph, state)
            elif console.enabled and hasattr(graph, "stream"):
                final_state = run_marked_graph(graph, state, console)
            else:
                console.marker("agent", "working through the request")
                final_state = graph.invoke(state)
            close_graph(graph)
        traces.finish_run(run_id, final_state.status)
    except (LMClientError, MemoryError, SandboxError) as exc:
        console.error(str(exc))
        traces.event(run_id, "error", {"message": str(exc)}, node="cli")
        traces.finish_run(run_id, "error")
        emit_run_output(args, None, traces, run_id, error=str(exc))
        return 1

    console.finish(final_state.status, run_id)
    emit_run_output(args, final_state, traces, run_id)
    return 0 if final_state.status == "done" else 1


def run_plan_task(
    args: argparse.Namespace,
    task: str,
    thread_id: Optional[str],
    workspace: Path,
) -> int:
    preflight = run_preflight_failure(args)
    if preflight:
        emit_preflight_failure(args, preflight)
        return 1

    console = RunConsole(args)
    traces = TraceStore(Path(args.trace_db))
    run_id = traces.start_run(task, str(workspace), thread_id=thread_id)
    console.start(args, task, workspace)
    state = HarnessState(
        task=task,
        workspace=str(workspace),
        thread_id=thread_id or run_id,
        run_id=run_id,
        token_budget=args.token_budget,
    )
    record_run_started_event(args, traces, run_id, task, workspace, thread_id)

    try:
        with ExitStack() as stack:
            console.marker(
                "setup",
                "loading memory and profile" if not args.no_memory else "memory disabled",
            )
            memory = (
                None
                if args.no_memory
                else stack.enter_context(Memory(Path(args.memory_db)))
            )
            profile_memory = (
                None
                if args.no_memory
                else stack.enter_context(Memory(Path(args.profile_db)))
            )
            runtime = build_runtime(args, workspace, memory, profile_memory, task)
            nodes = Nodes(build_client(args), traces, runtime)
            console.marker("plan", "building the implementation plan")
            final_state = nodes.memory_read(state)
            final_state = nodes.plan(final_state)
            final_state = nodes.memory_write(final_state)
            final_state = finalize_plan_only(final_state, traces)
            final_state = nodes.learn(final_state)
        traces.finish_run(run_id, final_state.status)
    except (LMClientError, MemoryError, SandboxError) as exc:
        console.error(str(exc))
        traces.event(run_id, "error", {"message": str(exc)}, node="cli")
        traces.finish_run(run_id, "error")
        emit_run_output(args, None, traces, run_id, error=str(exc))
        return 1

    console.finish(final_state.status, run_id)
    emit_run_output(args, final_state, traces, run_id)
    return 0 if final_state.status == "done" else 1


def record_run_started_event(
    args: argparse.Namespace,
    traces: TraceStore,
    run_id: str,
    task: str,
    workspace: Path,
    thread_id: Optional[str],
) -> None:
    traces.record_typed_event(
        RunStartedEvent(
            run_id=run_id,
            sequence=traces.next_sequence(run_id),
            node="cli",
            task=task,
            workspace=str(workspace),
            thread_id=thread_id or run_id,
            payload={
                "sandbox_enabled": not args.no_sandbox,
                "memory_enabled": not args.no_memory,
                "autonomy": args.autonomy,
            },
        )
    )


def finalize_plan_only(state: HarnessState, traces: TraceStore) -> HarnessState:
    state.status = "done"
    state.final_answer = render_user_plan(state.plan)
    traces.event(
        state.run_id,
        "final",
        {"final_answer": state.final_answer},
        node="plan",
    )
    completion_action = CompleteTaskAction(
        summary=state.final_answer,
        status=CompletionStatus.SUCCESS,
        verification="plan only; no workspace actions executed",
    )
    traces.record_typed_event(
        CompletionEvent.from_action(
            run_id=state.run_id,
            sequence=traces.next_sequence(state.run_id),
            node="plan",
            action=completion_action,
        )
    )
    return state


def render_user_plan(plan: TaskPlan) -> str:
    if not plan.steps:
        return "Implementation Plan\nNo plan steps were produced."
    lines = ["Implementation Plan"]
    for step in plan.steps:
        indent = "  " if step.parent_id else ""
        lines.append(f"{indent}{step.id}. {step.description}")
    return "\n".join(lines)


def emit_run_output(
    args: argparse.Namespace,
    final_state: Optional[HarnessState],
    traces: TraceStore,
    run_id: str,
    error: Optional[str] = None,
) -> None:
    if args.json_output:
        payload = run_output_payload(args, traces, run_id, final_state)
        if error:
            payload["error"] = error
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return
    if final_state and final_state.final_answer:
        print(final_state.final_answer)


def run_output_payload(
    args: argparse.Namespace,
    traces: TraceStore,
    run_id: str,
    final_state: Optional[HarnessState],
) -> dict:
    summary = traces.run_summary(run_id)
    response = (
        final_state.final_answer
        if final_state is not None and final_state.final_answer is not None
        else summary.get("final_answer")
    )
    return {
        "status": summary["status"],
        "response": response,
        "final_answer": response,
        "run_id": summary["run_id"],
        "thread_id": summary["thread_id"],
        "task": summary["task"],
        "workspace": summary["workspace"],
        "event_count": summary["event_count"],
        "trace_db": args.trace_db,
    }


def checkpoint_path(args: argparse.Namespace) -> Optional[Path]:
    if args.no_checkpoint or args.graph_backend == "simple":
        return None
    return Path(args.checkpoint_db)


def run_streaming_graph(graph, state: HarnessState) -> HarnessState:
    if not hasattr(graph, "stream"):
        return graph.invoke(state)
    final_state = None
    for update in graph.stream(state):
        print(json.dumps({"graph_update": update}, default=str, sort_keys=True))
        final_state = graph_state_from_update(update) or final_state
    return final_state or graph.invoke(state)


def run_marked_graph(graph, state: HarnessState, console: RunConsole) -> HarnessState:
    if not hasattr(graph, "stream"):
        console.marker("agent", "working through the request")
        return graph.invoke(state)
    final_state = None
    for update in graph.stream(state):
        console.graph_update(update)
        final_state = graph_state_from_update(update) or final_state
    return final_state or graph.invoke(state)


def graph_update_nodes(update: object) -> list[str]:
    if not isinstance(update, dict):
        return []
    return [node for node in update if isinstance(node, str)]


def graph_state_from_update(update: object) -> Optional[HarnessState]:
    if isinstance(update, HarnessState):
        return update
    if not isinstance(update, dict):
        return None
    for value in update.values():
        if isinstance(value, HarnessState):
            return value
        if isinstance(value, dict):
            return HarnessState.model_validate(value)
    return None


def close_graph(graph) -> None:
    close = getattr(graph, "close", None)
    if callable(close):
        close()
