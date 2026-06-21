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
    ANSI_RED,
    should_use_color,
    stream_is_tty,
    style_text,
    terminal_flag,
)
from rlm_harness.config import default_provider
from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import GraphRuntimeConfig, Nodes
from rlm_harness.kernel import AutonomyMode, CompletionEvent, RunStartedEvent
from rlm_harness.kernel.state import (
    RunState,
)
from rlm_harness.kernel.supervisor import (
    Supervisor,
    SupervisorConfig,
    page_history_between_turns,
)
from rlm_harness.mcp_config import MCP_CONFIG_PATH, MCPConfigStore, mcp_workflow_context
from rlm_harness.memory import Memory, MemoryError, MemoryPagingConfig
from rlm_harness.model_client import LMClientError
from rlm_harness.providers import normalize_provider
from rlm_harness.rlm import RLMRuntime
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


def streaming_output_enabled(stream: TextIO) -> bool:
    """Whether the model's response should stream to the terminal.

    Streaming (token-by-token output) is enabled only in a real TTY
    *without* an explicit ``HARNESS_PROGRESS`` override. When
    ``HARNESS_PROGRESS=on`` is set, the user wants the old
    single-line status mode, not streaming text. When
    ``HARNESS_PROGRESS=off`` or ``--json``/``--quiet`` is set, no
    output at all.
    """
    progress = terminal_flag(os.environ.get("HARNESS_PROGRESS"))
    if progress is not None:
        return False
    return stream_is_tty(stream)


def compact_marker_text(text: str, max_chars: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3].rstrip()}..."


def run_command_label(task: str) -> str:
    return f"harness run {shlex.quote(compact_marker_text(task))}"


def _extract_diff_from_code(code: str, stdout: str) -> str:
    """Detect file-edit operations in a REPL block and return a diff summary.

    The model writes files via ``write_file(path, content)`` or
    ``apply_patch(diff)``. When we see those calls in the code, we
    produce a one-line summary so the user knows what changed without
    having to inspect the full stdout.
    """
    if not code:
        return ""
    # ``write_file('path', ...)`` or ``write_file("path", ...)``
    import re

    write_match = re.search(r"write_file\(\s*['\"]([^'\"]+)['\"]", code)
    if write_match:
        path = write_match.group(1)
        return f"[edit] {path}"
    # ``apply_patch(diff)`` — try to extract changed file names from
    # the diff header lines in stdout.
    if "apply_patch" in code and stdout:
        files = re.findall(r"^(?:---|\+\+\+) [ab]/(.+)$", stdout, re.MULTILINE)
        if files:
            unique = list(dict.fromkeys(files))
            return f"[patch] {', '.join(unique[:5])}"
    # ``propose_file_change('path', ...)`` — show the proposal.
    propose_match = re.search(r"propose_file_change\(\s*['\"]([^'\"]+)['\"]", code)
    if propose_match:
        path = propose_match.group(1)
        return f"[proposed] {path}"
    return ""


def _first_error_line(stderr: str) -> str:
    """Extract the last line of a Python traceback (the actual error)."""
    if not stderr:
        return "unknown error"
    lines = stderr.strip().splitlines()
    # The error is usually the last non-empty line.
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("File ", "Traceback", "  ")):
            return stripped
    return lines[-1] if lines else "unknown error"


class RunConsole:
    def __init__(self, args: argparse.Namespace, stream: Optional[TextIO] = None):
        self.stream = stream or sys.stderr
        self.enabled = should_emit_run_markers(args, self.stream)
        self.color = should_use_color(self.stream)
        self.memory_enabled = not getattr(args, "no_memory", False)
        self.json_output = getattr(args, "json_output", False)
        self.quiet = getattr(args, "quiet", False)
        self._rich_status: Any = None
        self._line_active = False
        self._use_rich = (
            self.enabled
            and stream_is_tty(self.stream)
            and RichConsole is not None
            and RichStatus is not None
        )
        # Streaming state: when we're printing the model's response
        # token-by-token, we stop using the status line and write
        # directly to the stream so the user sees the text arrive.
        self._streaming_active = False
        self._turn_tokens = 0
        # Streaming output (token-by-token) is only enabled in a real
        # TTY without an explicit HARNESS_PROGRESS override. When the
        # old status-line mode is forced, we keep the single-line
        # contract.
        self.streaming = streaming_output_enabled(self.stream)

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

    def on_turn_event(self, event: Any) -> None:
        """Handle a streaming RLM turn event from the supervisor.

        This is the ``stream_sink`` callback. It renders the model's
        response, tool observations, and token usage to stderr in real
        time. When ``json_output`` or ``quiet`` is set, it is a no-op
        so the final answer stays clean for piping.
        """
        from rlm_harness.rlm.runtime import (
            IterationFinished,
            ObservationRecorded,
            TokenDelta,
            TurnFinished,
            TurnStarted,
        )

        if self.json_output or self.quiet:
            return
        if not self.enabled:
            return
        # Streaming token output and observation rendering are only
        # active in a real TTY. When the old status-line mode is
        # forced via HARNESS_PROGRESS, keep the single-line contract.
        if not self.streaming:
            return

        if isinstance(event, TurnStarted):
            # Stop any previous status line; the model is about to
            # stream its response.
            self._stop_streaming()
            self._turn_tokens = 0
        elif isinstance(event, TokenDelta):
            # Print the delta directly to the stream so the user sees
            # the model's response arrive token-by-token.
            self._write_delta(event.delta)
        elif isinstance(event, IterationFinished):
            # The model finished one iteration. If it emitted REPL
            # blocks, observations will follow — don't add a newline
            # yet. If it didn't, the response is the final answer.
            self._turn_tokens += int(event.usage.get("completion_tokens") or 0)
            if not event.repl_blocks:
                self._stop_streaming()
            else:
                self._stop_streaming()
                self._write_line("")
        elif isinstance(event, ObservationRecorded):
            self._render_observation(event.observation)
        elif isinstance(event, TurnFinished):
            result = event.result
            if result.status == "error":
                self._write_line(
                    style_text(f"[error] {result.final_answer}", ANSI_RED, self.color)
                )
            elif result.status == "stopped":
                self._write_line(
                    style_text(
                        f"[stopped] {result.iterations} iterations, "
                        f"{result.tokens_used} tokens",
                        ANSI_CYAN,
                        self.color,
                    )
                )
            elif result.status == "done":
                # Don't print the final answer here — it goes to
                # stdout via emit_run_output. Show a summary if
                # the turn used tokens.
                if result.tokens_used > 0:
                    self._write_line(
                        style_text(
                            f"[done] {result.iterations} iters, "
                            f"{result.tokens_used} tokens",
                            ANSI_CYAN,
                            self.color,
                        )
                    )

    def _write_delta(self, delta: str) -> None:
        """Write a streaming token delta to the stream."""
        if not self.enabled:
            return
        # Stop the status line before printing tokens.
        if self._rich_status is not None:
            self._rich_status.stop()
            self._rich_status = None
        self._streaming_active = True
        self.stream.write(delta)
        self.stream.flush()

    def _stop_streaming(self) -> None:
        if self._streaming_active:
            self.stream.write("\n")
            self.stream.flush()
            self._streaming_active = False

    def _write_line(self, text: str) -> None:
        if not self.enabled:
            return
        self._stop_streaming()
        if text:
            self.stream.write(text + "\n")
            self.stream.flush()

    def _render_observation(self, observation: Any) -> None:
        """Render a REPL observation (tool output) to the stream."""
        if not self.enabled:
            return
        status = getattr(observation, "status", "ok")
        stdout = getattr(observation, "stdout", "")
        stderr = getattr(observation, "stderr", "")
        code = getattr(observation, "code", "")

        # Detect file-edit operations and show a diff summary.
        diff_summary = _extract_diff_from_code(code, stdout)
        if diff_summary:
            self._write_line(style_text(diff_summary, ANSI_CYAN, self.color))
            return

        if status == "error" and stderr:
            # Show a clean one-line error, not the full traceback.
            first_error = _first_error_line(stderr)
            self._write_line(style_text(f"[tool error] {first_error}", ANSI_RED, self.color))
            return

        # For non-error observations, show a brief summary.
        if stdout:
            preview = stdout.strip().split("\n")[0][:120]
            self._write_line(style_text(f"[tool] {preview}", ANSI_CYAN, self.color))

    def finish(self, status: str, run_id: str) -> None:
        self._stop_streaming()
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
    if getattr(args, "ask", False):
        args.autonomy = AutonomyMode.ASK.value
        args.act_engine = "tool"
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
            if args.graph_backend in ("supervisor", "auto"):
                # The supervisor is the new control plane (Phase A).
                # It runs the RLM runtime per turn, pages memory
                # between turns, and emits typed events to the
                # trace. Returns a `HarnessState` for the rest of
                # the CLI. `auto` is the legacy default value and is
                # treated as an alias for `supervisor` so older
                # configs and pinned scripts keep the new path.
                # The supervisor receives the `run_id` this
                # function already created so the trace has
                # exactly one `runs` row per invocation.
                console.marker("supervisor", "running RLM turns")
                final_state = run_supervisor_graph(
                    args,
                    task,
                    thread_id,
                    workspace,
                    build_client(args),
                    traces,
                    memory,
                    profile_memory,
                    run_id=run_id,
                    console=console,
                )
                # The supervisor's `_finish_run` is the canonical
                # closer; it has the verification-aware status.
                # Skip the legacy `traces.finish_run` below.
            else:
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
    state.final_answer = render_user_plan(
        state.plan,
        task=state.task,
        workspace=Path(state.workspace),
    )
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


def render_user_plan(
    plan: TaskPlan,
    *,
    task: str = "",
    workspace: Optional[Path] = None,
) -> str:
    if not plan.steps:
        return "Implementation Plan\nNo plan steps were produced."
    if looks_like_generic_plan(plan):
        return render_workspace_grounded_plan(task, workspace)
    lines = ["Implementation Plan"]
    for step in plan.steps:
        indent = "  " if step.parent_id else ""
        lines.append(f"{indent}{step.id}. {step.description}")
    return "\n".join(lines)


def looks_like_generic_plan(plan: TaskPlan) -> bool:
    generic_phrases = (
        "inspect the task",
        "produce a concise response",
        "record the result",
    )
    descriptions = [step.description.strip().lower().rstrip(".") for step in plan.steps]
    if not descriptions or len(descriptions) > 4:
        return False
    matches = sum(
        1
        for description in descriptions
        if any(phrase in description for phrase in generic_phrases)
    )
    return matches >= min(2, len(descriptions))


def render_workspace_grounded_plan(task: str, workspace: Optional[Path]) -> str:
    files = workspace_files(workspace) if workspace is not None else []
    orientation = plan_orientation_files(files)
    verification = plan_verification_command(files)

    lines = ["Implementation Plan"]
    if task.strip():
        lines.append(f"For: {compact_marker_text(task, max_chars=140)}")
    if orientation:
        lines.append("Start with:")
        lines.extend(f"- {path}" for path in orientation)

    target = "the smallest owned surface"
    if orientation:
        target = orientation[-1]
    lines.extend(
        [
            "1. Confirm the requested behavior against the current project shape.",
            f"2. Inspect {target} and the nearby command or runtime code before editing.",
            "3. Add or update the focused regression test for the behavior you want.",
            "4. Make the smallest scoped change that satisfies the test and existing contracts.",
            f"5. Run `{verification}` and report the files changed plus the result.",
        ]
    )
    return "\n".join(lines)


def workspace_files(workspace: Optional[Path], max_files: int = 300) -> list[str]:
    if workspace is None or not workspace.exists():
        return []
    ignored = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".rlm_harness",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "dist",
        "build",
        "node_modules",
        "target",
    }
    files: list[str] = []
    for path in sorted(workspace.rglob("*")):
        try:
            relative = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in ignored for part in relative.parts):
            continue
        if path.is_file():
            files.append(str(relative))
            if len(files) >= max_files:
                break
    return files


def plan_orientation_files(files: list[str]) -> list[str]:
    preferred = [
        "README.md",
        "Cargo.toml",
        "pyproject.toml",
        "package.json",
        "src/main.rs",
        "src/lib.rs",
        "main.py",
        "app.py",
        "src/main.ts",
        "src/main.tsx",
        "src/App.tsx",
    ]
    selected = [path for path in preferred if path in files]
    selected.extend(
        path
        for path in files
        if path.startswith("crates/") and path.endswith(("/src/main.rs", "/src/lib.rs"))
    )
    selected.extend(
        path
        for path in files
        if path.endswith(("_cli.py", "/cli.py", "/main.py")) and path not in selected
    )
    return dedupe(selected)[:5]


def plan_verification_command(files: list[str]) -> str:
    if "Cargo.toml" in files or any(path.endswith(".rs") for path in files):
        return "cargo test"
    if any(path.startswith("tests/") for path in files):
        return "pytest"
    if any(Path(path).name.startswith("test_") and path.endswith(".py") for path in files):
        return "python -m unittest"
    if "pyproject.toml" in files or any(path.endswith(".py") for path in files):
        return "python -m pytest"
    if "package.json" in files:
        return "npm test"
    return "the project test command"


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


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


# --- Supervisor runner (Phase A.5) ---------------------------------------
# The supervisor is the new control plane. The existing graph runners
# in `graph/build.py` are preserved for backward compatibility; this
# module exposes the supervisor as a `--graph-backend supervisor`
# option that produces a `HarnessState` so the rest of the CLI works
# unchanged. The adapter is intentionally narrow: it converts the
# legacy `HarnessState` to the kernel `RunState`, runs the
# supervisor, and converts the result back.


def run_supervisor_graph(
    args: argparse.Namespace,
    task: str,
    thread_id: Optional[str],
    workspace: Path,
    client,
    traces: TraceStore,
    memory: Optional[Memory],
    profile_memory: Optional[Memory],
    run_id: Optional[str] = None,
    *,
    console: Optional[RunConsole] = None,
) -> HarnessState:
    """Run the supervisor as the control plane.

    Returns a `HarnessState` whose `status` is the harness status
    string. The caller owns the run lifecycle: it should pass the
    `run_id` it created with `traces.start_run(...)` so the trace has
    exactly one `runs` row per invocation. The supervisor's
    `_finish_run` is the canonical closer with the
    verification-aware status.

    If the caller does not provide a `run_id` (e.g. an internal
    test that drives the supervisor directly), the supervisor will
    create one. Production callers must pass it.
    """
    if run_id is None:
        run_id = traces.start_run(task, str(workspace), thread_id=thread_id)
        owned_run = True
    else:
        owned_run = False
    sandbox_config = SandboxConfig(
        image=args.sandbox_image,
        workspace=workspace,
        memory=args.sandbox_memory,
        cpus=args.sandbox_cpus,
        default_timeout_s=args.sandbox_timeout,
    )
    subcall_config = RLMSubcallConfig(
        max_depth=args.max_depth,
        max_subcalls=args.max_subcalls,
        token_budget=args.token_budget,
        max_tokens=args.subcall_max_tokens,
    )
    # `max_iterations` is per-turn model-call count (the RLM runtime
    # already enforces it). The supervisor adds `max_turns` on top.
    max_iterations = max(1, int(getattr(args, "max_iterations", 8) or 8))
    max_turns = max(1, int(getattr(args, "max_turns", 50) or 50))
    max_subcalls_per_turn = max(
        1, int(getattr(args, "max_subcalls_per_turn", 8) or 8)
    )
    runtime = RLMRuntime(
        client,
        workspace=workspace,
        max_iterations=max_iterations,
        max_depth=subcall_config.max_depth,
        sandbox_enabled=not args.no_sandbox,
        sandbox_config=sandbox_config,
        subcall_config=subcall_config,
    )
    state = HarnessState(
        task=task,
        workspace=str(workspace),
        thread_id=thread_id or run_id,
        run_id=run_id,
        token_budget=args.token_budget,
    )
    run_state = RunState.from_harness_state(state)
    # Bind the supervisor's `run_id` to the one the caller created.
    # This keeps every typed event under the same `runs` row.
    run_state.request.run_id = run_id
    run_state.request.thread_id = thread_id or run_id
    run_state.request.autonomy = AutonomyMode(args.autonomy)
    config = SupervisorConfig(
        max_turns=max_turns,
        max_subcalls_per_turn=max_subcalls_per_turn,
        between_turns=(
            page_history_between_turns(memory)
            if memory is not None
            else None
        ),
        stream_sink=console.on_turn_event if console is not None else None,
    )
    supervisor = Supervisor(
        runtime=runtime,
        traces=traces,
        config=config,
    )
    final = supervisor.run(run_state)
    harness_state = final.to_harness_state()
    if owned_run:
        # Internal callers that did not pass a `run_id` get a
        # best-effort finish so the trace is not left `running`.
        traces.finish_run(run_id, harness_state.status)
    return harness_state
