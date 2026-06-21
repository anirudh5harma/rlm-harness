from __future__ import annotations

import argparse
from typing import Optional

from rlm_harness import task_runtime
from rlm_harness.config import (
    default_memory_path,
    default_profile_path,
    default_trace_path,
)
from rlm_harness.kernel import AutonomyMode
from rlm_harness.mcp_config import MCP_CONFIG_PATH

PERMISSION_MODE_ALIASES = task_runtime.PERMISSION_MODE_ALIASES
GRAPH_NODE_MARKERS = task_runtime.GRAPH_NODE_MARKERS

RunConsole = task_runtime.RunConsole
apply_permission_aliases = task_runtime.apply_permission_aliases
build_client = task_runtime.build_client
build_runtime = task_runtime.build_runtime
checkpoint_path = task_runtime.checkpoint_path
close_graph = task_runtime.close_graph
compact_marker_text = task_runtime.compact_marker_text
emit_preflight_failure = task_runtime.emit_preflight_failure
emit_run_output = task_runtime.emit_run_output
finalize_plan_only = task_runtime.finalize_plan_only
graph_state_from_update = task_runtime.graph_state_from_update
graph_update_nodes = task_runtime.graph_update_nodes
record_run_started_event = task_runtime.record_run_started_event
render_user_plan = task_runtime.render_user_plan
run_command_label = task_runtime.run_command_label
run_marked_graph = task_runtime.run_marked_graph
run_output_payload = task_runtime.run_output_payload
run_plan_task = task_runtime.run_plan_task
run_preflight_failure = task_runtime.run_preflight_failure
run_streaming_graph = task_runtime.run_streaming_graph
run_task = task_runtime.run_task
should_emit_run_markers = task_runtime.should_emit_run_markers

__all__ = [
    "GRAPH_NODE_MARKERS",
    "PERMISSION_MODE_ALIASES",
    "RunConsole",
    "add_run_args",
    "add_task_commands",
    "apply_permission_aliases",
    "build_client",
    "build_runtime",
    "checkpoint_path",
    "close_graph",
    "compact_marker_text",
    "emit_preflight_failure",
    "emit_run_output",
    "finalize_plan_only",
    "graph_state_from_update",
    "graph_update_nodes",
    "record_run_started_event",
    "render_user_plan",
    "run_command_label",
    "run_marked_graph",
    "run_output_payload",
    "run_plan_task",
    "run_preflight_failure",
    "run_streaming_graph",
    "run_task",
    "should_emit_run_markers",
]


def add_run_args(
    command: argparse.ArgumentParser,
    add_model_args,
    workspace_default: Optional[str] = ".",
    include_thread_id: bool = True,
) -> None:
    command.add_argument("--workspace", default=workspace_default, help="Workspace path.")
    command.add_argument(
        "--trace-db",
        default=str(default_trace_path()),
        help=argparse.SUPPRESS,
    )
    command.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON.")
    command.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    command.add_argument("--stream", action="store_true", help="Print LangGraph update events.")
    if include_thread_id:
        command.add_argument(
            "--thread-id",
            default=None,
            help="Thread id for memory continuity.",
        )
    command.add_argument(
        "--memory-db",
        default=str(default_memory_path()),
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--profile-db",
        default=str(default_profile_path()),
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--mcp-config",
        default=str(MCP_CONFIG_PATH),
        help=argparse.SUPPRESS,
    )
    command.add_argument("--no-mcp", action="store_true", help=argparse.SUPPRESS)
    command.add_argument("--no-style-scan", action="store_true", help=argparse.SUPPRESS)
    command.add_argument(
        "--style-scan-max-files",
        type=int,
        default=400,
        help=argparse.SUPPRESS,
    )
    command.add_argument("--no-memory", action="store_true", help=argparse.SUPPRESS)
    command.add_argument(
        "--max-history-tokens",
        type=int,
        default=1600,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--preserve-recent-steps",
        type=int,
        default=4,
        help=argparse.SUPPRESS,
    )
    command.add_argument("--recall-limit", type=int, default=6, help=argparse.SUPPRESS)
    command.add_argument(
        "--archival-limit",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--summary-max-tokens",
        type=int,
        default=300,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--token-budget",
        type=int,
        default=100000,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--graph-backend",
        default="supervisor",
        choices=["auto", "supervisor", "simple", "langgraph"],
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--checkpoint-db",
        default=".rlm_harness/checkpoints.db",
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--no-checkpoint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    command.add_argument("--no-sandbox", action="store_true", help=argparse.SUPPRESS)
    command.add_argument(
        "--sandbox-image",
        default="rlm-harness-sandbox:latest",
        help=argparse.SUPPRESS,
    )
    command.add_argument("--sandbox-memory", default="512m", help=argparse.SUPPRESS)
    command.add_argument("--sandbox-cpus", type=float, default=1.0, help=argparse.SUPPRESS)
    command.add_argument(
        "--sandbox-timeout",
        type=float,
        default=60,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--max-subcalls",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--subcall-max-tokens",
        type=int,
        default=1024,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--max-action-retries",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--max-turns",
        dest="max_iterations",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--act-engine",
        choices=["rlm", "json", "tool"],
        default="tool",
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--mode",
        dest="autonomy",
        choices=[mode.value for mode in AutonomyMode],
        default=AutonomyMode.SANDBOX.value,
        help="Autonomy mode for typed tool execution.",
    )
    command.add_argument(
        "--permission-mode",
        choices=sorted(PERMISSION_MODE_ALIASES),
        default=None,
        help=(
            "Permission mode alias: ask, plan, propose, sandbox, trusted, "
            "standard, or auto-accept."
        ),
    )
    command.add_argument(
        "--plan",
        dest="plan_only",
        action="store_true",
        help="Plan only; do not edit files or run shell actions.",
    )
    command.add_argument(
        "--ask",
        dest="ask",
        action="store_true",
        help="Read-only mode: answer the task without editing files or running shell actions.",
    )
    command.add_argument(
        "--auto-accept",
        action="store_true",
        help="Use trusted tool execution for this run.",
    )
    command.add_argument(
        "-t",
        "--trust",
        action="store_true",
        help="Use trusted tool execution for this run.",
    )
    command.add_argument(
        "--yolo",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--skip-onboarding",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    add_model_args(command, public=True)


def add_task_commands(
    subparsers,
    add_model_args,
    *,
    cmd_ask,
    cmd_plan,
    cmd_run,
    cmd_work,
    cmd_resume,
    cmd_continue,
) -> None:
    ask = subparsers.add_parser("ask", help="Ask a read-only workspace question.")
    ask.add_argument("task")
    add_run_args(ask, add_model_args)
    ask.set_defaults(func=cmd_ask)

    plan_cmd = subparsers.add_parser("plan", help="Create a read-only implementation plan.")
    plan_cmd.add_argument("task")
    add_run_args(plan_cmd, add_model_args)
    plan_cmd.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="Run a task.")
    run.add_argument("task")
    add_run_args(run, add_model_args)
    run.set_defaults(func=cmd_run)

    work = subparsers.add_parser("work", help="Run a coding task with typed tools.")
    work.add_argument("task")
    add_run_args(work, add_model_args)
    work.set_defaults(func=cmd_work)

    resume = subparsers.add_parser("resume", help="Resume a thread.")
    resume.add_argument("thread_id")
    resume.add_argument("task", nargs="?")
    add_run_args(
        resume,
        add_model_args,
        workspace_default=None,
        include_thread_id=False,
    )
    resume.set_defaults(func=cmd_resume)

    continue_cmd = subparsers.add_parser("continue", help="Continue the latest thread.")
    continue_cmd.add_argument("task", nargs="?")
    add_run_args(
        continue_cmd,
        add_model_args,
        workspace_default=None,
        include_thread_id=False,
    )
    continue_cmd.set_defaults(func=cmd_continue)
