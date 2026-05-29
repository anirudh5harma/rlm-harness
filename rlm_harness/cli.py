from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import rlm_harness.mcp_config as mcp_config
import rlm_harness.shell_cli as shell
import rlm_harness.task_cli as task_workflow
from rlm_harness import onboarding_cli as onboarding
from rlm_harness.config import (
    CONFIG_PATH as _CONFIG_PATH,
)
from rlm_harness.kernel import AutonomyMode
from rlm_harness.maintenance_cli import (
    cmd_update as _cmd_update,
)
from rlm_harness.parser_cli import RootCommandCallbacks, build_parser
from rlm_harness.quality_cli import (
    build_eval_harness_command as _build_eval_harness_command,
)
from rlm_harness.quality_cli import (
    record_eval_failure_proposals as _record_eval_failure_proposals,
)
from rlm_harness.tracing import TraceStore

CONFIG_PATH = _CONFIG_PATH
MCPConfigStore = mcp_config.MCPConfigStore
PERMISSION_MODE_ALIASES = task_workflow.PERMISSION_MODE_ALIASES

cmd_config = onboarding.cmd_config
cmd_init = onboarding.cmd_init
cmd_readiness = onboarding.cmd_readiness
cmd_status = onboarding.cmd_status

RunConsole = task_workflow.RunConsole
apply_permission_aliases = task_workflow.apply_permission_aliases
build_client = task_workflow.build_client
build_runtime = task_workflow.build_runtime
checkpoint_path = task_workflow.checkpoint_path
close_graph = task_workflow.close_graph
emit_run_output = task_workflow.emit_run_output
finalize_plan_only = task_workflow.finalize_plan_only
graph_state_from_update = task_workflow.graph_state_from_update
graph_update_nodes = task_workflow.graph_update_nodes
record_run_started_event = task_workflow.record_run_started_event
render_user_plan = task_workflow.render_user_plan
run_marked_graph = task_workflow.run_marked_graph
run_output_payload = task_workflow.run_output_payload
run_plan_task = task_workflow.run_plan_task
run_preflight_failure = task_workflow.run_preflight_failure
run_streaming_graph = task_workflow.run_streaming_graph
run_task = task_workflow.run_task

ROOT_PRINT_FLAGS = shell.ROOT_PRINT_FLAGS
ROOT_BOOLEAN_RUN_FLAGS = shell.ROOT_BOOLEAN_RUN_FLAGS
ROOT_VALUE_RUN_FLAGS = shell.ROOT_VALUE_RUN_FLAGS
argv_has_option = shell.argv_has_option
normalize_argv = shell.normalize_argv
normalize_leading_run_flags = shell.normalize_leading_run_flags


def cmd_run(args: argparse.Namespace) -> int:
    return run_or_plan_task(args, args.task, args.thread_id, Path(args.workspace).resolve())


def cmd_ask(args: argparse.Namespace) -> int:
    args.act_engine = "tool"
    args.autonomy = AutonomyMode.ASK.value
    return run_task(args, args.task, args.thread_id, Path(args.workspace).resolve())


def cmd_plan(args: argparse.Namespace) -> int:
    args.act_engine = "tool"
    args.autonomy = AutonomyMode.PLAN.value
    return run_plan_task(args, args.task, args.thread_id, Path(args.workspace).resolve())


def cmd_work(args: argparse.Namespace) -> int:
    args.act_engine = "tool"
    return run_or_plan_task(args, args.task, args.thread_id, Path(args.workspace).resolve())


def cmd_resume(args: argparse.Namespace) -> int:
    traces = TraceStore(Path(args.trace_db))
    previous = traces.latest_run_for_thread(args.thread_id)
    if previous is None and args.task is None:
        print(
            f"Cannot resume {args.thread_id}: no previous run in {args.trace_db}",
            file=sys.stderr,
        )
        return 1
    task = args.task or str(previous["task"])
    workspace_value = args.workspace or (previous["workspace"] if previous else ".")
    workspace = Path(workspace_value).resolve()
    return run_or_plan_task(args, task, args.thread_id, workspace)


def cmd_continue(args: argparse.Namespace) -> int:
    traces = TraceStore(Path(args.trace_db))
    runs = traces.list_runs(limit=1)
    if not runs:
        print(f"Cannot continue: no previous runs in {args.trace_db}", file=sys.stderr)
        return 1
    previous = runs[0]
    thread_id = str(previous["thread_id"])
    task = args.task or str(previous["task"])
    workspace_value = args.workspace or previous["workspace"]
    workspace = Path(str(workspace_value)).resolve()
    return run_or_plan_task(args, task, thread_id, workspace)


def run_or_plan_task(
    args: argparse.Namespace,
    task: str,
    thread_id: Optional[str],
    workspace: Path,
) -> int:
    apply_permission_aliases(args)
    if args.autonomy == AutonomyMode.PLAN.value:
        args.act_engine = "tool"
        return run_plan_task(args, task, thread_id, workspace)
    return run_task(args, task, thread_id, workspace)


def cmd_update(args: argparse.Namespace) -> int:
    return _cmd_update(args)


def build_eval_harness_command(args: argparse.Namespace) -> list[str]:
    return _build_eval_harness_command(args)


def record_eval_failure_proposals(report, memory_path: Path) -> int:
    return _record_eval_failure_proposals(report, memory_path)


def parser() -> argparse.ArgumentParser:
    return build_parser(
        RootCommandCallbacks(
            cmd_ask=cmd_ask,
            cmd_plan=cmd_plan,
            cmd_run=cmd_run,
            cmd_work=cmd_work,
            cmd_resume=cmd_resume,
            cmd_continue=cmd_continue,
        )
    )


def interactive_loop() -> int:
    return shell.interactive_loop(main)


def main(argv: list[str] | None = None) -> int:
    argv = normalize_argv(argv)
    command_parser = parser()
    if argv == []:
        if sys.stdin.isatty():
            return interactive_loop()
        command_parser.print_help()
        return 0
    args = command_parser.parse_args(argv)
    if hasattr(args, "provider"):
        args.provider_explicit = argv_has_option(argv, "--provider")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
