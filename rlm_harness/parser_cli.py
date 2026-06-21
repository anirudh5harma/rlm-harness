from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from rlm_harness import __version__
from rlm_harness.cli_catalog import DEFAULT_PROG, PUBLIC_COMMANDS
from rlm_harness.config import default_base_url, default_model, default_provider
from rlm_harness.extension_cli import add_install_command
from rlm_harness.learning_cli import add_learning_commands
from rlm_harness.maintenance_cli import add_maintenance_commands
from rlm_harness.mcp_cli import add_mcp_command
from rlm_harness.memory_cli import add_memory_command
from rlm_harness.onboarding_cli import add_onboarding_commands
from rlm_harness.provider_cli import add_provider_commands
from rlm_harness.quality_cli import add_quality_commands
from rlm_harness.runtime_cli import add_runtime_commands
from rlm_harness.sandbox_cli import add_sandbox_command
from rlm_harness.surface_cli import add_surface_commands
from rlm_harness.task_cli import add_task_commands
from rlm_harness.taste_cli import add_taste_command
from rlm_harness.trace_cli import add_trace_command


@dataclass(frozen=True)
class RootCommandCallbacks:
    cmd_ask: Callable[[argparse.Namespace], int]
    cmd_plan: Callable[[argparse.Namespace], int]
    cmd_run: Callable[[argparse.Namespace], int]
    cmd_work: Callable[[argparse.Namespace], int]
    cmd_resume: Callable[[argparse.Namespace], int]
    cmd_continue: Callable[[argparse.Namespace], int]


def add_model_args(
    command: argparse.ArgumentParser,
    include_timeout: bool = True,
    public: bool = False,
) -> None:
    command.add_argument(
        "--provider",
        default=default_provider(),
        help="Model provider." if public else argparse.SUPPRESS,
    )
    command.add_argument(
        "-m",
        "--model",
        default=default_model(),
        help="Model name sent to the provider." if public else argparse.SUPPRESS,
    )
    command.add_argument(
        "--base-url",
        default=default_base_url(),
        help=argparse.SUPPRESS,
    )
    command.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    if include_timeout:
        command.add_argument(
            "--timeout",
            type=int,
            default=120,
            help=argparse.SUPPRESS,
        )


def build_parser(callbacks: RootCommandCallbacks) -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog=DEFAULT_PROG,
        description=(
            f'Local recursive coding-agent harness. Run a task with: {DEFAULT_PROG} "fix tests"'
        ),
    )
    root.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = root.add_subparsers(
        dest="command",
        metavar="{init,doctor,status,history,mcp,eval,update,install}",
        required=True,
    )

    add_surface_commands(subparsers)
    add_mcp_command(subparsers)
    add_task_commands(
        subparsers,
        add_model_args,
        cmd_ask=callbacks.cmd_ask,
        cmd_plan=callbacks.cmd_plan,
        cmd_run=callbacks.cmd_run,
        cmd_work=callbacks.cmd_work,
        cmd_resume=callbacks.cmd_resume,
        cmd_continue=callbacks.cmd_continue,
    )
    add_trace_command(subparsers, name="trace")
    add_trace_command(subparsers, name="history")
    add_install_command(subparsers)
    add_onboarding_commands(subparsers)
    add_provider_commands(subparsers)
    add_taste_command(subparsers, "profile", "Inspect or teach Harness taste.")
    add_taste_command(subparsers, "taste", "Manage taste learning and preferences.")
    add_learning_commands(subparsers)
    add_quality_commands(subparsers, add_model_args)
    add_maintenance_commands(subparsers)
    add_runtime_commands(subparsers, add_model_args)
    add_memory_command(subparsers)
    add_sandbox_command(subparsers, add_model_args)

    # Only the PUBLIC_COMMANDS are advertised in `--help`. Legacy
    # aliases are still registered (so `harness run`, `harness taste`,
    # `/ask`, etc. keep working) but hidden from the choices list.
    subparsers._choices_actions = [  # type: ignore[attr-defined]
        action
        for action in subparsers._choices_actions  # type: ignore[attr-defined]
        if action.dest in set(PUBLIC_COMMANDS)
    ]
    return root
