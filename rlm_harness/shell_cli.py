from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from typing import Optional

from rlm_harness.cli_catalog import (
    ALL_COMMANDS,
    ANSI_BLUE,
    ANSI_CYAN,
    render_slash_palette,
    should_use_color,
    style_text,
)
from rlm_harness.config import default_model, default_provider
from rlm_harness.kernel import AutonomyMode

ROOT_PRINT_FLAGS = {"-p", "--print"}
ROOT_BOOLEAN_RUN_FLAGS = {
    "--ask": "--ask",
    "--auto-accept": "--auto-accept",
    "-t": "--trust",
    "--trust": "--trust",
    "--yolo": "--yolo",
    "--dangerously-skip-permissions": "--dangerously-skip-permissions",
    "--skip-onboarding": "--skip-onboarding",
    "--json": "--json",
    "--quiet": "--quiet",
    "--stream": "--stream",
    "--no-memory": "--no-memory",
    "--no-sandbox": "--no-sandbox",
}
ROOT_VALUE_RUN_FLAGS = {
    "--permission-mode": "--permission-mode",
    "-m": "--model",
    "--model": "--model",
    "--provider": "--provider",
    "--workspace": "--workspace",
    "--trace-db": "--trace-db",
    "--memory-db": "--memory-db",
    "--profile-db": "--profile-db",
    "--max-turns": "--max-turns",
}


def interactive_loop(dispatch: Callable[[list[str]], int]) -> int:
    color = should_use_color(sys.stdout)
    name = style_text("Harness", ANSI_CYAN, color)
    print(f"{name} interactive mode. Type a coding task and press Enter. Type / for commands.")
    print(
        f"Using provider={style_text(default_provider(), ANSI_CYAN, color)} "
        f"model={style_text(default_model(), ANSI_BLUE, color)}"
    )
    print("Configure with /provider to choose a provider, then /model to select a model.")
    prompt = f"{style_text('harness', ANSI_CYAN, color)}{style_text('>', ANSI_BLUE, color)} "
    while True:
        try:
            task = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not task:
            continue
        if task in {"/q", "/quit", "quit", "exit"}:
            return 0
        if task in {"/", "/h", "/help", "help"}:
            print(render_slash_palette(color=color))
            continue
        if task.startswith("/"):
            try:
                slash_args = shlex.split(task[1:])
            except ValueError as exc:
                print(f"Invalid command: {exc}", file=sys.stderr)
                continue
            if not slash_args:
                continue
            exit_code = dispatch(slash_args)
        else:
            exit_code = dispatch([task])
        if exit_code != 0:
            print(f"Task exited with status {exit_code}", file=sys.stderr)


def normalize_argv(argv: list[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return argv
    if argv[0] in {"/", "/h", "/help"}:
        return ["palette", *argv[1:]]
    if argv[0] in {"-c", "--continue"}:
        return ["continue", *argv[1:]]
    if argv[0] in {"-r", "--resume"}:
        return ["resume", *argv[1:]]
    if argv[0].startswith("/") and len(argv[0]) > 1:
        argv = [argv[0][1:], *argv[1:]]
    if argv[0] == "help":
        return ["palette", *argv[1:]]
    if argv[0] == "--list-models":
        return ["model", *argv[1:]]
    if argv[0] in {"-h", "--help", "-v", "--version"}:
        return argv
    if argv[0] in ALL_COMMANDS or argv[0].startswith("-"):
        commanduse_args = normalize_leading_run_flags(argv)
        if commanduse_args is not None:
            return commanduse_args
        return argv
    return ["run", *argv]


def argv_has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def normalize_leading_run_flags(argv: list[str]) -> Optional[list[str]]:
    command = "run"
    run_flags: list[str] = []
    saw_leading_run_flag = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ROOT_PRINT_FLAGS:
            saw_leading_run_flag = True
            command = "run"
            index += 1
            continue
        if token == "--plan":
            saw_leading_run_flag = True
            command = "plan"
            index += 1
            continue
        if token in ROOT_BOOLEAN_RUN_FLAGS:
            saw_leading_run_flag = True
            run_flags.append(ROOT_BOOLEAN_RUN_FLAGS[token])
            index += 1
            continue
        if token in ROOT_VALUE_RUN_FLAGS:
            if index + 1 >= len(argv):
                return None
            saw_leading_run_flag = True
            value = argv[index + 1]
            normalized_flag = ROOT_VALUE_RUN_FLAGS[token]
            if normalized_flag == "--permission-mode" and value == AutonomyMode.PLAN.value:
                command = "plan"
            else:
                run_flags.extend([normalized_flag, value])
            index += 2
            continue
        if saw_leading_run_flag:
            return [command, token, *argv[index + 1 :], *run_flags]
        return None
    if saw_leading_run_flag:
        return [command, *run_flags]
    return None
