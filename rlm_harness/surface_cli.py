from __future__ import annotations

import argparse
import json
import sys

from rlm_harness.cli_catalog import (
    command_catalog,
    render_command_catalog,
    render_slash_palette,
    should_use_color,
    slash_command_catalog,
)
from rlm_harness.tools import default_tool_registry, render_tool_catalog


def cmd_commands(args: argparse.Namespace) -> int:
    commands = command_catalog(
        include_internal=args.include_internal,
        include_legacy=args.include_all,
    )
    if args.json_output:
        print(json.dumps(commands, sort_keys=True))
    else:
        print(render_command_catalog(commands, color=should_use_color(sys.stdout)))
    return 0


def cmd_palette(args: argparse.Namespace) -> int:
    include_internal = not args.public_only
    if args.json_output:
        payload = {"commands": slash_command_catalog(include_internal=include_internal)}
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            render_slash_palette(
                include_internal=include_internal,
                color=should_use_color(sys.stdout),
            )
        )
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    registry = default_tool_registry()
    if args.json_output:
        print(json.dumps(registry.payload(include_internal=args.include_internal), sort_keys=True))
    else:
        print(render_tool_catalog(registry, include_internal=args.include_internal))
    return 0


def add_surface_commands(subparsers) -> None:
    commands = subparsers.add_parser("commands", help="List the command surface.")
    commands.add_argument("--json", dest="json_output", action="store_true")
    commands.add_argument("--include-internal", action="store_true")
    commands.add_argument(
        "--all",
        dest="include_all",
        action="store_true",
        help="Include hidden compatibility aliases (run, ask, plan, taste, etc.).",
    )
    commands.set_defaults(func=cmd_commands)

    tools = subparsers.add_parser("tools", help="List action capabilities.")
    tools.add_argument("--json", dest="json_output", action="store_true")
    tools.add_argument(
        "--include-internal",
        action="store_true",
        help="Include runtime compatibility actions.",
    )
    tools.set_defaults(func=cmd_tools)

    palette = subparsers.add_parser("palette", help="Show slash commands.")
    palette.add_argument("--json", dest="json_output", action="store_true")
    palette.add_argument(
        "--public-only",
        action="store_true",
        help="Hide runtime/internal commands and compatibility tools.",
    )
    palette.set_defaults(func=cmd_palette)
