from __future__ import annotations

import os
from typing import TextIO

from rlm_harness.tools import render_tool_catalog

PUBLIC_COMMANDS = {
    "ask",
    "commands",
    "continue",
    "tools",
    "palette",
    "plan",
    "run",
    "work",
    "resume",
    "status",
    "trace",
    "doctor",
    "dogfood",
    "evolve",
    "feedback",
    "init",
    "model",
    "mcp",
    "provider",
    "profile",
    "readiness",
    "taste",
    "config",
    "eval",
    "update",
}
INTERNAL_COMMANDS = {
    "langgraph-plan",
    "benchmark-model",
    "check-model",
    "serve-mlx",
    "mem",
    "sandbox",
}
ALL_COMMANDS = PUBLIC_COMMANDS | INTERNAL_COMMANDS
DEFAULT_PROG = "harness"

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[31m"

COMMAND_CATALOG = [
    {
        "name": "ask",
        "usage": 'harness ask "what is this project?"',
        "group": "work",
        "summary": "Answer read-only questions using typed workspace tools.",
    },
    {
        "name": "plan",
        "usage": 'harness plan "how should we fix this?"',
        "group": "work",
        "summary": "Produce a read-only implementation plan.",
    },
    {
        "name": "run",
        "usage": 'harness "fix tests"',
        "group": "work",
        "summary": "Run a scoped coding task with sandboxed typed tools.",
    },
    {
        "name": "work",
        "usage": 'harness work "fix tests"',
        "group": "work",
        "summary": "Run a coding task with sandboxed typed tools.",
    },
    {
        "name": "resume",
        "usage": "harness resume <thread-id> [task]",
        "group": "work",
        "summary": "Continue a previous thread using the trace database.",
    },
    {
        "name": "continue",
        "usage": "harness continue [task]",
        "group": "work",
        "summary": "Continue the latest thread without copying its id.",
    },
    {
        "name": "trace",
        "usage": "harness trace list|report|events",
        "group": "inspect",
        "summary": "Inspect run history, reports, and event records.",
    },
    {
        "name": "status",
        "usage": "harness status",
        "group": "inspect",
        "summary": "Show provider, latest run, taste, and evolution status.",
    },
    {
        "name": "tools",
        "usage": "harness tools",
        "group": "inspect",
        "summary": "List action capabilities, risks, scopes, and confirmation requirements.",
    },
    {
        "name": "mcp",
        "usage": "harness mcp list|setup|add|show|tools|trust|enable",
        "group": "inspect",
        "summary": "Manage MCP servers, auth hints, and workflow purposes.",
    },
    {
        "name": "palette",
        "usage": "harness /",
        "group": "inspect",
        "summary": "Show slash commands and action tools in one view.",
    },
    {
        "name": "readiness",
        "usage": "harness readiness",
        "group": "setup",
        "summary": "Check whether Harness is ready for daily coding work.",
    },
    {
        "name": "init",
        "usage": "harness init [--provider name] [--api-key key]",
        "group": "setup",
        "summary": "Bootstrap provider config, project taste, and readiness.",
    },
    {
        "name": "doctor",
        "usage": "harness doctor",
        "group": "setup",
        "summary": "Print local dependency and sandbox health.",
    },
    {
        "name": "provider",
        "usage": "harness provider [name] [--api-key key]",
        "group": "setup",
        "summary": "Choose and save the active model provider.",
    },
    {
        "name": "model",
        "usage": "harness model [name]",
        "group": "setup",
        "summary": "List or save the active model.",
    },
    {
        "name": "config",
        "usage": "harness config",
        "group": "setup",
        "summary": "Show saved provider, model, and profile paths.",
    },
    {
        "name": "taste",
        "usage": "harness taste list|context|learn|scan|approve|reject",
        "group": "learn",
        "summary": "First-class taste learning and preference management.",
    },
    {
        "name": "profile",
        "usage": "harness profile list|context|learn|scan|approve|reject",
        "group": "learn",
        "summary": "Compatibility alias for learned taste records.",
    },
    {
        "name": "feedback",
        "usage": "harness feedback add|list",
        "group": "learn",
        "summary": "Record feedback so future runs adapt to your preferences.",
    },
    {
        "name": "evolve",
        "usage": "harness evolve list|propose|approve|reject",
        "group": "learn",
        "summary": "Review self-evolution proposals before they affect behavior.",
    },
    {
        "name": "eval",
        "usage": "harness eval <suite>",
        "group": "quality",
        "summary": "Run local harness evaluation suites.",
    },
    {
        "name": "dogfood",
        "usage": "harness dogfood",
        "group": "quality",
        "summary": "Run readiness, eval, and feedback proof checks.",
    },
    {
        "name": "update",
        "usage": "harness update",
        "group": "setup",
        "summary": "Upgrade the managed install and rebuild the sandbox image.",
    },
]

INTERNAL_COMMAND_CATALOG = [
    {
        "name": "langgraph-plan",
        "usage": "harness langgraph-plan",
        "group": "runtime",
        "summary": "Run the internal LangGraph planning diagnostic.",
    },
    {
        "name": "benchmark-model",
        "usage": "harness benchmark-model",
        "group": "runtime",
        "summary": "Measure provider latency with a small prompt set.",
    },
    {
        "name": "check-model",
        "usage": "harness check-model",
        "group": "runtime",
        "summary": "Send one probe prompt to the configured model.",
    },
    {
        "name": "serve-mlx",
        "usage": "harness serve-mlx",
        "group": "runtime",
        "summary": "Start a local MLX OpenAI-compatible model server.",
    },
    {
        "name": "mem",
        "usage": "harness mem pin|get|recall-append|recall-page|archive-add|search",
        "group": "runtime",
        "summary": "Inspect or mutate the low-level memory store.",
    },
    {
        "name": "sandbox",
        "usage": "harness sandbox build|run",
        "group": "runtime",
        "summary": "Build the sandbox image or run sandboxed code directly.",
    },
]


def style_text(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{ANSI_RESET}" if enabled else text


def terminal_flag(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "always"}:
        return True
    if normalized in {"0", "false", "no", "off", "never", "none"}:
        return False
    return None


def stream_is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def should_use_color(stream: TextIO) -> bool:
    color = terminal_flag(os.environ.get("HARNESS_COLOR"))
    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    if terminal_flag(os.environ.get("CLICOLOR_FORCE")) is True:
        return True
    if terminal_flag(os.environ.get("FORCE_COLOR")) is True:
        return True
    return stream_is_tty(stream) and os.environ.get("TERM") != "dumb"


def command_catalog(*, include_internal: bool = False) -> list[dict[str, str]]:
    commands = [dict(command) for command in COMMAND_CATALOG]
    if include_internal:
        commands.extend(dict(command) for command in INTERNAL_COMMAND_CATALOG)
    return commands


def render_command_catalog(commands: list[dict[str, str]], *, color: bool = False) -> str:
    lines = [style_text("Harness commands", ANSI_CYAN, color), ""]
    groups = ["work", "inspect", "learn", "quality", "setup", "runtime"]
    for group in groups:
        group_commands = [command for command in commands if command["group"] == group]
        if not group_commands:
            continue
        lines.append(style_text(group, ANSI_BLUE, color))
        for command in group_commands:
            lines.append(f"  {style_text(command['usage'], ANSI_CYAN, color)}")
            lines.append(f"    {command['summary']}")
        lines.append("")
    lines.append('Tip: `harness "fix tests"` is shorthand for `harness run "fix tests"`.')
    return "\n".join(lines).rstrip()


def render_slash_palette(*, include_internal: bool = True, color: bool = False) -> str:
    lines = [style_text("Harness slash palette", ANSI_CYAN, color), ""]
    lines.append(style_text("Commands", ANSI_BLUE, color))
    for command in command_catalog(include_internal=include_internal):
        usage = slash_usage(command)
        lines.append(f"  {style_text(usage, ANSI_CYAN, color)}")
        lines.append(f"    {command['summary']}")
    lines.append("")
    lines.append(render_tool_catalog(include_internal=include_internal))
    lines.append("")
    lines.append("Tip: Type `/command ...` to run a command, or a plain task to start work.")
    return "\n".join(lines).rstrip()


def slash_usage(command: dict[str, str]) -> str:
    if command["name"] == "run":
        return '/run "fix tests"'
    command_usage = command["usage"]
    if command_usage == "harness /":
        return "/"
    if command_usage.startswith("harness "):
        return f"/{command_usage.removeprefix('harness ')}"
    return command_usage
