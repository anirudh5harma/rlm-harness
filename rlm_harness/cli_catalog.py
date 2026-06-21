from __future__ import annotations

import os
from typing import TextIO

# The visible surface (shown in `harness --help`). Kept to a
# Claude Code / Codex-style primary set: one task verb, a couple
# of inspect/setup commands. Everything else is a hidden alias
# that still works for scripts and muscle memory but is no longer
# advertised.
PUBLIC_COMMANDS = {
    "init",
    "doctor",
    "status",
    "history",
    "mcp",
    "eval",
    "update",
    "install",
}

# Hidden aliases: still registered as top-level subcommands so
# existing scripts and slash commands keep working, but no longer
# shown in `--help`. They route to the canonical behavior.
LEGACY_COMMANDS = {
    "run",  # `harness "task"` is the primary shape
    "work",  # alias for `harness "task"`
    "ask",  # alias for `harness "task" --ask`
    "plan",  # alias for `harness "task" --plan`
    "continue",  # alias for `harness --continue`
    "resume",  # alias for `harness --resume`
    "trace",  # alias for `history`
    "commands",  # alias for the command catalog
    "tools",  # alias for the tool catalog
    "palette",  # alias for the slash palette
    "taste",  # taste/feedback/evolve learning surface
    "profile",  # legacy alias for `taste`
    "evolve",  # alias for `taste evolve`
    "feedback",  # alias for `taste feedback`
    "model",  # alias for `init model`
    "provider",  # alias for `init provider`
    "config",  # alias for `status config`
    "readiness",  # alias for `doctor`
    "dogfood",  # alias for `eval dogfood`
}
INTERNAL_COMMANDS = {
    "langgraph-plan",
    "benchmark-model",
    "check-model",
    "serve-mlx",
    "mem",
    "sandbox",
}
ALL_COMMANDS = PUBLIC_COMMANDS | LEGACY_COMMANDS | INTERNAL_COMMANDS
DEFAULT_PROG = "harness"
SLASH_HIDDEN_COMMANDS = {"tools"}

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_BLUE = ANSI_CYAN
ANSI_RED = "\033[31m"

COMMAND_CATALOG = [
    {
        "name": "run",
        "usage": 'harness "fix tests"',
        "group": "work",
        "summary": (
            "Run a coding task. Pass --plan for a read-only plan, "
            "--ask for a read-only answer."
        ),
    },
    {
        "name": "continue",
        "usage": "harness --continue [task]",
        "group": "work",
        "summary": "Continue the latest thread.",
    },
    {
        "name": "resume",
        "usage": "harness --resume <thread-id> [task]",
        "group": "work",
        "summary": "Resume a specific thread by id.",
    },
    {
        "name": "history",
        "usage": "harness history list|report|show|events|replay",
        "group": "inspect",
        "summary": "Inspect run history, timelines, and replay traces.",
    },
    {
        "name": "status",
        "usage": "harness status",
        "group": "inspect",
        "summary": "Show provider, latest run, taste, and recommended next actions.",
    },
    {
        "name": "mcp",
        "usage": "harness mcp list|setup|add|show|tools|trust|enable",
        "group": "inspect",
        "summary": "Manage MCP servers, auth hints, and workflow purposes.",
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
        "summary": "Check local dependencies and sandbox health.",
    },
    {
        "name": "update",
        "usage": "harness update",
        "group": "setup",
        "summary": "Upgrade the managed install and rebuild the sandbox image.",
    },
    {
        "name": "install",
        "usage": "harness install <source>",
        "group": "setup",
        "summary": "Install, refresh, or list harness extensions.",
    },
    {
        "name": "eval",
        "usage": "harness eval <suite>",
        "group": "quality",
        "summary": "Run local harness evaluation suites.",
    },
]

# Hidden aliases still registered as subcommands so existing scripts
# and slash commands keep working. Shown via `harness commands --all`.
LEGACY_COMMAND_CATALOG = [
    {
        "name": "ask",
        "usage": 'harness ask "what is this project?"',
        "group": "work",
        "summary": "Hidden alias for `harness \"task\" --ask`.",
    },
    {
        "name": "plan",
        "usage": 'harness plan "how should we fix this?"',
        "group": "work",
        "summary": "Hidden alias for `harness \"task\" --plan`.",
    },
    {
        "name": "work",
        "usage": 'harness work "fix tests"',
        "group": "work",
        "summary": "Hidden alias for `harness \"fix tests\"`.",
    },
    {
        "name": "trace",
        "usage": "harness trace list|report|show|events|replay",
        "group": "inspect",
        "summary": "Hidden alias for `history`.",
    },
    {
        "name": "commands",
        "usage": "harness commands",
        "group": "inspect",
        "summary": "Print this command surface (--all includes hidden aliases).",
    },
    {
        "name": "tools",
        "usage": "harness tools",
        "group": "inspect",
        "summary": "List action capabilities, risks, and confirmation requirements.",
    },
    {
        "name": "palette",
        "usage": "harness /",
        "group": "inspect",
        "summary": "Show the slash command palette.",
    },
    {
        "name": "readiness",
        "usage": "harness readiness",
        "group": "setup",
        "summary": "Hidden alias for `doctor`.",
    },
    {
        "name": "config",
        "usage": "harness config",
        "group": "setup",
        "summary": "Show saved provider, model, and profile paths.",
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
        "name": "taste",
        "usage": "harness taste list|context|learn|scan|approve|reject",
        "group": "learn",
        "summary": "Manage taste learning and preferences.",
    },
    {
        "name": "profile",
        "usage": "harness profile list|context|learn|scan|approve|reject",
        "group": "learn",
        "summary": "Compatibility alias for `taste`.",
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
        "name": "dogfood",
        "usage": "harness dogfood",
        "group": "quality",
        "summary": "Run readiness, eval, and feedback proof checks.",
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


def command_catalog(
    *, include_internal: bool = False, include_legacy: bool = False
) -> list[dict[str, str]]:
    commands = [dict(command) for command in COMMAND_CATALOG]
    if include_legacy:
        commands.extend(dict(command) for command in LEGACY_COMMAND_CATALOG)
    if include_internal:
        commands.extend(dict(command) for command in INTERNAL_COMMAND_CATALOG)
    return commands


def slash_command_catalog(*, include_internal: bool = False) -> list[dict[str, str]]:
    # The interactive slash palette stays rich: it includes the
    # hidden aliases so in-REPL `/ask`, `/plan`, `/taste`, etc. keep
    # working and stay discoverable inside the REPL even though they
    # are hidden from `--help`.
    return [
        command
        for command in command_catalog(
            include_internal=include_internal, include_legacy=True
        )
        if command["name"] not in SLASH_HIDDEN_COMMANDS
    ]


def render_command_catalog(commands: list[dict[str, str]], *, color: bool = False) -> str:
    lines = [style_text("Harness commands", ANSI_CYAN, color), ""]
    groups = ["work", "inspect", "setup", "quality", "learn", "runtime"]
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
    for command in slash_command_catalog(include_internal=include_internal):
        usage = slash_usage(command)
        lines.append(f"  {style_text(usage, ANSI_CYAN, color)}")
        lines.append(f"    {command['summary']}")
    lines.append("")
    lines.append("Tip: Type `/command ...` to run a command, or a plain task to start work.")
    return "\n".join(lines).rstrip()


def slash_usage(command: dict[str, str]) -> str:
    name = command["name"]
    if name == "run":
        return '/run "fix tests"'
    if name == "continue":
        return "/continue [task]"
    if name == "resume":
        return "/resume <thread-id> [task]"
    command_usage = command["usage"]
    if command_usage == "harness /":
        return "/"
    if command_usage.startswith("harness "):
        return f"/{command_usage.removeprefix('harness ')}"
    return command_usage
