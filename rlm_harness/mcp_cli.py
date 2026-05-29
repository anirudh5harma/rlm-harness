from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from rlm_harness.mcp_client import MCPClient, MCPClientError
from rlm_harness.mcp_config import (
    MCP_CONFIG_PATH,
    MCPAuthConfig,
    MCPConfigStore,
    MCPServerConfig,
    parse_key_values,
    render_mcp_catalog,
)


def cmd_mcp(args: argparse.Namespace) -> int:
    store = MCPConfigStore(Path(args.mcp_config))
    if args.mcp_command == "list":
        servers = store.list()
        if args.json_output:
            print(json.dumps([server.to_public_dict() for server in servers], sort_keys=True))
        else:
            print(render_mcp_catalog(servers))
        return 0
    if args.mcp_command == "show":
        server = store.get(args.name)
        if server is None:
            print(f"Unknown MCP server: {args.name}", file=sys.stderr)
            return 1
        if args.json_output:
            print(json.dumps(server.to_public_dict(), sort_keys=True))
        else:
            print(render_mcp_catalog([server]))
        return 0
    if args.mcp_command == "remove":
        if not store.remove(args.name):
            print(f"Unknown MCP server: {args.name}", file=sys.stderr)
            return 1
        print(f"removed\t{args.name}")
        return 0
    if args.mcp_command in {"trust", "untrust", "enable", "disable"}:
        changes = {}
        if args.mcp_command in {"trust", "untrust"}:
            changes["trusted"] = args.mcp_command == "trust"
        else:
            changes["enabled"] = args.mcp_command == "enable"
        server = store.update(args.name, **changes)
        if server is None:
            print(f"Unknown MCP server: {args.name}", file=sys.stderr)
            return 1
        if args.json_output:
            print(json.dumps(server.to_public_dict(), sort_keys=True))
        else:
            enabled = "enabled" if server.enabled else "disabled"
            trusted = "trusted" if server.trusted else "approval-gated"
            print(f"{server.name}\t{enabled}\t{trusted}")
        return 0
    if args.mcp_command == "tools":
        try:
            server = store.select(args.name, args.purpose)
            result = MCPClient(server, timeout_s=args.timeout).list_tools()
        except (ValueError, MCPClientError) as exc:
            print(f"MCP tools failed: {exc}", file=sys.stderr)
            return 1
        payload = {"server": server.name, **result}
        if args.json_output:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(render_mcp_tools(payload))
        return 0
    if args.mcp_command == "add":
        try:
            env = parse_key_values(args.env, option="--env")
            headers = parse_key_values(args.header, option="--header")
            server = MCPServerConfig(
                name=args.name,
                transport=args.transport,
                url=args.url,
                command=args.command,
                args=[*args.arg, *getattr(args, "command_args", [])],
                env=env,
                headers=headers,
                auth=MCPAuthConfig(
                    type=args.auth,
                    token_env=args.token_env,
                    api_key_env=args.api_key_env,
                    api_key_header=args.api_key_header,
                    scopes=args.scope,
                ),
                purposes=args.purpose,
                enabled=not args.disabled,
                trusted=args.trusted,
            )
        except ValueError as exc:
            print(f"MCP command failed: {exc}", file=sys.stderr)
            return 1
        store.add(server)
        print(f"added\t{server.name}")
        return 0
    if args.mcp_command == "setup":
        return cmd_mcp_setup(args, store)
    print(f"Unknown MCP command: {args.mcp_command}", file=sys.stderr)
    return 1


def cmd_mcp_setup(args: argparse.Namespace, store: MCPConfigStore) -> int:
    if not sys.stdin.isatty():
        print(
            "MCP setup needs an interactive terminal. "
            "Use `harness mcp add ...` for scripted setup.",
            file=sys.stderr,
        )
        return 1

    print("Harness MCP setup")
    try:
        name = args.name or prompt_required("Server name: ")
        transport = prompt_choice(
            ["http", "sse", "stdio"],
            "Transport [http/sse/stdio]",
            default="http",
        )
        url = None
        command = None
        if transport in {"http", "sse"}:
            url = prompt_required("Server URL: ")
        else:
            command = prompt_required("Command: ")
        purpose_text = prompt_text("Purposes, comma-separated", default=name) or name
        auth_default = "bearer_env" if transport in {"http", "sse"} else "none"
        auth_type = prompt_choice(
            ["none", "bearer_env", "api_key_env"],
            "Auth [none/bearer_env/api_key_env]",
            default=auth_default,
        )
        token_env = None
        api_key_env = None
        api_key_header = "x-api-key"
        if auth_type == "bearer_env":
            token_env = prompt_text(
                "Bearer token env var",
                default=default_mcp_credential_env(name, suffix="TOKEN"),
            )
        elif auth_type == "api_key_env":
            api_key_env = prompt_text(
                "API key env var",
                default=default_mcp_credential_env(name, suffix="API_KEY"),
            )
            api_key_header = prompt_text("API key header", default="x-api-key") or "x-api-key"
        trusted = bool(args.trusted)
        if not trusted:
            trusted = prompt_yes_no("Trust this MCP for autonomous workflow calls?", default=False)
        server = MCPServerConfig(
            name=name,
            transport=transport,
            url=url,
            command=command,
            auth=MCPAuthConfig(
                type=auth_type,
                token_env=token_env,
                api_key_env=api_key_env,
                api_key_header=api_key_header,
            ),
            purposes=parse_mcp_purposes(purpose_text),
            enabled=not args.disabled,
            trusted=trusted,
        )
    except ValueError as exc:
        print(f"MCP setup failed: {exc}", file=sys.stderr)
        return 1

    store.add(server)
    if args.json_output:
        print(json.dumps(server.to_public_dict(), sort_keys=True))
    else:
        print(f"added\t{server.name}")
        print(f"purpose\t{', '.join(server.purposes) or '-'}")
        print(f"auth\t{server.auth.summary()}")
        print(f"trust\t{'trusted' if server.trusted else 'approval-gated'}")
        print(f"next\tharness mcp tools {server.name}")
    return 0


def prompt_required(prompt: str) -> str:
    value = prompt_text(prompt)
    if not value:
        raise ValueError(f"{prompt.rstrip(': ')} is required")
    return value


def prompt_text(prompt: str, *, default: Optional[str] = None) -> Optional[str]:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return raw or default


def prompt_choice(options: list[str], prompt: str, *, default: str) -> str:
    value = (prompt_text(prompt, default=default) or default).strip().lower()
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < len(options):
            return options[index]
    if value not in options:
        raise ValueError(f"{prompt} must be one of: {', '.join(options)}")
    return value


def prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    value = prompt_text(f"{prompt} [{default_label}]")
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"y", "yes", "true", "1"}:
        return True
    if normalized in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"{prompt} expects yes or no")


def default_mcp_credential_env(name: str, *, suffix: str) -> str:
    prefix = "".join(char if char.isalnum() else "_" for char in name.upper()).strip("_")
    return f"{prefix or 'MCP'}_{suffix}"


def parse_mcp_purposes(text: str) -> list[str]:
    return [purpose.strip() for purpose in text.replace(";", ",").split(",") if purpose.strip()]


def render_mcp_tools(payload: dict[str, object]) -> str:
    server = str(payload.get("server") or "mcp")
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return f"No MCP tools returned by {server}."
    lines = [f"Harness MCP tools from {server}", ""]
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        description = str(tool.get("description") or "").strip()
        lines.append(f"{name}\t{description}" if description else name)
    return "\n".join(lines).rstrip()


def add_mcp_command(subparsers) -> None:
    mcp = subparsers.add_parser("mcp", help="Manage MCP servers.")
    mcp.add_argument("--mcp-config", default=str(MCP_CONFIG_PATH))
    mcp_subparsers = mcp.add_subparsers(dest="mcp_command", required=True)

    mcp_list = mcp_subparsers.add_parser("list")
    mcp_list.add_argument("--json", dest="json_output", action="store_true")
    mcp_list.set_defaults(func=cmd_mcp)

    mcp_show = mcp_subparsers.add_parser("show")
    mcp_show.add_argument("name")
    mcp_show.add_argument("--json", dest="json_output", action="store_true")
    mcp_show.set_defaults(func=cmd_mcp)

    mcp_remove = mcp_subparsers.add_parser("remove")
    mcp_remove.add_argument("name")
    mcp_remove.set_defaults(func=cmd_mcp)

    mcp_setup = mcp_subparsers.add_parser("setup")
    mcp_setup.add_argument("name", nargs="?")
    mcp_setup.add_argument("--trusted", action="store_true")
    mcp_setup.add_argument("--disabled", action="store_true")
    mcp_setup.add_argument("--json", dest="json_output", action="store_true")
    mcp_setup.set_defaults(func=cmd_mcp)

    mcp_tools = mcp_subparsers.add_parser("tools")
    mcp_tools.add_argument("name", nargs="?")
    mcp_tools.add_argument("--purpose", default=None)
    mcp_tools.add_argument("--timeout", type=float, default=30.0)
    mcp_tools.add_argument("--json", dest="json_output", action="store_true")
    mcp_tools.set_defaults(func=cmd_mcp)

    for command_name in ("trust", "untrust", "enable", "disable"):
        mcp_state = mcp_subparsers.add_parser(command_name)
        mcp_state.add_argument("name")
        mcp_state.add_argument("--json", dest="json_output", action="store_true")
        mcp_state.set_defaults(func=cmd_mcp)

    mcp_add = mcp_subparsers.add_parser("add")
    mcp_add.add_argument("name")
    mcp_add.add_argument("--transport", choices=["stdio", "http", "sse"], default="stdio")
    mcp_add.add_argument("--url", default=None)
    mcp_add.add_argument("--command", default=None)
    mcp_add.add_argument("--arg", action="append", default=[])
    mcp_add.add_argument(
        "--args",
        dest="command_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="All remaining arguments to pass to a stdio MCP command.",
    )
    mcp_add.add_argument("--env", action="append", default=[])
    mcp_add.add_argument("--header", action="append", default=[])
    mcp_add.add_argument("--purpose", action="append", default=[])
    mcp_add.add_argument(
        "--auth",
        choices=["none", "bearer_env", "api_key_env", "oauth"],
        default="none",
    )
    mcp_add.add_argument("--token-env", default=None)
    mcp_add.add_argument("--api-key-env", default=None)
    mcp_add.add_argument("--api-key-header", default="x-api-key")
    mcp_add.add_argument("--scope", action="append", default=[])
    mcp_add.add_argument("--trusted", action="store_true")
    mcp_add.add_argument("--disabled", action="store_true")
    mcp_add.set_defaults(func=cmd_mcp)
