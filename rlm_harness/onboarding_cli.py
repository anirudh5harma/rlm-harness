from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from rlm_harness import config as harness_config
from rlm_harness.config import (
    default_api_key,
    default_base_url,
    default_memory_path,
    default_model,
    default_profile_path,
    default_provider,
    default_trace_path,
    masked_secret,
    save_user_config,
)
from rlm_harness.mcp_config import MCP_CONFIG_PATH, MCPConfigStore
from rlm_harness.memory import Memory, MemoryError
from rlm_harness.memory.evolution import EvolutionProposalManager
from rlm_harness.memory.profile import TasteProfileStore
from rlm_harness.project_style import scan_project_style
from rlm_harness.providers import PROVIDERS, normalize_provider, provider_preset, static_models
from rlm_harness.readiness import build_readiness_report, render_readiness_report
from rlm_harness.tracing import TraceStore


def current_config_payload() -> dict[str, str]:
    provider = default_provider()
    return {
        "provider": provider,
        "model": default_model(),
        "base_url": default_base_url(),
        "api_key": masked_secret(default_api_key(provider)),
        "config_path": str(harness_config.CONFIG_PATH),
        "mcp_config_path": str(MCP_CONFIG_PATH),
        "profile_path": str(default_profile_path()),
    }


def print_current_config() -> None:
    config = current_config_payload()
    print(f"provider\t{config['provider']}")
    print(f"model\t{config['model']}")
    print(f"base_url\t{config['base_url']}")
    print(f"api_key\t{config['api_key']}")
    print(f"config\t{config['config_path']}")
    print(f"mcp_config\t{config['mcp_config_path']}")
    print(f"profile\t{config['profile_path']}")


def cmd_config(args: argparse.Namespace) -> int:
    if args.json_output:
        print(json.dumps(current_config_payload(), sort_keys=True))
    else:
        print_current_config()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    payload = status_payload(args)
    if args.json_output:
        print(json.dumps(payload, sort_keys=True))
    else:
        print_status_payload(payload)
    return 0


def status_payload(args: argparse.Namespace) -> dict[str, object]:
    provider = default_provider()
    profile_path = Path(args.profile_db)
    memory_path = Path(args.memory_db)
    trace_path = Path(args.trace_db)
    mcp_path = Path(getattr(args, "mcp_config", str(MCP_CONFIG_PATH)))
    payload: dict[str, object] = {
        "provider": provider,
        "model": default_model(),
        "base_url": default_base_url(),
        "api_key": "set" if default_api_key(provider) else "missing",
        "paths": {
            "config": str(harness_config.CONFIG_PATH),
            "mcp_config": str(mcp_path),
            "trace_db": str(trace_path),
            "memory_db": str(memory_path),
            "profile_db": str(profile_path),
        },
        "latest_run": latest_run_payload(trace_path),
        "mcp": mcp_status_counts(MCPConfigStore(mcp_path)),
        "taste": taste_status_counts(profile_path, memory_path),
        "evolution": evolution_status_counts(profile_path, memory_path),
    }
    payload["next"] = status_next_actions(payload)
    return payload


def status_next_actions(payload: dict[str, object]) -> list[str]:
    actions: list[str] = []
    provider = str(payload.get("provider") or "")
    api_key = str(payload.get("api_key") or "")
    latest = payload.get("latest_run")
    taste = payload.get("taste")
    evolution = payload.get("evolution")
    mcp = payload.get("mcp")

    if provider == "stub":
        actions.append("Run `harness init --provider openrouter --api-key <key>`.")
    elif api_key == "missing":
        actions.append(f"Run `harness /provider {provider} --api-key <key>`.")

    if isinstance(latest, dict):
        thread_id = latest.get("thread_id")
        if thread_id:
            actions.append("Run `harness continue` to resume the latest thread.")
    else:
        actions.append('Run `harness ask "what is this project?"` for orientation.')

    if isinstance(taste, dict) and int(taste.get("active", 0) or 0) == 0:
        actions.append("Run `harness taste scan` to learn this project's style.")

    if isinstance(evolution, dict) and int(evolution.get("pending", 0) or 0) > 0:
        actions.append("Review pending improvements with `harness evolve list`.")

    if isinstance(mcp, dict):
        configured = int(mcp.get("configured", 0) or 0)
        authenticated = int(mcp.get("authenticated", 0) or 0)
        credentials_present = int(mcp.get("credentials_present", 0) or 0)
        if configured == 0:
            actions.append("Optional: add workflow MCPs with `harness mcp setup`.")
        elif authenticated > credentials_present:
            actions.append("Set missing MCP credential env vars or disable unused MCPs.")

    result = []
    seen = set()
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        result.append(action)
    return result[:5]


def latest_run_payload(trace_path: Path) -> Optional[dict[str, object]]:
    if not trace_path.exists():
        return None
    try:
        runs = TraceStore(trace_path).list_runs(limit=1)
    except (OSError, sqlite3.Error, ValueError):
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "run_id": run["run_id"],
        "thread_id": run["thread_id"],
        "status": run["status"],
        "task": run["task"],
        "workspace": run["workspace"],
        "started_at": run["started_at"],
    }


def taste_status_counts(profile_path: Path, memory_path: Path) -> dict[str, int]:
    counts = {"active": 0, "pending": 0, "rejected": 0}
    for path in unique_existing_paths(profile_path, memory_path):
        try:
            with Memory(path) as memory:
                for status in counts:
                    counts[status] += len(TasteProfileStore(memory).records(status=status))
        except MemoryError:
            continue
    return counts


def evolution_status_counts(profile_path: Path, memory_path: Path) -> dict[str, int]:
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    memories = []
    with ExitStack() as stack:
        for path in unique_existing_paths(profile_path, memory_path):
            try:
                memories.append(stack.enter_context(Memory(path)))
            except MemoryError:
                continue
        user_memory = memories[0] if memories else None
        project_memory = memories[1] if len(memories) > 1 else None
        manager = EvolutionProposalManager(user_memory, project_memory)
        for status in counts:
            counts[status] = len(manager.proposals(status=status))
    return counts


def mcp_status_counts(store: MCPConfigStore) -> dict[str, int]:
    servers = store.list()
    return {
        "configured": len(servers),
        "enabled": sum(1 for server in servers if server.enabled),
        "authenticated": sum(1 for server in servers if server.auth.is_authenticated),
        "credentials_present": sum(1 for server in servers if server.auth.credential_available),
    }


def unique_existing_paths(*paths: Path) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        result.append(resolved)
    return result


def print_status_payload(payload: dict[str, object]) -> None:
    print("Harness status")
    print(f"provider\t{payload['provider']}")
    print(f"model\t{payload['model']}")
    print(f"api_key\t{payload['api_key']}")
    latest = payload.get("latest_run")
    if isinstance(latest, dict):
        print(f"latest_thread\t{latest.get('thread_id')}")
        print(f"latest_status\t{latest.get('status')}")
        print(f"latest_task\t{latest.get('task')}")
    else:
        print("latest_thread\t-")
    taste = payload.get("taste")
    if isinstance(taste, dict):
        print(
            "taste\t"
            f"active={taste.get('active', 0)} "
            f"pending={taste.get('pending', 0)} "
            f"rejected={taste.get('rejected', 0)}"
        )
    evolution = payload.get("evolution")
    if isinstance(evolution, dict):
        print(
            "evolution\t"
            f"pending={evolution.get('pending', 0)} "
            f"approved={evolution.get('approved', 0)} "
            f"rejected={evolution.get('rejected', 0)}"
        )
    mcp = payload.get("mcp")
    if isinstance(mcp, dict):
        print(
            "mcp\t"
            f"configured={mcp.get('configured', 0)} "
            f"enabled={mcp.get('enabled', 0)} "
            f"authenticated={mcp.get('authenticated', 0)} "
            f"credentials_present={mcp.get('credentials_present', 0)}"
        )
    paths = payload.get("paths")
    if isinstance(paths, dict):
        print(f"trace_db\t{paths.get('trace_db')}")
        print(f"profile_db\t{paths.get('profile_db')}")
        print(f"memory_db\t{paths.get('memory_db')}")
        print(f"mcp_config\t{paths.get('mcp_config')}")
    next_actions = payload.get("next")
    if isinstance(next_actions, list) and next_actions:
        print("next")
        for action in next_actions:
            print(f"  {action}")


def cmd_readiness(args: argparse.Namespace) -> int:
    report = build_readiness_report(Path.cwd(), check_docker=not args.no_docker)
    if args.json_output:
        print(report.to_json())
    else:
        print(render_readiness_report(report))
    return 0 if report.status in {"ready", "degraded"} else 1


def cmd_init(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    mcp_config_path = Path(args.mcp_config)
    profile_path = Path(args.profile_db)
    try:
        config_updates = init_config_updates(args)
    except ValueError as exc:
        print(f"Init failed: {exc}", file=sys.stderr)
        return 1
    if config_updates:
        save_user_config(config_updates)

    style_records = []
    if not args.no_style_scan:
        try:
            style_records = scan_project_style(workspace, max_files=args.max_files)
            with Memory(Path(args.memory_db)) as memory:
                store = TasteProfileStore(memory)
                style_records = [store.add(record) for record in style_records]
        except MemoryError as exc:
            print(f"Init failed: {exc}", file=sys.stderr)
            return 1

    readiness = build_readiness_report(workspace, check_docker=not args.no_docker)
    config_payload = current_config_payload()
    config_payload["mcp_config_path"] = str(mcp_config_path)
    config_payload["profile_path"] = str(profile_path)
    payload = {
        "config_updated": bool(config_updates),
        "config": config_payload,
        "mcp": mcp_status_counts(MCPConfigStore(mcp_config_path)),
        "workspace": str(workspace),
        "style_records": [record.to_dict() for record in style_records],
        "readiness": readiness.to_dict(),
        "next": init_next_action(readiness),
    }
    if args.json_output:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(render_init_report(payload))
    return 0


def init_config_updates(args: argparse.Namespace) -> dict[str, str]:
    updates: dict[str, str] = {}
    if args.provider:
        provider = normalize_provider(args.provider)
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider: {args.provider}")
        preset = provider_preset(provider)
        updates["provider"] = provider
        updates["base_url"] = args.base_url or preset.base_url
        if args.model:
            updates["model"] = args.model
        elif provider == "stub":
            updates["model"] = "stub"
        elif args.set_default_model:
            updates["model"] = static_models(provider)[0]
    elif args.model:
        updates["model"] = args.model
    if args.api_key:
        updates["api_key"] = args.api_key
    if args.base_url and "base_url" not in updates:
        updates["base_url"] = args.base_url
    return updates


def init_next_action(readiness) -> str:
    for check in readiness.checks:
        if check.status == "blocked" and check.action:
            return check.action
    if readiness.status == "ready":
        return 'Run `harness ask "what is this project?"` or `harness work "fix tests"`.'
    return "Review readiness warnings, then start with a small ask or plan command."


def render_init_report(payload: dict[str, object]) -> str:
    lines = ["Harness init"]
    lines.append(f"workspace\t{payload['workspace']}")
    lines.append(f"config_updated\t{payload['config_updated']}")
    style_records = payload.get("style_records")
    if isinstance(style_records, list):
        lines.append(f"project_style_records\t{len(style_records)}")
    mcp = payload.get("mcp")
    if isinstance(mcp, dict):
        lines.append(
            "mcp\t"
            f"configured={mcp.get('configured', 0)} "
            f"enabled={mcp.get('enabled', 0)} "
            f"authenticated={mcp.get('authenticated', 0)} "
            f"credentials_present={mcp.get('credentials_present', 0)}"
        )
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        lines.append(f"readiness\t{readiness.get('status')}")
        for check in readiness.get("checks", []):
            if not isinstance(check, dict):
                continue
            line = f"  {check.get('name')}\t{check.get('status')}\t{check.get('detail')}"
            lines.append(line)
            if check.get("action"):
                lines.append(f"    next\t{check.get('action')}")
    lines.append(f"next\t{payload['next']}")
    return "\n".join(lines)


def add_onboarding_commands(subparsers) -> None:
    status = subparsers.add_parser("status", help="Show current harness status.")
    status.add_argument("--json", dest="json_output", action="store_true")
    status.add_argument("--trace-db", default=str(default_trace_path()))
    status.add_argument(
        "--memory-db",
        default=str(default_memory_path()),
        help=argparse.SUPPRESS,
    )
    status.add_argument(
        "--profile-db",
        default=str(default_profile_path()),
        help=argparse.SUPPRESS,
    )
    status.add_argument(
        "--mcp-config",
        default=str(MCP_CONFIG_PATH),
        help=argparse.SUPPRESS,
    )
    status.set_defaults(func=cmd_status)

    init = subparsers.add_parser("init", help="Bootstrap Harness for this workspace.")
    init.add_argument("--provider", default=None)
    init.add_argument("--model", default=None)
    init.add_argument("--api-key", default=None)
    init.add_argument("--base-url", default=None)
    init.add_argument("--workspace", default=".")
    init.add_argument("--memory-db", default=str(default_memory_path()))
    init.add_argument("--profile-db", default=str(default_profile_path()))
    init.add_argument("--mcp-config", default=str(MCP_CONFIG_PATH))
    init.add_argument("--max-files", type=int, default=400)
    init.add_argument("--no-style-scan", action="store_true")
    init.add_argument("--no-docker", action="store_true")
    init.add_argument("--json", dest="json_output", action="store_true")
    init.add_argument(
        "--keep-model",
        dest="set_default_model",
        action="store_false",
        default=True,
        help="Keep the current model when switching providers.",
    )
    init.set_defaults(func=cmd_init)

    config_cmd = subparsers.add_parser("config", help="Show saved harness configuration.")
    config_cmd.add_argument("--json", dest="json_output", action="store_true")
    config_cmd.set_defaults(func=cmd_config)

    readiness = subparsers.add_parser(
        "readiness",
        help="Check whether Harness is ready for daily coding work.",
    )
    readiness.add_argument("--json", dest="json_output", action="store_true")
    readiness.add_argument(
        "--no-docker",
        action="store_true",
        help="Skip Docker and sandbox checks.",
    )
    readiness.set_defaults(func=cmd_readiness)
