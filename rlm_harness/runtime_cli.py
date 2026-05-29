from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from rlm_harness.config import default_api_key
from rlm_harness.maintenance_cli import module_status
from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.model_server import MLXServer, MLXServerConfig, MLXServerError
from rlm_harness.providers import normalize_provider
from rlm_harness.types import Msg


def build_client(args: argparse.Namespace) -> LMClient:
    provider = normalize_provider(args.provider)
    return LMClient(
        provider=provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or default_api_key(provider),
        timeout_s=args.timeout,
    )


def cmd_langgraph_plan(args: argparse.Namespace) -> int:
    scopes = [
        {
            "scope": "durable_checkpoints",
            "status": module_status("langgraph.checkpoint.sqlite"),
            "next": "Enabled for LangGraph runs via --checkpoint-db.",
        },
        {
            "scope": "state_reducers",
            "status": "open",
            "next": "Optional: migrate HarnessState to a TypedDict with annotated reducers.",
        },
        {
            "scope": "tool_nodes",
            "status": "partial",
            "next": "Action execution is a graph node; LangGraph ToolNode remains optional.",
        },
        {
            "scope": "streaming_observability",
            "status": "ok",
            "next": "Use --stream with LangGraph backend.",
        },
    ]
    if args.json_output:
        print(json.dumps(scopes, sort_keys=True))
    else:
        for item in scopes:
            print(f"{item['scope']}\t{item['status']}\t{item['next']}")
    return 0


def cmd_benchmark_model(args: argparse.Namespace) -> int:
    client = build_client(args)
    prompts = [
        "Reply with exactly: ok",
        "Return JSON with keys language and purpose for Python.",
        "Write a three-step plan for fixing a failing unit test.",
    ]
    latencies = []
    for prompt in prompts:
        started = time.perf_counter()
        completion = client.complete(
            [Msg(role="user", content=prompt)],
            max_tokens=args.max_tokens,
            temperature=0,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        latencies.append(elapsed_ms)
        print(f"Prompt: {prompt}")
        print(f"Latency: {elapsed_ms}ms")
        print(f"Response: {completion.content.strip()}")
        print()

    print(f"Latency mean: {int(statistics.mean(latencies))}ms")
    print(f"Latency min/max: {min(latencies)}ms/{max(latencies)}ms")
    return 0


def cmd_check_model(args: argparse.Namespace) -> int:
    client = build_client(args)
    try:
        completion = client.complete(
            [Msg(role="user", content=args.prompt)],
            max_tokens=args.max_tokens,
            temperature=0,
        )
    except LMClientError as exc:
        print(f"Model check failed: {exc}", file=sys.stderr)
        return 1
    print(completion.content.strip())
    print(
        f"model={completion.model} provider={completion.provider} "
        f"latency_ms={completion.latency_ms}"
    )
    return 0


def cmd_serve_mlx(args: argparse.Namespace) -> int:
    config = MLXServerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        executable=args.executable,
        extra_args=tuple(args.extra_arg or ()),
    )
    server = MLXServer(config)
    print("Starting MLX server: {}".format(" ".join(config.command())))
    try:
        server.start(wait=True, timeout_s=args.ready_timeout)
    except MLXServerError as exc:
        print(f"Model server failed: {exc}", file=sys.stderr)
        return 1

    print(f"Ready at {config.base_url}")
    try:
        return server.wait()
    except KeyboardInterrupt:
        server.stop()
        return 130


def add_runtime_commands(subparsers, add_model_args) -> None:
    langgraph_plan = subparsers.add_parser(
        "langgraph-plan",
        help=argparse.SUPPRESS,
    )
    langgraph_plan.add_argument("--json", dest="json_output", action="store_true")
    langgraph_plan.set_defaults(func=cmd_langgraph_plan)

    benchmark = subparsers.add_parser("benchmark-model", help=argparse.SUPPRESS)
    benchmark.add_argument("--max-tokens", type=int, default=300)
    add_model_args(benchmark)
    benchmark.set_defaults(func=cmd_benchmark_model)

    check_model = subparsers.add_parser("check-model", help=argparse.SUPPRESS)
    check_model.add_argument("--prompt", default="Reply with exactly: hello")
    check_model.add_argument("--max-tokens", type=int, default=64)
    add_model_args(check_model)
    check_model.set_defaults(func=cmd_check_model)

    serve_mlx = subparsers.add_parser("serve-mlx", help=argparse.SUPPRESS)
    serve_mlx.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
    )
    serve_mlx.add_argument("--host", default="127.0.0.1")
    serve_mlx.add_argument("--port", type=int, default=8080)
    serve_mlx.add_argument("--executable", default="mlx_lm.server")
    serve_mlx.add_argument("--ready-timeout", type=float, default=60)
    serve_mlx.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument passed through to mlx_lm.server. Repeat for multiple args.",
    )
    serve_mlx.set_defaults(func=cmd_serve_mlx)
