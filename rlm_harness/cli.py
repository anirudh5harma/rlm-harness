from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import Nodes
from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.model_server import MLXServer, MLXServerConfig, MLXServerError
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg


def default_trace_path() -> Path:
    return Path(os.environ.get("RLM_HARNESS_TRACE_DB", ".rlm_harness/traces.db"))


def build_client(args: argparse.Namespace) -> LMClient:
    return LMClient(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY"),
        timeout_s=args.timeout,
    )


def cmd_run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    traces = TraceStore(Path(args.trace_db))
    run_id = traces.start_run(args.task, str(workspace), thread_id=args.thread_id)
    state = HarnessState(
        task=args.task,
        workspace=str(workspace),
        thread_id=args.thread_id or run_id,
        run_id=run_id,
        token_budget=args.token_budget,
    )
    traces.event(
        run_id,
        "run_started",
        {"task": args.task, "workspace": str(workspace)},
        node="cli",
    )

    try:
        graph = build_graph(Nodes(build_client(args), traces), backend=args.graph_backend)
        final_state = graph.invoke(state)
        traces.finish_run(run_id, final_state.status)
    except LMClientError as exc:
        traces.event(run_id, "error", {"message": str(exc)}, node="cli")
        traces.finish_run(run_id, "error")
        print(f"Run failed: {exc}", file=sys.stderr)
        print(traces.render_report(run_id))
        return 1

    if final_state.final_answer:
        print(final_state.final_answer)
    print()
    print(traces.render_report(run_id))
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
    completion = client.complete(
        [Msg(role="user", content=args.prompt)],
        max_tokens=args.max_tokens,
        temperature=0,
    )
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rlm-harness")
    subparsers = root.add_subparsers(dest="command", required=True)

    def add_model_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--provider", default="stub", choices=["stub", "openai-compatible"])
        command.add_argument("--model", default="stub")
        command.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
        command.add_argument("--api-key", default=None)
        command.add_argument("--timeout", type=int, default=120)

    run = subparsers.add_parser("run")
    run.add_argument("task")
    run.add_argument("--workspace", default=".")
    run.add_argument("--trace-db", default=str(default_trace_path()))
    run.add_argument("--thread-id", default=None)
    run.add_argument("--token-budget", type=int, default=100000)
    run.add_argument("--graph-backend", default="auto", choices=["auto", "simple", "langgraph"])
    add_model_args(run)
    run.set_defaults(func=cmd_run)

    benchmark = subparsers.add_parser("benchmark-model")
    benchmark.add_argument("--max-tokens", type=int, default=300)
    add_model_args(benchmark)
    benchmark.set_defaults(func=cmd_benchmark_model)

    check_model = subparsers.add_parser("check-model")
    check_model.add_argument("--prompt", default="Reply with exactly: hello")
    check_model.add_argument("--max-tokens", type=int, default=64)
    add_model_args(check_model)
    check_model.set_defaults(func=cmd_check_model)

    serve_mlx = subparsers.add_parser("serve-mlx")
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
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
