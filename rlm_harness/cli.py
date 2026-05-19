from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import GraphRuntimeConfig, Nodes
from rlm_harness.memory import Memory, MemoryError, MemoryPagingConfig
from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.model_server import MLXServer, MLXServerConfig, MLXServerError
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg


def default_trace_path() -> Path:
    return Path(os.environ.get("RLM_HARNESS_TRACE_DB", ".rlm_harness/traces.db"))


def default_memory_path() -> Path:
    return Path(os.environ.get("RLM_HARNESS_MEMORY_DB", ".rlm_harness/memory.db"))


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
        {
            "task": args.task,
            "workspace": str(workspace),
            "sandbox_enabled": not args.no_sandbox,
            "memory_enabled": not args.no_memory,
        },
        node="cli",
    )

    try:
        memory_context = (
            nullcontext(None) if args.no_memory else Memory(Path(args.memory_db))
        )
        with memory_context as memory:
            runtime = GraphRuntimeConfig(
                sandbox_enabled=not args.no_sandbox,
                sandbox_config=SandboxConfig(
                    image=args.sandbox_image,
                    workspace=workspace,
                    memory=args.sandbox_memory,
                    cpus=args.sandbox_cpus,
                    default_timeout_s=args.sandbox_timeout,
                ),
                subcall_config=RLMSubcallConfig(
                    max_depth=args.max_depth,
                    max_subcalls=args.max_subcalls,
                    token_budget=args.token_budget,
                    max_tokens=args.subcall_max_tokens,
                ),
                max_action_retries=args.max_action_retries,
                memory=memory,
                memory_paging=MemoryPagingConfig(
                    max_history_tokens=args.max_history_tokens,
                    preserve_recent_steps=args.preserve_recent_steps,
                    recall_limit=args.recall_limit,
                    archival_limit=args.archival_limit,
                    summary_max_tokens=args.summary_max_tokens,
                ),
            )
            graph = build_graph(
                Nodes(build_client(args), traces, runtime),
                backend=args.graph_backend,
            )
            final_state = graph.invoke(state)
        traces.finish_run(run_id, final_state.status)
    except (LMClientError, MemoryError, SandboxError) as exc:
        traces.event(run_id, "error", {"message": str(exc)}, node="cli")
        traces.finish_run(run_id, "error")
        print(f"Run failed: {exc}", file=sys.stderr)
        print(traces.render_report(run_id))
        return 1

    if final_state.final_answer:
        print(final_state.final_answer)
    print()
    print(traces.render_report(run_id))
    return 0 if final_state.status == "done" else 1


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


def open_memory(args: argparse.Namespace) -> Memory:
    return Memory(Path(args.memory_db))


def cmd_mem_pin(args: argparse.Namespace) -> int:
    try:
        with open_memory(args) as memory:
            item = memory.core_set(args.key, args.value)
    except MemoryError as exc:
        print(f"Memory command failed: {exc}", file=sys.stderr)
        return 1
    print(f"pinned {item.key}")
    return 0


def cmd_mem_get(args: argparse.Namespace) -> int:
    try:
        with open_memory(args) as memory:
            value = memory.core_get(args.key)
    except MemoryError as exc:
        print(f"Memory command failed: {exc}", file=sys.stderr)
        return 1
    if value is None:
        return 1
    print(value)
    return 0


def cmd_mem_recall_append(args: argparse.Namespace) -> int:
    try:
        with open_memory(args) as memory:
            event = memory.recall_append(args.thread_id, args.role, args.content)
    except MemoryError as exc:
        print(f"Memory command failed: {exc}", file=sys.stderr)
        return 1
    print(f"recall {event.id}")
    return 0


def cmd_mem_recall_page(args: argparse.Namespace) -> int:
    try:
        with open_memory(args) as memory:
            events = memory.recall_page(args.thread_id, query=args.query or "", k=args.limit)
    except MemoryError as exc:
        print(f"Memory command failed: {exc}", file=sys.stderr)
        return 1
    for event in events:
        print(f"{event.id}\t{event.ts}\t{event.role}\t{event.content}")
    return 0


def cmd_mem_archive_add(args: argparse.Namespace) -> int:
    try:
        with open_memory(args) as memory:
            item = memory.archival_add(
                args.kind,
                args.content,
                source_thread=args.source_thread,
            )
    except MemoryError as exc:
        print(f"Memory command failed: {exc}", file=sys.stderr)
        return 1
    print(f"archival {item.id}")
    return 0


def cmd_mem_search(args: argparse.Namespace) -> int:
    try:
        with open_memory(args) as memory:
            results = memory.archival_search(
                args.query,
                k=args.limit,
                kind=args.kind,
                source_thread=args.source_thread,
            )
    except MemoryError as exc:
        print(f"Memory command failed: {exc}", file=sys.stderr)
        return 1
    for result in results:
        item = result.memory
        print(f"{item.id}\t{result.score:.4f}\t{item.kind}\t{item.content}")
    return 0


def cmd_sandbox_build(args: argparse.Namespace) -> int:
    try:
        DockerREPL.build_image(
            image=args.image,
            dockerfile=Path(args.dockerfile),
            context=Path(args.context),
        )
    except SandboxError as exc:
        print(f"Sandbox build failed: {exc}", file=sys.stderr)
        return 1
    print(f"built {args.image}")
    return 0


def cmd_sandbox_run(args: argparse.Namespace) -> int:
    config = SandboxConfig(
        image=args.image,
        workspace=Path(args.workspace),
        memory=args.memory,
        cpus=args.cpus,
        default_timeout_s=args.cell_timeout,
    )
    subcall_config = RLMSubcallConfig(
        max_depth=args.max_depth,
        max_subcalls=args.max_subcalls,
        token_budget=args.token_budget,
        max_tokens=args.max_tokens,
    )
    try:
        with DockerREPL(
            config,
            completion_client=build_client(args),
            subcall_config=subcall_config,
        ) as repl:
            result = repl.execute(args.code, timeout_s=args.cell_timeout)
    except SandboxError as exc:
        print(f"Sandbox run failed: {exc}", file=sys.stderr)
        return 1

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return 0 if result.ok else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rlm-harness")
    subparsers = root.add_subparsers(dest="command", required=True)

    def add_model_args(command: argparse.ArgumentParser, include_timeout: bool = True) -> None:
        command.add_argument("--provider", default="stub", choices=["stub", "openai-compatible"])
        command.add_argument("--model", default="stub")
        command.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
        command.add_argument("--api-key", default=None)
        if include_timeout:
            command.add_argument("--timeout", type=int, default=120)

    run = subparsers.add_parser("run")
    run.add_argument("task")
    run.add_argument("--workspace", default=".")
    run.add_argument("--trace-db", default=str(default_trace_path()))
    run.add_argument("--thread-id", default=None)
    run.add_argument("--memory-db", default=str(default_memory_path()))
    run.add_argument("--no-memory", action="store_true")
    run.add_argument("--max-history-tokens", type=int, default=1600)
    run.add_argument("--preserve-recent-steps", type=int, default=4)
    run.add_argument("--recall-limit", type=int, default=6)
    run.add_argument("--archival-limit", type=int, default=3)
    run.add_argument("--summary-max-tokens", type=int, default=300)
    run.add_argument("--token-budget", type=int, default=100000)
    run.add_argument("--graph-backend", default="auto", choices=["auto", "simple", "langgraph"])
    run.add_argument("--no-sandbox", action="store_true")
    run.add_argument("--sandbox-image", default="rlm-harness-sandbox:latest")
    run.add_argument("--sandbox-memory", default="512m")
    run.add_argument("--sandbox-cpus", type=float, default=1.0)
    run.add_argument("--sandbox-timeout", type=float, default=60)
    run.add_argument("--max-depth", type=int, default=3)
    run.add_argument("--max-subcalls", type=int, default=32)
    run.add_argument("--subcall-max-tokens", type=int, default=512)
    run.add_argument("--max-action-retries", type=int, default=1)
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

    mem = subparsers.add_parser("mem")
    mem.add_argument("--memory-db", default=str(default_memory_path()))
    mem_subparsers = mem.add_subparsers(dest="mem_command", required=True)

    mem_pin = mem_subparsers.add_parser("pin")
    mem_pin.add_argument("key")
    mem_pin.add_argument("value")
    mem_pin.set_defaults(func=cmd_mem_pin)

    mem_get = mem_subparsers.add_parser("get")
    mem_get.add_argument("key")
    mem_get.set_defaults(func=cmd_mem_get)

    mem_recall_append = mem_subparsers.add_parser("recall-append")
    mem_recall_append.add_argument("thread_id")
    mem_recall_append.add_argument("role")
    mem_recall_append.add_argument("content")
    mem_recall_append.set_defaults(func=cmd_mem_recall_append)

    mem_recall_page = mem_subparsers.add_parser("recall-page")
    mem_recall_page.add_argument("thread_id")
    mem_recall_page.add_argument("--query", default="")
    mem_recall_page.add_argument("--limit", type=int, default=5)
    mem_recall_page.set_defaults(func=cmd_mem_recall_page)

    mem_archive_add = mem_subparsers.add_parser("archive-add")
    mem_archive_add.add_argument("kind")
    mem_archive_add.add_argument("content")
    mem_archive_add.add_argument("--source-thread", default=None)
    mem_archive_add.set_defaults(func=cmd_mem_archive_add)

    mem_search = mem_subparsers.add_parser("search")
    mem_search.add_argument("query")
    mem_search.add_argument("--kind", default=None)
    mem_search.add_argument("--source-thread", default=None)
    mem_search.add_argument("--limit", type=int, default=5)
    mem_search.set_defaults(func=cmd_mem_search)

    sandbox = subparsers.add_parser("sandbox")
    sandbox_subparsers = sandbox.add_subparsers(dest="sandbox_command", required=True)

    sandbox_build = sandbox_subparsers.add_parser("build")
    sandbox_build.add_argument("--image", default="rlm-harness-sandbox:latest")
    sandbox_build.add_argument("--dockerfile", default="docker/sandbox.Dockerfile")
    sandbox_build.add_argument("--context", default=".")
    sandbox_build.set_defaults(func=cmd_sandbox_build)

    sandbox_run = sandbox_subparsers.add_parser("run")
    sandbox_run.add_argument("code")
    sandbox_run.add_argument("--image", default="rlm-harness-sandbox:latest")
    sandbox_run.add_argument("--workspace", default=".")
    sandbox_run.add_argument("--memory", default="512m")
    sandbox_run.add_argument("--cpus", type=float, default=1.0)
    sandbox_run.add_argument("--timeout", dest="cell_timeout", type=float, default=60)
    sandbox_run.add_argument("--max-depth", type=int, default=3)
    sandbox_run.add_argument("--max-subcalls", type=int, default=32)
    sandbox_run.add_argument("--token-budget", type=int, default=200000)
    sandbox_run.add_argument("--max-tokens", type=int, default=512)
    sandbox_run.add_argument("--model-timeout", dest="timeout", type=int, default=120)
    add_model_args(sandbox_run, include_timeout=False)
    sandbox_run.set_defaults(func=cmd_sandbox_run)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
