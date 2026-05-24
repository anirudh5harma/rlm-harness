from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from rlm_harness.config import (
    CONFIG_PATH,
    default_api_key,
    default_base_url,
    default_memory_path,
    default_model,
    default_provider,
    default_trace_path,
    masked_secret,
    save_user_config,
)
from rlm_harness.evals.langsmith import (
    LangSmithExperimentUploader,
    LangSmithUploadConfig,
    collect_run_metadata,
)
from rlm_harness.evals.runner import EvalRunner
from rlm_harness.evals.suite import EvalSuiteFileLoader
from rlm_harness.graph.build import build_graph
from rlm_harness.graph.nodes import GraphRuntimeConfig, Nodes
from rlm_harness.memory import Memory, MemoryError, MemoryPagingConfig
from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.model_server import MLXServer, MLXServerConfig, MLXServerError
from rlm_harness.providers import (
    PROVIDERS,
    fetch_provider_models,
    normalize_provider,
    provider_names,
    provider_preset,
    static_models,
)
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError
from rlm_harness.tracing import TraceStore
from rlm_harness.types import HarnessState, Msg

PUBLIC_COMMANDS = {
    "run",
    "resume",
    "trace",
    "doctor",
    "model",
    "provider",
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


def build_client(args: argparse.Namespace) -> LMClient:
    provider = normalize_provider(args.provider)
    return LMClient(
        provider=provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or default_api_key(provider),
        timeout_s=args.timeout,
    )


def build_runtime(
    args: argparse.Namespace,
    workspace: Path,
    memory: Optional[Memory],
) -> GraphRuntimeConfig:
    return GraphRuntimeConfig(
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
        max_iterations=args.max_iterations,
        act_engine=args.act_engine,
        memory=memory,
        memory_paging=MemoryPagingConfig(
            max_history_tokens=args.max_history_tokens,
            preserve_recent_steps=args.preserve_recent_steps,
            recall_limit=args.recall_limit,
            archival_limit=args.archival_limit,
            summary_max_tokens=args.summary_max_tokens,
        ),
    )


def cmd_run(args: argparse.Namespace) -> int:
    return run_task(args, args.task, args.thread_id, Path(args.workspace).resolve())


def cmd_resume(args: argparse.Namespace) -> int:
    traces = TraceStore(Path(args.trace_db))
    previous = traces.latest_run_for_thread(args.thread_id)
    if previous is None and args.task is None:
        print(
            f"Cannot resume {args.thread_id}: no previous run in {args.trace_db}",
            file=sys.stderr,
        )
        return 1
    task = args.task or str(previous["task"])
    workspace_value = args.workspace or (previous["workspace"] if previous else ".")
    workspace = Path(workspace_value).resolve()
    return run_task(args, task, args.thread_id, workspace)


def run_task(
    args: argparse.Namespace,
    task: str,
    thread_id: Optional[str],
    workspace: Path,
) -> int:
    traces = TraceStore(Path(args.trace_db))
    run_id = traces.start_run(task, str(workspace), thread_id=thread_id)
    state = HarnessState(
        task=task,
        workspace=str(workspace),
        thread_id=thread_id or run_id,
        run_id=run_id,
        token_budget=args.token_budget,
    )
    traces.event(
        run_id,
        "run_started",
        {
            "task": task,
            "workspace": str(workspace),
            "sandbox_enabled": not args.no_sandbox,
            "memory_enabled": not args.no_memory,
        },
        node="cli",
    )

    try:
        memory_context = nullcontext(None) if args.no_memory else Memory(Path(args.memory_db))
        with memory_context as memory:
            runtime = build_runtime(args, workspace, memory)
            graph = build_graph(
                Nodes(build_client(args), traces, runtime),
                backend=args.graph_backend,
                checkpoint_path=checkpoint_path(args),
            )
            if args.stream:
                final_state = run_streaming_graph(graph, state)
            else:
                final_state = graph.invoke(state)
            close_graph(graph)
        traces.finish_run(run_id, final_state.status)
    except (LMClientError, MemoryError, SandboxError) as exc:
        traces.event(run_id, "error", {"message": str(exc)}, node="cli")
        traces.finish_run(run_id, "error")
        emit_run_output(args, None, traces, run_id, error=str(exc))
        return 1

    emit_run_output(args, final_state, traces, run_id)
    return 0 if final_state.status == "done" else 1


def emit_run_output(
    args: argparse.Namespace,
    final_state: Optional[HarnessState],
    traces: TraceStore,
    run_id: str,
    error: Optional[str] = None,
) -> None:
    if args.json_output:
        payload = run_output_payload(args, traces, run_id, final_state)
        if error:
            payload["error"] = error
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return
    if final_state and final_state.final_answer:
        print(final_state.final_answer)


def run_output_payload(
    args: argparse.Namespace,
    traces: TraceStore,
    run_id: str,
    final_state: Optional[HarnessState],
) -> dict:
    summary = traces.run_summary(run_id)
    response = (
        final_state.final_answer
        if final_state is not None and final_state.final_answer is not None
        else summary.get("final_answer")
    )
    return {
        "status": summary["status"],
        "response": response,
        "final_answer": response,
        "run_id": summary["run_id"],
        "thread_id": summary["thread_id"],
        "task": summary["task"],
        "workspace": summary["workspace"],
        "event_count": summary["event_count"],
        "trace_db": args.trace_db,
    }


def checkpoint_path(args: argparse.Namespace) -> Optional[Path]:
    if args.no_checkpoint or args.graph_backend == "simple":
        return None
    return Path(args.checkpoint_db)


def run_streaming_graph(graph, state: HarnessState) -> HarnessState:
    if not hasattr(graph, "stream"):
        return graph.invoke(state)
    final_state = None
    for update in graph.stream(state):
        print(json.dumps({"graph_update": update}, default=str, sort_keys=True))
        if isinstance(update, dict):
            for value in update.values():
                if isinstance(value, HarnessState):
                    final_state = value
                elif isinstance(value, dict):
                    final_state = HarnessState.model_validate(value)
    return final_state or graph.invoke(state)


def close_graph(graph) -> None:
    close = getattr(graph, "close", None)
    if callable(close):
        close()


def cmd_trace_list(args: argparse.Namespace) -> int:
    traces = TraceStore(Path(args.trace_db))
    try:
        runs = traces.list_runs(limit=args.limit, thread_id=args.thread_id)
    except ValueError as exc:
        print(f"Trace command failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(runs, sort_keys=True))
        return 0
    for run in runs:
        print("{run_id}\t{thread_id}\t{status}\t{started_at}\t{task}".format(**run))
    return 0


def cmd_trace_report(args: argparse.Namespace) -> int:
    traces = TraceStore(Path(args.trace_db))
    try:
        if args.json_output:
            print(json.dumps(traces.run_summary(args.run_id), sort_keys=True))
        else:
            print(traces.render_report(args.run_id))
    except KeyError as exc:
        print(f"Trace command failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_trace_events(args: argparse.Namespace) -> int:
    traces = TraceStore(Path(args.trace_db))
    events = traces.events(args.run_id)
    if args.json_output:
        print(json.dumps(events, sort_keys=True))
    else:
        for event in events:
            print(
                "[{kind}] {node} {payload}".format(
                    kind=event["kind"],
                    node=event["node"] or "-",
                    payload=json.dumps(event["payload"], sort_keys=True),
                )
            )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = {
        "python": sys.version.split()[0],
        "harness_cli": shutil.which("harness") or "not installed as command yet",
        "docker_cli": "ok" if shutil.which("docker") else "missing",
        "langgraph": module_status("langgraph"),
        "sqlite_vec": module_status("sqlite_vec"),
    }
    if shutil.which("docker"):
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        checks["docker_daemon"] = (
            completed.stdout.strip() if completed.returncode == 0 else "unavailable"
        )
        image_check = subprocess.run(
            ["docker", "image", "inspect", "rlm-harness-sandbox:latest"],
            text=True,
            capture_output=True,
            check=False,
        )
        checks["sandbox_image"] = "ok" if image_check.returncode == 0 else "missing"
    if args.json_output:
        print(json.dumps(checks, sort_keys=True))
    else:
        for name, value in checks.items():
            print(f"{name}\t{value}")
    failing = {"missing", "unavailable"}
    return 0 if all(value not in failing for value in checks.values()) else 1


def module_status(module: str) -> str:
    try:
        __import__(module)
    except ImportError:
        return "missing"
    return "ok"


def current_config_payload() -> dict[str, str]:
    provider = default_provider()
    return {
        "provider": provider,
        "model": default_model(),
        "base_url": default_base_url(),
        "api_key": masked_secret(default_api_key(provider)),
        "config_path": str(CONFIG_PATH),
    }


def print_current_config() -> None:
    config = current_config_payload()
    print(f"provider\t{config['provider']}")
    print(f"model\t{config['model']}")
    print(f"base_url\t{config['base_url']}")
    print(f"api_key\t{config['api_key']}")
    print(f"config\t{config['config_path']}")


def prompt_numbered_choice(options: list[str], prompt: str) -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    try:
        raw = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not raw:
        return None
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index]
    return raw


def print_provider_options() -> None:
    print("Available providers:")
    for index, name in enumerate(provider_names(), start=1):
        preset = PROVIDERS[name]
        print(f"  {index}. {name}\t{preset.label}\t{preset.description}")


def cmd_config(args: argparse.Namespace) -> int:
    if args.json_output:
        print(json.dumps(current_config_payload(), sort_keys=True))
    else:
        print_current_config()
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    app_dir = Path(os.environ.get("HARNESS_APP_DIR", Path.home() / ".local/share/harness"))
    src_dir = app_dir / "src"
    repo_url = os.environ.get(
        "HARNESS_REPO_URL",
        "https://github.com/anirudh5harma/rlm-harness.git",
    )
    ref = os.environ.get("HARNESS_REF", "main")
    venv_dir = app_dir / "venv"
    pip_bin = venv_dir / "bin" / "pip"

    if args.in_place:
        return _update_in_place(args)

    if not (src_dir / ".git").is_dir():
        print(
            f"Update requires an install-managed copy at {src_dir}. "
            f"Re-run the install script or use --in-place for a local checkout.",
            file=sys.stderr,
        )
        return 1

    print(f"Fetching {repo_url} ({ref})...")
    result = subprocess.run(
        ["git", "-C", str(src_dir), "fetch", "--tags", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"git fetch failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    subprocess.run(
        ["git", "-C", str(src_dir), "checkout", ref],
        text=True,
        capture_output=True,
        check=False,
    )

    branch_check = subprocess.run(
        ["git", "-C", str(src_dir), "symbolic-ref", "-q", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if branch_check.returncode == 0:
        pull_result = subprocess.run(
            ["git", "-C", str(src_dir), "pull", "--ff-only", "origin", ref],
            text=True,
            capture_output=True,
            check=False,
        )
        if pull_result.returncode != 0:
            print(f"git pull failed: {pull_result.stderr.strip()}", file=sys.stderr)
            return 1

    if not pip_bin.exists():
        print(f"pip not found at {pip_bin}. Re-run the install script.", file=sys.stderr)
        return 1

    print("Upgrading package...")
    pip_result = subprocess.run(
        [str(pip_bin), "install", "--upgrade", f"{src_dir}[graph]"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pip_result.returncode != 0:
        print(f"pip install failed: {pip_result.stderr.strip()}", file=sys.stderr)
        return 1

    print("harness updated to latest.")
    if not args.no_sandbox_rebuild and shutil.which("docker"):
        print("Rebuilding sandbox image...")
        subprocess.run(
            [
                str(venv_dir / "bin" / "harness"),
                "sandbox",
                "build",
                "--dockerfile",
                str(src_dir / "docker/sandbox.Dockerfile"),
                "--context",
                str(src_dir),
            ],
            check=False,
        )
    return 0


def _update_in_place(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").is_dir():
        print(f"No .git directory found at {repo_root}", file=sys.stderr)
        return 1

    print("Fetching origin...")
    result = subprocess.run(
        ["git", "-C", str(repo_root), "fetch", "--tags", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"git fetch failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    branch_result = subprocess.run(
        ["git", "-C", str(repo_root), "symbolic-ref", "-q", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if branch_result.returncode != 0:
        print("HEAD is detached; cannot pull.", file=sys.stderr)
        return 1

    branch = branch_result.stdout.strip().removeprefix("refs/heads/")
    pull_result = subprocess.run(
        ["git", "-C", str(repo_root), "pull", "--ff-only", "origin", branch],
        text=True,
        capture_output=True,
        check=False,
    )
    if pull_result.returncode != 0:
        print(f"git pull failed: {pull_result.stderr.strip()}", file=sys.stderr)
        return 1

    print("Reinstalling package...")
    pip_result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{repo_root}[dev,graph]"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pip_result.returncode != 0:
        print(f"pip install failed: {pip_result.stderr.strip()}", file=sys.stderr)
        return 1

    print("harness updated to latest (in-place).")
    return 0


def build_eval_harness_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from rlm_harness.cli import main; "
        "raise SystemExit(main())"
    )
    command = [sys.executable, "-c", bootstrap, "run"]
    if args.no_sandbox:
        command.append("--no-sandbox")
    command.extend(["--provider", args.provider, "--model", args.model])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    return command


def cmd_eval(args: argparse.Namespace) -> int:
    work_root = Path(args.work_root).resolve()
    suite = EvalSuiteFileLoader().load_suite(Path(args.path), work_root)

    harness_command = build_eval_harness_command(args)
    runner = EvalRunner(harness_command=harness_command, timeout_s=args.eval_timeout)
    metadata = collect_run_metadata(args, Path(__file__).resolve().parents[1])
    report = runner.run(suite, metadata=metadata)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report.to_json(), encoding="utf-8")
    if args.langsmith_upload or args.langsmith_required:
        config = LangSmithUploadConfig.from_env(
            dataset_name=args.langsmith_dataset,
            experiment_name=args.langsmith_experiment,
        )
        try:
            upload_response = LangSmithExperimentUploader(config).upload(
                report,
                required=args.langsmith_required,
            )
        except RuntimeError as exc:
            if args.langsmith_required:
                print(f"LangSmith upload failed: {exc}", file=sys.stderr)
                return 1
            print(f"LangSmith upload warning: {exc}", file=sys.stderr)
        else:
            if upload_response.get("skipped"):
                print(f"LangSmith upload skipped: {upload_response['skipped']}", file=sys.stderr)
            elif not args.json_output:
                print("langsmith_upload\tok")
    if args.json_output:
        print(report.to_json())
    else:
        print(f"suite\t{report.suite}")
        print(f"run_id\t{report.run_id}")
        print(f"pass_rate\t{report.pass_rate:.3f}")
        for result in report.results:
            print(f"{result.case_id}\t{result.status}\t{result.score:.1f}\t{result.latency_ms}ms")
    return 0 if all(result.passed for result in report.results) else 1


def cmd_model(args: argparse.Namespace) -> int:
    provider = normalize_provider(args.provider or default_provider())
    base_url = args.base_url
    if not base_url:
        base_url = (
            default_base_url()
            if provider == default_provider()
            else provider_preset(provider).base_url
        )
    if args.model:
        save_user_config({"model": args.model})
        print(f"model set to {args.model}")
        return 0

    models = (
        static_models(provider)
        if args.offline
        else fetch_provider_models(provider, base_url, default_api_key(provider))
    )
    if args.json_output:
        print(json.dumps({"provider": provider, "models": models}, sort_keys=True))
        return 0

    print(f"Available models for {provider}:")
    for index, model_name in enumerate(models, start=1):
        current = " *" if model_name == default_model() else ""
        print(f"  {index}. {model_name}{current}")
    selected = prompt_numbered_choice(
        models,
        "Select model number/name, or press Enter to keep current: ",
    )
    if selected:
        save_user_config({"model": selected})
        print(f"model set to {selected}")
    return 0


def cmd_provider(args: argparse.Namespace) -> int:
    provider_arg = " ".join(args.provider) if isinstance(args.provider, list) else args.provider
    if not provider_arg:
        print_provider_options()
        selected = prompt_numbered_choice(
            provider_names(),
            "Select provider number/name, or press Enter to keep current: ",
        )
        if not selected:
            print(f"current provider\t{default_provider()}")
            print(f"base_url\t{default_base_url()}")
            print(f"api_key\t{masked_secret(default_api_key())}")
            return 0
        provider_arg = selected

    provider = normalize_provider(provider_arg)
    if provider not in PROVIDERS:
        print(f"Unknown provider: {provider_arg}", file=sys.stderr)
        print_provider_options()
        return 1

    preset = provider_preset(provider)
    updates = {
        "provider": provider,
        "base_url": args.base_url or preset.base_url,
    }
    if provider == "stub":
        updates["model"] = "stub"
    elif args.set_default_model:
        updates["model"] = static_models(provider)[0]

    if args.api_key:
        updates["api_key"] = args.api_key
    elif provider != "stub" and args.prompt_key and sys.stdin.isatty():
        entered = getpass.getpass(f"{preset.label} API key: ").strip()
        if entered:
            updates["api_key"] = entered

    save_user_config(updates)
    print(f"provider set to {provider}")
    print(f"base_url set to {updates['base_url']}")
    if updates.get("model"):
        print(f"model set to {updates['model']}")
    if updates.get("api_key"):
        print("api_key saved")
    elif provider != "stub" and not default_api_key(provider):
        env_hint = preset.api_key_env[0] if preset.api_key_env else "HARNESS_API_KEY"
        print(f"api_key not set; run: harness /provider {provider} --api-key <key>")
        print(f"or export {env_hint}=<key>")
    return 0


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
    root = argparse.ArgumentParser(
        prog=DEFAULT_PROG,
        description=(
            f'Local recursive coding-agent harness. Run a task with: {DEFAULT_PROG} "fix tests"'
        ),
    )
    subparsers = root.add_subparsers(
        dest="command",
        metavar="{run,resume,trace,doctor,model,provider,config,eval,update}",
        required=True,
    )

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

    def add_run_args(
        command: argparse.ArgumentParser,
        workspace_default: Optional[str] = ".",
        include_thread_id: bool = True,
    ) -> None:
        command.add_argument("--workspace", default=workspace_default, help="Workspace path.")
        command.add_argument(
            "--trace-db",
            default=str(default_trace_path()),
            help=argparse.SUPPRESS,
        )
        command.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON.")
        command.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
        command.add_argument("--stream", action="store_true", help="Print LangGraph update events.")
        if include_thread_id:
            command.add_argument(
                "--thread-id",
                default=None,
                help="Thread id for memory continuity.",
            )
        command.add_argument(
            "--memory-db",
            default=str(default_memory_path()),
            help=argparse.SUPPRESS,
        )
        command.add_argument("--no-memory", action="store_true", help=argparse.SUPPRESS)
        command.add_argument(
            "--max-history-tokens",
            type=int,
            default=1600,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--preserve-recent-steps",
            type=int,
            default=4,
            help=argparse.SUPPRESS,
        )
        command.add_argument("--recall-limit", type=int, default=6, help=argparse.SUPPRESS)
        command.add_argument(
            "--archival-limit",
            type=int,
            default=3,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--summary-max-tokens",
            type=int,
            default=300,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--token-budget",
            type=int,
            default=100000,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--graph-backend",
            default="auto",
            choices=["auto", "simple", "langgraph"],
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--checkpoint-db",
            default=".rlm_harness/checkpoints.db",
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--no-checkpoint",
            action="store_true",
            help=argparse.SUPPRESS,
        )
        command.add_argument("--no-sandbox", action="store_true", help=argparse.SUPPRESS)
        command.add_argument(
            "--sandbox-image",
            default="rlm-harness-sandbox:latest",
            help=argparse.SUPPRESS,
        )
        command.add_argument("--sandbox-memory", default="512m", help=argparse.SUPPRESS)
        command.add_argument("--sandbox-cpus", type=float, default=1.0, help=argparse.SUPPRESS)
        command.add_argument(
            "--sandbox-timeout",
            type=float,
            default=60,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--max-depth",
            type=int,
            default=3,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--max-subcalls",
            type=int,
            default=32,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--subcall-max-tokens",
            type=int,
            default=512,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--max-action-retries",
            type=int,
            default=1,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--max-iterations",
            type=int,
            default=3,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--act-engine",
            choices=["rlm", "json"],
            default="json",
            help=argparse.SUPPRESS,
        )
        add_model_args(command, public=True)

    run = subparsers.add_parser("run", help="Run a task.")
    run.add_argument("task")
    add_run_args(run)
    run.set_defaults(func=cmd_run)

    resume = subparsers.add_parser("resume", help="Resume a thread.")
    resume.add_argument("thread_id")
    resume.add_argument("task", nargs="?")
    add_run_args(resume, workspace_default=None, include_thread_id=False)
    resume.set_defaults(func=cmd_resume)

    trace = subparsers.add_parser("trace", help="Inspect traces.")
    trace.add_argument("--trace-db", default=str(default_trace_path()))
    trace_subparsers = trace.add_subparsers(dest="trace_command", required=True)

    trace_list = trace_subparsers.add_parser("list")
    trace_list.add_argument("--limit", type=int, default=20)
    trace_list.add_argument("--thread-id", default=None)
    trace_list.add_argument("--json", dest="json_output", action="store_true")
    trace_list.set_defaults(func=cmd_trace_list)

    trace_report = trace_subparsers.add_parser("report")
    trace_report.add_argument("run_id")
    trace_report.add_argument("--json", dest="json_output", action="store_true")
    trace_report.set_defaults(func=cmd_trace_report)

    trace_events = trace_subparsers.add_parser("events")
    trace_events.add_argument("run_id")
    trace_events.add_argument("--json", dest="json_output", action="store_true")
    trace_events.set_defaults(func=cmd_trace_events)

    doctor = subparsers.add_parser("doctor", help="Check local setup.")
    doctor.add_argument("--json", dest="json_output", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    model = subparsers.add_parser("model", help="List or set the default model.")
    model.add_argument("model", nargs="?", help="Model name, e.g. qwen/qwen3.7-max.")
    model.add_argument("--provider", default=None)
    model.add_argument("--base-url", default=None, help=argparse.SUPPRESS)
    model.add_argument("--json", dest="json_output", action="store_true")
    model.add_argument("--offline", action="store_true", help="Use bundled model suggestions only.")
    model.set_defaults(func=cmd_model)

    provider = subparsers.add_parser("provider", help="Choose provider and save API key.")
    provider.add_argument(
        "provider",
        nargs="*",
        help="Provider to use. Omit to choose from a list.",
    )
    provider.add_argument(
        "--api-key",
        default=None,
        help="API key to save in ~/.harness/config.json.",
    )
    provider.add_argument(
        "--base-url",
        default=None,
        help="Override the provider base URL.",
    )
    provider.add_argument(
        "--keep-model",
        dest="set_default_model",
        action="store_false",
        default=True,
        help="Keep the current model when switching providers.",
    )
    provider.add_argument(
        "--no-prompt",
        dest="prompt_key",
        action="store_false",
        default=True,
        help="Do not prompt for an API key.",
    )
    provider.set_defaults(func=cmd_provider)

    config_cmd = subparsers.add_parser("config", help="Show saved harness configuration.")
    config_cmd.add_argument("--json", dest="json_output", action="store_true")
    config_cmd.set_defaults(func=cmd_config)

    update = subparsers.add_parser("update", help="Fetch latest harness from GitHub and upgrade.")
    update.add_argument(
        "--in-place",
        action="store_true",
        help="Update from the local dev checkout instead of the managed install.",
    )
    update.add_argument(
        "--no-sandbox-rebuild",
        action="store_true",
        help="Skip sandbox image rebuild after update.",
    )
    update.set_defaults(func=cmd_update)

    eval_cmd = subparsers.add_parser("eval", help="Run harness evaluation suites.")
    eval_cmd.add_argument("path", help="Path to a local YAML/JSON eval suite.")
    eval_cmd.add_argument("--work-root", default=".harness-evals/work")
    eval_cmd.add_argument("--output", default=None)
    eval_cmd.add_argument("--eval-timeout", type=int, default=900)
    eval_cmd.add_argument("--json", dest="json_output", action="store_true")
    eval_cmd.add_argument(
        "--langsmith-upload",
        action="store_true",
        help="Upload the completed local eval report to LangSmith as an external experiment.",
    )
    eval_cmd.add_argument(
        "--langsmith-required",
        action="store_true",
        help="Fail the eval command if LangSmith upload is not configured or fails.",
    )
    eval_cmd.add_argument("--langsmith-dataset", default="rlm-harness")
    eval_cmd.add_argument("--langsmith-experiment", default=None)
    eval_cmd.add_argument("--no-sandbox", action="store_true", help=argparse.SUPPRESS)
    add_model_args(eval_cmd, public=True)
    eval_cmd.set_defaults(func=cmd_eval)

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

    mem = subparsers.add_parser("mem", help=argparse.SUPPRESS)
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

    sandbox = subparsers.add_parser("sandbox", help=argparse.SUPPRESS)
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

    subparsers._choices_actions = [  # type: ignore[attr-defined]
        action
        for action in subparsers._choices_actions  # type: ignore[attr-defined]
        if action.dest in PUBLIC_COMMANDS
    ]
    return root


def interactive_loop() -> int:
    print("Harness interactive mode. Type a coding task and press Enter. Type /help or /quit.")
    print(f"Using provider={default_provider()} model={default_model()}")
    print("Configure with /provider to choose a provider, then /model to select a model.")
    while True:
        try:
            task = input("harness> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not task:
            continue
        if task in {"/q", "/quit", "quit", "exit"}:
            return 0
        if task in {"/h", "/help", "help"}:
            print("Tasks: type any natural-language coding task.")
            print(
                "Slash commands: /provider [name] [--api-key key], /model [name], "
                "/config, /doctor, /update, /quit"
            )
            continue
        if task.startswith("/"):
            try:
                slash_args = shlex.split(task[1:])
            except ValueError as exc:
                print(f"Invalid command: {exc}", file=sys.stderr)
                continue
            if not slash_args:
                continue
            exit_code = main(slash_args)
        else:
            exit_code = main([task])
        if exit_code != 0:
            print(f"Task exited with status {exit_code}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = normalize_argv(argv)
    command_parser = parser()
    if argv == []:
        if sys.stdin.isatty():
            return interactive_loop()
        command_parser.print_help()
        return 0
    args = command_parser.parse_args(argv)
    return args.func(args)


def normalize_argv(argv: list[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return argv
    if argv[0].startswith("/") and len(argv[0]) > 1:
        argv = [argv[0][1:], *argv[1:]]
    if argv[0] in ALL_COMMANDS or argv[0].startswith("-"):
        return argv
    return ["run", *argv]


if __name__ == "__main__":
    raise SystemExit(main())
