from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlm_harness.runtime_cli import build_client
from rlm_harness.sandbox import DockerREPL, RLMSubcallConfig, SandboxConfig, SandboxError


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


def add_sandbox_command(subparsers, add_model_args) -> None:
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
    sandbox_run.add_argument("--max-tokens", type=int, default=1024)
    sandbox_run.add_argument("--model-timeout", dest="timeout", type=int, default=120)
    add_model_args(sandbox_run, include_timeout=False)
    sandbox_run.set_defaults(func=cmd_sandbox_run)
