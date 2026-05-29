from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlm_harness.config import default_memory_path
from rlm_harness.dogfood import render_dogfood_report, run_dogfood
from rlm_harness.evals.langsmith import (
    LangSmithExperimentUploader,
    LangSmithUploadConfig,
    collect_run_metadata,
)
from rlm_harness.evals.runner import EvalRunner
from rlm_harness.evals.suite import EvalSuiteFileLoader
from rlm_harness.memory import Memory
from rlm_harness.memory.evolution import EvolutionProposal, EvolutionProposalManager
from rlm_harness.readiness import build_readiness_report


def build_eval_harness_command(args: argparse.Namespace) -> list[str]:
    return build_harness_run_command(args, no_sandbox=args.no_sandbox)


def build_harness_run_command(
    args: argparse.Namespace,
    no_sandbox: bool = False,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from rlm_harness.cli import main; "
        "raise SystemExit(main())"
    )
    command = [sys.executable, "-c", bootstrap, "run"]
    if no_sandbox:
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
    recorded_failures = 0
    if args.record_failures:
        recorded_failures = record_eval_failure_proposals(report, Path(args.memory_db))
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
        if args.record_failures:
            print(f"failure_proposals\t{recorded_failures}")
        for result in report.results:
            print(f"{result.case_id}\t{result.status}\t{result.score:.1f}\t{result.latency_ms}ms")
    return 0 if all(result.passed for result in report.results) else 1


def cmd_dogfood(args: argparse.Namespace) -> int:
    readiness = build_readiness_report(Path.cwd(), check_docker=not args.no_docker)
    report = run_dogfood(
        readiness=readiness,
        work_root=Path(args.work_root).resolve(),
        sandbox_harness_command=build_harness_run_command(args, no_sandbox=False),
        no_sandbox_harness_command=build_harness_run_command(args, no_sandbox=True),
        timeout_s=args.eval_timeout,
        no_docker=args.no_docker,
        strict_readiness=args.strict_readiness,
        install_smoke=args.install_smoke,
        repo_root=Path(__file__).resolve().parents[1],
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report.to_json(), encoding="utf-8")
    if args.json_output:
        print(report.to_json())
    else:
        print(render_dogfood_report(report))
    return 0 if report.status == "passed" else 1


def record_eval_failure_proposals(report, memory_path: Path) -> int:
    failures = [result for result in report.results if not result.passed]
    if not failures:
        return 0
    with Memory(memory_path) as memory:
        store = EvolutionProposalManager(None, memory)
        for result in failures:
            prompt = str(result.metadata.get("prompt") or result.case_id)
            store.add(
                EvolutionProposal.create(
                    scope="project",
                    kind="eval_case",
                    title=f"Improve failing eval: {result.case_id}",
                    body=(
                        f"Investigate eval `{report.suite}/{result.case_id}` and add "
                        "a prompt, tool, or verification improvement so this case passes."
                    ),
                    rationale=(
                        "Daily-driver quality requires repeated coding evals to pass. "
                        f"Observed status `{result.status}` with score {result.score:.2f}."
                    ),
                    evidence={
                        "suite": report.suite,
                        "case_id": result.case_id,
                        "prompt": prompt[:500],
                        "status": result.status,
                        "score": result.score,
                        "output": result.output[:1000],
                        "harness_stderr": result.harness_stderr[:1000],
                    },
                )
            )
    return len(failures)


def add_quality_commands(subparsers, add_model_args) -> None:
    dogfood = subparsers.add_parser(
        "dogfood",
        help="Run readiness, eval, and feedback proof checks.",
    )
    dogfood.add_argument("--work-root", default=".harness-evals/dogfood")
    dogfood.add_argument("--output", default=None)
    dogfood.add_argument("--eval-timeout", type=int, default=900)
    dogfood.add_argument("--json", dest="json_output", action="store_true")
    dogfood.add_argument(
        "--no-docker",
        action="store_true",
        help="Skip Docker-dependent dogfood checks.",
    )
    dogfood.add_argument(
        "--strict-readiness",
        action="store_true",
        help="Fail if readiness has setup blockers.",
    )
    dogfood.add_argument(
        "--install-smoke",
        action="store_true",
        help="Install Harness into a fresh venv and run a bundled eval.",
    )
    add_model_args(dogfood, public=True)
    dogfood.set_defaults(func=cmd_dogfood)

    eval_cmd = subparsers.add_parser("eval", help="Run harness evaluation suites.")
    eval_cmd.add_argument(
        "path",
        help="Path to a local YAML/JSON eval suite, or built-in: daily-driver, taste-regression.",
    )
    eval_cmd.add_argument("--work-root", default=".harness-evals/work")
    eval_cmd.add_argument("--output", default=None)
    eval_cmd.add_argument("--eval-timeout", type=int, default=900)
    eval_cmd.add_argument("--json", dest="json_output", action="store_true")
    eval_cmd.add_argument(
        "--record-failures",
        action="store_true",
        help="Create pending evolution proposals for failing eval cases.",
    )
    eval_cmd.add_argument(
        "--memory-db",
        default=str(default_memory_path()),
        help=argparse.SUPPRESS,
    )
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
