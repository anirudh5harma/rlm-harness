from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from rlm_harness.config import default_memory_path, default_profile_path, default_trace_path
from rlm_harness.memory import Memory, MemoryError
from rlm_harness.memory.evolution import EvolutionProposal, EvolutionProposalManager
from rlm_harness.memory.feedback import (
    FeedbackRecord,
    FeedbackStore,
    infer_evolution_from_feedback,
    infer_taste_from_feedback,
)
from rlm_harness.memory.profile import TasteProfileStore
from rlm_harness.tracing import TraceStore


def cmd_evolve_list(args: argparse.Namespace) -> int:
    try:
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(Path(args.profile_db)))
            project_memory = stack.enter_context(Memory(Path(args.memory_db)))
            manager = EvolutionProposalManager(profile_memory, project_memory)
            proposals = manager.proposals(
                scope=args.scope,
                status=args.status,
                kind=args.kind,
            )
    except MemoryError as exc:
        print(f"Evolution command failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps([proposal.to_dict() for proposal in proposals], sort_keys=True))
        return 0
    if not proposals:
        print("No evolution proposals yet.")
        return 0
    for proposal in proposals:
        print(
            f"{proposal.id}\t{proposal.status}\t{proposal.scope}\t"
            f"{proposal.kind}\t{proposal.title}\t{proposal.body}"
        )
    return 0


def cmd_evolve_propose(args: argparse.Namespace) -> int:
    try:
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(Path(args.profile_db)))
            project_memory = stack.enter_context(Memory(Path(args.memory_db)))
            manager = EvolutionProposalManager(profile_memory, project_memory)
            proposal = manager.add(
                EvolutionProposal.create(
                    scope=args.scope,
                    kind=args.kind,
                    title=args.title,
                    body=args.body,
                    rationale=args.rationale,
                    status="approved" if args.approved else "pending",
                    evidence={"source": "cli"},
                )
            )
    except MemoryError as exc:
        print(f"Evolution command failed: {exc}", file=sys.stderr)
        return 1
    if proposal is None:
        print(f"No writable memory store for scope: {args.scope}", file=sys.stderr)
        return 1
    print(f"{proposal.status}\t{proposal.id}\t{proposal.title}")
    return 0


def cmd_evolve_approve(args: argparse.Namespace) -> int:
    try:
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(Path(args.profile_db)))
            project_memory = stack.enter_context(Memory(Path(args.memory_db)))
            proposal = EvolutionProposalManager(
                profile_memory,
                project_memory,
            ).approve(args.proposal_id)
    except MemoryError as exc:
        print(f"Evolution command failed: {exc}", file=sys.stderr)
        return 1
    if proposal is None:
        print(f"Unknown evolution proposal: {args.proposal_id}", file=sys.stderr)
        return 1
    print(f"approved\t{proposal.id}\t{proposal.title}")
    return 0


def cmd_evolve_reject(args: argparse.Namespace) -> int:
    try:
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(Path(args.profile_db)))
            project_memory = stack.enter_context(Memory(Path(args.memory_db)))
            proposal = EvolutionProposalManager(
                profile_memory,
                project_memory,
            ).reject(args.proposal_id)
    except MemoryError as exc:
        print(f"Evolution command failed: {exc}", file=sys.stderr)
        return 1
    if proposal is None:
        print(f"Unknown evolution proposal: {args.proposal_id}", file=sys.stderr)
        return 1
    print(f"rejected\t{proposal.id}\t{proposal.title}")
    return 0


def cmd_feedback_list(args: argparse.Namespace) -> int:
    try:
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(Path(args.profile_db)))
            project_memory = stack.enter_context(Memory(Path(args.memory_db)))
            stores = [FeedbackStore(profile_memory), FeedbackStore(project_memory)]
            records = []
            for store in stores:
                records.extend(store.records(scope=args.scope, rating=args.rating))
            records.sort(key=lambda record: -record.created_at)
    except MemoryError as exc:
        print(f"Feedback command failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps([record.to_dict() for record in records], sort_keys=True))
        return 0
    if not records:
        print("No feedback yet.")
        return 0
    for record in records:
        run = record.run_id or "-"
        print(f"{record.id}\t{record.rating}\t{record.scope}\t{run}\t{record.comment}")
    return 0


def cmd_feedback_add(args: argparse.Namespace) -> int:
    evidence = {"source": "cli"}
    thread_id = None
    if args.run_id:
        try:
            summary = TraceStore(Path(args.trace_db)).run_summary(args.run_id)
        except (KeyError, OSError) as exc:
            print(f"Feedback command failed: {exc}", file=sys.stderr)
            return 1
        thread_id = str(summary.get("thread_id") or "")
        evidence["run"] = {
            "task": str(summary.get("task") or "")[:500],
            "status": str(summary.get("status") or ""),
            "final_answer": str(summary.get("final_answer") or "")[:1000],
        }

    try:
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(Path(args.profile_db)))
            project_memory = stack.enter_context(Memory(Path(args.memory_db)))
            feedback = FeedbackRecord.create(
                scope=args.scope,
                rating=args.rating,
                comment=args.comment,
                run_id=args.run_id,
                thread_id=thread_id,
                evidence=evidence,
            )
            feedback_store = FeedbackStore(
                project_memory if feedback.scope == "project" else profile_memory
            )
            feedback = feedback_store.add(feedback)

            taste_store = TasteProfileStore(
                project_memory if feedback.scope == "project" else profile_memory
            )
            learned = [
                taste_store.add(record)
                for record in infer_taste_from_feedback(feedback, active=args.active)
            ]

            evolution = EvolutionProposalManager(profile_memory, project_memory)
            proposals = [
                proposal
                for proposal in (
                    evolution.add(proposal)
                    for proposal in infer_evolution_from_feedback(feedback)
                )
                if proposal is not None
            ]
    except MemoryError as exc:
        print(f"Feedback command failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"feedback\t{feedback.id}\t"
        f"taste={len(learned)}\tproposals={len(proposals)}"
    )
    return 0


def add_learning_commands(subparsers) -> None:
    add_evolve_command(subparsers)
    add_feedback_command(subparsers)


def add_evolve_command(subparsers) -> None:
    evolve = subparsers.add_parser(
        "evolve",
        help="Inspect and approve self-evolution proposals.",
    )
    evolve.add_argument(
        "--profile-db",
        default=str(default_profile_path()),
        help=argparse.SUPPRESS,
    )
    evolve.add_argument(
        "--memory-db",
        default=str(default_memory_path()),
        help=argparse.SUPPRESS,
    )
    evolve_subparsers = evolve.add_subparsers(dest="evolve_command", required=False)

    evolve_list = evolve_subparsers.add_parser("list")
    evolve_list.add_argument("--scope", choices=["user", "project"], default=None)
    evolve_list.add_argument(
        "--status",
        choices=["pending", "approved", "rejected"],
        default="pending",
    )
    evolve_list.add_argument(
        "--kind",
        choices=["prompt_rule", "verification_policy", "eval_case", "tooling"],
        default=None,
    )
    evolve_list.add_argument("--json", dest="json_output", action="store_true")
    evolve_list.set_defaults(func=cmd_evolve_list)

    evolve_propose = evolve_subparsers.add_parser("propose")
    evolve_propose.add_argument("--scope", choices=["user", "project"], default="user")
    evolve_propose.add_argument(
        "--kind",
        choices=["prompt_rule", "verification_policy", "eval_case", "tooling"],
        default="prompt_rule",
    )
    evolve_propose.add_argument("--title", required=True)
    evolve_propose.add_argument("--body", required=True)
    evolve_propose.add_argument("--rationale", required=True)
    evolve_propose.add_argument("--approved", action="store_true")
    evolve_propose.set_defaults(func=cmd_evolve_propose)

    evolve_approve = evolve_subparsers.add_parser("approve")
    evolve_approve.add_argument("proposal_id")
    evolve_approve.set_defaults(func=cmd_evolve_approve)

    evolve_reject = evolve_subparsers.add_parser("reject")
    evolve_reject.add_argument("proposal_id")
    evolve_reject.set_defaults(func=cmd_evolve_reject)
    evolve.set_defaults(
        func=cmd_evolve_list,
        scope=None,
        status="pending",
        kind=None,
        json_output=False,
    )


def add_feedback_command(subparsers) -> None:
    feedback = subparsers.add_parser(
        "feedback",
        help="Record feedback so Harness can learn your taste.",
    )
    feedback.add_argument(
        "--profile-db",
        default=str(default_profile_path()),
        help=argparse.SUPPRESS,
    )
    feedback.add_argument(
        "--memory-db",
        default=str(default_memory_path()),
        help=argparse.SUPPRESS,
    )
    feedback.add_argument(
        "--trace-db",
        default=str(default_trace_path()),
        help=argparse.SUPPRESS,
    )
    feedback_subparsers = feedback.add_subparsers(dest="feedback_command", required=False)

    feedback_add = feedback_subparsers.add_parser("add")
    feedback_add.add_argument("comment")
    feedback_add.add_argument(
        "--rating",
        choices=["positive", "negative", "neutral", "good", "bad", "ok"],
        default="neutral",
    )
    feedback_add.add_argument("--scope", choices=["user", "project"], default="user")
    feedback_add.add_argument("--run-id", default=None)
    feedback_add.add_argument("--active", action="store_true")
    feedback_add.set_defaults(func=cmd_feedback_add)

    feedback_list = feedback_subparsers.add_parser("list")
    feedback_list.add_argument("--scope", choices=["user", "project"], default=None)
    feedback_list.add_argument(
        "--rating",
        choices=["positive", "negative", "neutral"],
        default=None,
    )
    feedback_list.add_argument("--json", dest="json_output", action="store_true")
    feedback_list.set_defaults(func=cmd_feedback_list)
    feedback.set_defaults(
        func=cmd_feedback_list,
        scope=None,
        rating=None,
        json_output=False,
    )
