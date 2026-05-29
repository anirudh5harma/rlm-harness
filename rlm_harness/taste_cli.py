from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from rlm_harness.config import default_memory_path, default_profile_path
from rlm_harness.memory import Memory, MemoryError
from rlm_harness.memory.evolution import EvolutionProposalManager
from rlm_harness.memory.profile import TasteProfileManager, TasteProfileStore, TasteRecord
from rlm_harness.project_style import scan_project_style


def open_profile(args: argparse.Namespace) -> Memory:
    return Memory(Path(args.profile_db))


def open_project_profile(args: argparse.Namespace) -> Memory:
    return Memory(Path(getattr(args, "memory_db", default_memory_path())))


def taste_memory_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        Path(args.profile_db),
        Path(getattr(args, "memory_db", default_memory_path())),
    )


def cmd_profile_list(args: argparse.Namespace) -> int:
    try:
        profile_path, project_path = taste_memory_paths(args)
        records = []
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(profile_path))
            profile_store = TasteProfileStore(profile_memory)
            if args.scope != "project":
                records.extend(
                    profile_store.records(
                        scope="user" if args.scope == "user" else None,
                        status=args.status,
                        kind=args.kind,
                    )
                )
            if args.scope != "user":
                if args.scope == "project" or project_path.exists():
                    project_memory = stack.enter_context(Memory(project_path))
                    records.extend(
                        TasteProfileStore(project_memory).records(
                            scope="project",
                            status=args.status,
                            kind=args.kind,
                        )
                    )
        records = sorted(
            records,
            key=lambda r: (r.status != "active", -r.confidence, -r.updated_at),
        )
    except MemoryError as exc:
        print(f"Profile command failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps([record.to_dict() for record in records], sort_keys=True))
        return 0
    if not records:
        print("No taste records yet.")
        return 0
    for record in records:
        print(
            f"{record.id}\t{record.status}\t{record.scope}\t"
            f"{record.kind}\t{record.confidence:.2f}\t{record.text}"
        )
    return 0


def cmd_profile_context(args: argparse.Namespace) -> int:
    try:
        profile_path, project_path = taste_memory_paths(args)
        with ExitStack() as stack:
            profile_memory = stack.enter_context(Memory(profile_path))
            project_memory = (
                stack.enter_context(Memory(project_path))
                if project_path.exists()
                else None
            )
            taste_context = TasteProfileManager(
                profile_memory,
                project_memory,
            ).render_context(max_records=args.max_records)
            evolution_context = EvolutionProposalManager(
                profile_memory,
                project_memory,
            ).render_context(max_proposals=args.max_proposals)
        context_parts = [part for part in (taste_context, evolution_context) if part.strip()]
        context = "\n\n".join(context_parts)
        prompt_context = f"Taste context:\n{context}" if context else ""
    except MemoryError as exc:
        print(f"Profile command failed: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(
            json.dumps(
                {
                    "context": prompt_context,
                    "empty": not bool(prompt_context),
                },
                sort_keys=True,
            )
        )
        return 0
    if not prompt_context:
        print("No active taste context yet.")
        return 0
    print(prompt_context)
    return 0


def cmd_profile_learn(args: argparse.Namespace) -> int:
    try:
        opener = open_project_profile if args.scope == "project" else open_profile
        with opener(args) as memory:
            record = TasteProfileStore(memory).add(
                TasteRecord.create(
                    scope=args.scope,
                    kind=args.kind,
                    text=args.text,
                    confidence=args.confidence,
                    status="active" if args.active else "pending",
                    evidence={"source": "cli"},
                )
            )
    except MemoryError as exc:
        print(f"Profile command failed: {exc}", file=sys.stderr)
        return 1
    print(f"{record.status}\t{record.id}\t{record.text}")
    return 0


def cmd_profile_scan(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    try:
        records = scan_project_style(workspace, max_files=args.max_files)
        with open_project_profile(args) as memory:
            store = TasteProfileStore(memory)
            stored = [store.add(record) for record in records]
    except MemoryError as exc:
        print(f"Profile command failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps([record.to_dict() for record in stored], sort_keys=True))
        return 0
    if not stored:
        print("No project style conventions detected.")
        return 0
    for record in stored:
        print(f"{record.status}\t{record.id}\t{record.kind}\t{record.text}")
    return 0


def cmd_profile_approve(args: argparse.Namespace) -> int:
    try:
        record = update_taste_status(args, "active")
    except MemoryError as exc:
        print(f"Profile command failed: {exc}", file=sys.stderr)
        return 1
    if record is None:
        print(f"Unknown taste record: {args.record_id}", file=sys.stderr)
        return 1
    print(f"active\t{record.id}\t{record.text}")
    return 0


def cmd_profile_reject(args: argparse.Namespace) -> int:
    try:
        record = update_taste_status(args, "rejected")
    except MemoryError as exc:
        print(f"Profile command failed: {exc}", file=sys.stderr)
        return 1
    if record is None:
        print(f"Unknown taste record: {args.record_id}", file=sys.stderr)
        return 1
    print(f"rejected\t{record.id}\t{record.text}")
    return 0


def update_taste_status(args: argparse.Namespace, status: str) -> Optional[TasteRecord]:
    profile_path, project_path = taste_memory_paths(args)
    with ExitStack() as stack:
        profile_memory = stack.enter_context(Memory(profile_path))
        stores = [TasteProfileStore(profile_memory)]
        if project_path.exists():
            project_memory = stack.enter_context(Memory(project_path))
            stores.append(TasteProfileStore(project_memory))
        for store in stores:
            if status == "active":
                record = store.approve(args.record_id)
            else:
                record = store.reject(args.record_id)
            if record is not None:
                return record
    return None


def add_taste_command(subparsers, name: str, help_text: str) -> None:
    command = subparsers.add_parser(name, help=help_text)
    command.add_argument(
        "--profile-db",
        default=str(default_profile_path()),
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--memory-db",
        default=str(default_memory_path()),
        help=argparse.SUPPRESS,
    )
    taste_subparsers = command.add_subparsers(dest=f"{name}_command", required=False)

    taste_list = taste_subparsers.add_parser("list")
    taste_list.add_argument("--scope", choices=["user", "project"], default=None)
    taste_list.add_argument(
        "--status",
        choices=["active", "pending", "rejected"],
        default="active",
    )
    taste_list.add_argument("--kind", default=None)
    taste_list.add_argument("--json", dest="json_output", action="store_true")
    taste_list.set_defaults(func=cmd_profile_list)

    taste_context = taste_subparsers.add_parser("context")
    taste_context.add_argument("--max-records", type=int, default=16)
    taste_context.add_argument("--max-proposals", type=int, default=8)
    taste_context.add_argument("--json", dest="json_output", action="store_true")
    taste_context.set_defaults(func=cmd_profile_context)

    taste_learn = taste_subparsers.add_parser("learn")
    taste_learn.add_argument("text")
    taste_learn.add_argument("--scope", choices=["user", "project"], default="user")
    taste_learn.add_argument("--kind", default="preference")
    taste_learn.add_argument("--confidence", type=float, default=0.9)
    taste_learn.add_argument("--active", action="store_true")
    taste_learn.set_defaults(func=cmd_profile_learn)

    taste_scan = taste_subparsers.add_parser("scan")
    taste_scan.add_argument("--workspace", default=".")
    taste_scan.add_argument("--max-files", type=int, default=400)
    taste_scan.add_argument("--json", dest="json_output", action="store_true")
    taste_scan.set_defaults(func=cmd_profile_scan)

    taste_approve = taste_subparsers.add_parser("approve")
    taste_approve.add_argument("record_id")
    taste_approve.set_defaults(func=cmd_profile_approve)

    taste_reject = taste_subparsers.add_parser("reject")
    taste_reject.add_argument("record_id")
    taste_reject.set_defaults(func=cmd_profile_reject)
    command.set_defaults(
        func=cmd_profile_list,
        scope=None,
        status="active",
        kind=None,
        json_output=False,
    )
