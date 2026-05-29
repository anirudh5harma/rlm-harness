from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlm_harness.config import default_memory_path
from rlm_harness.memory import Memory, MemoryError


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


def add_memory_command(subparsers) -> None:
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
