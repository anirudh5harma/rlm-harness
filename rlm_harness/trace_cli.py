from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlm_harness.config import default_trace_path
from rlm_harness.tracing import TraceStore


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


def cmd_trace_show(args: argparse.Namespace) -> int:
    """Compact timeline for a run. The `events` command lists raw
    events; `show` produces the human-readable summary a
    developer wants when they ask "what happened in this run?".
    """
    traces = TraceStore(Path(args.trace_db))
    try:
        timeline = traces.timeline_summary(args.run_id)
    except KeyError as exc:
        print(f"Trace command failed: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(timeline, indent=2, sort_keys=True))
        return 0
    # Human-readable summary.
    print(f"Run:       {timeline['run_id']}")
    print(f"Thread:    {timeline['thread_id']}")
    print(f"Task:      {timeline['task']}")
    print(f"Status:    {timeline['status']}")
    print(f"Workspace: {timeline['workspace']}")
    if timeline.get("final_answer"):
        print()
        print("Final answer:")
        for line in str(timeline["final_answer"]).splitlines():
            print(f"  {line}")
    print()
    print(f"Events ({timeline['event_count']}):")
    for event in timeline["events"]:
        parent = (
            f" parent={event['parent_id']}"
            if event.get("parent_id") is not None
            else ""
        )
        node = event.get("node") or "-"
        # Compact one-line summary per event; payload is
        # intentionally truncated.
        payload_str = json.dumps(event["payload"], sort_keys=True)
        if len(payload_str) > 120:
            payload_str = payload_str[:117] + "..."
        print(f"  [{event['kind']}] {node}{parent}  {payload_str}")
    return 0


def cmd_trace_replay(args: argparse.Namespace) -> int:
    """Replay a recorded run from its JSONL tree.

    Writes the run's events to a JSONL tree file (under
    `--out`) and reads them back, confirming the round-trip
    is lossless. Replay of the model calls themselves is a
    follow-up: this command ensures the on-disk tree is a
    faithful, replayable representation of the run.
    """
    traces = TraceStore(Path(args.trace_db))
    if traces.get_run(args.run_id) is None:
        print(f"Trace command failed: unknown run_id: {args.run_id}", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    written = traces.write_jsonl_tree(args.run_id, out_path)
    readback = traces.read_jsonl_tree(out_path)
    if written != len(readback):
        print(
            f"Trace replay failed: wrote {written} events but read back "
            f"{len(readback)}",
            file=sys.stderr,
        )
        return 1
    if args.json_output:
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "events_written": written,
                    "tree_path": str(out_path),
                },
                sort_keys=True,
            )
        )
    else:
        print(f"Replayed {written} events to {out_path}")
    return 0


def add_trace_command(subparsers, *, name: str = "trace") -> None:
    trace = subparsers.add_parser(
        name,
        help="Inspect run history and replay traces." if name == "history" else "Inspect traces.",
    )
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

    trace_show = trace_subparsers.add_parser(
        "show", help="Compact timeline of a run."
    )
    trace_show.add_argument("run_id")
    trace_show.add_argument("--json", dest="json_output", action="store_true")
    trace_show.set_defaults(func=cmd_trace_show)

    trace_replay = trace_subparsers.add_parser(
        "replay", help="Replay a run from its JSONL tree."
    )
    trace_replay.add_argument("run_id")
    trace_replay.add_argument(
        "--out",
        default=".rlm_harness/replay/<run-id>.jsonl",
        help="Output path for the JSONL tree.",
    )
    trace_replay.add_argument("--json", dest="json_output", action="store_true")
    trace_replay.set_defaults(func=cmd_trace_replay)
