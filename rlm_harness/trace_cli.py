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


def add_trace_command(subparsers) -> None:
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
