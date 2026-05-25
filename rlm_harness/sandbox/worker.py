from __future__ import annotations

import contextlib
import io
import json
import signal
import sys
import time
import traceback

from rlm_shim import RLMBridge
from sandbox_tools import (
    ToolError,
    apply_patch,
    git_diff,
    git_log,
    git_status,
    list_files,
    project_audit,
    project_overview,
    project_summary,
    read_file,
    read_first_existing,
    run_shell,
    search_code,
    tool_help,
    tool_names,
    write_file,
)


class CellTimeout(Exception):
    pass


def handle_timeout(signum, frame):
    raise CellTimeout("cell timed out")


def execute_cell(code: str, timeout_s: float, namespace: dict) -> dict:
    stdout = io.StringIO()
    stderr = io.StringIO()
    started = time.perf_counter()
    status = "ok"
    timed_out = False

    old_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace)
            answer = namespace.get("answer")
            if isinstance(answer, dict) and answer.get("ready"):
                print("__RLM_FINAL_ANSWER__" + json.dumps(str(answer.get("content", ""))))
    except CellTimeout:
        status = "timeout"
        timed_out = True
        stderr.write(f"Execution timed out after {timeout_s:g}s\n")
    except ToolError as exc:
        status = "tool_error"
        stderr.write(f"ToolError: {exc}\n")
    except BaseException:
        status = "error"
        stderr.write(traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

    return {
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "status": status,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "timed_out": timed_out,
    }


def main() -> int:
    bridge = RLMBridge()
    namespace = {
        "__name__": "__sandbox__",
        "rlm": bridge,
        "llm_query": bridge.llm_query,
        "llm_query_batched": lambda prompts, context=None, model=None, max_tokens=None: [
            bridge.llm_query(prompt, context=context, model=model, max_tokens=max_tokens)
            for prompt in prompts
        ],
        "rlm_query": bridge.completion,
        "rlm_query_batched": lambda prompts, context=None, model=None, max_tokens=None: [
            bridge.completion(prompt, context=context, model=model, max_tokens=max_tokens)
            for prompt in prompts
        ],
        "read_file": read_file,
        "read_first_existing": read_first_existing,
        "list_files": list_files,
        "project_overview": project_overview,
        "project_summary": project_summary,
        "project_audit": project_audit,
        "write_file": write_file,
        "apply_patch": apply_patch,
        "run_shell": run_shell,
        "git_status": git_status,
        "git_diff": git_diff,
        "git_log": git_log,
        "search_code": search_code,
        "tool_help": tool_help,
        "tool_names": tool_names,
    }
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            request_id = request["id"]
            code = request["code"]
            timeout_s = float(request.get("timeout_s", 60))
            if timeout_s <= 0:
                raise ValueError("timeout_s must be positive")
            result = execute_cell(code, timeout_s, namespace)
            result["id"] = request_id
            result["type"] = "execute_result"
        except BaseException as exc:
            result = {
                "id": None,
                "type": "execute_result",
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}\n",
                "status": "protocol_error",
                "elapsed_ms": 0,
                "timed_out": False,
            }
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
