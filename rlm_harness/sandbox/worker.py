from __future__ import annotations

import contextlib
import io
import json
import signal
import sys
import time
import traceback


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
    except CellTimeout:
        status = "timeout"
        timed_out = True
        stderr.write(f"Execution timed out after {timeout_s:g}s\n")
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
    namespace = {"__name__": "__sandbox__"}
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
        except BaseException as exc:
            result = {
                "id": None,
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
