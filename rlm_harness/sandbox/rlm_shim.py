from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Optional


class RLMCompletionError(RuntimeError):
    pass


class RLMBridge:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.__stdin__
        self.stdout = stdout or sys.__stdout__

    def completion(
        self,
        query: str,
        context: Any,
        depth_hint: int = -1,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        request_id = uuid.uuid4().hex
        request = {
            "type": "rlm_completion_request",
            "id": request_id,
            "query": query,
            "context": serialize_context(context),
            "depth_hint": depth_hint,
            "max_tokens": max_tokens,
        }
        self.stdout.write(json.dumps(request, sort_keys=True) + "\n")
        self.stdout.flush()

        for raw_line in self.stdin:
            response = json.loads(raw_line)
            if response.get("type") != "rlm_completion_response":
                continue
            if response.get("id") != request_id:
                continue
            if response.get("error"):
                raise RLMCompletionError(str(response["error"]))
            return str(response.get("content", ""))

        raise RLMCompletionError("host closed the RLM completion channel")


def serialize_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, sort_keys=True)
    except TypeError:
        return repr(context)
