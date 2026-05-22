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

    def llm_query(
        self,
        prompt: str,
        context: Any = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._request_completion(
            request_type="llm_completion_request",
            response_type="llm_completion_response",
            query=prompt,
            context=context,
            depth_hint=-1,
            model=model,
            max_tokens=max_tokens,
        )

    def completion(
        self,
        query: str,
        context: Any,
        depth_hint: int = -1,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._request_completion(
            request_type="rlm_completion_request",
            response_type="rlm_completion_response",
            query=query,
            context=context,
            depth_hint=depth_hint,
            model=model,
            max_tokens=max_tokens,
        )

    def _request_completion(
        self,
        request_type: str,
        response_type: str,
        query: str,
        context: Any,
        depth_hint: int,
        model: Optional[str],
        max_tokens: Optional[int],
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        request_id = uuid.uuid4().hex
        request = {
            "type": request_type,
            "id": request_id,
            "query": query,
            "context": serialize_context(context),
            "depth_hint": depth_hint,
            "model": model,
            "max_tokens": max_tokens,
        }
        self.stdout.write(json.dumps(request, sort_keys=True) + "\n")
        self.stdout.flush()

        for raw_line in self.stdin:
            response = json.loads(raw_line)
            if response.get("type") != response_type:
                continue
            if response.get("id") != request_id:
                continue
            if response.get("error"):
                raise RLMCompletionError(str(response["error"]))
            return str(response.get("content", ""))

        raise RLMCompletionError("host closed the completion channel")


def serialize_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, sort_keys=True)
    except TypeError:
        return repr(context)
