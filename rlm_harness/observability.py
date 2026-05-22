from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def langsmith_tracing_enabled() -> bool:
    value = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2")
    return str(value).lower() in {"1", "true", "yes", "on"}


def maybe_traceable(name: str, run_type: str = "chain", **metadata: Any):
    """Return LangSmith's traceable decorator when available, otherwise a no-op.

    Harness stays dependency-light and local-first: setting LANGSMITH_TRACING=true
    and installing the optional langsmith extra enables industry-standard traces;
    without both, runtime behavior is unchanged.
    """

    def decorator(func: Callable):
        if not langsmith_tracing_enabled():
            return func
        try:
            from langsmith.run_helpers import traceable
        except ImportError:
            return func
        kwargs: dict[str, Any] = {"name": name, "run_type": run_type}
        if metadata:
            kwargs["metadata"] = metadata
        return traceable(**kwargs)(func)

    return decorator
