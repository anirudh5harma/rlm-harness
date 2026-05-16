"""Compatibility import path for the harness model client."""

from rlm_harness.model_client import LMClient, LMClientError
from rlm_harness.types import Completion, Msg

__all__ = ["Completion", "LMClient", "LMClientError", "Msg"]
