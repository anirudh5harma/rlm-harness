from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Msg(BaseModel):
    role: str
    content: str


class Completion(BaseModel):
    content: str
    model: str
    provider: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class HarnessState(BaseModel):
    task: str
    workspace: str
    thread_id: str
    run_id: str
    plan: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)
    depth: int = 0
    token_budget: int = 100000
    status: str = "new"
    final_answer: Optional[str] = None
