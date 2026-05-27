from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


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


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    id: str
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    dependencies: list[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    result_summary: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3


class TaskPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    current_step_id: Optional[str] = None
    strategy: str = "sequential"

    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == PlanStepStatus.COMPLETED)

    def total_count(self) -> int:
        return len(self.steps)

    def has_more(self) -> bool:
        return any(s.status in (PlanStepStatus.PENDING, PlanStepStatus.IN_PROGRESS) for s in self.steps)

    @classmethod
    def from_flat_steps(cls, descriptions: list[str]) -> TaskPlan:
        """Convert a flat list of step descriptions into a TaskPlan."""
        steps = [
            PlanStep(id=str(i + 1), description=desc)
            for i, desc in enumerate(descriptions)
        ]
        for i in range(1, len(steps)):
            steps[i].dependencies.append(steps[i - 1].id)
        return cls(steps=steps, current_step_id=steps[0].id if steps else None)


class ExecutionBudget(BaseModel):
    token_limit: int = 100_000
    tokens_used: int = 0
    iteration_limit: int = 6
    iterations_used: int = 0

    @property
    def token_fraction(self) -> float:
        return self.tokens_used / self.token_limit if self.token_limit else 0.0

    @property
    def iteration_fraction(self) -> float:
        return self.iterations_used / self.iteration_limit if self.iteration_limit else 0.0

    @property
    def is_warning(self) -> bool:
        return self.token_fraction >= 0.8 or self.iteration_fraction >= 0.8

    @property
    def is_exhausted(self) -> bool:
        return self.iterations_used >= self.iteration_limit or self.tokens_used >= self.token_limit

    @property
    def progress_summary(self) -> str:
        return (
            f"iteration {self.iterations_used}/{self.iteration_limit}, "
            f"tokens {self.tokens_used}/{self.token_limit} ({self.token_fraction:.0%})"
        )


class HarnessState(BaseModel):
    task: str
    workspace: str
    thread_id: str
    run_id: str
    plan: TaskPlan = Field(default_factory=TaskPlan)
    history: list[dict[str, Any]] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)
    depth: int = 0
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    status: str = "new"
    final_answer: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_plan(cls, data: Any) -> Any:
        if isinstance(data, dict):
            plan_val = data.get("plan")
            if isinstance(plan_val, list):
                data["plan"] = TaskPlan.from_flat_steps(plan_val)
            budget_val = data.get("budget")
            token_val = data.get("token_budget")
            if budget_val is None and isinstance(token_val, int):
                data["budget"] = ExecutionBudget(token_limit=token_val)
        return data
