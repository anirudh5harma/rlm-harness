from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rlm_harness.kernel.events import AnyRunEvent
from rlm_harness.types import ExecutionBudget, HarnessState, TaskPlan


class AutonomyMode(str, Enum):
    ASK = "ask"
    PLAN = "plan"
    PROPOSE = "propose"
    SANDBOX = "sandbox"
    TRUSTED = "trusted"


class RunPhase(str, Enum):
    NEW = "new"
    CONTEXT = "context"
    PLAN = "plan"
    SELECT_ACTION = "select_action"
    AUTHORIZE = "authorize"
    EXECUTE = "execute"
    OBSERVE = "observe"
    VERIFY = "verify"
    REFLECT = "reflect"
    LEARN = "learn"
    FINALIZE = "finalize"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    workspace: str
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
    autonomy: AutonomyMode = AutonomyMode.SANDBOX
    provider: Optional[str] = None
    model: Optional[str] = None
    resume_id: Optional[str] = None


class RunContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_map: str = ""
    selected_files: list[str] = Field(default_factory=list)
    memory_context: str = ""
    taste_context: str = ""
    capabilities: list[str] = Field(default_factory=list)
    token_budget: int = 100_000
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    request: RunRequest
    phase: RunPhase = RunPhase.NEW
    plan: TaskPlan = Field(default_factory=TaskPlan)
    context: RunContext = Field(default_factory=RunContext)
    events: list[AnyRunEvent] = Field(default_factory=list)
    event_cursor: int = 0
    changed_files: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    verification: Optional[dict[str, Any]] = None
    final_answer: Optional[str] = None
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    scratch: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_harness_state(cls, state: HarnessState) -> RunState:
        return cls(
            request=RunRequest(
                task=state.task,
                workspace=state.workspace,
                thread_id=state.thread_id,
                run_id=state.run_id,
            ),
            phase=run_phase_from_status(state.status),
            plan=state.plan,
            final_answer=state.final_answer,
            budget=state.budget,
            scratch=dict(state.scratch),
        )

    def to_harness_state(self) -> HarnessState:
        run_id = self.request.run_id or ""
        thread_id = self.request.thread_id or run_id
        return HarnessState(
            task=self.request.task,
            workspace=self.request.workspace,
            thread_id=thread_id,
            run_id=run_id,
            plan=self.plan,
            scratch=dict(self.scratch),
            budget=self.budget,
            status=harness_status_from_phase(self.phase),
            final_answer=self.final_answer,
        )


def run_phase_from_status(status: str) -> RunPhase:
    if status == "done":
        return RunPhase.DONE
    if status == "failed":
        return RunPhase.FAILED
    if status == "blocked":
        return RunPhase.BLOCKED
    return RunPhase.NEW


def harness_status_from_phase(phase: RunPhase) -> str:
    if phase == RunPhase.DONE:
        return "done"
    if phase == RunPhase.FAILED:
        return "failed"
    if phase == RunPhase.BLOCKED:
        return "blocked"
    return "new"
