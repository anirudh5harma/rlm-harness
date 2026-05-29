from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rlm_harness.actions.base import (
    AnyAction,
    AnyObservation,
    CompleteTaskAction,
    CompletionStatus,
    VerificationStatus,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_event_id() -> str:
    return f"evt_{uuid4().hex}"


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=new_event_id)
    run_id: str
    sequence: int
    kind: str
    created_at: datetime = Field(default_factory=utc_now)
    node: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RunStartedEvent(RunEvent):
    kind: Literal["run_started"] = "run_started"
    task: str
    workspace: str
    thread_id: str


class ContextBuiltEvent(RunEvent):
    kind: Literal["context_built"] = "context_built"
    sections: list[str] = Field(default_factory=list)
    token_estimate: int = 0


class PlanCreatedEvent(RunEvent):
    kind: Literal["plan_created"] = "plan_created"
    strategy: str = "sequential"
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ActionSelectedEvent(RunEvent):
    kind: Literal["action_selected"] = "action_selected"
    action: AnyAction


class AuthorizationEvent(RunEvent):
    kind: Literal["authorization"] = "authorization"
    action_id: str
    decision: Literal["approved", "denied", "needs_confirmation"]
    reason: str


class ObservationRecordedEvent(RunEvent):
    kind: Literal["observation_recorded"] = "observation_recorded"
    observation: AnyObservation


class VerificationEvent(RunEvent):
    kind: Literal["verification"] = "verification"
    status: VerificationStatus
    checks: list[dict[str, Any]] = Field(default_factory=list)


class CompletionEvent(RunEvent):
    kind: Literal["completion"] = "completion"
    status: CompletionStatus
    final_answer: str
    verification: Optional[str] = None

    @classmethod
    def from_action(
        cls,
        *,
        run_id: str,
        sequence: int,
        action: CompleteTaskAction,
        node: Optional[str] = None,
    ) -> CompletionEvent:
        return cls(
            run_id=run_id,
            sequence=sequence,
            node=node,
            status=action.status,
            final_answer=action.summary,
            verification=action.verification,
            payload={"action_id": action.action_id},
        )


AnyRunEvent = Annotated[
    Union[
        RunStartedEvent,
        ContextBuiltEvent,
        PlanCreatedEvent,
        ActionSelectedEvent,
        AuthorizationEvent,
        ObservationRecordedEvent,
        VerificationEvent,
        CompletionEvent,
    ],
    Field(discriminator="kind"),
]


def parse_event(payload: dict[str, Any]) -> AnyRunEvent:
    return TypeAdapter(AnyRunEvent).validate_python(payload)
