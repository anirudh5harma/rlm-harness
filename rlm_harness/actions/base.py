from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ActionRisk(str, Enum):
    READ = "read"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"


class ActionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ObservationStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"
    UNVERIFIED = "unverified"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


class CompletionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class BaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(default_factory=lambda: new_id("act"))
    kind: str
    created_at: datetime = Field(default_factory=utc_now)
    risk: ActionRisk = ActionRisk.LOW
    requires_confirmation: bool = False
    reason: Optional[str] = None


class ReadFileAction(BaseAction):
    kind: Literal["read_file"] = "read_file"
    path: str
    max_bytes: int = 64_000
    risk: ActionRisk = ActionRisk.READ


class ReadFileSliceAction(BaseAction):
    kind: Literal["read_file_slice"] = "read_file_slice"
    path: str
    start: int = 0
    max_bytes: int = 16_000
    risk: ActionRisk = ActionRisk.READ


class ChunkFileAction(BaseAction):
    kind: Literal["chunk_file"] = "chunk_file"
    path: str
    chunk_chars: int = 8_000
    max_chunks: int = 12
    risk: ActionRisk = ActionRisk.READ


class ReadFirstExistingAction(BaseAction):
    kind: Literal["read_first_existing"] = "read_first_existing"
    paths: list[str]
    max_bytes: int = 64_000
    risk: ActionRisk = ActionRisk.READ


class ListFilesAction(BaseAction):
    kind: Literal["list_files"] = "list_files"
    path: str = "."
    max_depth: int = 4
    max_count: int = 300
    risk: ActionRisk = ActionRisk.READ


class SearchCodeAction(BaseAction):
    kind: Literal["search_code"] = "search_code"
    pattern: str
    path: str = "."
    max_count: int = 100
    risk: ActionRisk = ActionRisk.READ


class ApplyPatchAction(BaseAction):
    kind: Literal["apply_patch"] = "apply_patch"
    diff: str
    timeout_s: float = 60.0
    risk: ActionRisk = ActionRisk.MEDIUM


class WriteFileAction(BaseAction):
    kind: Literal["write_file"] = "write_file"
    path: str
    content: str
    risk: ActionRisk = ActionRisk.HIGH
    requires_confirmation: bool = True


class RunShellAction(BaseAction):
    kind: Literal["run_shell"] = "run_shell"
    command: str
    timeout_s: float = 60.0
    risk: ActionRisk = ActionRisk.MEDIUM


class PythonReplAction(BaseAction):
    kind: Literal["python_repl"] = "python_repl"
    code: str
    timeout_s: float = 60.0
    risk: ActionRisk = ActionRisk.MEDIUM


class RLMTaskAction(BaseAction):
    kind: Literal["rlm_task"] = "rlm_task"
    task: str
    engine: Literal["rlm"] = "rlm"
    risk: ActionRisk = ActionRisk.MEDIUM


class GitStatusAction(BaseAction):
    kind: Literal["git_status"] = "git_status"
    risk: ActionRisk = ActionRisk.READ


class GitDiffAction(BaseAction):
    kind: Literal["git_diff"] = "git_diff"
    path: Optional[str] = None
    risk: ActionRisk = ActionRisk.READ


class GitLogAction(BaseAction):
    kind: Literal["git_log"] = "git_log"
    n: int = 10
    risk: ActionRisk = ActionRisk.READ


class ProjectOverviewAction(BaseAction):
    kind: Literal["project_overview"] = "project_overview"
    max_files: int = 300
    max_read_bytes: int = 64_000
    risk: ActionRisk = ActionRisk.READ


class ProjectSummaryAction(BaseAction):
    kind: Literal["project_summary"] = "project_summary"
    max_files: int = 300
    max_read_bytes: int = 64_000
    risk: ActionRisk = ActionRisk.READ


class ProjectAuditAction(BaseAction):
    kind: Literal["project_audit"] = "project_audit"
    max_files: int = 300
    max_read_bytes: int = 64_000
    risk: ActionRisk = ActionRisk.READ


class PlanOrientationAction(BaseAction):
    kind: Literal["plan_orientation"] = "plan_orientation"
    risk: ActionRisk = ActionRisk.READ


class ProposeChangeAction(BaseAction):
    kind: Literal["propose_file_change"] = "propose_file_change"
    path: str
    content: str
    reason: str
    risk: ActionRisk = ActionRisk.MEDIUM


class ListPendingChangesAction(BaseAction):
    kind: Literal["list_pending_changes"] = "list_pending_changes"
    risk: ActionRisk = ActionRisk.READ


class ApplyPendingChangeAction(BaseAction):
    kind: Literal["apply_pending_change"] = "apply_pending_change"
    change_id: str
    risk: ActionRisk = ActionRisk.HIGH
    requires_confirmation: bool = True


class ClearPendingChangesAction(BaseAction):
    kind: Literal["clear_pending_changes"] = "clear_pending_changes"
    risk: ActionRisk = ActionRisk.LOW


class RecordMemoryAction(BaseAction):
    kind: Literal["record_memory"] = "record_memory"
    content: str
    scope: Literal["user", "project"] = "project"
    source_run_id: Optional[str] = None
    risk: ActionRisk = ActionRisk.LOW


class MCPListToolsAction(BaseAction):
    kind: Literal["mcp_list_tools"] = "mcp_list_tools"
    server: Optional[str] = None
    purpose: Optional[str] = None
    timeout_s: float = 30.0
    risk: ActionRisk = ActionRisk.READ


class MCPCallToolAction(BaseAction):
    kind: Literal["mcp_call_tool"] = "mcp_call_tool"
    server: Optional[str] = None
    purpose: Optional[str] = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 30.0
    risk: ActionRisk = ActionRisk.MEDIUM


class CompleteTaskAction(BaseAction):
    kind: Literal["complete_task"] = "complete_task"
    summary: str
    status: CompletionStatus = CompletionStatus.SUCCESS
    verification: Optional[str] = None
    risk: ActionRisk = ActionRisk.LOW


AnyAction = Annotated[
    Union[
        ReadFileAction,
        ReadFileSliceAction,
        ChunkFileAction,
        ReadFirstExistingAction,
        ListFilesAction,
        SearchCodeAction,
        ApplyPatchAction,
        WriteFileAction,
        RunShellAction,
        PythonReplAction,
        RLMTaskAction,
        GitStatusAction,
        GitDiffAction,
        GitLogAction,
        ProjectOverviewAction,
        ProjectSummaryAction,
        ProjectAuditAction,
        PlanOrientationAction,
        ProposeChangeAction,
        ListPendingChangesAction,
        ApplyPendingChangeAction,
        ClearPendingChangesAction,
        RecordMemoryAction,
        MCPListToolsAction,
        MCPCallToolAction,
        CompleteTaskAction,
    ],
    Field(discriminator="kind"),
]


class BaseObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(default_factory=lambda: new_id("obs"))
    action_id: Optional[str] = None
    kind: str
    created_at: datetime = Field(default_factory=utc_now)
    status: ObservationStatus = ObservationStatus.OK
    summary: Optional[str] = None


class TextObservation(BaseObservation):
    kind: Literal["text"] = "text"
    text: str


class DataObservation(BaseObservation):
    kind: Literal["data"] = "data"
    data: Any


class FileObservation(BaseObservation):
    kind: Literal["file"] = "file"
    path: str
    content: str
    truncated: bool = False


class CommandObservation(BaseObservation):
    kind: Literal["command"] = "command"
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: Optional[int] = None


class PatchObservation(BaseObservation):
    kind: Literal["patch"] = "patch"
    changed_files: list[str] = Field(default_factory=list)
    diff_summary: Optional[str] = None


class VerificationObservation(BaseObservation):
    kind: Literal["verification"] = "verification"
    verification_status: VerificationStatus
    checks: list[dict[str, Any]] = Field(default_factory=list)


class PermissionObservation(BaseObservation):
    kind: Literal["permission"] = "permission"
    status: ObservationStatus = ObservationStatus.DENIED
    decision: Literal["approved", "denied", "needs_confirmation"]
    reason: str


class ErrorObservation(BaseObservation):
    kind: Literal["error"] = "error"
    status: ObservationStatus = ObservationStatus.ERROR
    error_type: str
    message: str
    recoverable: bool = True


AnyObservation = Annotated[
    Union[
        TextObservation,
        DataObservation,
        FileObservation,
        CommandObservation,
        PatchObservation,
        VerificationObservation,
        PermissionObservation,
        ErrorObservation,
    ],
    Field(discriminator="kind"),
]


def parse_action(payload: dict[str, Any]) -> AnyAction:
    return TypeAdapter(AnyAction).validate_python(payload)


def parse_observation(payload: dict[str, Any]) -> AnyObservation:
    return TypeAdapter(AnyObservation).validate_python(payload)
