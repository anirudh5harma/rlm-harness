from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rlm_harness.actions import ActionRisk


class ToolScope(str, Enum):
    WORKSPACE = "workspace"
    SHELL = "shell"
    GIT = "git"
    PROJECT = "project"
    CONTROL = "control"
    MEMORY = "memory"
    MCP = "mcp"
    RUNTIME = "runtime"


class SideEffect(str, Enum):
    NONE = "none"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    COMMAND = "command"
    GIT_READ = "git_read"
    MEMORY_WRITE = "memory_write"
    MCP_TOOL = "mcp_tool"
    COMPLETION = "completion"


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    action_kind: str
    summary: str
    scope: ToolScope
    risk: ActionRisk
    side_effect: SideEffect = SideEffect.NONE
    sandbox_required: bool = True
    requires_confirmation: bool = False
    timeout_s: Optional[float] = None
    parameters: dict[str, str] = Field(default_factory=dict)
    public: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_kind": self.action_kind,
            "summary": self.summary,
            "scope": self.scope.value,
            "risk": self.risk.value,
            "side_effect": self.side_effect.value,
            "sandbox_required": self.sandbox_required,
            "requires_confirmation": self.requires_confirmation,
            "timeout_s": self.timeout_s,
            "parameters": dict(self.parameters),
        }


class ToolRegistry:
    def __init__(self, descriptors: list[ToolDescriptor]):
        self._by_name: dict[str, ToolDescriptor] = {}
        self._by_action_kind: dict[str, ToolDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.name in self._by_name:
                raise ValueError(f"duplicate tool name: {descriptor.name}")
            if descriptor.action_kind in self._by_action_kind:
                raise ValueError(f"duplicate action kind: {descriptor.action_kind}")
            self._by_name[descriptor.name] = descriptor
            self._by_action_kind[descriptor.action_kind] = descriptor

    def all(self, *, include_internal: bool = False) -> list[ToolDescriptor]:
        descriptors = list(self._by_name.values())
        if not include_internal:
            descriptors = [descriptor for descriptor in descriptors if descriptor.public]
        # Sorted for cross-version stability; the model sees the
        # same tool list regardless of dict insertion order.
        descriptors.sort(key=lambda d: d.name)
        return descriptors

    def names(self, *, include_internal: bool = False) -> list[str]:
        return [descriptor.name for descriptor in self.all(include_internal=include_internal)]

    def get(self, name: str) -> ToolDescriptor:
        return self._by_name[name]

    def for_action_kind(self, action_kind: str) -> ToolDescriptor:
        return self._by_action_kind[action_kind]

    def payload(self, *, include_internal: bool = False) -> list[dict[str, Any]]:
        return [
            descriptor.to_public_dict()
            for descriptor in self.all(include_internal=include_internal)
        ]

    def public_payload(self) -> list[dict[str, Any]]:
        return self.payload()


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(DEFAULT_TOOL_DESCRIPTORS)


def render_tool_names(
    registry: Optional[ToolRegistry] = None,
    *,
    include_internal: bool = False,
) -> str:
    active = registry or default_tool_registry()
    return ", ".join(active.names(include_internal=include_internal))


def render_tool_catalog(
    registry: Optional[ToolRegistry] = None,
    *,
    include_internal: bool = False,
) -> str:
    active = registry or default_tool_registry()
    lines = ["Harness tools", ""]
    scopes = [
        ToolScope.WORKSPACE,
        ToolScope.PROJECT,
        ToolScope.GIT,
        ToolScope.SHELL,
        ToolScope.CONTROL,
        ToolScope.MEMORY,
        ToolScope.MCP,
        ToolScope.RUNTIME,
    ]
    for scope in scopes:
        descriptors = [
            tool
            for tool in active.all(include_internal=include_internal)
            if tool.scope == scope
        ]
        if not descriptors:
            continue
        lines.append(scope.value)
        for descriptor in descriptors:
            suffix = ""
            if descriptor.requires_confirmation:
                suffix = " (confirmation)"
            lines.append(f"  {descriptor.name} [{descriptor.risk.value}]{suffix}")
            lines.append(f"    {descriptor.summary}")
        lines.append("")
    return "\n".join(lines).rstrip()


DEFAULT_TOOL_DESCRIPTORS = [
    ToolDescriptor(
        name="read_file",
        action_kind="read_file",
        summary="Read a UTF-8 text file from the workspace.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.READ,
        side_effect=SideEffect.FILE_READ,
        parameters={"path": "workspace-relative path", "max_bytes": "optional byte cap"},
    ),
    ToolDescriptor(
        name="read_file_slice",
        action_kind="read_file_slice",
        summary="Read a bounded byte slice from a workspace text file.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.READ,
        side_effect=SideEffect.FILE_READ,
        parameters={
            "path": "workspace-relative path",
            "start": "optional zero-based byte offset",
            "max_bytes": "optional byte count",
        },
    ),
    ToolDescriptor(
        name="chunk_file",
        action_kind="chunk_file",
        summary="Split a workspace text file into bounded chunks.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.READ,
        side_effect=SideEffect.FILE_READ,
        parameters={
            "path": "workspace-relative path",
            "chunk_chars": "optional chunk size",
            "max_chunks": "optional chunk cap",
        },
    ),
    ToolDescriptor(
        name="read_first_existing",
        action_kind="read_first_existing",
        summary="Read the first existing file from a candidate path list.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.READ,
        side_effect=SideEffect.FILE_READ,
        parameters={"paths": "candidate paths", "max_bytes": "optional byte cap"},
    ),
    ToolDescriptor(
        name="list_files",
        action_kind="list_files",
        summary="List workspace files while skipping generated dependency directories.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.READ,
        parameters={
            "path": "optional workspace-relative root",
            "max_depth": "optional traversal depth",
            "max_count": "optional result cap",
        },
    ),
    ToolDescriptor(
        name="search_code",
        action_kind="search_code",
        summary="Search workspace text with ripgrep.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.READ,
        parameters={"pattern": "regex pattern", "path": "optional path", "max_count": "cap"},
    ),
    ToolDescriptor(
        name="write_file",
        action_kind="write_file",
        summary="Write UTF-8 text to a workspace file, creating parent directories.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.HIGH,
        side_effect=SideEffect.FILE_WRITE,
        requires_confirmation=True,
        parameters={"path": "workspace-relative path", "content": "new file content"},
    ),
    ToolDescriptor(
        name="apply_patch",
        action_kind="apply_patch",
        summary="Apply a unified diff to the workspace.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.MEDIUM,
        side_effect=SideEffect.FILE_WRITE,
        timeout_s=60.0,
        parameters={"diff": "unified diff"},
    ),
    ToolDescriptor(
        name="propose_file_change",
        action_kind="propose_file_change",
        summary="Queue a file change and return a diff for review before applying.",
        scope=ToolScope.WORKSPACE,
        risk=ActionRisk.MEDIUM,
        side_effect=SideEffect.FILE_WRITE,
        parameters={
            "path": "workspace-relative path",
            "content": "proposed full file content",
            "reason": "reason for proposal",
        },
    ),
    ToolDescriptor(
        name="project_overview",
        action_kind="project_overview",
        summary="Return files, common docs/config, git status, and recent commits.",
        scope=ToolScope.PROJECT,
        risk=ActionRisk.READ,
        parameters={"max_files": "optional file cap", "max_read_bytes": "per-file cap"},
    ),
    ToolDescriptor(
        name="project_summary",
        action_kind="project_summary",
        summary="Return a concise human-readable summary of the workspace project.",
        scope=ToolScope.PROJECT,
        risk=ActionRisk.READ,
        parameters={"max_files": "optional file cap", "max_read_bytes": "per-file cap"},
    ),
    ToolDescriptor(
        name="project_audit",
        action_kind="project_audit",
        summary="Return evidence-backed project risks, gaps, and next steps.",
        scope=ToolScope.PROJECT,
        risk=ActionRisk.READ,
        parameters={"max_files": "optional file cap", "max_read_bytes": "per-file cap"},
    ),
    ToolDescriptor(
        name="git_status",
        action_kind="git_status",
        summary="Return git status --short for the workspace.",
        scope=ToolScope.GIT,
        risk=ActionRisk.READ,
        side_effect=SideEffect.GIT_READ,
    ),
    ToolDescriptor(
        name="git_diff",
        action_kind="git_diff",
        summary="Return git diff, optionally scoped to one workspace path.",
        scope=ToolScope.GIT,
        risk=ActionRisk.READ,
        side_effect=SideEffect.GIT_READ,
        parameters={"path": "optional workspace-relative path"},
    ),
    ToolDescriptor(
        name="git_log",
        action_kind="git_log",
        summary="Return recent one-line git commits.",
        scope=ToolScope.GIT,
        risk=ActionRisk.READ,
        side_effect=SideEffect.GIT_READ,
        parameters={"n": "optional positive commit count"},
    ),
    ToolDescriptor(
        name="run_shell",
        action_kind="run_shell",
        summary="Run a shell command in the sandbox workspace.",
        scope=ToolScope.SHELL,
        risk=ActionRisk.MEDIUM,
        side_effect=SideEffect.COMMAND,
        timeout_s=60.0,
        parameters={"cmd": "shell command", "timeout": "optional seconds"},
    ),
    ToolDescriptor(
        name="list_pending_changes",
        action_kind="list_pending_changes",
        summary="List queued file-change proposals with diffs.",
        scope=ToolScope.CONTROL,
        risk=ActionRisk.READ,
    ),
    ToolDescriptor(
        name="apply_pending_change",
        action_kind="apply_pending_change",
        summary="Apply one queued file-change proposal after approval.",
        scope=ToolScope.CONTROL,
        risk=ActionRisk.HIGH,
        side_effect=SideEffect.FILE_WRITE,
        requires_confirmation=True,
        parameters={"change_id": "pending change id"},
    ),
    ToolDescriptor(
        name="clear_pending_changes",
        action_kind="clear_pending_changes",
        summary="Discard all queued file-change proposals.",
        scope=ToolScope.CONTROL,
        risk=ActionRisk.LOW,
    ),
    ToolDescriptor(
        name="complete_task",
        action_kind="complete_task",
        summary="Signal that the requested task is complete, partial, blocked, or failed.",
        scope=ToolScope.CONTROL,
        risk=ActionRisk.LOW,
        side_effect=SideEffect.COMPLETION,
        parameters={
            "summary": "user-facing summary",
            "status": "success, partial, blocked, or failed",
            "verification": "optional verification evidence",
        },
    ),
    ToolDescriptor(
        name="record_memory",
        action_kind="record_memory",
        summary="Record scoped user or project memory for future runs.",
        scope=ToolScope.MEMORY,
        risk=ActionRisk.LOW,
        side_effect=SideEffect.MEMORY_WRITE,
        sandbox_required=False,
        parameters={"content": "memory text", "scope": "user or project"},
    ),
    ToolDescriptor(
        name="mcp_list_tools",
        action_kind="mcp_list_tools",
        summary="List tools from a configured MCP server or designated purpose.",
        scope=ToolScope.MCP,
        risk=ActionRisk.READ,
        side_effect=SideEffect.NONE,
        sandbox_required=False,
        parameters={"server": "optional MCP server name", "purpose": "optional purpose label"},
    ),
    ToolDescriptor(
        name="mcp_call_tool",
        action_kind="mcp_call_tool",
        summary="Call a tool exposed by a configured MCP server.",
        scope=ToolScope.MCP,
        risk=ActionRisk.MEDIUM,
        side_effect=SideEffect.MCP_TOOL,
        sandbox_required=False,
        timeout_s=30.0,
        parameters={
            "server": "optional MCP server name",
            "purpose": "optional purpose label",
            "tool_name": "remote MCP tool name",
            "arguments": "tool arguments object",
        },
    ),
    ToolDescriptor(
        name="python_repl",
        action_kind="python_repl",
        summary="Internal compatibility action for executing Python cells in the sandbox.",
        scope=ToolScope.RUNTIME,
        risk=ActionRisk.MEDIUM,
        side_effect=SideEffect.COMMAND,
        public=False,
    ),
    ToolDescriptor(
        name="rlm_task",
        action_kind="rlm_task",
        summary="Internal compatibility action for recursive runtime execution.",
        scope=ToolScope.RUNTIME,
        risk=ActionRisk.MEDIUM,
        side_effect=SideEffect.COMMAND,
        public=False,
    ),
]
