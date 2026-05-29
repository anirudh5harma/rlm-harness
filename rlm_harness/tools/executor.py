from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from rlm_harness.actions import (
    AnyAction,
    AnyObservation,
    ApplyPatchAction,
    ApplyPendingChangeAction,
    ChunkFileAction,
    ClearPendingChangesAction,
    CommandObservation,
    CompleteTaskAction,
    DataObservation,
    ErrorObservation,
    FileObservation,
    GitDiffAction,
    GitLogAction,
    GitStatusAction,
    ListFilesAction,
    ListPendingChangesAction,
    ObservationStatus,
    PatchObservation,
    PermissionObservation,
    ProjectAuditAction,
    ProjectOverviewAction,
    ProjectSummaryAction,
    ProposeChangeAction,
    ReadFileAction,
    ReadFileSliceAction,
    ReadFirstExistingAction,
    RunShellAction,
    SearchCodeAction,
    TextObservation,
    WriteFileAction,
)
from rlm_harness.kernel import AutonomyMode
from rlm_harness.sandbox import tools as sandbox_tools
from rlm_harness.tools.authorization import authorize_tool_action
from rlm_harness.tools.registry import ToolRegistry, default_tool_registry


class ToolExecutor:
    def __init__(
        self,
        workspace: Path,
        registry: Optional[ToolRegistry] = None,
        autonomy: AutonomyMode | str = AutonomyMode.SANDBOX,
    ):
        self.workspace = workspace.resolve()
        self.registry = registry or default_tool_registry()
        self.autonomy = autonomy

    def execute(self, action: AnyAction, *, approved: bool = False) -> AnyObservation:
        descriptor = self.registry.for_action_kind(action.kind)
        decision = authorize_tool_action(action, descriptor, self.autonomy)
        if not decision.allowed:
            return PermissionObservation(
                action_id=action.action_id,
                decision="denied",
                reason=decision.reason,
                summary=descriptor.summary,
            )
        if descriptor.requires_confirmation and not approved:
            return PermissionObservation(
                action_id=action.action_id,
                decision="needs_confirmation",
                reason=f"{descriptor.name} requires confirmation before execution.",
                summary=descriptor.summary,
            )

        started = time.monotonic()
        try:
            with workspace_context(self.workspace):
                return self._execute(action, elapsed_ms=lambda: elapsed_ms(started))
        except Exception as exc:
            return ErrorObservation(
                action_id=action.action_id,
                error_type=type(exc).__name__,
                message=str(exc),
                summary=f"{action.kind} failed",
            )

    def _execute(self, action: AnyAction, *, elapsed_ms) -> AnyObservation:
        if isinstance(action, ReadFileAction):
            return FileObservation(
                action_id=action.action_id,
                path=action.path,
                content=sandbox_tools.read_file(action.path, max_bytes=action.max_bytes),
            )
        if isinstance(action, ReadFileSliceAction):
            payload = sandbox_tools.read_file_slice(
                action.path,
                start=action.start,
                max_bytes=action.max_bytes,
            )
            return FileObservation(
                action_id=action.action_id,
                path=action.path,
                content=str(payload.get("content") or ""),
                truncated=bool(payload.get("truncated")),
                summary=slice_summary(payload),
            )
        if isinstance(action, ChunkFileAction):
            return DataObservation(
                action_id=action.action_id,
                data=sandbox_tools.chunk_file(
                    action.path,
                    chunk_chars=action.chunk_chars,
                    max_chunks=action.max_chunks,
                ),
                summary=f"chunked {action.path}",
            )
        if isinstance(action, ReadFirstExistingAction):
            payload = sandbox_tools.read_first_existing(
                action.paths,
                max_bytes=action.max_bytes,
            )
            return FileObservation(
                action_id=action.action_id,
                path=str(payload.get("path") or ""),
                content=str(payload.get("content") or ""),
                summary="read first existing file",
            )
        if isinstance(action, ListFilesAction):
            return DataObservation(
                action_id=action.action_id,
                data=sandbox_tools.list_files(
                    action.path,
                    max_depth=action.max_depth,
                    max_count=action.max_count,
                ),
                summary=f"listed files under {action.path}",
            )
        if isinstance(action, SearchCodeAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.search_code(
                    action.pattern,
                    path=action.path,
                    max_count=action.max_count,
                ),
                summary=f"searched {action.path}",
            )
        if isinstance(action, WriteFileAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.write_file(action.path, action.content),
                summary=f"wrote {action.path}",
            )
        if isinstance(action, ApplyPatchAction):
            return PatchObservation(
                action_id=action.action_id,
                diff_summary=sandbox_tools.apply_patch(
                    action.diff,
                    timeout=action.timeout_s,
                ),
                changed_files=changed_files_from_patch(action.diff),
                summary="applied patch",
            )
        if isinstance(action, RunShellAction):
            result = sandbox_tools.run_shell(action.command, timeout=action.timeout_s)
            timed_out = bool(result.get("timed_out"))
            returncode = int(result.get("returncode") or 0)
            return CommandObservation(
                action_id=action.action_id,
                command=action.command,
                exit_code=returncode,
                stdout=str(result.get("stdout") or ""),
                stderr=str(result.get("stderr") or ""),
                duration_ms=elapsed_ms(),
                status=command_status(returncode, timed_out),
                summary="command completed" if returncode == 0 else "command failed",
            )
        if isinstance(action, GitStatusAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.git_status(),
                summary="git status",
            )
        if isinstance(action, GitDiffAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.git_diff(action.path),
                summary="git diff",
            )
        if isinstance(action, GitLogAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.git_log(action.n),
                summary="git log",
            )
        if isinstance(action, ProjectOverviewAction):
            return DataObservation(
                action_id=action.action_id,
                data=sandbox_tools.project_overview(
                    max_files=action.max_files,
                    max_read_bytes=action.max_read_bytes,
                ),
                summary="project overview",
            )
        if isinstance(action, ProjectSummaryAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.project_summary(
                    max_files=action.max_files,
                    max_read_bytes=action.max_read_bytes,
                ),
                summary="project summary",
            )
        if isinstance(action, ProjectAuditAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.project_audit(
                    max_files=action.max_files,
                    max_read_bytes=action.max_read_bytes,
                ),
                summary="project audit",
            )
        if isinstance(action, ProposeChangeAction):
            return DataObservation(
                action_id=action.action_id,
                data=sandbox_tools.propose_file_change(
                    action.path,
                    action.content,
                    reason=action.reason,
                ),
                summary=f"proposed change for {action.path}",
            )
        if isinstance(action, ListPendingChangesAction):
            return DataObservation(
                action_id=action.action_id,
                data=sandbox_tools.list_pending_changes(),
                summary="listed pending changes",
            )
        if isinstance(action, ApplyPendingChangeAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.apply_pending_change(action.change_id),
                summary=f"applied pending change {action.change_id}",
            )
        if isinstance(action, ClearPendingChangesAction):
            return TextObservation(
                action_id=action.action_id,
                text=sandbox_tools.clear_pending_changes(),
                summary="cleared pending changes",
            )
        if isinstance(action, CompleteTaskAction):
            return TextObservation(
                action_id=action.action_id,
                text=action.summary,
                summary=action.status.value,
            )

        return ErrorObservation(
            action_id=action.action_id,
            error_type="UnsupportedAction",
            message=f"no executor for action kind: {action.kind}",
            recoverable=False,
        )


@contextmanager
def workspace_context(workspace: Path) -> Iterator[None]:
    previous = sandbox_tools.WORKSPACE
    sandbox_tools.WORKSPACE = workspace
    try:
        yield
    finally:
        sandbox_tools.WORKSPACE = previous


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def command_status(returncode: int, timed_out: bool) -> ObservationStatus:
    if timed_out:
        return ObservationStatus.TIMEOUT
    return ObservationStatus.OK if returncode == 0 else ObservationStatus.ERROR


def changed_files_from_patch(diff: str) -> list[str]:
    files = []
    seen = set()
    for line in diff.splitlines():
        path = ""
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/").strip()
        elif line.startswith("--- a/"):
            path = line.removeprefix("--- a/").strip()
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                path = parts[3].removeprefix("b/").strip()
        if not path or path == "/dev/null" or path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def slice_summary(payload: dict) -> str:
    start = payload.get("start", 0)
    end = payload.get("end", 0)
    return f"read bytes {start}-{end}"
