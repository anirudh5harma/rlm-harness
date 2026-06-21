from __future__ import annotations

import ast
import json

from rlm_harness.actions import (
    AnyAction,
    AnyObservation,
    CommandObservation,
    DataObservation,
    ErrorObservation,
    FileObservation,
    PatchObservation,
    PermissionObservation,
    TextObservation,
    parse_action,
)
from rlm_harness.kernel import AutonomyMode
from rlm_harness.sandbox import tools as sandbox_tools
from rlm_harness.tools import default_tool_registry
from rlm_harness.tools.authorization import authorize_tool_action


class ActionParseError(ValueError):
    pass


ToolActionParseError = ActionParseError
HOST_TYPED_TOOL_KINDS = {"mcp_list_tools", "mcp_call_tool"}


def parse_typed_tool_action(text: str) -> AnyAction:
    payload = parse_action_payload(text)
    if payload.get("type") == "tool" and isinstance(payload.get("action"), dict):
        payload = payload["action"]
    elif payload.get("type") == "tool" and isinstance(payload.get("name"), str):
        payload = {**payload, "kind": payload["name"]}
        payload.pop("type", None)
        payload.pop("name", None)
    elif "action_kind" in payload and "kind" not in payload:
        payload = {**payload, "kind": payload["action_kind"]}
        payload.pop("action_kind", None)
    elif "name" in payload and "kind" not in payload:
        payload = {**payload, "kind": payload["name"]}
        payload.pop("name", None)

    try:
        action = parse_action(payload)
    except Exception as exc:
        raise ActionParseError(f"tool action did not match a known schema: {exc}") from exc

    allowed_action_kinds = set(sandbox_tools.tool_names()) | HOST_TYPED_TOOL_KINDS
    if action.kind not in allowed_action_kinds:
        raise ActionParseError(f"tool action is not executable in this runtime: {action.kind}")
    return action


def parse_action_payload(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = parse_embedded_json_object(text, exc)

    if not isinstance(payload, dict):
        raise ActionParseError("action must be a JSON object")
    return payload


def parse_embedded_json_object(text: str, original: json.JSONDecodeError) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1]).strip()
            payload = parse_jsonish_mapping(inner)
            if payload is not None:
                return payload

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        payload = parse_jsonish_mapping(stripped[start : end + 1])
        if payload is not None:
            return payload
    raise ActionParseError("action was not valid JSON") from original


def parse_jsonish_mapping(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    return payload if isinstance(payload, dict) else None


def render_typed_observation(observation: AnyObservation) -> str:

    payload: dict[str, object] = {
        "status": legacy_status_from_observation(observation),
        "stdout": "",
        "stderr": "",
        "observation_kind": observation.kind,
        "observation_id": observation.observation_id,
        "action_id": observation.action_id,
    }
    if observation.summary:
        payload["summary"] = observation.summary

    if isinstance(observation, TextObservation):
        payload["stdout"] = observation.text
    elif isinstance(observation, FileObservation):
        payload["stdout"] = observation.content
        payload["path"] = observation.path
        payload["truncated"] = observation.truncated
    elif isinstance(observation, DataObservation):
        payload["stdout"] = json.dumps(observation.data, sort_keys=True, indent=2)
        payload["data"] = observation.data
    elif isinstance(observation, CommandObservation):
        payload["stdout"] = observation.stdout
        payload["stderr"] = observation.stderr
        payload["command"] = observation.command
        payload["returncode"] = observation.exit_code
        payload["elapsed_ms"] = observation.duration_ms or 0
    elif isinstance(observation, PatchObservation):
        patch_lines = []
        if observation.changed_files:
            patch_lines.append(
                "Changed files:\n"
                + "\n".join(f"- {path}" for path in observation.changed_files)
            )
        if observation.diff_summary:
            patch_lines.append(observation.diff_summary)
        payload["stdout"] = "\n".join(patch_lines)
        payload["changed_files"] = observation.changed_files
    elif isinstance(observation, PermissionObservation):
        payload["stderr"] = observation.reason
        payload["decision"] = observation.decision
    elif isinstance(observation, ErrorObservation):
        payload["stderr"] = observation.message
        payload["error_type"] = observation.error_type
        payload["recoverable"] = observation.recoverable

    return render_observation(payload)


def render_observation(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2)


def legacy_status_from_observation(observation: AnyObservation) -> str:
    if isinstance(observation, PermissionObservation):
        return "permission_denied"
    return observation.status.value


def executable_tool_payload(autonomy: AutonomyMode | str = AutonomyMode.SANDBOX) -> list[dict]:
    registry = default_tool_registry()
    executable_names = set(sandbox_tools.tool_names()) | HOST_TYPED_TOOL_KINDS
    payload = []
    for descriptor in registry.all():
        if descriptor.name not in executable_names:
            continue
        action = parse_action(default_action_payload(descriptor.action_kind))
        decision = authorize_tool_action(action, descriptor, autonomy)
        if decision.allowed:
            payload.append(descriptor.to_public_dict())
    return payload


def default_action_payload(action_kind: str) -> dict:
    payloads = {
        "read_file": {"kind": "read_file", "path": "."},
        "read_file_slice": {"kind": "read_file_slice", "path": "."},
        "chunk_file": {"kind": "chunk_file", "path": "."},
        "read_first_existing": {"kind": "read_first_existing", "paths": ["."]},
        "list_files": {"kind": "list_files"},
        "search_code": {"kind": "search_code", "pattern": "x"},
        "write_file": {"kind": "write_file", "path": ".", "content": ""},
        "apply_patch": {"kind": "apply_patch", "diff": ""},
        "run_shell": {"kind": "run_shell", "command": "true"},
        "git_status": {"kind": "git_status"},
        "git_diff": {"kind": "git_diff"},
        "git_log": {"kind": "git_log"},
        "project_overview": {"kind": "project_overview"},
        "project_summary": {"kind": "project_summary"},
        "project_audit": {"kind": "project_audit"},
        "plan_orientation": {"kind": "plan_orientation"},
        "propose_file_change": {
            "kind": "propose_file_change",
            "path": ".",
            "content": "",
            "reason": "proposal",
        },
        "list_pending_changes": {"kind": "list_pending_changes"},
        "apply_pending_change": {"kind": "apply_pending_change", "change_id": "pending"},
        "clear_pending_changes": {"kind": "clear_pending_changes"},
        "mcp_list_tools": {"kind": "mcp_list_tools", "server": "server-name"},
        "mcp_call_tool": {
            "kind": "mcp_call_tool",
            "server": "server-name",
            "tool_name": "tool",
            "arguments": {},
        },
        "complete_task": {"kind": "complete_task", "summary": "done"},
    }
    return payloads[action_kind]
